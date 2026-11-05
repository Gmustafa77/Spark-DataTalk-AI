"""Deterministic entity resolution for free-text names in questions (no LLM calls).

Why this exists
---------------
High-cardinality columns such as ``Provider_Name`` are not part of the grounding
values sent to the planner / SQL agent (``core.data.VALUE_CONTEXT_FIELDS``). An
exact name ("Sunrise Hospital") works because the model copies it verbatim into
the SQL, but partial names, aliases, abbreviations or several names in one
question ("Sunrise or Sun") produce ``= 'Sun'`` filters that match nothing, or an
``INVALID_FILTER`` plan.

This module turns the mentions in a question into the exact database values,
before any AI call:

    question ─▶ extract mentions ─▶ resolve each mention against the distinct
              names (exact → alias/acronym → token-prefix → substring → fuzzy)
              ─▶ grounding block added to ``available_values`` for THIS question

Backward compatibility
----------------------
* :meth:`EntityResolver.augment` returns the original ``available_values``
  object untouched when nothing is found **or** when every mention is already an
  exact full name. Those questions therefore send byte-identical payloads, keep
  their LLM-memo keys and behave exactly as before.
* Nothing in the planner, SQL agent, SQL guard, insight composer, charts, QA
  store or UI changes.

Performance
-----------
The index is built once per loaded dataset (distinct values only) and reused.
Exact/alias/prefix lookups are dictionary + ``bisect`` operations; substring and
fuzzy matching run only for mentions the cheaper stages did not resolve, and use
``rapidfuzz`` (C++) when installed, with a candidate-pool ``difflib`` fallback.

No Streamlit imports.
"""
from __future__ import annotations

import bisect
import logging
import re
import threading
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

try:  # optional, much faster fuzzy matching
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover
    fuzz = process = None  # type: ignore[assignment]

import difflib

log = logging.getLogger("datatalk.entity_resolver")

# ═══════════════════════════════════════════════════════════════════════════════
# Vocabulary
# ═══════════════════════════════════════════════════════════════════════════════
# Words that describe an organisation type. They never identify a provider on
# their own ("hospital" must not match every hospital).
GENERIC_NAME_TOKENS = frozenset({
    "hospital", "hospitals", "clinic", "clinics", "medical", "center", "centre", "centers", "health",
    "healthcare", "care", "group", "associates", "assoc", "services", "service", "system", "systems",
    "partners", "practice", "physicians", "physician", "inc", "llc", "ltd", "pc", "pllc", "pa", "corp",
    "co", "company", "the", "of", "and", "at", "for", "regional", "general", "memorial", "community",
    "network", "md", "dr", "st", "saint", "llp", "lp", "plc",
})
_LEGAL_SUFFIX = frozenset({"inc", "llc", "ltd", "pc", "pllc", "corp", "co", "llp", "lp", "plc", "pa"})

# Words in a question that are never part of a provider mention.
STOP_TOKENS = frozenset({
    # analytics / domain
    "show", "me", "give", "list", "what", "which", "who", "is", "are", "was", "were", "the", "a", "an",
    "average", "avg", "mean", "median", "total", "count", "number", "rate", "rates", "tat", "turnaround",
    "time", "provider", "providers", "appeal", "appeals", "case", "cases", "overturn", "overturned",
    "overturns", "volume", "compare", "comparison", "versus", "vs", "trend", "trends", "monthly",
    "weekly", "daily", "yearly", "by", "per", "each", "all", "top", "bottom", "highest", "lowest",
    "best", "worst", "of", "for", "and", "or", "with", "in", "on", "at", "to", "from", "please",
    "containing", "contains", "contain", "named", "called", "like", "matching", "match", "starting",
    "start", "starts", "beginning", "begins", "name", "names", "their", "its", "them", "those",
    "compliance", "denial", "denials", "upheld", "closed", "open", "pending", "status", "sla",
    "days", "day", "week", "weeks", "month", "months", "quarter", "quarters", "year", "years",
    "last", "past", "previous", "prior", "this", "current", "next", "ytd", "mtd", "qtd", "since",
    "between", "during", "over", "today", "yesterday",
})

# A mention segment starts after one of these cues …
_CUE_RE = re.compile(
    r"\b(?:of|for|from|at|about|regarding|between|containing|contains|named|called|like|matching|"
    r"starting\s+with|begin(?:ning|s)?\s+with|providers?)\b\s+",
    re.I,
)
# … and stops at one of these (time scope, grouping, another clause).
_STOP_RE = re.compile(
    r"\s+(?:by|per|in|during|over|since|last|past|previous|this|grouped|where|when|which|that|"
    r"broken|split|and\s+(?:show|what|how|which|also|then)|compared)\b"
    r"|[?.;:!()\n]",
    re.I,
)
_SPLIT_RE = re.compile(r"\s*(?:,|/|&|\+|\bor\b|\band\b|\bvs\.?\b|\bversus\b)\s*", re.I)
_QUOTED_RE = re.compile(r"[\"'\u2018\u2019\u201c\u201d`]([^\"'\u2018\u2019\u201c\u201d`]{2,80})[\"'\u2018\u2019\u201c\u201d`]")


def normalize(text: Any) -> str:
    """Case-folded, accent-insensitive, punctuation-free form used for matching."""
    t = unicodedata.normalize("NFKD", str(text or ""))
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).casefold()
    t = t.replace("&", " and ").replace("'", "")
    t = re.sub(r"[^0-9a-z]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _tokens(norm: str) -> list[str]:
    return norm.split() if norm else []


def _sql_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration + results
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ResolverConfig:
    column: str = "Provider_Name"              # raw SQL column the names come from
    max_values_per_mention: int = 25           # above this the SQL hint switches to LIKE
    max_values_total: int = 50                 # hard cap on values added to the payload
    fuzzy_cutoff: float = 88.0                 # 0–100; typo tolerance (rapidfuzz WRatio)
    min_partial_len: int = 3                   # shortest partial term that may match ("Sun")
    aliases: dict = field(default_factory=dict)  # {"SRH": "Sunrise Regional Hospital", "Sun": [..]}


@dataclass
class MentionMatch:
    text: str                 # the mention as written by the user
    match_type: str           # exact | alias | acronym | partial | substring | fuzzy
    values: list[str]         # exact database values
    total_matches: int        # before capping
    like_pattern: str = ""    # set when the mention is partial (used for very wide matches)

    @property
    def truncated(self) -> bool:
        return self.total_matches > len(self.values)


# ═══════════════════════════════════════════════════════════════════════════════
# Resolver
# ═══════════════════════════════════════════════════════════════════════════════
class EntityResolver:
    """Index of the distinct values of one column, plus question-level resolution."""

    def __init__(self, names: Iterable[Any], cfg: ResolverConfig | None = None) -> None:
        self.cfg = cfg or ResolverConfig()
        uniq = sorted({str(n).strip() for n in names if n is not None and str(n).strip()
                       and str(n).strip().lower() not in {"nan", "none", "null", "unknown"}})
        self.names: list[str] = uniq
        self.norm: list[str] = [normalize(n) for n in uniq]
        self._exact: dict[str, list[int]] = {}
        self._token_ids: dict[str, set[int]] = {}
        self._acronym: dict[str, list[int]] = {}
        for i, nm in enumerate(self.norm):
            self._exact.setdefault(nm, []).append(i)
            stripped = " ".join(t for t in _tokens(nm) if t not in _LEGAL_SUFFIX)
            if stripped and stripped != nm:
                self._exact.setdefault(stripped, []).append(i)
            toks = _tokens(nm)
            for t in toks:
                self._token_ids.setdefault(t, set()).add(i)
            acr = "".join(t[0] for t in toks if t not in _LEGAL_SUFFIX and not t.isdigit())
            if len(acr) >= 2:
                self._acronym.setdefault(acr, []).append(i)
        self._sorted_tokens: list[str] = sorted(self._token_ids)
        self._alias: dict[str, list[int]] = self._build_aliases(self.cfg.aliases)
        log.info("entity resolver ready: column=%s distinct=%d tokens=%d",
                 self.cfg.column, len(self.names), len(self._sorted_tokens))

    # ── construction helpers ────────────────────────────────────────────────────
    @classmethod
    def from_frame(cls, df: pd.DataFrame, cfg: ResolverConfig | None = None) -> EntityResolver | None:
        cfg = cfg or ResolverConfig()
        if df is None or cfg.column not in df.columns:
            return None
        return cls(df[cfg.column].dropna().unique().tolist(), cfg)

    def _build_aliases(self, aliases: Mapping) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for alias, targets in (aliases or {}).items():
            targets = [targets] if isinstance(targets, str) else list(targets or [])
            ids = sorted({i for t in targets for i in self._exact.get(normalize(t), [])})
            if ids:
                out[normalize(alias)] = ids
        return out

    # ── single-term resolution ──────────────────────────────────────────────────
    def _prefix_ids(self, token: str) -> set[int]:
        lo = bisect.bisect_left(self._sorted_tokens, token)
        ids: set[int] = set()
        for t in self._sorted_tokens[lo:]:
            if not t.startswith(token):
                break
            ids |= self._token_ids[t]
        return ids

    def resolve_term(self, raw: str) -> tuple[str, list[int]]:
        """Return ``(match_type, name_ids)`` for one mention; ``("", [])`` when unresolved."""
        term = normalize(raw)
        if not term:
            return "", []
        if term in self._exact:
            return "exact", list(self._exact[term])
        if term in self._alias:
            return "alias", list(self._alias[term])

        toks = [t for t in _tokens(term) if t not in STOP_TOKENS]
        significant = [t for t in toks if t not in GENERIC_NAME_TOKENS]
        if not significant:
            return "", []  # "hospital", "the clinic" … never match on their own
        if len(toks) == 1 and toks[0] in self._acronym and raw.strip().isupper():
            return "acronym", list(self._acronym[toks[0]])

        # Token-prefix containment: every significant token is a prefix of some name token.
        if all(len(t) >= self.cfg.min_partial_len for t in significant):
            ids: set[int] | None = None
            for t in significant:
                hit = self._prefix_ids(t)
                ids = hit if ids is None else ids & hit
                if not ids:
                    break
            if ids:
                # Keep generic words the user typed ("Sun Hospital" ≠ "Sun Clinic").
                generic = [t for t in toks if t in GENERIC_NAME_TOKENS]
                if generic:
                    narrowed = {i for i in ids if all(g in _tokens(self.norm[i]) for g in generic)}
                    ids = narrowed or ids
                return "partial", sorted(ids)

        # Substring inside a token ("rise" → "Sunrise …").
        joined = " ".join(significant)
        if len(joined) >= max(4, self.cfg.min_partial_len):
            ids = {i for i, nm in enumerate(self.norm) if joined in nm}
            if ids:
                return "substring", sorted(ids)

        # Typo tolerance.
        return self._fuzzy(" ".join(toks))

    def _fuzzy(self, term: str) -> tuple[str, list[int]]:
        if len(term) < 4 or not self.norm:
            return "", []
        cap = self.cfg.max_values_per_mention
        if process is not None:
            hits = process.extract(term, self.norm, scorer=fuzz.WRatio,
                                   score_cutoff=self.cfg.fuzzy_cutoff, limit=cap)
            # also token-level typo ("sunrize" vs token "sunrise")
            tok_hits = process.extract(term, self._sorted_tokens, scorer=fuzz.ratio,
                                       score_cutoff=self.cfg.fuzzy_cutoff, limit=5) if " " not in term else []
            ids = {h[2] for h in hits}
            for tok, _score, _ in tok_hits:
                if tok not in GENERIC_NAME_TOKENS:
                    ids |= self._token_ids[tok]
            return ("fuzzy", sorted(ids)) if ids else ("", [])
        # difflib fallback on a candidate pool sharing a 2-char token prefix
        pool = sorted({i for t in term.split() if len(t) >= 2 for i in self._prefix_ids(t[:2])})
        cands = [self.norm[i] for i in pool] or self.norm[:5000]
        close = difflib.get_close_matches(term, cands, n=cap, cutoff=self.cfg.fuzzy_cutoff / 100)
        ids = sorted({i for c in close for i in self._exact.get(c, [])})
        return ("fuzzy", ids) if ids else ("", [])

    # ── question-level resolution ───────────────────────────────────────────────
    @staticmethod
    def extract_mentions(question: str) -> list[str]:
        """Candidate entity mentions: quoted strings + cue-introduced name lists."""
        q = " ".join(str(question or "").split())
        found: list[str] = [m.group(1).strip() for m in _QUOTED_RE.finditer(q)]
        unquoted = _QUOTED_RE.sub(lambda m: f" {m.group(1)} ", q)
        for cue in _CUE_RE.finditer(unquoted):
            tail = unquoted[cue.end():]
            stop = _STOP_RE.search(tail)
            segment = tail[: stop.start()] if stop else tail
            for part in _SPLIT_RE.split(segment):
                words = part.strip(" -").split()
                while words and words[0].lower() in STOP_TOKENS:
                    words.pop(0)  # "providers containing Sun" → "Sun"
                while words and words[-1].lower() in STOP_TOKENS:
                    words.pop()
                if words:
                    found.append(" ".join(words))
        out, seen = [], set()
        for m in found:
            k = normalize(m)
            if k and k not in seen:
                seen.add(k)
                out.append(m)
        return out[:12]

    def resolve(self, question: str) -> list[MentionMatch]:
        cfg = self.cfg
        matches: list[MentionMatch] = []
        mentions = self.extract_mentions(question)
        for mention in mentions:
            key = normalize(mention)
            kind, ids = self.resolve_term(mention)
            if not ids:
                continue
            values = [self.names[i] for i in ids]
            like = ""
            if kind in {"partial", "substring"}:
                like = "%" + " ".join(t for t in _tokens(key) if t not in STOP_TOKENS).upper() + "%"
            matches.append(MentionMatch(mention, kind, values[: cfg.max_values_per_mention],
                                        len(values), like))
        return matches

    def augment(self, question: str, available_values: dict) -> dict:
        """Return ``available_values`` plus resolved names for this question.

        Returns the **same object, unchanged** when nothing resolves or when every
        mention is already an exact full name (existing behaviour preserved).
        """
        try:
            matches = self.resolve(question)
        except Exception:  # resolution must never break answering
            log.exception("entity resolution failed; continuing without it")
            return available_values
        if not matches or all(m.match_type == "exact" and len(m.values) == 1 for m in matches):
            return available_values

        col = self.cfg.column
        values = list(dict.fromkeys(v for m in matches for v in m.values))[: self.cfg.max_values_total]
        mentions = []
        for m in matches:
            item: dict[str, Any] = {"mention": m.text, "match_type": m.match_type,
                                    "resolved_values": m.values, "total_matches": m.total_matches}
            if m.truncated and m.like_pattern:
                item["sql_condition"] = f"upper(trim(`{col}`)) LIKE '{_sql_literal(m.like_pattern)}'"
            mentions.append(item)

        out = dict(available_values or {})
        out[col] = values
        out["entity_resolution"] = {
            "column": col,
            "mentions": mentions,
            "instructions": (
                f"The user referred to {col} values by partial name, alias, abbreviation or several names. "
                f"They were resolved deterministically to the exact database values listed above; these "
                f"values ARE valid filter values (do not mark the question INVALID_FILTER). Filter with "
                f"`{col}` IN (...) using every resolved value of every mention, verbatim. When a mention "
                f"has `sql_condition`, use that condition (OR-ed with the IN list) instead of listing values. "
                f"When more than one value is resolved, GROUP BY `{col}` so each provider is reported "
                f"separately, unless the user asked for one combined figure."
            ),
        }
        log.info("entity resolution: %d mention(s) → %d value(s) [%s]", len(matches), len(values),
                 ", ".join(f"{m.text}:{m.match_type}:{m.total_matches}" for m in matches))
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Process-wide cache (one index per loaded dataset)
# ═══════════════════════════════════════════════════════════════════════════════
class EntityResolverCache:
    """Builds the resolver lazily and rebuilds it only when the dataset changes."""

    def __init__(self, cfg: ResolverConfig | None = None) -> None:
        self.cfg = cfg or ResolverConfig()
        self._lock = threading.Lock()
        self._key: tuple | None = None
        self._resolver: EntityResolver | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> EntityResolverCache:
        return cls(ResolverConfig(
            column=getattr(settings, "entity_resolution_column", "Provider_Name") or "Provider_Name",
            fuzzy_cutoff=float(getattr(settings, "entity_resolution_fuzzy_cutoff", 88.0) or 88.0),
            max_values_per_mention=int(getattr(settings, "entity_resolution_max_values", 25) or 25),
            aliases=dict(getattr(settings, "entity_aliases", {}) or {}),
        ))

    def get(self, df: pd.DataFrame | None) -> EntityResolver | None:
        if df is None or self.cfg.column not in getattr(df, "columns", []):
            return None
        key = (id(df), len(df))
        with self._lock:
            if key != self._key:
                self._resolver = EntityResolver.from_frame(df, self.cfg)
                self._key = key
            return self._resolver

    def augment(self, question: str, df: pd.DataFrame | None, available_values: dict) -> dict:
        try:
            resolver = self.get(df)
        except Exception:
            log.exception("entity resolver could not be built; continuing without it")
            return available_values
        return resolver.augment(question, available_values) if resolver else available_values


__all__ = ["EntityResolver", "EntityResolverCache", "ResolverConfig", "MentionMatch", "normalize"]
