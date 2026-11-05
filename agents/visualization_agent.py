
"""Governed chart builder, preserving the original public entry points.

metric_metadata example:
 {"rate": {"unit": "fraction", "aggregation": "weighted_rate",
           "numerator": "met", "denominator": "eligible"},
  "cases": {"unit": "count", "aggregation": "sum", "additive": True}}
Unknown units remain raw. No aggregation is inferred. Upstream result_hash is
provenance unless ChartConfig.result_hash_validator(df, hash) is supplied.
"""
from __future__ import annotations
import hashlib
import logging
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
import numpy as np
import pandas as pd
import plotly.graph_objects as go

log = logging.getLogger("datatalk.visualization")
PALETTE = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#06b6d4"]
_SYNONYMS = {"tat": "turnaround", "turnaround": "tat", "overturns": "overturn", "volume": "count"}
_GENERIC = {"avg", "average", "total", "count", "number", "rate", "of", "the"}
_COUNT_WORDS = {"count", "cases", "case", "volume", "total", "number", "num", "appeals", "records", "claims"}
_RATE_AVG_WORDS = {"rate", "percent", "percentage", "pct", "share", "avg", "average", "mean", "median", "ratio", "score"}
_PERIOD = re.compile(r"^(?:\d{4}|\d{4}-(?:0[1-9]|1[0-2])|\d{4}-?Q[1-4])$")

@dataclass
class ChartConfig:
    max_charts_per_question: int = 2
    max_bars: int = 15
    max_pie_slices: int = 6
    min_periods_for_trend: int = 3
    style_fn: Callable | None = None
    color_sequence: list = field(default_factory=lambda: list(PALETTE))
    max_series: int = 6
    max_heatmap_cells: int = 400
    max_points: int = 20000
    result_hash_validator: Callable | None = None

    def __post_init__(self):
        for name in ("max_charts_per_question", "max_bars", "max_pie_slices", "min_periods_for_trend", "max_series", "max_heatmap_cells", "max_points"):
            v = getattr(self, name)
            minimum = 0 if name == "max_charts_per_question" else 1
            if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not self.color_sequence: raise ValueError("color_sequence must not be empty")
        self.color_sequence = list(self.color_sequence)
        for color in self.color_sequence: go.Scatter(marker_color=color)
        for name in ("style_fn", "result_hash_validator"):
            if getattr(self, name) is not None and not callable(getattr(self, name)):
                raise ValueError(f"{name} must be callable")

@dataclass
class ChartSource:
    sub_question_id: str
    df: pd.DataFrame
    result_hash: str
    question_text: str = ""
    sql: str = ""
    primary_metric: str | None = None
    metric_metadata: dict = field(default_factory=dict)
    dimensions: list[str] = field(default_factory=list)
    temporal_columns: list[str] = field(default_factory=list)
    chart_spec: dict = field(default_factory=dict)  # Trusted application configuration, not LLM assertions.

def _words(name):
    return re.findall(r"[a-z]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name)).lower())

def _is_count(col, s):
    w = set(_words(col))
    return bool(w & _COUNT_WORDS) and not bool(w & _RATE_AVG_WORDS)

def _order_by_first(sql):
    """Conservative lexer: only simple outer ORDER BY aliases are accepted."""
    tokens, depth, i = [], 0, 0
    text = sql or ""
    while i < len(text):
        ch = text[i]
        if text.startswith("--", i):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end + 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0: return ""
            i = end + 2
            continue
        if ch in "'\"`[":
            closing = "]" if ch == "[" else ch
            i += 1
            value = ""
            while i < len(text):
                if text[i] == closing:
                    if i + 1 < len(text) and text[i+1] == closing:
                        value += closing
                        i += 2
                        continue
                    break
                value += text[i]
                i += 1
            if i == len(text): return ""
            if depth == 0: tokens.append(("literal" if ch == "'" else "identifier", value))
            i += 1
            continue
        if ch == "(":
            if depth == 0: tokens.append(("symbol", "("))
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth -= 1
            if depth < 0: return ""
            if depth == 0: tokens.append(("symbol", ")"))
            i += 1
            continue
        match = re.match(r"[A-Za-z_][A-Za-z0-9_$]*|\d+", text[i:])
        if match:
            if depth == 0: tokens.append(("word", match.group()))
            i += len(match.group())
        else:
            if depth == 0 and not ch.isspace(): tokens.append(("symbol", ch))
            i += 1
    if depth: return ""
    positions = [j for j in range(len(tokens)-1) if tokens[j][0] == tokens[j+1][0] == "word" and tokens[j][1].lower() == "order" and tokens[j+1][1].lower() == "by"]
    if len(positions) != 1: return ""
    expr = []
    for kind, value in tokens[positions[0]+2:]:
        if value in (",", ";") or (kind == "word" and value.lower() in {"limit", "offset", "fetch"}): break
        expr.append((kind, value))
    if len(expr) >= 2 and [v.lower() for _, v in expr[-2:]] in (["nulls", "first"], ["nulls", "last"]): expr = expr[:-2]
    if expr and expr[-1][0] == "word" and expr[-1][1].lower() in {"asc", "desc"}: expr = expr[:-1]
    if len(expr) == 3 and expr[1] == ("symbol", ".") and expr[0][0] in {"word", "identifier"}: expr = [expr[-1]]
    if len(expr) == 1 and expr[0][0] in {"word", "identifier"} and not expr[0][1].isdigit(): return expr[0][1].casefold()
    return ""

def order_measures(df, numeric, question="", sql=""):
    if not numeric: return []
    hint = _order_by_first(sql)
    primary = next((c for c in numeric if c.casefold() == hint), None)
    if primary is None:
        qw = set(_words(question))
        qw |= {_SYNONYMS[w] for w in qw if w in _SYNONYMS}
        scores = sorted([(len((set(_words(c)) - _GENERIC) & qw), i, c) for i, c in enumerate(numeric)], key=lambda t: (-t[0], t[1]))
        if scores[0][0]: primary = scores[0][2]
    if primary is None: primary = next((c for c in numeric if not _is_count(c, df[c])), numeric[0])
    rest = sorted((c for c in numeric if c != primary), key=lambda c: (not _is_count(c, df[c]), c.casefold(), c))
    return [primary, *rest]

def color_map(labels, palette):
    if not palette: raise ValueError("palette must not be empty")
    return {v: palette[i % len(palette)] for i, v in enumerate(sorted({str(v) for v in labels}))}

def _fingerprint(df):
    if not isinstance(df, pd.DataFrame): raise ValueError("Source must be a DataFrame")
    if not df.columns.is_unique or any(not isinstance(c, str) for c in df): raise ValueError("Column names must be unique strings")
    digest = hashlib.sha256(repr([(c, str(df[c].dtype)) for c in df]).encode())
    digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return digest.hexdigest()

def _safe_sum(values):
    values = list(values)
    if all(isinstance(v, (int, np.integer)) for v in values):
        return sum(int(v) for v in values)
    return math.fsum(float(v) for v in values)

def _same(actual, expected):
    if actual is None or expected is None: return actual is expected
    raw_a, raw_b = np.asarray(actual), np.asarray(expected)
    a, b = np.asarray(actual, dtype=object), np.asarray(expected, dtype=object)
    if raw_a.dtype.kind == "M" or raw_b.dtype.kind == "M":
        if a.shape != b.shape: return False
        try:
            da = pd.to_datetime(raw_a.ravel(), utc=True)
            db = pd.to_datetime(raw_b.ravel(), utc=True)
            return bool(((da == db) | (da.isna() & db.isna())).all())
        except (ValueError, TypeError, OverflowError):
            return False
    if a.shape != b.shape: return False
    for x, y in zip(a.flat, b.flat):
        xn, yn = bool(pd.isna(x)), bool(pd.isna(y))
        if xn or yn:
            if xn != yn: return False
            continue
        if isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number)):
            if float(x) != float(y): return False
        elif x != y: return False
    return True

@dataclass
class _Binding:
    kind: str
    name: str
    fields: dict
    source_columns: tuple

class ChartBuilder:
    def __init__(self, cfg=None): self.cfg = cfg or ChartConfig()
    @staticmethod
    def _label(c): return str(c).replace("_", " ").strip().title()
    @staticmethod
    def _looks_temporal(c):
        words = set(_words(c))
        if words & {"avg", "average", "mean", "duration", "turnaround", "elapsed"}: return False
        return bool(words & {"date", "month", "year", "quarter", "week", "period", "time", "timestamp", "datetime"})
    @staticmethod
    def _looks_period(s):
        v = s.dropna().astype(str)
        return not v.empty and bool(v.map(lambda x: bool(_PERIOD.fullmatch(x))).all())
    @staticmethod
    def _is_additive(col): return bool(set(_words(col)) & _COUNT_WORDS) and not bool(set(_words(col)) & _RATE_AVG_WORDS)
    @staticmethod
    def _mixed_scale(df, cols):
        mags = [float(df[c].abs().max()) for c in cols if df[c].notna().any() and df[c].abs().max() > 0]
        return bool(mags) and max(mags) / min(mags) > 25

    def _prepare(self, c, notes):
        df = c.df.copy(deep=True)
        metadata = getattr(c, "metric_metadata", {}) or {}
        if not isinstance(metadata, Mapping): raise ValueError("metric_metadata must be a mapping")
        dimensions = list(getattr(c, "dimensions", []) or [])
        temporal = list(getattr(c, "temporal_columns", []) or [])
        if any(col not in df for col in dimensions + temporal + list(metadata)): raise ValueError("Metadata references a missing column")
        if set(dimensions) & set(temporal): raise ValueError("Conflicting dimension and temporal metadata")
        for col, spec in metadata.items():
            if not isinstance(spec, Mapping): raise ValueError(f"Metadata for {col} must be a mapping")
            if spec.get("aggregation") not in {None, "sum", "min", "max", "mean", "weighted_rate"}: raise ValueError(f"Unsupported aggregation for {col}")
            if not isinstance(spec.get("unit", "raw"), str): raise ValueError(f"Invalid unit for {col}")
        times, numeric = [], []
        for col in df:
            s, words = df[col], set(_words(col))
            if col in dimensions: continue
            if pd.api.types.is_datetime64_any_dtype(s):
                times.append(col)
                continue
            if col in temporal or (col not in metadata and self._looks_temporal(col)):
                if words & {"year", "month", "quarter", "week", "period"} and (self._looks_period(s) or pd.api.types.is_numeric_dtype(s)):
                    if pd.api.types.is_numeric_dtype(s):
                        values = s.dropna().astype(float)
                        if not np.isfinite(values).all() or not (values == values.round()).all(): raise ValueError(f"Invalid numeric period in {col}")
                        if "month" in words and not values.between(1, 12).all(): raise ValueError("Month must be in 1-12")
                        if "quarter" in words and not values.between(1, 4).all(): raise ValueError("Quarter must be in 1-4")
                        if "week" in words and not values.between(1, 53).all(): raise ValueError("Week must be in 1-53")
                    else:
                        # Normalize quarter spelling so chronological sorting is stable.
                        df[col] = s.map(lambda v: re.sub(r"(\d{4})Q", r"\1-Q", str(v)) if pd.notna(v) else None)
                    times.append(col)
                    continue
                if not pd.api.types.is_numeric_dtype(s):
                    conv = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
                    if s.notna().any() and conv[s.notna()].notna().all():
                        df[col] = conv
                        times.append(col)
                        continue
                if col in temporal: raise ValueError(f"Invalid temporal values in {col}")
                notes.append(f"{col}: temporal conversion not applied; original values retained as a dimension.")
                dimensions.append(col)
                continue
            if pd.api.types.is_bool_dtype(s) or (col not in metadata and words & {"id", "identifier", "code", "zip", "postal"}): continue
            if pd.api.types.is_complex_dtype(s): raise ValueError(f"Complex values unsupported in {col}")
            if pd.api.types.is_numeric_dtype(s): numeric.append(col)
            elif col in metadata or (s.notna().any() and s.dropna().map(lambda v: isinstance(v, Decimal)).all()):
                converted = pd.to_numeric(s, errors="coerce")
                if not converted[s.notna()].notna().all(): raise ValueError(f"Invalid numeric values in {col}")
                df[col] = converted
                numeric.append(col)
            elif words & (_COUNT_WORDS | _RATE_AVG_WORDS | {"amount", "cost", "days", "duration", "value", "metric", "revenue", "balance", "tat"}) and s.notna().any() and s.dropna().map(lambda v: isinstance(v, str) and bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", v.strip())) and not bool(re.match(r"^[+-]?0\d", v.strip()))).all():
                df[col] = pd.to_numeric(s, errors="raise")
                numeric.append(col)
                notes.append(f"{col}: numeric strings converted without dropping values.")
        for col in numeric:
            values = df[col].dropna()
            if not np.isfinite(values.astype(float)).all(): raise ValueError(f"Non-finite numeric values in {col}")
            if values.map(lambda v: abs(v) > 2**53 and float(v).is_integer()).any(): raise ValueError(f"{col} exceeds chart renderer exact integer range")
        numeric = order_measures(df, numeric, str(getattr(c, "question_text", "") or ""), str(getattr(c, "sql", "") or ""))
        primary = getattr(c, "primary_metric", None)
        if primary is not None:
            if primary not in numeric: raise ValueError("primary_metric must reference a numeric measure")
            numeric = [primary] + [m for m in numeric if m != primary]
        categories = [col for col in df if col not in numeric and col not in times]
        for col in categories:
            nonnull = df[col].dropna()
            labels = nonnull.map(str)
            if labels.nunique() != nonnull.nunique(): raise ValueError(f"Ambiguous category labels in {col}")
            missing = "(Missing)"
            while missing in set(labels): missing = "_" + missing
            df[col] = df[col].map(lambda v: missing if pd.isna(v) else str(v))
        return df, numeric, times, categories

    def _aggregate(self, df, keys, measures, metadata, notes):
        if not df.duplicated(keys).any(): return df.copy()
        rows = []
        for key, group in df.groupby(keys, sort=True, dropna=False, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            row = dict(zip(keys, key))
            for m in measures:
                spec = metadata.get(m, {})
                agg = spec.get("aggregation")
                if agg is None: raise ValueError(f"Duplicate dimension keys require explicit aggregation metadata for {m}")
                if group[m].isna().any() and agg != "weighted_rate": raise ValueError(f"Cannot aggregate missing values in {m} without a null policy")
                if agg == "sum":
                    if spec.get("unit") in {"fraction", "percent"} or set(_words(m)) & _RATE_AVG_WORDS: raise ValueError(f"Summing a rate/average is not permitted: {m}")
                    value = _safe_sum(group[m].tolist())
                elif agg == "weighted_rate":
                    n, d = spec.get("numerator"), spec.get("denominator")
                    if n not in df or d not in df or n == d: raise ValueError(f"Valid numerator/denominator columns required for {m}")
                    if not pd.api.types.is_numeric_dtype(group[n]) or not pd.api.types.is_numeric_dtype(group[d]): raise ValueError("Weighted rate components must be numeric")
                    if group[[n, d]].isna().any().any() or (group[d] < 0).any() or (group[n] < 0).any(): raise ValueError("Weighted rate components must be non-null and nonnegative")
                    denominator = _safe_sum(group[d].tolist())
                    value = _safe_sum(group[n].tolist()) / denominator if denominator > 0 else np.nan
                    if denominator == 0: notes.append(f"{m}: zero denominator retained as a missing value, not zero.")
                    if spec.get("unit") == "percent": value *= 100
                    elif spec.get("unit") != "fraction": raise ValueError("Weighted rates require fraction or percent units")
                elif agg == "mean": value = _safe_sum(group[m].tolist()) / len(group)
                else: value = getattr(group[m], agg)()
                if pd.notna(value) and (not math.isfinite(float(value)) or (abs(value) > 2**53 and float(value).is_integer())): raise ValueError(f"Aggregation produced an unsupported value for {m}")
                row[m] = value
            rows.append(row)
        notes.append("Duplicate dimension keys aggregated using explicit metric metadata.")
        return pd.DataFrame(rows, columns=[*keys, *measures])

    def _binding(self, kind, name, columns, **fields):
        return _Binding(kind, name, {k: np.asarray(v, dtype=object).copy() for k, v in fields.items()}, tuple(columns))

    @staticmethod
    def _verify(fig, bindings):
        if len(fig.data) != len(bindings): return False
        for trace, binding in zip(fig.data, bindings):
            if trace.type != binding.kind or trace.name != binding.name: return False
            for field, expected in binding.fields.items():
                actual = trace
                for part in field.split("."): actual = getattr(actual, part, None)
                if not _same(actual, expected): return False
        return True

    @staticmethod
    def _semantic_trace(trace):
        """Allow styling, but not added sizes, labels, values, or transforms."""
        obj = trace.to_plotly_json()
        for key in ("line", "textfont", "hoverlabel", "opacity", "legendrank", "showlegend", "uid"):
            obj.pop(key, None)
        marker = obj.get("marker")
        if isinstance(marker, dict):
            for key in ("color", "colors", "line", "opacity", "colorscale", "showscale", "colorbar", "cmin", "cmax", "cmid", "coloraxis"):
                marker.pop(key, None)
            if not marker: obj.pop("marker", None)
        # JSON handles NumPy arrays using Plotly's own encoder.
        from plotly.utils import PlotlyJSONEncoder
        import json
        return json.dumps(obj, cls=PlotlyJSONEncoder, sort_keys=True)

    @staticmethod
    def _show_metric_values(fig):
        """Presentation-only labels; preserve all source arrays and hover settings."""
        def label(value):
            if value is None or pd.isna(value):
                return ""
            if isinstance(value, (int, np.integer)):
                return format(int(value), ",")
            if isinstance(value, (float, np.floating)):
                return format(float(value), ",")
            return str(value)

        for trace in fig.data:
            if trace.type == "bar":
                values = trace.x if trace.orientation == "h" else trace.y
                if values is not None:
                    trace.update(
                        text=[label(value) for value in values],
                        texttemplate="%{text}", textposition="outside",
                        cliponaxis=False, constraintext="none",
                    )
            elif trace.type == "scatter":
                if trace.y is not None:
                    mode = trace.mode or "lines"
                    if mode == "none":
                        mode = "text"
                    elif "text" not in mode.split("+"):
                        mode += "+text"
                    trace.update(
                        mode=mode, text=[label(value) for value in trace.y],
                        texttemplate="%{text}", textposition="top center",
                        cliponaxis=False,
                    )
            elif trace.type == "heatmap":
                if trace.z is not None:
                    trace.update(
                        text=[[label(value) for value in row] for row in trace.z],
                        texttemplate="%{text}",
                    )
            elif trace.type == "pie":
                trace.update(
                    textinfo="label+value+percent", textposition="outside",
                    automargin=True,
                )
            # Indicator traces already display their number; leave them unchanged.
        return fig

    def _stamp(self, candidates, c, fingerprint, notes):
        out = []
        for fig, bindings in candidates[:self.cfg.max_charts_per_question]:
            try:
                if not self._verify(fig, bindings): raise ValueError("Chart/source reconciliation failed")
                verified = go.Figure(fig)
                if self.cfg.style_fn:
                    try:
                        styled = self.cfg.style_fn(go.Figure(verified))
                        if not isinstance(styled, go.Figure) or not self._verify(styled, bindings): raise ValueError("Styling changed governed chart data")
                        if [self._semantic_trace(t) for t in styled.data] != [self._semantic_trace(t) for t in verified.data]: raise ValueError("Styling changed trace semantics")
                        fig = styled
                    except Exception:
                        log.exception("Chart styling rejected; retaining verified default figure")
                        notes.append("Custom chart styling failed validation; verified default styling retained.")
                        fig = verified
                # Apply labels after the existing style validation so custom
                # styling cannot hide them. Recheck all governed data bindings.
                fig = self._show_metric_values(fig)
                if not self._verify(fig, bindings):
                    raise ValueError("Value labels changed governed chart data")
                fig.update_layout(meta={"result_hash": str(getattr(c, "result_hash", "") or ""), "sql_sub_question": str(getattr(c, "sub_question_id", "") or ""), "source_dataframe_sha256": fingerprint, "reconciliation": "passed", "upstream_hash_verified": self.cfg.result_hash_validator is not None, "source_columns": [list(b.source_columns) for b in bindings], "validation_version": "2.0"})
                fig.to_json()
                out.append(fig)
            except Exception:
                log.exception("Chart discarded after validation failure")
                notes.append("A chart failed source reconciliation and was omitted; use the result table.")
        if _fingerprint(c.df) != fingerprint:
            notes.append("Source changed during styling; all charts omitted.")
            return []
        return out

    def _kpi_figure(self, value, label, unit="raw"):
        if pd.isna(value): raise ValueError(f"No KPI value for {label}")
        display = float(value) * (100 if unit == "fraction" else 1)
        if not math.isfinite(display): raise ValueError("KPI conversion produced a non-finite value")
        percent = unit in {"fraction", "percent"}
        options = dict(mode="number", value=display, name=label, number={"valueformat": ",.2f", "suffix": "%" if percent else ""}, title={"text": self._label(label)})
        if percent and 0 <= display <= 100: options.update(mode="gauge+number", gauge={"axis": {"range": [0, 100]}, "bar": {"color": self.cfg.color_sequence[0]}})
        return go.Figure(go.Indicator(**options))

    def build_requested(self, c, chart_spec, analysis_type=""):
        """Adapt an existing result contract without mutating it.

        chart_spec must come from trusted application/semantic configuration.
        This method does not authorize access to case-level records. Existing
        row-level permissions and sensitive-column checks must run upstream.
        """
        source = ChartSource(
            sub_question_id=getattr(c,"sub_question_id",""), df=c.df,
            result_hash=getattr(c,"result_hash",""),
            question_text=getattr(c,"question_text",""), sql=getattr(c,"sql",""),
            primary_metric=getattr(c,"primary_metric",None),
            metric_metadata=getattr(c,"metric_metadata",{}) or {},
            dimensions=getattr(c,"dimensions",[]) or [],
            temporal_columns=getattr(c,"temporal_columns",[]) or [],
            chart_spec=chart_spec,
        )
        if getattr(c,"truncated",False):
            return [], ["Truncated result: requested chart omitted; the table is retained."]
        return self.build(source,analysis_type)

    def _advanced(self, c, spec, fingerprint, notes):
        """Explicit, trusted chart contracts. No sampling, inferred targets or units."""
        df = c.df.copy(deep=True)
        if not df.columns.is_unique:
            raise ValueError("Duplicate column names")
        kind = spec.get("type")
        fig, bindings = go.Figure(), []
        def require(flag):
            if spec.get(flag) is not True:
                raise ValueError(f"Trusted chart specification must declare {flag}=True")
        def column(key):
            col = spec.get(key)
            if not isinstance(col, str) or col not in df:
                raise ValueError(f"Missing or invalid {key} column")
            return col
        def numbers(col, nonnegative=False):
            v = df[col]
            if pd.api.types.is_bool_dtype(v) or not pd.api.types.is_numeric_dtype(v):
                raise ValueError(f"{col} must be numeric, not text or boolean")
            if pd.api.types.is_integer_dtype(v) and any(abs(int(x)) > 2**53 for x in v.dropna()):
                raise ValueError("Integer exceeds exact chart numeric precision")
            a = v.to_numpy(dtype=float)
            if not np.isfinite(a).all() or (nonnegative and (a < 0).any()):
                raise ValueError(f"{col} contains missing, infinite or invalid negative values")
            return a
        def categories(col):
            if df[col].isna().any():
                raise ValueError(f"Missing category in {col}")
            vals = df[col].map(str)
            if vals.nunique() != df[col].nunique():
                raise ValueError("Category labels collide after conversion to text")
            return vals
        def add(trace, columns, **fields):
            fig.add_trace(trace)
            bindings.append(self._binding(trace.type, trace.name, columns, **fields))
        def limit(n, maximum):
            if n > maximum:
                raise ValueError("Chart exceeds configured category/series limit; no truncation applied")
        if kind in {"stacked_bar", "stacked_100"}:
            require("mutually_exclusive")
            require("additive_counts")
            group, segment, metric = column("category"), column("segment"), column("value")
            if len({group, segment, metric}) != 3:
                raise ValueError("Category, segment and value must be distinct")
            categories(group); categories(segment)
            values = numbers(metric, True)
            if not np.equal(values, np.floor(values)).all():
                raise ValueError("Stacked status counts must be integers")
            if df.duplicated([group, segment]).any():
                raise ValueError("Provide one reconciled count per category/segment")
            groups, segments = list(pd.unique(df[group])), list(pd.unique(df[segment]))
            limit(len(groups), self.cfg.max_bars); limit(len(segments), self.cfg.max_series)
            # Missing combinations are unknown, not implicitly zero.
            if len(df) != len(groups) * len(segments):
                raise ValueError("Provide all category/segment combinations, including explicit zeros")
            matrix = df.pivot(index=group, columns=segment, values=metric).reindex(index=groups, columns=segments)
            totals = matrix.sum(axis=1).to_numpy(dtype=float)
            if kind == "stacked_100":
                require("exhaustive_segments")
                if spec.get("denominator") != "sum_of_segments":
                    raise ValueError("Declare denominator='sum_of_segments'")
                if (totals <= 0).any():
                    raise ValueError("100% stacks require positive denominators")
            for seg in segments:
                raw = matrix[seg].to_numpy(dtype=float)
                shown = raw / totals * 100 if kind == "stacked_100" else raw
                custom = np.column_stack([raw, totals])
                text = [f"{v:.2f}% ({n:,.0f})" for v,n in zip(shown,raw)] if kind == "stacked_100" else [f"{v:,.0f}" for v in raw]
                add(go.Bar(name=str(seg), x=list(map(str,groups)), y=shown, customdata=custom,
                           text=text, texttemplate="%{text}", textposition="inside",
                           hovertemplate="%{x}<br>%{fullData.name}: %{customdata[0]}<br>Denominator: %{customdata[1]}<extra></extra>"),
                    [group,segment,metric], x=list(map(str,groups)), y=shown, customdata=custom)
            fig.update_layout(barmode="stack", yaxis_title="Percent of supplied status counts" if kind == "stacked_100" else self._label(metric))
            if kind == "stacked_100": fig.update_yaxes(range=[0,100], ticksuffix="%")
            notes.append("Stacked categories are declared mutually exclusive by trusted application configuration.")
        elif kind == "bullet":
            require("target_approved")
            metric, target = column("value"), column("target")
            if metric == target or len(df) != 1:
                raise ValueError("Bullet requires one row and distinct actual/target columns")
            actual, goal = numbers(metric,True)[0], numbers(target,True)[0]
            unit = spec.get("unit")
            if unit not in {"fraction","percent","count","days","raw"}:
                raise ValueError("Declare the actual/target unit explicitly")
            if unit == "fraction": actual,goal=actual*100,goal*100
            percent = unit in {"fraction","percent"}
            if percent and max(actual,goal)>100:
                raise ValueError("Percent actual/target must be between 0 and 100")
            maximum = 100 if percent else max(actual,goal,1)*1.2
            add(go.Indicator(name=metric,mode="number+gauge",value=actual,
                number={"suffix":"%" if percent else "", "valueformat":",.2f"},
                gauge={"shape":"bullet","axis":{"range":[0,maximum]},"threshold":{"value":goal,"line":{"color":"#ef4444","width":3}}}),
                [metric,target], value=actual, **{"gauge.threshold.value":goal,"gauge.axis.range":[0,maximum]})
            fig.add_annotation(x=.5,y=1.15,xref="paper",yref="paper",text=f"Approved target: {goal:,.2f}"+("%" if percent else ""),showarrow=False)
        elif kind == "histogram":
            require("observations")
            metric=column("value"); values=numbers(metric)
            edges=spec.get("bin_edges")
            if not isinstance(edges,list) or len(edges)<2 or any(isinstance(x,bool) for x in edges):
                raise ValueError("Provide explicit numeric bin_edges")
            if any(not isinstance(x, (int,float)) or isinstance(x,bool) for x in edges):
                raise ValueError("Bin edges must be numeric")
            edges=np.asarray(edges,dtype=float)
            if not np.isfinite(edges).all() or not (np.diff(edges)>0).all():
                raise ValueError("Bin edges must be finite and strictly increasing")
            limit(len(edges)-1,self.cfg.max_bars)
            if values.min()<edges[0] or values.max()>edges[-1]:
                raise ValueError("Bins must cover every observation")
            counts,_=np.histogram(values,bins=edges)
            if int(counts.sum())!=len(values): raise ValueError("Histogram reconciliation failed")
            centers=(edges[:-1]+edges[1:])/2
            bounds=np.column_stack([edges[:-1],edges[1:]])
            add(go.Bar(name="Observations",x=centers,y=counts,width=np.diff(edges),customdata=bounds,
                text=counts,texttemplate="%{text}",textposition="outside",
                hovertemplate="Bin: %{customdata[0]} to %{customdata[1]}<br>Count: %{y}<extra></extra>"),
                [metric],x=centers,y=counts,width=np.diff(edges),customdata=bounds)
            fig.update_layout(bargap=0,xaxis_title=self._label(metric),yaxis_title="Observation count")
            notes.append("Histogram bins are left-inclusive/right-exclusive; the final bin includes its right edge. No deduplication or sampling.")
        elif kind == "box_plot":
            require("observations")
            metric=column("value");numbers(metric)
            group=spec.get("category")
            if group is not None:
                group=column("category");categories(group)
                if group==metric:raise ValueError("Group and value must differ")
                groups=list(pd.unique(df[group]))
            else: groups=[None]
            limit(len(groups),self.cfg.max_series)
            for g in groups:
                vals=df.loc[df[group]==g,metric].to_numpy(dtype=float) if group else numbers(metric)
                if len(vals)<2:raise ValueError("Each box requires at least two observations")
                q1,median,q3=np.quantile(vals,[.25,.5,.75],method="linear")
                iqr=q3-q1;low=float(vals[vals>=q1-1.5*iqr].min());high=float(vals[vals<=q3+1.5*iqr].max())
                name=str(g) if group else self._label(metric)
                add(go.Box(name=name,q1=[q1],median=[median],q3=[q3],lowerfence=[low],upperfence=[high],boxpoints=False),
                    [metric]+([group] if group else []),q1=[q1],median=[median],q3=[q3],lowerfence=[low],upperfence=[high])
                fig.add_annotation(x=name,y=high,text=f"Q1 {q1:,.2f}<br>Median {median:,.2f}<br>Q3 {q3:,.2f}",showarrow=False,yshift=30)
            fig.update_layout(yaxis_title=self._label(metric))
            notes.append("Box summaries use linear quartiles and 1.5-IQR observed fences. Outlier points are not drawn; the source table retains all observations.")
        elif kind == "pareto":
            require("mutually_exclusive");require("additive_counts")
            group,metric=column("category"),column("value")
            labels=categories(group);values=numbers(metric,True)
            if group==metric or labels.duplicated().any():raise ValueError("Provide one count per distinct category")
            if not np.equal(values,np.floor(values)).all():raise ValueError("Pareto requires integer counts")
            limit(len(df),self.cfg.max_bars)
            order=np.argsort(-values,kind="stable");values=values[order];labels=labels.iloc[order].tolist()
            total=math.fsum(values)
            if total<=0:raise ValueError("Pareto total must be positive")
            cumulative=np.cumsum(values)/total*100
            add(go.Bar(name=metric,x=labels,y=values,text=values,texttemplate="%{text}",textposition="outside"),[group,metric],x=labels,y=values)
            add(go.Scatter(name="Cumulative %",x=labels,y=cumulative,yaxis="y2",mode="lines+markers+text",
                text=[f"{v:.1f}%" for v in cumulative],textposition="top center"),[group,metric],x=labels,y=cumulative,yaxis="y2")
            fig.update_layout(yaxis2={"overlaying":"y","side":"right","range":[0,100],"ticksuffix":"%"},yaxis_title=self._label(metric))
            notes.append("Cumulative percentage uses all supplied categories; no top-N truncation.")
        elif kind == "waterfall":
            require("reconciled_movements")
            group,metric,measure=column("category"),column("value"),column("measure")
            if len({group,metric,measure})!=3:raise ValueError("Waterfall columns must be distinct")
            labels=categories(group).tolist();values=numbers(metric);roles=df[measure].tolist()
            limit(len(df),self.cfg.max_bars)
            if len(df)<3 or roles[0]!="absolute" or roles[-1]!="total" or any(x!="relative" for x in roles[1:-1]):
                raise ValueError("Expected opening absolute, signed relative movements, closing total in supplied row order")
            closing=math.fsum(values[:-1])
            if not math.isclose(closing,float(values[-1]),rel_tol=1e-10,abs_tol=1e-8):
                raise ValueError("Opening plus signed movements does not equal supplied closing balance")
            if values[0]<0 or closing<0 or (np.cumsum(values[:-1]) < -1e-8).any():
                raise ValueError("Backlog balance cannot become negative")
            rendered=values.copy();rendered[-1]=0
            add(go.Waterfall(name=metric,x=labels,y=rendered,measure=roles,customdata=values,
                text=[f"{v:,.2f}" for v in values],textposition="outside",
                hovertemplate="%{x}: %{customdata}<extra></extra>"),[group,metric,measure],x=labels,y=rendered,measure=roles,customdata=values)
            notes.append("Waterfall follows supplied row order; opening plus signed movements reconciles to the supplied closing balance.")
        elif kind == "treemap":
            require("additive_counts");require("leaf_rows")
            path=spec.get("path");metric=column("value")
            if not isinstance(path,list) or not path or len(set(path))!=len(path) or any(not isinstance(x,str) or x not in df or x==metric for x in path):
                raise ValueError("Provide distinct hierarchy path columns separate from value")
            for col in path:categories(col)
            values=numbers(metric,True)
            if not np.equal(values,np.floor(values)).all():raise ValueError("Treemap requires integer leaf counts")
            if df.duplicated(path).any():raise ValueError("Provide one additive leaf count per complete hierarchy path")
            if math.fsum(values)<=0:raise ValueError("Treemap total must be positive")
            nodes={}
            for row,value in zip(df[path].itertuples(index=False,name=None),values):
                for depth in range(1,len(path)+1):
                    key=tuple(map(str,row[:depth]));nodes.setdefault(key,[]).append(float(value))
            limit(len(nodes),self.cfg.max_heatmap_cells)
            keys=list(nodes);ids=[str(i) for i in range(len(keys))];index=dict(zip(keys,ids))
            labels=[key[-1] for key in keys];parents=[index[key[:-1]] if len(key)>1 else "" for key in keys]
            totals=[math.fsum(nodes[key]) for key in keys]
            add(go.Treemap(name=metric,ids=ids,labels=labels,parents=parents,values=totals,branchvalues="total",textinfo="label+value+percent root"),
                path+[metric],ids=ids,labels=labels,parents=parents,values=totals,branchvalues="total")
            notes.append("Hierarchy nodes sum declared additive leaf counts; parent totals are not added again to the grand total.")
        else:
            raise ValueError("Unsupported explicit chart type")
        fig.update_layout(title=self._label(kind),template="plotly_white",colorway=list(self.cfg.color_sequence))
        # Advanced labels/annotations are semantic output, not cosmetic style.
        # Preserve them and all layout/trace settings against style callbacks.
        if self.cfg.style_fn:
            original = go.Figure(fig)
            protected = ("barmode", "barnorm", "annotations", "shapes", "yaxis2")
            try:
                candidate = go.Figure(fig)
                styled = self.cfg.style_fn(candidate)
                if styled is not None:
                    candidate = styled
                if not isinstance(candidate, go.Figure) or not self._verify(candidate, bindings):
                    raise ValueError("Style changed governed values")
                if len(candidate.data) != len(original.data) or any(
                    self._semantic_trace(x) != self._semantic_trace(y)
                    for x,y in zip(original.data,candidate.data)
                ):
                    raise ValueError("Style changed advanced chart semantics")
                old_layout, new_layout = original.layout.to_plotly_json(), candidate.layout.to_plotly_json()
                if any(old_layout.get(k) != new_layout.get(k) for k in protected):
                    raise ValueError("Style changed advanced chart layout semantics")
                for axis in ("xaxis","yaxis"):
                    for prop in ("range","type","overlaying","side"):
                        if old_layout.get(axis,{}).get(prop) != new_layout.get(axis,{}).get(prop):
                            raise ValueError("Style changed advanced chart axes")
                fig = candidate
            except Exception:
                fig = original
                notes.append("Advanced chart retained verified default styling after an incompatible style callback.")
        if not self._verify(fig,bindings):raise ValueError("Advanced chart reconciliation failed")
        if _fingerprint(c.df)!=fingerprint:raise ValueError("Source changed during advanced chart generation")
        fig.update_layout(meta={"result_hash":str(getattr(c,"result_hash","") or ""),
            "sql_sub_question":str(getattr(c,"sub_question_id","") or ""),
            "source_dataframe_sha256":fingerprint,"reconciliation":"passed",
            "upstream_hash_verified":self.cfg.result_hash_validator is not None,
            "source_columns":[list(b.source_columns) for b in bindings],
            "validation_version":"3.0","chart_type":kind})
        return [fig]

    def build(self, c, analysis_type=""):
        notes = []
        if analysis_type == "drilldown": return [], ["Row-level result: table only, by design."]
        if self.cfg.max_charts_per_question == 0: return [], ["Charts disabled by configuration."]
        try:
            fingerprint = _fingerprint(c.df)
            if c.df.empty: return [], ["Empty result: table only."]
            if len(c.df) > self.cfg.max_points: return [], [f"Result exceeds the {self.cfg.max_points}-row chart safety limit; no sampling applied."]
            if self.cfg.result_hash_validator and not self.cfg.result_hash_validator(c.df.copy(deep=True), c.result_hash): raise ValueError("Upstream result hash validation failed")
            spec = getattr(c, "chart_spec", {}) or {}
            if not isinstance(spec, Mapping): raise ValueError("chart_spec must be a mapping")
            advanced_types = {"stacked_bar", "stacked_100", "bullet", "histogram", "box_plot", "pareto", "waterfall", "treemap"}
            if spec or analysis_type in advanced_types:
                spec = dict(spec)
                if analysis_type in advanced_types and spec.get("type",analysis_type) != analysis_type:
                    raise ValueError("Conflicting explicit chart types")
                spec.setdefault("type", analysis_type)
                return self._advanced(c, spec, fingerprint, notes), notes
            df, num, times, cat = self._prepare(c, notes)
            if not num: return [], notes + ["No numeric measure to chart; use the result table."]
            metadata = getattr(c, "metric_metadata", {}) or {}
            candidates = []
            if len(df) == 1:
                for m in num[:self.cfg.max_charts_per_question]:
                    if pd.isna(df[m].iloc[0]):
                        notes.append(f"{m}: null KPI omitted.")
                        continue
                    unit = metadata.get(m, {}).get("unit", "raw")
                    if unit == "raw" and set(_words(m)) & {"rate", "percent", "percentage", "pct"}: notes.append(f"{m}: percentage unit unspecified; raw value displayed without conversion.")
                    fig = self._kpi_figure(df[m].iloc[0], m, unit)
                    expected = float(df[m].iloc[0]) * (100 if unit == "fraction" else 1)
                    if unit in {"fraction", "percent"} and not 0 <= expected <= 100: notes.append(f"{m}: outside 0-100%; actual value retained without a gauge.")
                    candidates.append((fig, [self._binding("indicator", m, [m], value=expected)]))
            elif times:
                if len(times) > 1 or len(cat) > 1: raise ValueError("Multiple time/group dimensions require an explicit projection before charting")
                x, m = times[0], num[0]
                if df[x].isna().any(): raise ValueError("Null time keys cannot be safely plotted")
                keys = [x] + cat
                plot = self._aggregate(df, keys, num, metadata, notes).sort_values(keys, kind="mergesort")
                if cat and plot[cat[0]].nunique() > self.cfg.max_series: raise ValueError("Too many time-series groups; no groups were merged or dropped")
                periods = plot[x].nunique()
                bars = periods < self.cfg.min_periods_for_trend
                if bars: notes.append(f"Only {periods} periods available; shown as grouped bars.")
                metrics = [m] if cat else num[:self.cfg.max_charts_per_question]
                for metric in metrics:
                    fig, bindings = go.Figure(), []
                    groups = [(metric, plot)] if not cat else [(label, plot.loc[plot[cat[0]] == label]) for label in sorted(plot[cat[0]].unique())]
                    cmap = color_map([label for label, _ in groups], self.cfg.color_sequence)
                    for label, part in groups:
                        if bars: fig.add_trace(go.Bar(x=part[x], y=part[metric], name=label, marker_color=cmap[label]))
                        else: fig.add_trace(go.Scatter(x=part[x], y=part[metric], name=label, mode="lines+markers", connectgaps=False, line_color=cmap[label]))
                        bindings.append(self._binding("bar" if bars else "scatter", label, [*keys, metric], x=part[x], y=part[metric]))
                    fig.update_layout(title=f"{self._label(metric)} over {self._label(x)}", barmode="group", hovermode="x unified", xaxis_title=self._label(x), yaxis_title=self._label(metric))
                    candidates.append((fig, bindings))
            elif len(cat) == 2:
                a, b = cat
                m = num[0]
                if df[a].nunique() * df[b].nunique() > self.cfg.max_heatmap_cells: raise ValueError("Heatmap exceeds configured cell limit")
                plot = self._aggregate(df, [a, b], [m], metadata, notes)
                pivot = plot.pivot(index=a, columns=b, values=m).sort_index().sort_index(axis=1)
                fig = go.Figure(go.Heatmap(x=pivot.columns.tolist(), y=pivot.index.tolist(), z=pivot.to_numpy(), name=m, colorscale="Blues", hoverongaps=False))
                fig.update_layout(title=f"{self._label(m)}: {self._label(a)} / {self._label(b)}")
                candidates.append((fig, [self._binding("heatmap", m, [a, b, m], x=pivot.columns, y=pivot.index, z=pivot.to_numpy())]))
            elif len(cat) == 1:
                d, m = cat[0], num[0]
                plot = self._aggregate(df, [d], num, metadata, notes).sort_values([m, d], ascending=[False, True], kind="mergesort", na_position="last")
                total = len(plot)
                plot = plot.head(self.cfg.max_bars)
                truncated = len(plot) < total
                if truncated: notes.append(f"Chart shows the top {len(plot)} of {total} categories by {self._label(m)}; the table is complete.")
                colours = color_map(plot[d], self.cfg.color_sequence)
                for metric in num[:self.cfg.max_charts_per_question]:
                    horizontal = total > 8
                    x, y = (plot[metric], plot[d]) if horizontal else (plot[d], plot[metric])
                    fig = go.Figure(go.Bar(x=x, y=y, name=metric, orientation="h" if horizontal else "v", marker_color=[colours[v] for v in plot[d]]))
                    fig.update_layout(title=f"{self._label(metric)} by {self._label(d)}", showlegend=False)
                    candidates.append((fig, [self._binding("bar", metric, [d, metric], x=x, y=y)]))
                spec = metadata.get(m, {})
                if len(candidates) < self.cfg.max_charts_per_question and not truncated and 2 <= total <= self.cfg.max_pie_slices and spec.get("additive") is True and spec.get("unit") not in {"fraction", "percent"} and not set(_words(m)) & _RATE_AVG_WORDS and plot[m].notna().all() and (plot[m] >= 0).all() and plot[m].sum() > 0:
                    fig = go.Figure(go.Pie(labels=plot[d], values=plot[m], name=m, hole=.52, sort=False, marker_colors=[colours[v] for v in plot[d]]))
                    fig.update_layout(title=f"Share of {self._label(m)} by {self._label(d)}")
                    candidates.append((fig, [self._binding("pie", m, [d, m], labels=plot[d], values=plot[m])]))
            elif not cat and len(num) >= 2 and len(df) >= 10:
                x, y = num[:2]
                plot = df.sort_values(num, kind="mergesort")
                name = f"{y} vs {x}"
                fig = go.Figure(go.Scatter(x=plot[x], y=plot[y], name=name, mode="markers", marker_color=self.cfg.color_sequence[0]))
                fig.update_layout(title=f"{self._label(y)} vs {self._label(x)}", xaxis_title=self._label(x), yaxis_title=self._label(y))
                candidates.append((fig, [self._binding("scatter", name, [x, y], x=plot[x], y=plot[y])]))
            else: return [], notes + ["No safe chart fits this result shape; the table is the complete answer."]
            if _fingerprint(c.df) != fingerprint: raise ValueError("Source dataframe changed during chart construction")
            return self._stamp(candidates, c, fingerprint, notes), notes
        except Exception as exc:
            log.exception("Chart generation declined")
            notes.append(f"Chart omitted: {exc}. The result table is unchanged." if isinstance(exc, ValueError) else "Chart generation failed safely; use the result table and check application logs.")
            return [], notes

def fallback_charts(df, question="", style_fn=None, max_charts=2):
    """Same governed validation on fallback paths; ordinary errors return []."""
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty: return []
        cfg = ChartConfig(max_charts_per_question=max_charts, style_fn=style_fn)
        figures, notes = ChartBuilder(cfg).build(ChartSource("fallback", df, "", question_text=question))
        if notes: log.info("Fallback visualization notes: %s", notes)
        return figures
    except Exception:
        log.exception("Fallback chart generation failed safely")
        return []
