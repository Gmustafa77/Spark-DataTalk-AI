"""Visualization recommendation engine + deterministic renderer (no LLM, no Streamlit).

Pipeline (one returned result table):

    profile_dataset(df, context)          column roles, temporal parsing, cardinality,
            │                             sign, nulls, additivity, percent units
            ▼
    recommend_visualizations(df, context) -> VisualizationPlan
            │   recommended_chart, compatible_charts, x_axis, y_axis, series,
            │   time_column, measure_columns, aggregation, reason
            ▼
    render_chart(df, plan, chart_type)    -> (plotly Figure | None, notes)

Design rules
------------
* **Semantic, not just dtype.** A ``month`` column that arrives from SQL as text
  ("Jan", "2026-01", "Q1 2026", "FY2026", "Monday" …) is recognised as time and
  ordered chronologically. The question wording ("trend", "monthly", "over time")
  and the SQL agent's ``grain``/``dimensions`` reinforce the decision.
* **Strict time-series rule.** Time-series data is recommended as Line → Area.
  Pie, Donut and Treemap are never offered for it; Bar is offered only as a
  secondary option and never as the recommendation when ≥2 periods exist.
* **Meaningful, not merely renderable.** Pie/Donut require an additive,
  non-negative measure with 2–6 categories; Histogram/Box require raw
  observations; Heatmap requires two bounded dimensions; Treemap requires an
  additive, non-negative measure.
* **Analytical result is never changed.** Rendering only sorts, limits the
  number of bars drawn (with a note) or pivots. It never sums a rate/average;
  duplicated keys of a non-additive measure disable the charts that would need it.
* **Deterministic.** Same data + same context → same plan and same figure.

Chart catalogue (v1.1)
----------------------
KPI cards, Column, Bar, Line, Area, Clustered/Stacked, Pie, Donut, Combo (bars + line on a
secondary axis), Waterfall, Funnel, Scatter/Bubble, Matrix (pivot with totals and shading),
Heatmap, Treemap, Histogram, Box, Map (US states / countries / latitude-longitude) and Table.
Each is offered only when the data supports it (see ``_augment``). The Slicer is a UI
control that filters the stored result before drawing; it lives in ``components.viz_switcher``.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.plot_style import PALETTE, style_figure

log = logging.getLogger("datatalk.viz_recommender")

VIZ_VERSION = "viz-recommender-1.1"

# ═══════════════════════════════════════════════════════════════════════════════
# Chart catalogue
# ═══════════════════════════════════════════════════════════════════════════════
# Keys are stored in QA.json plans, so they never change. Labels follow BI-tool wording:
# "bar" is drawn vertically (a *column* chart) and "hbar" horizontally (a *bar* chart).
CHART_LABELS: dict[str, str] = {
    "line": "Line", "area": "Area", "bar": "Column", "hbar": "Bar",
    "grouped_bar": "Grouped bar", "stacked_bar": "Stacked bar", "pie": "Pie", "donut": "Donut",
    "scatter": "Scatter", "histogram": "Histogram", "box": "Box plot", "treemap": "Treemap",
    "heatmap": "Heatmap", "kpi": "KPI cards", "combo": "Combo", "waterfall": "Waterfall",
    "funnel": "Funnel", "matrix": "Matrix", "table": "Table", "map": "Map",
}
CHART_ICONS: dict[str, str] = {
    "line": "📈", "area": "🏔️", "bar": "📊", "hbar": "📶", "grouped_bar": "📊", "stacked_bar": "🧱",
    "pie": "🥧", "donut": "🍩", "scatter": "✳️", "histogram": "📉", "box": "📦", "treemap": "🟩",
    "heatmap": "🔥", "kpi": "🔢", "combo": "🧩", "waterfall": "🌊", "funnel": "🔻", "matrix": "🧮",
    "table": "📋", "map": "🗺️",
}
_TIME_FORBIDDEN = {"pie", "donut", "treemap", "histogram"}


@dataclass
class VizLimits:
    max_pie_slices: int = 6
    min_pie_slices: int = 2
    max_bars: int = 25
    max_series: int = 8
    max_heatmap_cells: int = 600
    max_treemap_nodes: int = 60
    min_rows_distribution: int = 8
    min_rows_scatter: int = 5
    max_render_rows: int = 20000
    high_cardinality: int = 50
    min_waterfall_steps: int = 3
    max_waterfall_steps: int = 15
    min_funnel_stages: int = 3
    max_funnel_stages: int = 10
    max_matrix_rows: int = 50
    max_matrix_cols: int = 15
    max_table_rows: int = 500


@dataclass
class ColumnInfo:
    name: str
    role: str                      # time | dimension | measure | id | text | coord | empty
    dtype: str = ""
    n_unique: int = 0
    null_pct: float = 0.0
    time_kind: str = ""            # date | year | month | quarter | week | weekday | fiscal | sequence
    additive: bool = False
    percent: bool = False
    has_negative: bool = False
    integer: bool = False
    ordered: bool = False          # categorical labels with intrinsic order ("0-15d", "16-30d")


@dataclass
class VisualizationPlan:
    recommended_chart: str | None
    compatible_charts: list[str]
    x_axis: str | None = None
    y_axis: str | None = None
    series: str | None = None
    time_column: str | None = None
    dimension_columns: list[str] = field(default_factory=list)
    measure_columns: list[str] = field(default_factory=list)
    aggregation: str = "none"      # none | pre-aggregated | sum | observations
    is_time_series: bool = False
    reason: str = ""
    notes: list[str] = field(default_factory=list)
    question: str = ""
    # Context kept so a stored plan renders exactly like the one first built (and can be upgraded).
    grain: str = ""
    declared_dimensions: list[str] = field(default_factory=list)
    geo_column: str | None = None      # dimension holding US states / countries
    geo_kind: str = ""                 # us_abbr | us_name | country | iso3 | latlon
    lat_column: str | None = None
    lon_column: str | None = None
    version: str = VIZ_VERSION

    @property
    def switchable(self) -> bool:
        return len(self.compatible_charts) >= 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> VisualizationPlan | None:
        if not isinstance(d, dict) or "compatible_charts" not in d:
            return None
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


# ═══════════════════════════════════════════════════════════════════════════════
# Vocabulary
# ═══════════════════════════════════════════════════════════════════════════════
_TIME_NAME = {"date", "month", "year", "quarter", "week", "period", "day", "yearmonth", "time", "timestamp",
              "fy", "fiscal", "yyyy", "mm", "weekday", "dow", "qtr"}
_NOT_TIME_NAME = {"avg", "average", "mean", "median", "days", "duration", "tat", "turnaround", "age", "aging",
                  "elapsed", "count", "rate", "total"}
_ID_NAME = {"id", "identifier", "key", "uuid", "guid", "code", "zip", "postal", "npi", "no"}
_PCT_NAME = {"rate", "percent", "percentage", "pct", "share", "ratio"}
_AVG_NAME = {"avg", "average", "mean", "median", "score", "index", "min", "max", "minimum", "maximum"}
_ADD_NAME = {"count", "cases", "case", "total", "volume", "num", "appeals", "records", "claims", "items", "sum",
             "amount", "revenue", "cost", "paid", "spend", "closed", "open", "overdue", "escalated", "pending",
             "overturned", "upheld", "compliant", "denials", "transferred", "n"}
_DUR_NAME = {"days", "hours", "tat", "turnaround", "age", "aging", "duration"}

_TEMPORAL_Q = re.compile(
    r"\b(trend|trends|trending|over time|time series|timeline|history|historical|monthly|weekly|daily|"
    r"quarterly|yearly|annual|annually|month over month|month-over-month|year over year|yoy|mom|"
    r"(each|every|per|by) (day|week|month|quarter|year|date|period)|"
    r"(last|past|previous) \d+ (days|weeks|months|quarters|years)|seasonal|seasonality)\b",
    re.I,
)
_FUNNEL_Q = re.compile(r"\b(funnel|conversion|pipeline|drop[- ]?off|attrition|stage[- ]by[- ]stage|progression)\b", re.I)
_WATERFALL_Q = re.compile(r"\b(waterfall|bridge|walk|build[- ]?up|cumulative|running total|net change)\b", re.I)
_MAP_Q = re.compile(r"\b(map|maps|geograph\w*|geospatial|choropleth)\b", re.I)
_KPI_Q = re.compile(r"\b(kpis?|scorecard|headline|at a glance|key metrics?)\b", re.I)
_MATRIX_Q = re.compile(r"\b(matrix|pivot|cross-?tab\w*)\b", re.I)
_TABLE_Q = re.compile(r"\b(as a table|in a table|tabular|table view)\b", re.I)
_DELTA_NAME = {"change", "delta", "variance", "diff", "difference", "net", "growth", "movement"}
_STAGE_NAME = {"stage", "step", "phase", "level", "tier", "status", "funnel", "pipeline"}
_COORD_NAME = {"lat", "latitude", "lon", "lng", "longitude"}
_STATE_NAME = {"state", "states", "st"}
_COUNTRY_NAME = {"country", "countries", "nation"}
_US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
_US_NAME_TO_ABBR = {v.lower(): k for k, v in _US_STATES.items()}
_DISTRIBUTION_Q = re.compile(r"\b(distribution|spread|histogram|outlier|outliers|variability|variance)\b", re.I)
_SHARE_Q = re.compile(r"\b(share|breakdown|mix|proportion|composition|split|percentage of|contribution)\b", re.I)
_COMPARE_Q = re.compile(r"\b(compare|comparison|vs|versus|against|relationship|correlat\w*)\b", re.I)

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_WEEKDAYS = {d: i for i, d in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"], 1)}
_ORDERED_CAT = re.compile(r"^\s*(?:[<>≤≥]=?\s*)?\d+(?:\.\d+)?\s*(?:[-–to+]+\s*\d*(?:\.\d+)?)?\s*[a-z%]*\s*$", re.I)


def _words(name: Any) -> set[str]:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return set(re.findall(r"[a-z]+", s.lower()))


def pretty(name: Any) -> str:
    s = str(name).strip()
    return s.replace("_", " ").title() if ("_" in s or s.islower()) else s


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal parsing  (value → chronological sort key)
# ═══════════════════════════════════════════════════════════════════════════════
def _year(y: str) -> int:
    v = int(y)
    return v + 2000 if v < 100 else v


def _parse_one(v: str, kind: str) -> float | None:
    s = v.strip().lower().replace("’", "'")
    if kind == "year":
        m = re.fullmatch(r"(\d{4})", s)
        return float(m.group(1)) if m and 1900 <= int(m.group(1)) <= 2100 else None
    if kind == "fiscal":
        m = re.fullmatch(r"fy\s*'?(\d{2}|\d{4})(?:\s*[-/ ]?\s*q([1-4]))?", s)
        return _year(m.group(1)) * 4 + (int(m.group(2)) - 1 if m.group(2) else 0) if m else None
    if kind == "quarter":
        for pat, yi, qi in ((r"(\d{4})\s*[-/ ]?\s*q([1-4])", 1, 2), (r"q([1-4])\s*[-/ ']?\s*(\d{2}|\d{4})", 2, 1)):
            m = re.fullmatch(pat, s)
            if m:
                return _year(m.group(yi)) * 4 + int(m.group(qi)) - 1
        m = re.fullmatch(r"q([1-4])|quarter\s*([1-4])", s)
        return float(int(m.group(1) or m.group(2))) if m else None
    if kind == "yearmonth":
        m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})", s)
        return _year(m.group(1)) * 12 + int(m.group(2)) - 1 if m and 1 <= int(m.group(2)) <= 12 else None
    if kind == "month":
        m = re.fullmatch(r"([a-z]{3})[a-z]*\.?(?:[\s\-/']+(\d{2}|\d{4}))?", s)
        if not m or m.group(1) not in _MONTHS:
            return None
        year = _year(m.group(2)) if m.group(2) else 0
        return year * 12 + _MONTHS[m.group(1)] - 1
    if kind == "week":
        m = re.fullmatch(r"(\d{4})\s*-?\s*w(\d{1,2})", s)
        if m:
            return _year(m.group(1)) * 54 + int(m.group(2))
        m = re.fullmatch(r"(?:w|wk|week)\s*(\d{1,2})", s)
        return float(m.group(1)) if m and 1 <= int(m.group(1)) <= 53 else None
    if kind == "weekday":
        m = re.fullmatch(r"([a-z]{3})[a-z]*", s)
        return float(_WEEKDAYS[m.group(1)]) if m and m.group(1) in _WEEKDAYS else None
    return None


_STRING_KINDS = ("year", "fiscal", "quarter", "yearmonth", "month", "week", "weekday")


def temporal_sort_keys(s: pd.Series, name: str = "", hinted: bool = False) -> tuple[pd.Series | None, str]:
    """Return ``(sort_key_series, kind)`` when ≥90% of non-null values are one temporal form."""
    nonnull = s.dropna()
    if nonnull.empty:
        return None, ""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.map(lambda t: t.value if pd.notna(t) else np.nan).astype("float64"), "date"
    w = _words(name)
    if pd.api.types.is_bool_dtype(s):
        return None, ""
    if pd.api.types.is_numeric_dtype(s):
        v = nonnull.astype(float)
        if not np.allclose(v, v.round()) or (w & _NOT_TIME_NAME):
            return None, ""
        if w & {"year", "fy", "yyyy"} and v.between(1900, 2100).all():
            return s.astype("float64"), "year"
        if w & {"month", "mm"} and v.between(1, 12).all():
            return s.astype("float64"), "month"
        if w & {"quarter", "qtr"} and v.between(1, 4).all():
            return s.astype("float64"), "quarter"
        if w & {"week"} and v.between(1, 53).all():
            return s.astype("float64"), "week"
        return None, ""
    txt = nonnull.astype(str)
    if txt.str.len().max() > 40:
        return None, ""
    # A column named like time tolerates a few malformed values (they are left out of the chart).
    need = 0.6 if (w & _TIME_NAME) else 0.9
    for kind in _STRING_KINDS:
        keys = txt.map(lambda x, k=kind: _parse_one(x, k))
        if keys.notna().mean() >= need:
            if kind == "year" and not (hinted or w & _TIME_NAME):
                continue  # a bare 4-digit text column is only time with a name/question hint
            out = pd.Series(np.nan, index=s.index, dtype="float64")
            out.loc[keys.index] = keys.astype("float64")
            return out, kind
    if txt.str.contains(r"\d").mean() >= need and (hinted or w & _TIME_NAME or
                                                   txt.str.match(r"^\d{4}-\d{2}-\d{2}").mean() >= 0.9):
        parsed = pd.to_datetime(txt, errors="coerce", format="mixed")
        if parsed.notna().mean() >= need:
            out = pd.Series(np.nan, index=s.index, dtype="float64")
            out.loc[parsed.index] = parsed.map(lambda t: t.value if pd.notna(t) else np.nan).astype("float64")
            return out, "date"
    return None, ""


# ═══════════════════════════════════════════════════════════════════════════════
# Profiling
# ═══════════════════════════════════════════════════════════════════════════════
def _numeric_text(s: pd.Series) -> tuple[pd.Series | None, bool]:
    """'1,234', '45%', '$1,200' → numbers when ≥95% parse. Returns (series, is_percent)."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return None, False
    txt = s.dropna().astype(str).str.strip()
    if txt.empty:
        return None, False
    pct = txt.str.endswith("%").mean() >= 0.9
    cleaned = txt.str.replace(r"[,$€£₹%\s]", "", regex=True).str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    if cleaned.str.match(r"^[+-]?0\d").mean() > 0.1:  # leading zeros → codes, not numbers
        return None, False
    num = pd.to_numeric(cleaned, errors="coerce")
    if num.notna().mean() < 0.95:
        return None, False
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[num.index] = num
    return out, pct


def profile_dataset(df: pd.DataFrame, context: dict | None = None) -> tuple[pd.DataFrame, dict[str, ColumnInfo]]:
    """Return a typed working copy (numeric text converted) and one ColumnInfo per column."""
    ctx = context or {}
    question = str(ctx.get("question") or "")
    grain = str(ctx.get("grain") or "").lower()
    temporal_intent = bool(_TEMPORAL_Q.search(question)) or grain in {"day", "week", "month", "quarter", "year"}
    declared_dims = {str(d) for d in (ctx.get("dimensions") or [])}

    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    d = d.loc[:, ~d.columns.duplicated()]
    n = len(d)
    infos: dict[str, ColumnInfo] = {}
    time_taken = False
    for c in d.columns:
        s = d[c]
        w = _words(c)
        info = ColumnInfo(c, "dimension", dtype=str(s.dtype), n_unique=int(s.nunique(dropna=True)),
                          null_pct=float(s.isna().mean()) if n else 0.0)
        if s.isna().all():
            info.role = "empty"
            infos[c] = info
            continue
        if pd.api.types.is_numeric_dtype(s) and s.dropna().map(
                lambda v: isinstance(v, float) and math.isinf(v)).any():
            d[c] = s.replace([np.inf, -np.inf], np.nan)
            s = d[c]
        hinted = temporal_intent or bool(w & _TIME_NAME) or c in declared_dims and bool(w & _TIME_NAME)
        if not time_taken:
            keys, kind = temporal_sort_keys(s, c, hinted=hinted)
            if keys is not None and (kind not in {"year", "month", "quarter", "week"}
                                     or not pd.api.types.is_numeric_dtype(s) or w & _TIME_NAME):
                info.role, info.time_kind, info.ordered = "time", kind, True
                time_taken = True
                infos[c] = info
                continue
        if pd.api.types.is_bool_dtype(s):
            infos[c] = info
            continue
        conv, pct_text = _numeric_text(s)
        if conv is not None:
            d[c] = conv
            s = conv
        if pd.api.types.is_numeric_dtype(s):
            v = s.dropna().astype(float)
            info.integer = bool(len(v)) and bool(np.allclose(v, v.round()))
            looks_id = bool(w & _ID_NAME) and not (w & (_ADD_NAME | _PCT_NAME | _AVG_NAME)) and \
                n > 1 and info.n_unique >= 0.9 * len(v)
            is_coord = bool(w & _COORD_NAME) and bool(len(v)) and bool(v.between(-180, 180).all())
            if is_coord:
                info.role = "coord"  # latitude / longitude: positions for a map, never a measure
            elif looks_id:
                info.role = "id"
            else:
                info.role = "measure"
                info.has_negative = bool((v < 0).any())
                info.percent = pct_text or bool(w & _PCT_NAME) or "%" in c
                avg_like = bool(w & (_AVG_NAME | _PCT_NAME)) or info.percent
                dur_like = bool(w & _DUR_NAME) and not (w & {"total", "sum"})
                info.additive = (not avg_like) and (not dur_like) and (
                    bool(w & _ADD_NAME) or (info.integer and not info.has_negative))
            infos[c] = info
            continue
        txt = s.dropna().astype(str)
        if (w & _ID_NAME) and n > 1 and info.n_unique >= 0.9 * len(txt):
            info.role = "id"
        elif not txt.empty and txt.str.len().mean() > 60 and info.n_unique >= 0.8 * len(txt):
            info.role = "text"
        else:
            info.ordered = bool(len(txt)) and txt.map(lambda x: bool(_ORDERED_CAT.match(x))).mean() >= 0.8
        infos[c] = info
    # A temporal question whose only remaining dimension names a period but did not parse
    # ("Period A", "Wave 1"): treat as an ordered sequence in returned order.
    if not time_taken and temporal_intent:
        seq = next((c for c, i in infos.items() if i.role == "dimension" and _words(c) & _TIME_NAME), None)
        if seq is not None:
            infos[seq].role, infos[seq].time_kind, infos[seq].ordered = "time", "sequence", True
    return d, infos


# ═══════════════════════════════════════════════════════════════════════════════
# Recommendation
# ═══════════════════════════════════════════════════════════════════════════════
_SYN = {"tat": "turnaround", "turnaround": "tat", "volume": "count", "overturns": "overturn"}
_GENERIC = {"avg", "average", "total", "count", "number", "rate", "of", "the", "by"}


def _primary_measure(measures: list[str], infos: dict, question: str, sql: str = "") -> str:
    m = re.search(r"\border\s+by\s+`?([^`,\s]+(?:\s[^`,]+)?)`?", sql or "", re.I)
    if m:
        hint = m.group(1).strip().lower()
        hit = next((c for c in measures if c.lower() == hint), None)
        if hit:
            return hit
    qw = _words(question)
    qw |= {_SYN[x] for x in qw if x in _SYN}
    scored = sorted(((len((_words(c) - _GENERIC) & qw), -i, c) for i, c in enumerate(measures)), reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][2]
    non_add = [c for c in measures if not infos[c].additive]
    return (non_add or measures)[0]


def _mixed_scale(df: pd.DataFrame, cols: list[str]) -> bool:
    mags = [float(df[c].abs().max()) for c in cols if df[c].notna().any() and float(df[c].abs().max()) > 0]
    return len(mags) >= 2 and max(mags) / min(mags) > 25


def _recommend_core(df: pd.DataFrame | None, ctx: dict, lim: VizLimits, sink: dict) -> VisualizationPlan:
    """Base recommendation (time series / categorical / numeric). ``sink`` receives the profiled
    frame so :func:`_augment` can reuse it instead of profiling twice."""
    question = str(ctx.get("question") or "")
    none = VisualizationPlan(None, [], question=question)
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            none.reason = "Empty result."
            return none
        if str(ctx.get("analysis_type") or "") == "drilldown":
            none.reason = "Row-level drill-down: table only."
            return none
        if len(df) > lim.max_render_rows:
            none.reason = f"More than {lim.max_render_rows:,} rows: table only."
            return none
        d, infos = profile_dataset(df, ctx)
        sink["d"], sink["infos"] = d, infos
        times = [c for c, i in infos.items() if i.role == "time"]
        dims = [c for c, i in infos.items() if i.role == "dimension"]
        measures = [c for c, i in infos.items() if i.role == "measure" and d[c].notna().any()]
        if not measures:
            none.reason = "No numeric measure to plot."
            return none
        preferred = str(ctx.get("preferred_measure") or "")  # set when a stored plan is upgraded
        primary = preferred if preferred in measures else _primary_measure(
            measures, infos, question, str(ctx.get("sql") or ""))
        measures = [primary] + [m for m in measures if m != primary]
        mi = infos[primary]
        vals = d[primary].dropna()
        n = len(d)
        plan = VisualizationPlan(None, [], y_axis=primary, measure_columns=measures, dimension_columns=dims,
                                 question=question)

        # ── KPI ───────────────────────────────────────────────────────────────
        if n == 1:
            plan.recommended_chart, plan.compatible_charts = "kpi", ["kpi"]
            plan.aggregation = "pre-aggregated"
            plan.reason = "Single-row result: a KPI is the only meaningful visual."
            return plan

        # ── Time series ──────────────────────────────────────────────────────
        if times:
            t = times[0]
            plan.time_column = plan.x_axis = t
            plan.is_time_series = True
            periods = int(d[t].nunique())
            tkeys, _ = temporal_sort_keys(d[t], t, hinted=True)
            bad = int((tkeys.isna() & d[t].notna()).sum()) if tkeys is not None else 0
            if bad:
                plan.notes.append(f"{bad} {pretty(t)} value(s) could not be read as time periods and are "
                                  "left out of the chart; the table is unchanged.")
            series = next((c for c in dims if 1 < d[c].nunique() <= lim.max_series), None)
            keys = [t] + ([series] if series else [])
            dup = d.duplicated(subset=keys).any()
            if dup and not mi.additive:
                plan.notes.append(f"{pretty(t)} repeats with a non-additive measure; values are not combined.")
                plan.compatible_charts = ["box"] if n >= lim.min_rows_distribution else []
                plan.recommended_chart = plan.compatible_charts[0] if plan.compatible_charts else None
                plan.aggregation = "observations"
                plan.reason = "Repeated periods of a rate/average: distribution per period only."
                return plan
            plan.aggregation = "sum" if dup else "pre-aggregated"
            plan.series = series
            if periods < 2:
                plan.compatible_charts = ["bar", "hbar"] if series else ["kpi"]
                plan.recommended_chart = plan.compatible_charts[0]
                plan.reason = "Only one period: no trend to show."
                return plan
            charts = ["line", "area"]
            if not mi.has_negative and series and mi.additive:
                charts.append("stacked_bar")
            charts += ["grouped_bar"] if series else ["bar"]
            if series and periods * d[series].nunique() <= lim.max_heatmap_cells:
                charts.append("heatmap")
            if mi.has_negative:
                charts = [c for c in charts if c not in {"area", "stacked_bar"}]
            plan.compatible_charts = [c for c in charts if c not in _TIME_FORBIDDEN]
            plan.recommended_chart = "line"
            plan.reason = (f"{pretty(t)} is a temporal dimension ({infos[t].time_kind}) with {periods} periods "
                           f"and a continuous measure: a trend line communicates change over time best.")
            if periods == 2:
                plan.notes.append("Only two periods: the line shows a single change.")
            return plan

        # ── Categorical ──────────────────────────────────────────────────────
        cat_dims = [c for c in dims if d[c].nunique() > 1]
        if cat_dims:
            a = cat_dims[0]
            n_a = int(d[a].nunique())
            plan.x_axis = a
            b = next((c for c in cat_dims[1:] if d[c].nunique() <= lim.max_series), None)
            keys = [a] + ([b] if b else [])
            dup = d.duplicated(subset=keys).any()
            if dup and not mi.additive:
                charts = ["box"] if n >= lim.min_rows_distribution else []
                plan.compatible_charts, plan.recommended_chart = charts, (charts[0] if charts else None)
                plan.aggregation = "observations"
                plan.reason = (f"{pretty(a)} repeats and {pretty(primary)} cannot be added up: "
                               "only its distribution per group is meaningful.")
                return plan
            plan.aggregation = "sum" if dup else "pre-aggregated"
            additive_share = (mi.additive and not mi.has_negative and float(vals.sum()) > 0
                              and vals.notna().all())
            if b:  # two dimensions
                plan.series = b
                cells = n_a * int(d[b].nunique())
                charts = []
                if n_a <= lim.max_bars:
                    charts.append("grouped_bar")
                    if additive_share:
                        charts.append("stacked_bar")
                if cells <= lim.max_heatmap_cells:
                    charts.append("heatmap")
                if additive_share and cells <= lim.max_treemap_nodes:
                    charts.append("treemap")
                plan.compatible_charts = charts
                plan.recommended_chart = ("heatmap" if "heatmap" in charts and n_a > 8 and d[b].nunique() > 4
                                          else (charts[0] if charts else None))
                plan.reason = (f"Two categorical dimensions ({pretty(a)} × {pretty(b)}) with one measure: "
                               "grouped comparison or a matrix view.")
                return plan
            long_labels = d[a].astype(str).str.len().max() > 14
            charts = ["bar", "hbar"]
            recommended = "hbar" if (n_a > 8 or long_labels) else "bar"
            if n_a > lim.high_cardinality:
                charts = ["hbar"]
                recommended = "hbar"
                plan.notes.append(f"{n_a} categories: the chart shows the top {lim.max_bars}; the table is complete.")
            if infos[a].ordered:
                charts = ["bar", "line"] if n_a >= 3 else ["bar"]
                recommended = "bar"
            share_ok = additive_share and lim.min_pie_slices <= n_a <= lim.max_pie_slices and not mi.percent
            if share_ok:
                charts += ["donut", "pie"]
                if _SHARE_Q.search(question):
                    recommended = "donut"
            if len(measures) >= 2:
                m2 = measures[1]
                if n_a <= 15 and not _mixed_scale(d, measures[:3]):
                    charts.append("grouped_bar")
                if n_a >= lim.min_rows_scatter and d[[primary, m2]].dropna().shape[0] >= lim.min_rows_scatter:
                    charts.append("scatter")
                    if _COMPARE_Q.search(question) and len(_words(question) & (_words(m2) - _GENERIC)):
                        recommended = "scatter"
            plan.compatible_charts = list(dict.fromkeys(charts))
            plan.recommended_chart = recommended
            plan.reason = (f"Categorical dimension {pretty(a)} ({n_a} values) with numeric measure "
                           f"{pretty(primary)}: ranked bars compare categories directly"
                           + ("; share charts are valid because the measure is additive." if share_ok else "."))
            return plan

        # ── Numeric only ─────────────────────────────────────────────────────
        charts = []
        if len(measures) >= 2 and d[measures[:2]].dropna().shape[0] >= lim.min_rows_scatter:
            charts.append("scatter")
            plan.x_axis, plan.y_axis = measures[0], measures[1]
        if vals.size >= lim.min_rows_distribution and vals.nunique() > 1:
            charts += ["histogram", "box"]
        plan.compatible_charts = charts
        plan.aggregation = "observations"
        if charts:
            plan.recommended_chart = ("histogram" if _DISTRIBUTION_Q.search(question) and "histogram" in charts
                                      else charts[0])
            plan.reason = ("Numeric observations without a category or time axis: relationship and "
                           "distribution views are meaningful.")
        else:
            plan.reason = "No category or time axis and too few observations for a distribution."
        return plan
    except Exception:  # recommendation must never break an answer
        log.exception("visualization recommendation failed")
        none.reason = "Recommendation unavailable."
        return none


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point + extended chart types (KPI, combo, waterfall, funnel, matrix, map, table)
# ═══════════════════════════════════════════════════════════════════════════════
def _stackable(info: ColumnInfo | None) -> bool:
    """A measure that may be stacked / summed: additive, not a rate, never negative."""
    return info is not None and info.additive and not info.percent and not info.has_negative


def _combo_roles(measures: list[str], infos: dict) -> tuple[str, list[str]]:
    """Bars = first additive volume measure; lines = the other measures (max 2) on a secondary axis."""
    ms = list(measures[:3])
    bar = next((c for c in ms if infos.get(c) is not None and infos[c].additive and not infos[c].percent), ms[0])
    return bar, [c for c in ms if c != bar][:2]


def _geo_keys(s: pd.Series, kind: str) -> pd.Series:
    t = s.astype("string").str.strip()
    if kind == "us_abbr":
        return t.str.upper()
    if kind == "us_name":
        return t.str.lower().map(_US_NAME_TO_ABBR)
    return t


def detect_geo(d: pd.DataFrame, infos: dict[str, ColumnInfo]) -> dict[str, str]:
    """Find a geography in the result: latitude/longitude pair, US states or countries ({} if none)."""
    coords = [c for c, i in infos.items() if i.role == "coord"]
    lat = next((c for c in coords if _words(c) & {"lat", "latitude"} and bool(d[c].dropna().between(-90, 90).all())),
               None)
    lon = next((c for c in coords if _words(c) & {"lon", "lng", "longitude"}), None)
    if lat and lon:
        return {"kind": "latlon", "lat": lat, "lon": lon}
    for c, i in infos.items():
        if i.role != "dimension" or pd.api.types.is_numeric_dtype(d[c]) or pd.api.types.is_bool_dtype(d[c]):
            continue
        vals = d[c].dropna().astype(str).str.strip()
        if vals.empty or vals.nunique() < 2:
            continue
        w = _words(c)
        up = vals.str.upper()
        if bool(up.isin(_US_STATES).mean() >= 0.9) and (w & _STATE_NAME or vals.nunique() >= 4):
            return {"kind": "us_abbr", "column": c}
        if bool(vals.str.lower().isin(_US_NAME_TO_ABBR).mean() >= 0.9):
            return {"kind": "us_name", "column": c}
        if w & _COUNTRY_NAME:
            return {"kind": "iso3" if bool(up.str.fullmatch(r"[A-Z]{3}").all()) else "country", "column": c}
    return {}


def _funnel_ok(d: pd.DataFrame, x: str | None, m: str, mi: ColumnInfo, question: str,
               plan: VisualizationPlan, lim: VizLimits) -> bool:
    """A funnel needs ordered stages and a measure that can only shrink along them."""
    if not x or plan.series or plan.is_time_series:
        return False
    if not mi.additive or mi.has_negative or mi.percent:
        return False
    if not lim.min_funnel_stages <= int(d[x].nunique()) <= lim.max_funnel_stages:
        return False
    asks = bool(_FUNNEL_Q.search(question))
    if not (asks or _words(x) & _STAGE_NAME):
        return False
    vals = d.groupby(x, sort=False)[m].sum().tolist()
    return asks or all(a >= b for a, b in zip(vals, vals[1:]))


def _augment(plan: VisualizationPlan, d: pd.DataFrame, infos: dict[str, ColumnInfo], ctx: dict,
             lim: VizLimits) -> VisualizationPlan:
    """Add the extended chart types the data supports to a base plan. Never removes a chart."""
    question = plan.question
    measures = [m for m in plan.measure_columns if m in d.columns]
    if not measures:
        return plan
    primary = measures[0]
    mi = infos[primary]
    x = plan.x_axis if (plan.x_axis in d.columns and plan.x_axis not in measures) else None
    series = plan.series if plan.series in d.columns else None
    n_x = int(d[x].nunique()) if x else 0
    charts = list(plan.compatible_charts)
    base_empty = not charts
    aggregated = plan.aggregation in {"pre-aggregated", "sum"}
    multi = aggregated and len(d) > 1 and not (plan.is_time_series and n_x < 2)
    extras: list[str] = []
    geo_plan: dict[str, str] = {}

    if multi and x:
        if not series and 2 <= n_x <= lim.max_bars and len(measures) >= 2:
            extras.append("combo")
        stack = [c for c in measures[:3] if _stackable(infos.get(c))]
        if (not series and len(stack) >= 2 and n_x <= lim.max_bars and not _mixed_scale(d, stack)
                and "stacked_bar" not in charts):
            charts.append("stacked_bar")
        wf_ok = not mi.percent and (mi.additive or bool(_words(primary) & _DELTA_NAME))
        if wf_ok and not series and lim.min_waterfall_steps <= n_x <= lim.max_waterfall_steps:
            extras.append("waterfall")
        if _funnel_ok(d, x, primary, mi, question, plan, lim):
            extras.append("funnel")
        if series and "heatmap" in charts:
            n_s = int(d[series].nunique())
            if min(n_x, n_s) <= lim.max_matrix_cols and max(n_x, n_s) <= lim.max_matrix_rows:
                extras.append("matrix")
        if not series or mi.additive:
            extras.append("kpi")

    geo = detect_geo(d, infos)
    if geo and not plan.is_time_series:
        if geo["kind"] == "latlon":
            geo_plan = {"geo_kind": "latlon", "lat_column": geo["lat"], "lon_column": geo["lon"]}
        elif aggregated and geo["column"] in d.columns and (mi.additive or not d.duplicated(subset=[geo["column"]]).any()):
            geo_plan = {"geo_kind": geo["kind"], "geo_column": geo["column"]}
        if geo_plan:
            extras.append("map")

    if base_empty and "map" not in extras:
        return plan  # nothing meaningful to chart: keep the result table-only
    for c in extras:
        if c not in charts:
            charts.append(c)
    charts.append("table")

    recommended, reason = plan.recommended_chart, plan.reason

    def pick(chart: str, why: str) -> None:
        nonlocal recommended, reason
        if chart in charts:
            recommended, reason = chart, f"{reason} {why}".strip()

    if base_empty:
        pick("map", "Geographic result: a map shows where the values concentrate.")
    else:
        for rx, chart, why in (
            (_FUNNEL_Q, "funnel", "The question describes a staged flow, so a funnel shows the drop between stages."),
            (_WATERFALL_Q, "waterfall", "The question asks for a build-up, so a waterfall shows each step's contribution."),
            (_MAP_Q, "map", "The question asks for a geographic view."),
            (_MATRIX_Q, "matrix", "The question asks for a pivot, so a matrix is shown."),
            (_KPI_Q, "kpi", "The question asks for headline numbers."),
            (_TABLE_Q, "table", "The question asks for a table."),
        ):
            if chart in extras + ["table"] and rx.search(question):
                pick(chart, why)
                break
        else:
            if "combo" in extras and not plan.is_time_series and recommended in {"bar", "hbar"} and n_x <= 15:
                bar_m, line_ms = _combo_roles(measures, infos)
                if infos[bar_m].additive and any(infos[c].percent for c in line_ms):
                    pick("combo", "A combo chart pairs the volume (bars) with the rate (line).")

    plan.compatible_charts = list(dict.fromkeys(charts))
    plan.recommended_chart, plan.reason = recommended, reason
    for k, v in geo_plan.items():
        setattr(plan, k, v)
    return plan


def recommend_visualizations(df: pd.DataFrame | None, query_context: dict | None = None,
                             limits: VizLimits | None = None) -> VisualizationPlan:
    """Deterministic chart recommendation for one result table.

    ``query_context`` keys (all optional): ``question``, ``sql``, ``grain``,
    ``dimensions``, ``analysis_type``, ``is_aggregate``, ``preferred_measure``.
    """
    lim = limits or VizLimits()
    ctx = query_context or {}
    sink: dict = {}
    plan = _recommend_core(df, ctx, lim, sink)
    plan.grain = str(ctx.get("grain") or "").lower()
    plan.declared_dimensions = [str(c) for c in (ctx.get("dimensions") or [])]
    if sink and plan.measure_columns:
        try:
            plan = _augment(plan, sink["d"], sink["infos"], ctx, lim)
        except Exception:  # extended options are optional; the base plan stays valid
            log.exception("extended chart options failed; keeping the base recommendation")
    return plan


# ═══════════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════════
def _fmt(v: float, info: ColumnInfo | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if info is not None and info.percent:
        return f"{v:,.1f}%"
    if abs(v - round(v)) < 1e-9:
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def _time_frame(d: pd.DataFrame, t: str, info: ColumnInfo) -> pd.DataFrame:
    keys, _ = temporal_sort_keys(d[t], t, hinted=True)
    out = d.copy()
    out["__k"] = keys if keys is not None else np.arange(len(d), dtype=float)
    out = out.dropna(subset=["__k"]) if info.time_kind != "sequence" else out
    return out.sort_values("__k", kind="mergesort")


def _combine(d: pd.DataFrame, keys: list[str], m: str, plan: VisualizationPlan) -> pd.DataFrame:
    """Combine duplicate keys only when the plan allows (additive measure)."""
    if plan.aggregation == "sum" and d.duplicated(subset=keys).any():
        extra = ["__k"] if "__k" in d.columns else []
        agg = {m: "sum", **{e: "min" for e in extra}}
        return d.groupby(keys, as_index=False, sort=False, dropna=False).agg(agg)
    return d


def _by_key(d: pd.DataFrame, key: str, m: str, additive: bool) -> pd.DataFrame | None:
    """One row per ``key``. Duplicates are summed only for additive measures (else ``None``)."""
    if not d.duplicated(subset=[key]).any():
        return d
    if not additive:
        return None
    agg = {m: "sum", **({"__k": "min"} if "__k" in d.columns else {})}
    out = d.groupby(key, sort=False, dropna=True, as_index=False).agg(agg)
    return out.sort_values("__k", kind="mergesort") if "__k" in out.columns else out


def _num_fmt(v: float, percent: bool) -> str:
    return ",.1f" if percent else (",.0f" if abs(v - round(v)) < 1e-9 else ",.2f")


def _kpi_cards(d: pd.DataFrame, infos: dict[str, ColumnInfo], p: VisualizationPlan) -> list[dict]:
    """Up to four headline numbers. Never adds up a rate or average: non-additive measures
    show real values only (latest, highest, lowest)."""
    ms = [c for c in p.measure_columns if c in d.columns]
    cards: list[dict] = []

    def card(title: str, value: float, m: str, ref: float | None = None, whole: bool = False) -> None:
        pct = bool(infos.get(m) and infos[m].percent)
        cards.append({"title": title, "value": float(value), "ref": ref, "suffix": "%" if pct else "",
                      "fmt": ",.0f" if whole else _num_fmt(float(value), pct)})

    if len(d) == 1:
        for c in ms[:4]:
            v = d[c].dropna()
            if not v.empty:
                card(pretty(c), v.iloc[0], c)
        return cards
    m = ms[0]
    mi = infos[m]
    t, x = p.time_column, p.x_axis
    if t and t in d.columns:
        per = _by_key(_time_frame(d, t, infos[t]), t, m, mi.additive)
        if per is None:
            return cards
        per = per.dropna(subset=[m])
        if per.empty:
            return cards
        last = per.iloc[-1]
        prev = float(per[m].iloc[-2]) if len(per) > 1 else None
        card(f"Latest · {last[t]}", last[m], m, ref=prev if prev not in (None, 0.0) else None)
        hi = per.loc[per[m].idxmax()]
        card(f"Peak · {hi[t]}", hi[m], m)
        if mi.additive:
            card(f"Total · {len(per)} periods", per[m].sum(), m)
        else:
            lo = per.loc[per[m].idxmin()]
            card(f"Lowest · {lo[t]}", lo[m], m)
        return cards
    if x and x in d.columns:
        per = _by_key(d, x, m, mi.additive)
        if per is None:
            return cards
        per = per.dropna(subset=[m])
        if per.empty:
            return cards
        hi, lo = per.loc[per[m].idxmax()], per.loc[per[m].idxmin()]
        if mi.additive:
            card(f"Total · {pretty(m)}", per[m].sum(), m)
            card(f"Top · {hi[x]}", hi[m], m)
        else:
            card(f"Highest · {hi[x]}", hi[m], m)
            card(f"Lowest · {lo[x]}", lo[m], m)
        card(f"{pretty(x)} values", len(per), m, whole=True)
        cards[-1]["suffix"] = ""
    return cards


def _shade(v: float, lo: float, hi: float) -> tuple[str, str]:
    """(fill, font) colour for a matrix cell: white → blue by value."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "#ffffff", "#6b7280"
    f = 0.0 if hi <= lo else min(1.0, max(0.0, (v - lo) / (hi - lo)))
    r, g, b = (int(255 + (c - 255) * f) for c in (31, 119, 180))
    return f"#{r:02x}{g:02x}{b:02x}", ("#ffffff" if f > 0.6 else "#111827")


def _table_figure(d: pd.DataFrame, infos: dict[str, ColumnInfo], lim: VizLimits) -> tuple[go.Figure, int, list[str]]:
    shown = d.head(lim.max_table_rows)
    cols = [c for c in shown.columns if c != "__k"]
    values = []
    for c in cols:
        info = infos.get(c)
        if pd.api.types.is_numeric_dtype(shown[c]) and not pd.api.types.is_bool_dtype(shown[c]):
            values.append([_fmt(float(v), info) if pd.notna(v) else "" for v in shown[c]])
        else:
            values.append(["" if pd.isna(v) else str(v) for v in shown[c]])
    fig = go.Figure(go.Table(
        header=dict(values=[pretty(c) for c in cols], fill_color="#e8eef6", align="left",
                    font=dict(color="#111827")),
        cells=dict(values=values, align="left", fill_color="#ffffff", font=dict(color="#111827"))))
    notes = [f"Showing the first {lim.max_table_rows:,} of {len(d):,} rows."] if len(d) > lim.max_table_rows else []
    return fig, len(shown), notes


def render_chart(df: pd.DataFrame, plan: VisualizationPlan | dict, chart_type: str,
                 *, height: int = 460, limits: VizLimits | None = None) -> tuple[go.Figure | None, list[str]]:
    """Draw ``chart_type`` from the already-returned ``df``. Never raises; no LLM, no SQL."""
    lim = limits or VizLimits()
    p = plan if isinstance(plan, VisualizationPlan) else VisualizationPlan.from_dict(plan)
    notes: list[str] = []
    if p is None or df is None or df.empty or chart_type not in p.compatible_charts:
        return None, notes
    try:
        d, infos = profile_dataset(df, {"question": p.question, "grain": p.grain, "dimensions": p.declared_dimensions})
        m = p.y_axis if p.y_axis in d.columns else (p.measure_columns[0] if p.measure_columns else None)
        if m is None:
            return None, notes
        mi = infos.get(m)
        x, s = p.x_axis, p.series
        t = p.time_column
        fig = go.Figure()
        title = ""

        if chart_type == "kpi":
            cards = _kpi_cards(d, infos, p)
            if not cards:
                return None, notes
            k = len(cards)
            for i, c in enumerate(cards):
                extra = {}
                if c["ref"] is not None:
                    extra = {"delta": {"reference": c["ref"], "relative": True, "valueformat": ".1%"}}
                fig.add_trace(go.Indicator(
                    mode="number+delta" if extra else "number", value=c["value"],
                    number={"valueformat": c["fmt"], "suffix": c["suffix"]}, title={"text": c["title"]},
                    domain={"x": [i / k + 0.01, (i + 1) / k - 0.01], "y": [0, 1]}, **extra))
            return style_figure(fig, 240), notes

        if chart_type == "table":
            fig, _, tnotes = _table_figure(d, infos, lim)
            notes += tnotes
            return style_figure(fig, min(700, 120 + 30 * min(len(d), lim.max_table_rows))), notes

        if t and chart_type in {"line", "area", "bar", "hbar", "grouped_bar", "stacked_bar", "heatmap", "box",
                                "combo", "waterfall", "matrix"}:
            d = _time_frame(d, t, infos[t])
            order = list(dict.fromkeys(d[t].astype(str)))
        else:
            order = None

        if chart_type in {"line", "area"}:
            if t is None:  # ordered categories drawn as a line
                d = d.copy()
                order = list(dict.fromkeys(d[x].astype(str)))
            xcol = t or x
            groups = [(None, d)] if not s else [(g, d[d[s] == g]) for g in sorted(d[s].dropna().unique(), key=str)]
            for i, (g, gd) in enumerate(groups):
                gd = _combine(gd, [xcol], m, p)
                fig.add_trace(go.Scatter(
                    x=gd[xcol].astype(str) if infos[xcol].time_kind != "date" else gd[xcol], y=gd[m],
                    name=str(g) if g is not None else pretty(m),
                    mode="lines+markers", line=dict(width=3 if not s else 2, color=PALETTE[i % len(PALETTE)]),
                    fill=("tonexty" if (s and i) else "tozeroy") if chart_type == "area" else None,
                    stackgroup=("one" if (chart_type == "area" and s and mi and mi.additive) else None),
                    connectgaps=False,
                    hovertemplate=f"%{{x}}<br>{pretty(m)}: %{{y:,.2f}}<extra>{g if g is not None else ''}</extra>",
                ))
            if infos[xcol].time_kind != "date" and order:
                fig.update_xaxes(type="category", categoryorder="array", categoryarray=order)
            fig.update_layout(hovermode="x unified", showlegend=bool(s))
            title = f"{pretty(m)} over {pretty(xcol)}" + (f" by {pretty(s)}" if s else "")

        elif chart_type in {"bar", "hbar"}:
            xcol = t or x
            plot = _combine(d, [xcol], m, p).dropna(subset=[m])
            if not t and not infos[xcol].ordered:
                plot = plot.sort_values([m, xcol], ascending=[False, True], kind="mergesort")
            total = len(plot)
            if total > lim.max_bars:
                plot = plot.head(lim.max_bars)
                notes.append(f"Showing the top {lim.max_bars} of {total} {pretty(xcol)} values; "
                             "the table is complete.")
            labels = plot[xcol].astype(str).tolist()
            text = [_fmt(v, mi) for v in plot[m]]
            if chart_type == "hbar":
                fig.add_trace(go.Bar(y=labels, x=plot[m], orientation="h", text=text, textposition="outside",
                                     marker_color=PALETTE[0], cliponaxis=False))
                fig.update_yaxes(autorange="reversed", type="category")
                height = max(height, min(900, 80 + 28 * len(plot)))
            else:
                fig.add_trace(go.Bar(x=labels, y=plot[m], text=text, textposition="outside",
                                     marker_color=PALETTE[0], cliponaxis=False))
                fig.update_xaxes(type="category", tickangle=-25 if len(plot) > 6 else 0)
            fig.update_layout(showlegend=False)
            title = f"{pretty(m)} by {pretty(xcol)}"

        elif chart_type in {"grouped_bar", "stacked_bar"}:
            xcol = t or x
            if s:
                plot = _combine(d, [xcol, s], m, p)
                xs = order or list(dict.fromkeys(
                    plot.groupby(xcol, sort=False)[m].sum().sort_values(ascending=False).index.astype(str)))
                for i, g in enumerate(sorted(plot[s].dropna().unique(), key=str)):
                    gd = plot[plot[s] == g].set_index(plot[plot[s] == g][xcol].astype(str)).reindex(xs)
                    fig.add_trace(go.Bar(x=xs, y=gd[m], name=str(g), marker_color=PALETTE[i % len(PALETTE)]))
                title = f"{pretty(m)} by {pretty(xcol)} and {pretty(s)}"
            else:  # several measures side by side
                ms = [c for c in p.measure_columns[:3] if c in d.columns]
                if chart_type == "stacked_bar":
                    ms = [c for c in ms if _stackable(infos.get(c))]  # never stack a rate onto a count
                if not ms:
                    return None, notes
                keep_order = bool(t) or infos[xcol].ordered
                plot = (d if keep_order else d.sort_values(ms[0], ascending=False, kind="mergesort")).head(lim.max_bars)
                for i, mm in enumerate(ms):
                    fig.add_trace(go.Bar(x=plot[xcol].astype(str), y=plot[mm], name=pretty(mm),
                                         marker_color=PALETTE[i % len(PALETTE)]))
                title = f"{', '.join(pretty(c) for c in ms)} by {pretty(xcol)}"
            fig.update_xaxes(type="category")
            fig.update_layout(barmode="stack" if chart_type == "stacked_bar" else "group",
                              legend=dict(orientation="h", y=1.1))

        elif chart_type in {"pie", "donut"}:
            plot = _combine(d, [x], m, p).dropna(subset=[m]).sort_values([m, x], ascending=[False, True])
            fig.add_trace(go.Pie(labels=plot[x].astype(str), values=plot[m], sort=False,
                                 hole=0.55 if chart_type == "donut" else 0, textinfo="label+percent",
                                 marker=dict(colors=PALETTE[: len(plot)])))
            title = f"Share of {pretty(m)} by {pretty(x)}"

        elif chart_type == "treemap":
            path = [x] + ([s] if s else [])
            plot = d.dropna(subset=path + [m])
            plot = plot[plot[m] > 0]
            ids, labels, parents, values = [], [], [], []
            for top, grp in plot.groupby(x, sort=True):
                ids.append(str(top)); labels.append(str(top)); parents.append(""); values.append(float(grp[m].sum()))
                if s:
                    for leaf, lg in grp.groupby(s, sort=True):
                        ids.append(f"{top}/{leaf}"); labels.append(str(leaf)); parents.append(str(top))
                        values.append(float(lg[m].sum()))
            fig.add_trace(go.Treemap(ids=ids, labels=labels, parents=parents, values=values,
                                     branchvalues="total", textinfo="label+value+percent root"))
            title = f"{pretty(m)} by {' › '.join(pretty(c) for c in path)}"

        elif chart_type == "heatmap":
            row = t or x
            plot = _combine(d, [row, s], m, p)
            piv = plot.pivot(index=s, columns=row, values=m)
            cols = order or sorted(piv.columns.astype(str))
            piv.columns = piv.columns.astype(str)
            piv = piv.reindex(columns=cols).sort_index()
            fig.add_trace(go.Heatmap(z=piv.values, x=cols, y=piv.index.astype(str), colorscale="Blues",
                                     hoverongaps=False, texttemplate="%{z:,.1f}"))
            fig.update_xaxes(type="category")
            title = f"{pretty(m)}: {pretty(s)} × {pretty(row)}"

        elif chart_type == "scatter":
            xm = p.x_axis if (p.x_axis in p.measure_columns) else p.measure_columns[0]
            ym = p.y_axis if p.y_axis != xm else p.measure_columns[1]
            if xm == p.measure_columns[0] and ym == xm:
                return None, notes
            if xm not in p.measure_columns:
                xm, ym = p.measure_columns[1], p.measure_columns[0]
            label = x if x in d.columns and x not in (xm, ym) else None
            plot = d.dropna(subset=[xm, ym])
            trace = go.Scattergl if len(plot) > 2000 else go.Scatter
            size_col = next((c for c in p.measure_columns if c not in (xm, ym) and c in plot.columns
                             and plot[c].notna().all() and (plot[c] >= 0).all() and plot[c].max() > 0), None)
            if size_col is not None and len(plot) <= 2000:  # bubble chart: third measure = marker area
                sizes = (8 + 32 * np.sqrt(plot[size_col] / plot[size_col].max())).tolist()
                notes.append(f"Bubble size shows {pretty(size_col)}.")
            else:
                sizes = 10
            fig.add_trace(trace(x=plot[xm], y=plot[ym], mode="markers+text" if label and len(plot) <= 20 else "markers",
                                text=plot[label].astype(str) if label else None, textposition="top center",
                                marker=dict(size=sizes, color=PALETTE[0], opacity=.8)))
            fig.update_layout(xaxis_title=pretty(xm), yaxis_title=pretty(ym), showlegend=False)
            title = f"{pretty(ym)} vs {pretty(xm)}"

        elif chart_type == "histogram":
            v = d[m].dropna()
            fig.add_trace(go.Histogram(x=v, nbinsx=min(40, max(5, int(math.sqrt(len(v))))),
                                       marker_color=PALETTE[0]))
            fig.update_layout(bargap=0.05, xaxis_title=pretty(m), yaxis_title="Count", showlegend=False)
            title = f"Distribution of {pretty(m)}"

        elif chart_type == "box":
            grp = t or x
            if grp:
                for i, g in enumerate(order or sorted(d[grp].dropna().astype(str).unique())):
                    fig.add_trace(go.Box(y=d.loc[d[grp].astype(str) == g, m], name=g,
                                         marker_color=PALETTE[i % len(PALETTE)], boxmean=True))
                title = f"{pretty(m)} distribution by {pretty(grp)}"
            else:
                fig.add_trace(go.Box(y=d[m].dropna(), name=pretty(m), marker_color=PALETTE[0], boxmean=True))
                title = f"{pretty(m)} distribution"
            fig.update_layout(showlegend=False)
        elif chart_type == "combo":
            xcol = t or x
            bar_m, line_ms = _combo_roles(p.measure_columns, infos)
            plot = d if (t or infos[xcol].ordered) else d.sort_values(bar_m, ascending=False, kind="mergesort")
            plot = _by_key(plot, xcol, bar_m, bool(infos[bar_m].additive))
            if plot is None:
                return None, notes
            plot = plot.head(lim.max_bars)
            labels = plot[xcol].astype(str).tolist()
            fig.add_trace(go.Bar(x=labels, y=plot[bar_m], name=pretty(bar_m), marker_color=PALETTE[0],
                                 yaxis="y", hovertemplate=f"%{{x}}<br>{pretty(bar_m)}: %{{y:,.2f}}<extra></extra>"))
            for i, lm in enumerate(line_ms, 1):
                fig.add_trace(go.Scatter(x=labels, y=plot[lm], name=pretty(lm), mode="lines+markers", yaxis="y2",
                                         line=dict(width=3, color=PALETTE[i % len(PALETTE)]),
                                         hovertemplate=f"%{{x}}<br>{pretty(lm)}: %{{y:,.2f}}<extra></extra>"))
            pct_lines = all(infos[c].percent for c in line_ms)
            fig.update_xaxes(type="category", **({"categoryorder": "array", "categoryarray": labels}))
            fig.update_layout(
                yaxis=dict(title=pretty(bar_m)),
                yaxis2=dict(title=", ".join(pretty(c) for c in line_ms), overlaying="y", side="right",
                            showgrid=False, ticksuffix="%" if pct_lines else ""),
                legend=dict(orientation="h", y=1.1), hovermode="x unified")
            notes.append("Two axes: bars use the left axis, lines the right axis.")
            title = f"{pretty(bar_m)} and {', '.join(pretty(c) for c in line_ms)} by {pretty(xcol)}"

        elif chart_type == "waterfall":
            xcol = t or x
            plot = _by_key(d, xcol, m, bool(mi and mi.additive))
            if plot is None:
                return None, notes
            plot = plot.dropna(subset=[m])
            if len(plot) < 2:
                return None, notes
            if not t and not infos[xcol].ordered:
                plot = plot.sort_values([m, xcol], ascending=[False, True], kind="mergesort")
            vals = plot[m].astype(float).tolist()
            labels = plot[xcol].astype(str).tolist()
            fig.add_trace(go.Waterfall(
                x=labels + ["Total"], y=vals + [0], measure=["relative"] * len(vals) + ["total"],
                text=[_fmt(v, mi) for v in vals] + [_fmt(sum(vals), mi)], textposition="outside",
                connector={"line": {"width": 1}}, cliponaxis=False,
                increasing={"marker": {"color": "#2e9e6b"}}, decreasing={"marker": {"color": "#d64545"}},
                totals={"marker": {"color": PALETTE[0]}}))
            fig.update_xaxes(type="category")
            fig.update_yaxes(title=pretty(m))
            fig.update_layout(showlegend=False)
            notes.append("Each bar moves the running total; the last bar is the overall total.")
            title = f"{pretty(m)} build-up by {pretty(xcol)}"

        elif chart_type == "funnel":
            plot = _by_key(d, x, m, True)
            plot = plot.dropna(subset=[m])
            vals = plot[m].astype(float).tolist()
            in_order = all(a >= b for a, b in zip(vals, vals[1:])) or bool(_words(x) & _STAGE_NAME) \
                or bool(_FUNNEL_Q.search(p.question))
            if not in_order:
                plot = plot.sort_values([m, x], ascending=[False, True], kind="mergesort")
                notes.append("Stages are ordered by value because the result has no stage order.")
            fig.add_trace(go.Funnel(
                y=plot[x].astype(str), x=plot[m], textinfo="value+percent initial",
                marker=dict(color=[PALETTE[i % len(PALETTE)] for i in range(len(plot))])))
            fig.update_layout(showlegend=False)
            title = f"{pretty(m)} by {pretty(x)}"

        elif chart_type == "matrix":
            a, b = t or x, s
            n_a, n_b = int(d[a].nunique()), int(d[b].nunique())
            rdim, cdim = (a, b) if n_b <= n_a else (b, a)  # fewer values across the top
            plot = _combine(d, [a, b], m, p)
            piv = plot.groupby([rdim, cdim], sort=False)[m].first().unstack(cdim)

            def _order(col: str, labels: pd.Index) -> list[str]:
                present = [str(v) for v in labels]
                if col == t:
                    return [v for v in order if v in present]
                if infos[col].ordered:
                    return [v for v in dict.fromkeys(d[col].dropna().astype(str)) if v in present]
                return sorted(present)

            piv.index, piv.columns = piv.index.astype(str), piv.columns.astype(str)
            piv = piv.reindex(index=_order(rdim, piv.index), columns=_order(cdim, piv.columns))
            if mi and mi.additive and not mi.percent and not (t is None and infos[rdim].ordered):
                if rdim != t:
                    piv = piv.loc[piv.sum(axis=1, min_count=1).sort_values(ascending=False, kind="mergesort").index]
            if len(piv) > lim.max_matrix_rows:
                notes.append(f"Showing the first {lim.max_matrix_rows} of {len(piv)} {pretty(rdim)} values; "
                             "the table is complete.")
                piv = piv.head(lim.max_matrix_rows)
            inner = piv.to_numpy(dtype=float)
            lo, hi = (float(np.nanmin(inner)), float(np.nanmax(inner))) if np.isfinite(inner).any() else (0.0, 0.0)
            totals = bool(mi and mi.additive and not mi.percent)
            heads = [f"{pretty(rdim)} \\ {pretty(cdim)}"] + list(piv.columns) + (["Total"] if totals else [])
            cols_vals, cols_fill, cols_font = [list(piv.index) + (["Total"] if totals else [])], [], []
            cols_fill.append(["#f3f4f6"] * (len(piv) + int(totals)))
            cols_font.append(["#111827"] * (len(piv) + int(totals)))
            for j, cname in enumerate(piv.columns):
                col = piv[cname]
                txt = [_fmt(v, mi) if pd.notna(v) else "–" for v in col]
                shades = [_shade(v, lo, hi) for v in col]
                if totals:
                    txt.append(_fmt(float(col.sum()), mi))
                    shades.append(("#eef2f7", "#111827"))
                cols_vals.append(txt)
                cols_fill.append([sh[0] for sh in shades])
                cols_font.append([sh[1] for sh in shades])
            if totals:
                rt = piv.sum(axis=1, min_count=1)
                cols_vals.append([_fmt(float(v), mi) if pd.notna(v) else "–" for v in rt] + [_fmt(float(rt.sum()), mi)])
                cols_fill.append(["#eef2f7"] * (len(piv) + 1))
                cols_font.append(["#111827"] * (len(piv) + 1))
            fig.add_trace(go.Table(
                header=dict(values=heads, fill_color="#e8eef6", align="left", font=dict(color="#111827")),
                cells=dict(values=cols_vals, fill_color=cols_fill, font=dict(color=cols_font), align="right")))
            height = max(240, min(900, 90 + 30 * (len(piv) + int(totals))))
            title = f"{pretty(m)}: {pretty(rdim)} × {pretty(cdim)}"

        elif chart_type == "map":
            if p.geo_kind == "latlon" and p.lat_column in d.columns and p.lon_column in d.columns:
                plot = d.dropna(subset=[p.lat_column, p.lon_column, m])
                if plot.empty:
                    return None, notes
                vmax = float(plot[m].clip(lower=0).max())
                sizes = (8 + 30 * np.sqrt(plot[m].clip(lower=0) / vmax)).tolist() if vmax > 0 else 12
                label_col = next((c for c in p.dimension_columns if c in plot.columns), None)
                fig.add_trace(go.Scattergeo(
                    lat=plot[p.lat_column], lon=plot[p.lon_column], mode="markers",
                    text=[f"{lbl}: {_fmt(v, mi)}" for lbl, v in zip(plot[label_col].astype(str) if label_col else
                                                                       [pretty(m)] * len(plot), plot[m])],
                    hoverinfo="text", marker=dict(size=sizes, color=PALETTE[0], opacity=.75,
                                                  line=dict(width=.5, color="white"))))
                fig.update_geos(fitbounds="locations", showcountries=True)
                title = f"{pretty(m)} by location"
            elif p.geo_column in d.columns:
                keys = _geo_keys(d[p.geo_column], p.geo_kind)
                frame = pd.DataFrame({"k": keys, "v": d[m]}).dropna()
                if frame.empty:
                    return None, notes
                frame = frame.groupby("k", sort=False)["v"].sum() if (mi and mi.additive) else \
                    frame.groupby("k", sort=False)["v"].first()
                is_us = p.geo_kind in {"us_abbr", "us_name"}
                fig.add_trace(go.Choropleth(
                    locations=frame.index.tolist(), z=frame.tolist(), colorscale="Blues",
                    locationmode="USA-states" if is_us else ("ISO-3" if p.geo_kind == "iso3" else "country names"),
                    colorbar=dict(title=pretty(m)), hovertemplate="%{location}<br>" + pretty(m) + ": %{z:,.2f}<extra></extra>"))
                if is_us:
                    fig.update_geos(scope="usa")
                else:
                    fig.update_geos(fitbounds="locations")
                title = f"{pretty(m)} by {pretty(p.geo_column)}"
            else:
                return None, notes
            height = max(height, 480)

        else:
            return None, notes

        if chart_type not in {"pie", "donut", "treemap", "heatmap", "scatter", "histogram", "combo", "waterfall",
                              "funnel", "matrix", "map", "table", "kpi"}:
            fig.update_yaxes(title=pretty(m) if chart_type != "hbar" else None,
                             ticksuffix="%" if mi and mi.percent and chart_type != "hbar" else None)
            if chart_type == "hbar":
                fig.update_xaxes(title=pretty(m))
        fig.update_layout(title=title, meta={"chart_type": chart_type, "viz_version": VIZ_VERSION})
        return style_figure(fig, height), notes
    except Exception:
        log.exception("chart rendering failed for %s", chart_type)
        return None, ["This visualization could not be drawn for the result; try another type."]


def existing_chart_type(figures: list) -> str | None:
    """Best-effort chart kind of an existing Plotly figure list (for the 'Auto' view)."""
    for f in figures or []:
        try:
            tr = f.data[0]
        except (AttributeError, IndexError):
            continue
        kind = tr.type
        if kind == "scatter":
            return "area" if getattr(tr, "fill", None) or getattr(tr, "stackgroup", None) else (
                "line" if "lines" in str(tr.mode or "lines") else "scatter")
        if kind == "bar":
            return "hbar" if tr.orientation == "h" else "bar"
        if kind == "pie":
            return "donut" if (tr.hole or 0) > 0 else "pie"
        return {"indicator": "kpi", "heatmap": "heatmap", "treemap": "treemap", "histogram": "histogram",
                "box": "box", "table": "table", "waterfall": "waterfall", "funnel": "funnel",
                "choropleth": "map", "scattergeo": "map"}.get(kind, kind)
    return None


__all__ = ["VisualizationPlan", "VizLimits", "ColumnInfo", "CHART_LABELS", "CHART_ICONS", "VIZ_VERSION",
           "profile_dataset", "recommend_visualizations", "render_chart", "temporal_sort_keys",
           "existing_chart_type", "pretty", "detect_geo"]