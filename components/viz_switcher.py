"""Dynamic visualization switching for one chat answer (Streamlit UI only).

Renders a compact **Visualization ▾** selector above the answer's chart. Only
chart types the recommendation engine marked as compatible are listed, and the
selector is hidden when there is a single meaningful visual.

Switching is local and cheap:

* runs inside ``st.fragment`` → only this chart area reruns, not the page;
* draws from the table already stored in the answer → **no LLM call, no SQL,
  no supervisor call**; the answer text, insight panel, tables and filters are
  untouched;
* figures are memoised with ``st.cache_data`` (same table + plan + type → same
  figure);
* the selection lives in ``st.session_state`` under a per-answer key, so it
  survives reruns and never leaks into another answer.

"Auto" shows the figures the answer was generated with (unchanged behaviour).
One safety correction applies: when the data is a time series but the original
figure is a bar/pie chart, Auto shows the recommended line chart instead.

Every visual has an **⬇️ Export** menu (interactive HTML, PNG when ``kaleido`` is installed, and the
CSV of the rows shown), so a chart can be shared outside the app.

Chart catalogue: KPI cards, Column, Bar, Line, Area, Grouped/Stacked, Pie, Donut, Combo,
Waterfall, Funnel, Scatter/Bubble, Matrix, Heatmap, Treemap, Histogram, Box, Map and Table.
The **Slicer** (🎚️) is a filter control next to the selector: it narrows the rows used by
*this visual only* (multi-select per dimension, no SQL). While a slicer is active, "Auto"
draws the recommended chart from the filtered rows, because the original figure cannot be
filtered. Plans stored by an older recommender version are upgraded on the fly, so cached
answers also get the full chart list.
"""
from __future__ import annotations

import hashlib
import html
import importlib.util
import io
import logging
import re
import zipfile
from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from core.plot_style import PLOTLY_CONFIG
from core.viz_recommender import (
    CHART_ICONS,
    CHART_LABELS,
    VIZ_VERSION,
    VisualizationPlan,
    existing_chart_type,
    pretty,
    recommend_visualizations,
    render_chart,
)

log = logging.getLogger("datatalk.viz_switcher")

AUTO = "auto"
TABLE = "table"
_DERIVE_ROUTES = {"analytics_engine", "direct_sql_fallback"}
RenderOriginal = Callable[[list, str], None]
_TIME_SERIES_BAD_DEFAULTS = {"bar", "hbar", "pie", "donut", "treemap"}
_SLICER_MAX_VALUES = 60   # a dimension with more distinct values is not sliceable
_SLICER_MAX_COLS = 3

_CSS = """
<style>
.dt-viz-bar{display:flex;align-items:center;gap:8px;margin:2px 0 0;}
.dt-viz-reason{font-size:.74rem;opacity:.68;line-height:1.35;}
div[class*="st-key-dtviz_"] [data-testid="stSelectbox"]{min-width:170px;}
div[class*="st-key-dtviz_"] [data-testid="stSelectbox"] div[data-baseweb="select"]>div{min-height:34px;
  font-size:.84rem;border-radius:9px;}
div[class*="st-key-dtviz_"] [data-testid="stPopover"] button{min-height:34px;font-size:.84rem;border-radius:9px;}
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _as_frame(table: Any) -> pd.DataFrame | None:
    if isinstance(table, pd.DataFrame):
        return table
    if isinstance(table, dict) and "columns" in table:  # restored from QA.json
        from core.qa_store import table_from_payload
        return table_from_payload(table)
    return None


@st.cache_data(show_spinner=False, max_entries=128, ttl=3600)
def _recommend_cached(df: pd.DataFrame, ctx: dict) -> dict:
    return recommend_visualizations(df, ctx).to_dict()


def _current_plan(plan: VisualizationPlan, df: pd.DataFrame) -> VisualizationPlan:
    """Plans stored by an older recommender (e.g. cached QA.json answers) are rebuilt from the
    stored table so they offer the current chart list. The stored primary measure is kept."""
    if plan.version == VIZ_VERSION:
        return plan
    try:
        ctx = {"question": plan.question, "grain": plan.grain, "dimensions": plan.declared_dimensions}
        if plan.measure_columns:
            ctx["preferred_measure"] = plan.measure_columns[0]
        upgraded = VisualizationPlan.from_dict(_recommend_cached(df, ctx))
        return upgraded if upgraded is not None and upgraded.compatible_charts else plan
    except Exception:
        log.exception("stored visualization plan could not be upgraded; using it as stored")
        return plan


def _entries(answer: Any) -> list[dict]:
    """One entry per chartable result: from ``meta['viz']`` or derived from ``answer.table``."""
    out = []
    for e in (getattr(answer, "meta", {}) or {}).get("viz") or []:
        if not isinstance(e, dict):
            continue
        plan = VisualizationPlan.from_dict(e.get("plan"))
        df = _as_frame(e.get("table"))
        if df is None and e.get("primary"):
            df = answer.table
        if plan is not None and df is not None and not df.empty:
            out.append({"id": str(e.get("id") or "q1"), "title": str(e.get("title") or ""),
                        "plan": _current_plan(plan, df), "df": df})
    if out:
        return out
    # Derive a plan from the answer table only where the table IS the charted result.
    # Quick actions (fast_path) often pair a row-level table with an aggregated chart,
    # so they keep their original figures unless a plan was stored explicitly.
    table = getattr(answer, "table", None)
    if (isinstance(table, pd.DataFrame) and not table.empty
            and getattr(answer, "route", "") in _DERIVE_ROUTES):
        question = str(((getattr(answer, "insight", None) or {}).get("question")) or "")
        plan = VisualizationPlan.from_dict(_recommend_cached(table, {"question": question}))
        if plan is not None and plan.compatible_charts:
            return [{"id": "q1", "title": "", "plan": plan, "df": table}]
    return []


def _figures_for(answer: Any, entry_id: str, single: bool) -> list:
    figs = list(getattr(answer, "figures", None) or [])
    if single:
        return figs
    return [f for f in figs if _fig_sub_question(f) == entry_id]


def _fig_sub_question(fig: Any) -> str | None:
    meta = getattr(getattr(fig, "layout", None), "meta", None)
    return meta.get("sql_sub_question") if isinstance(meta, dict) else None


def _key(card_key: str, entry_id: str) -> str:
    raw = f"{card_key}|{entry_id}"
    return "dtviz_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


@st.cache_data(show_spinner=False, max_entries=256, ttl=3600)
def _render_cached(df: pd.DataFrame, plan_dict: dict, chart: str):
    return render_chart(df, plan_dict, chart)


def _safe_render(df: pd.DataFrame, plan: VisualizationPlan, chart: str):
    """Cached render that can never raise into the page (hashing or plotting problems degrade)."""
    try:
        return _render_cached(df, plan.to_dict(), chart)
    except Exception:
        log.exception("chart render failed for %s", chart)
        return None, ["This visualization could not be drawn for the result; try another type."]


def _label(option: str, plan: VisualizationPlan, auto_kind: str | None) -> str:
    if option == AUTO:
        shown = auto_kind or plan.recommended_chart
        return f"✨ Auto · {CHART_LABELS.get(shown, str(shown).title())}"
    star = " ★" if option == plan.recommended_chart else ""
    return f"{CHART_ICONS.get(option, '')} {CHART_LABELS.get(option, option)}{star}"


def _plot(fig, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config=PLOTLY_CONFIG)


# ═══════════════════════════════════════════════════════════════════════════════
# export (share a chart anywhere)
# ═══════════════════════════════════════════════════════════════════════════════
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40] or "chart"


def _figs_html(figs: list) -> bytes:
    """Interactive, shareable page. Plotly.js loads from the CDN, so the file stays small (needs internet to open)."""
    cfg = {"displaylogo": False, "responsive": True}
    if len(figs) == 1:
        return figs[0].to_html(full_html=True, include_plotlyjs="cdn", config=cfg).encode("utf-8")
    parts = [f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False, config=cfg)
             for i, f in enumerate(figs)]
    return ("<!doctype html><html><head><meta charset='utf-8'><title>DataTalk.AI charts</title></head><body>"
            + "<hr>".join(parts) + "</body></html>").encode("utf-8")


def _figs_png(figs: list) -> bytes:
    """PNG (one chart) or ZIP of PNGs (several). Needs ``kaleido``; slow, so only built on request."""
    if len(figs) == 1:
        return figs[0].to_image(format="png", width=1200, height=650, scale=2)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, f in enumerate(figs, 1):
            zf.writestr(f"chart_{i}.png", f.to_image(format="png", width=1200, height=650, scale=2))
    return buf.getvalue()


def _csv_bytes(data: pd.DataFrame) -> bytes:
    """UTF-8 (with BOM so Excel reads accents). Text cells that start a spreadsheet formula are defused."""
    safe = data.copy()
    for c in safe.select_dtypes(include="object").columns:
        safe[c] = safe[c].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "@") else v)
    return safe.to_csv(index=False).encode("utf-8-sig")


def _export_controls(figs: list, data: pd.DataFrame | None, name: str, key: str) -> None:
    """Small ⬇️ Export menu under a chart: interactive HTML, PNG (if kaleido is installed) and the CSV
    of the rows the chart uses. Never raises into the page."""
    figs = [f for f in (figs or []) if hasattr(f, "to_html")]
    has_data = isinstance(data, pd.DataFrame) and not data.empty
    if not figs and not has_data:
        return
    try:
        base = f"datatalk_{_slug(name)}_{date.today().isoformat()}"
        _, right = st.columns([5, 1.4])
        with right, st.popover("⬇️ Export", help="Download this chart to share it anywhere."):
            if figs:
                st.download_button("Interactive chart (.html)", _figs_html(figs), file_name=f"{base}.html",
                                   mime="text/html", key=f"{key}_dl_html", on_click="ignore", width="stretch")
                if importlib.util.find_spec("kaleido") is not None:
                    png_key, sig = f"{key}_png", f"{name}|{len(figs)}|{len(data) if has_data else 0}"
                    if st.button("Prepare image (.png)", key=f"{key}_png_btn", width="stretch"):
                        with st.spinner("Rendering image…"):
                            try:
                                st.session_state[png_key] = (sig, _figs_png(figs))
                            except Exception:
                                log.exception("PNG export failed")
                                st.session_state.pop(png_key, None)
                                st.warning("Image export is not available on this server.")
                    ready = st.session_state.get(png_key)
                    if ready and ready[0] == sig:
                        st.download_button("Download image", ready[1], key=f"{key}_dl_png", on_click="ignore",
                                           file_name=f"{base}.{'png' if len(figs) == 1 else 'zip'}",
                                           mime="image/png" if len(figs) == 1 else "application/zip",
                                           width="stretch")
            if has_data:
                st.download_button("Data shown (.csv)", _csv_bytes(data), file_name=f"{base}.csv",
                                   mime="text/csv", key=f"{key}_dl_csv", on_click="ignore", width="stretch")
    except Exception:
        log.exception("export controls unavailable")


# ═══════════════════════════════════════════════════════════════════════════════
# slicer
# ═══════════════════════════════════════════════════════════════════════════════
def _slicer_columns(plan: VisualizationPlan, df: pd.DataFrame) -> list[str]:
    """Dimensions (time first) with a manageable number of distinct values."""
    cols: list[str] = []
    for c in [plan.time_column, plan.series, *plan.dimension_columns]:
        if c and c in df.columns and c not in cols and 2 <= df[c].nunique(dropna=True) <= _SLICER_MAX_VALUES:
            cols.append(c)
    return cols[:_SLICER_MAX_COLS]


def _slicer(df: pd.DataFrame, cols: list[str], key: str) -> pd.DataFrame:
    """Draw the slicer popover and return the rows that match it (``df`` itself when nothing is filtered)."""
    options = {c: list(dict.fromkeys(df[c].dropna().tolist())) for c in cols}
    keys = {c: f"{key}_slc_{i}" for i, c in enumerate(cols)}
    for c in cols:  # initialise once; the user's selection then lives in session state
        if keys[c] not in st.session_state:
            st.session_state[keys[c]] = list(options[c])

    def _reset() -> None:
        for c in cols:
            st.session_state[keys[c]] = list(options[c])

    with st.popover("🎚️ Slicer", help="Filter the rows used by this visual only. The answer, result table "
                                       "and insight panel are unchanged."):
        for c in cols:
            st.multiselect(pretty(c), options[c], key=keys[c], format_func=str)
        st.button("Reset", key=f"{key}_slc_reset", on_click=_reset)
    mask = pd.Series(True, index=df.index)
    for c in cols:
        chosen = st.session_state.get(keys[c], options[c])
        if len(chosen) != len(options[c]):
            mask &= df[c].isin(chosen)
    return df if bool(mask.all()) else df[mask]


# ═══════════════════════════════════════════════════════════════════════════════
# public API
# ═══════════════════════════════════════════════════════════════════════════════
def _default_original(figs: list, key: str) -> None:
    for i, fig in enumerate(figs):
        _plot(fig, f"{key}_f{i}")


def render_answer_visuals(answer: Any, card_key: str, render_original: RenderOriginal | None = None) -> None:
    """Draw the answer's charts with a compatible-chart selector.

    Drop-in replacement for the ``for fig in answer.figures: st.plotly_chart(...)``
    loop. With no stored data or a single meaningful chart it renders exactly the
    original figures, through ``render_original(figures, key)`` when supplied (so the
    page keeps its own layout, e.g. a two-column grid).
    """
    original = render_original or _default_original
    try:
        entries = _entries(answer)
    except Exception:
        log.exception("viz entries unavailable; showing original figures")
        entries = []
    if not entries or not any(e["plan"].switchable for e in entries):
        figures = list(getattr(answer, "figures", None) or [])
        original(figures, card_key)
        _export_controls(figures, getattr(answer, "table", None), "chart", f"{_key(card_key, 'x')}_all")
        return
    st.markdown(_CSS, unsafe_allow_html=True)
    single = len(entries) == 1
    for entry in entries:
        figs = _figures_for(answer, entry["id"], single)
        _entry_fragment(entry, figs, card_key, not single, original)
    if not single:  # figures not tied to any result (should be rare) are still shown
        ids = {e["id"] for e in entries}
        extra = [f for f in (getattr(answer, "figures", None) or []) if _fig_sub_question(f) not in ids]
        if extra:
            original(extra, f"{card_key}_extra")


@st.fragment
def _entry_fragment(entry: dict, original_figs: list, card_key: str, show_title: bool,
                    original: RenderOriginal) -> None:
    plan: VisualizationPlan = entry["plan"]
    df: pd.DataFrame = entry["df"]
    key = _key(card_key, entry["id"])
    auto_kind = existing_chart_type(original_figs)
    auto_override = (plan.is_time_series and auto_kind in _TIME_SERIES_BAD_DEFAULTS
                     and plan.recommended_chart in plan.compatible_charts)
    if auto_override or not original_figs:
        auto_kind = plan.recommended_chart

    if not plan.switchable:
        if original_figs and not auto_override:
            original(original_figs, f"{key}_orig")
            _export_controls(original_figs, df, plan.recommended_chart or "chart", f"{key}_x")
        elif plan.recommended_chart:
            fig, _ = _safe_render(df, plan, plan.recommended_chart)
            if fig is not None:
                _plot(fig, f"{key}_rec")
                _export_controls([fig], df, plan.recommended_chart, f"{key}_x")
        return

    options = [AUTO] + [c for c in plan.compatible_charts]
    slicer_cols = _slicer_columns(plan, df)
    with st.container(key=key):
        if slicer_cols:
            left, mid, right = st.columns([3.4, 1.1, 1.5], vertical_alignment="center")
        else:
            left, right = st.columns([4, 1.25], vertical_alignment="center")
            mid = None
        with left:
            head = f"<b>{html.escape(entry['title'])}</b> · " if show_title and entry["title"] else ""
            # titles / reasons embed question text and column names: always escaped (unsafe_allow_html)
            st.markdown(f"<div class='dt-viz-reason'>{head}{html.escape(plan.reason)}</div>",
                        unsafe_allow_html=True)
        view = df
        if mid is not None:
            with mid:
                view = _slicer(df, slicer_cols, key)
        filtered = view is not df
        if filtered:  # the original figure cannot be filtered: Auto = recommended chart on the rows kept
            auto_kind = plan.recommended_chart
        with right:
            choice = st.selectbox(
                "Visualization", options, index=0, key=f"{key}_sel",
                format_func=lambda o: _label(o, plan, auto_kind),
                help="Change the chart type. The data, filters and answer stay the same — no new AI call.",
            )

    if filtered:
        st.caption(f"🎚️ Slicer active · this visual uses {len(view):,} of {len(df):,} rows.")
        if view.empty:
            st.info("No rows match the slicer selection. Use Reset in the Slicer menu.")
            return

    if choice == AUTO and original_figs and not auto_override and not filtered:
        original(original_figs, f"{key}_orig")
        _export_controls(original_figs, view, auto_kind or "chart", f"{key}_x")
        return
    chart = plan.recommended_chart if choice == AUTO else choice
    if chart == TABLE:  # native, sortable, searchable grid instead of a static figure
        st.dataframe(view, width="stretch", hide_index=True, key=f"{key}_tbl")
        _export_controls([], view, "table", f"{key}_x")
        return
    fig, notes = _safe_render(view, plan, chart)
    if fig is None:
        if original_figs and not filtered:
            st.info("This chart type could not be drawn for the result; showing the original chart.")
            original(original_figs, f"{key}_fallback")
        else:
            st.info("This chart type could not be drawn for the result; showing the data table.")
            st.dataframe(view, width="stretch", hide_index=True, key=f"{key}_fbtbl")
        return
    _plot(fig, f"{key}_{chart}")
    for note in list(dict.fromkeys((notes or []) + list(plan.notes or [])))[:3]:
        st.caption(note)
    _export_controls([fig], view, chart, f"{key}_x")


__all__ = ["render_answer_visuals"]