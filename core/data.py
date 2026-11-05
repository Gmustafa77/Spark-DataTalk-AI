"""Appeals dataset: Databricks access, derived business columns and KPI maths.

Pure pandas/Databricks code — no Streamlit. ``core.runtime`` adds Streamlit caching.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.settings import (
    ESCALATION_LEVEL,
    RAW_COLUMNS,
    SLA_DAYS,
    Settings,
)

CASE_TYPE_LABELS = {"A": "Appeal", "C": "Complaint"}
BEHALF_OF_LABELS = {"P": "Provider", "M": "Member"}
SOURCE_LABELS = {"PO": "Pre-Service", "PR": "Post-Service"}
PROVIDER_RENAME = {
    "Provider_ID": "Provider ID", "Provider_Name": "Provider Name",
    "Provider_Type": "Provider Type", "Provider_Specialty": "Provider Specialty",
    "Network_Status": "Network Status",
}
DATE_COLUMNS = ("Received Date", "Posted Date", "Closed Date (CaseHistory)",
                "Case Item Level Posted Date Timestamp", "Routed Timestamp")
# Categorical fields whose distinct values are shown to the LLM as filter grounding.
VALUE_CONTEXT_FIELDS = ("Case Status", "Work Item Status Short Name", "Determination", "Behalf Of",
                        "Case Type Code", "Network_Status", "Provider_Type", "Provider_Specialty",
                        "Current Unit Name", "Routed To Site Name", "ASO Grouping")

KPI_META: dict[str, tuple[str, str]] = {
    "Total Appeals": ("total_appeals", "📁"),
    "Closed Appeals": ("closed_count", "✅"),
    "Open Appeals": ("open_count", "🔓"),
    "Compliance Rate": ("compliance_rate", "🎯"),
    "Avg Turnaround (Days)": ("avg_turnaround", "⏱️"),
    "Overdue Appeals": ("overdue_count", "⚠️"),
}


def databricks_connect(settings: Settings):
    """Open a Databricks SQL connection (context manager)."""
    from databricks import sql as dbsql  # lazy: keeps module import cheap

    return dbsql.connect(
        server_hostname=settings.databricks_server_hostname,
        http_path=settings.databricks_http_path,
        access_token=settings.databricks_access_token.get_secret_value(),
    )


def full_load_sql(settings: Settings) -> str:
    return f"SELECT {', '.join(f'`{c}`' for c in RAW_COLUMNS)} FROM {settings.fq_table}"


def enrich(raw: pd.DataFrame) -> pd.DataFrame:
    """Add every pandas-only derived column used by dashboards and fast paths."""
    df = raw.copy()
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if isinstance(df[col].dtype, pd.DatetimeTZDtype):
                df[col] = df[col].dt.tz_localize(None)

    if "Received Date" in df.columns:
        df["ReceivedMonth"] = df["Received Date"].dt.to_period("M").astype(str)
        df["ReceivedYear"] = df["Received Date"].dt.year
        df["Quarter"] = df["Received Date"].dt.to_period("Q").astype(str)

    if "Case Status" in df.columns:
        df["IsClosed"] = df["Case Status"].astype(str).str.strip().str.upper().eq("CLOSED")
    elif "Work Item Status Short Name" in df.columns:
        df["IsClosed"] = df["Work Item Status Short Name"].astype(str).str.upper().eq("CLOSED")
    else:
        df["IsClosed"] = False

    now_ts = pd.Timestamp.now().tz_localize(None)
    if {"Closed Date (CaseHistory)", "Received Date"} <= set(df.columns):
        closed_age = (df["Closed Date (CaseHistory)"] - df["Received Date"]).dt.days
        open_age = (now_ts - df["Received Date"]).dt.days
        age = pd.Series(np.where(df["IsClosed"], closed_age, open_age), index=df.index, dtype="float64")
        df["CaseAgeDays"] = age.clip(lower=0).fillna(0)
    else:
        df["CaseAgeDays"] = 0.0
    df["OpenAgingDays"] = np.where(df["IsClosed"], 0, df["CaseAgeDays"])

    if "Closed In Compliance Indicator" in df.columns:
        df["IsCompliant"] = df["Closed In Compliance Indicator"].astype(str).str.strip().str.upper().eq("Y")
    else:
        df["IsCompliant"] = True

    df["IsOverdue"] = df["CaseAgeDays"] > SLA_DAYS
    level = df["Level Type Code"] if "Level Type Code" in df.columns else pd.Series(0, index=df.index)
    df["IsEscalated"] = pd.to_numeric(level, errors="coerce").fillna(0).astype(int).eq(ESCALATION_LEVEL)
    status = df["Work Item Status Short Name"] if "Work Item Status Short Name" in df.columns else pd.Series("", index=df.index)
    df["IsTransferred"] = status.astype(str).str.upper().eq("TRANSFERRED")
    df["ComplianceStatus"] = np.where(df["IsClosed"], np.where(df["IsCompliant"], "Compliant", "Non-Compliant"),
                                      "Pending")
    risk = (0.05 + 0.30 * (~df["IsCompliant"]).astype(float) + 0.25 * df["IsOverdue"].astype(float)
            + 0.25 * df["IsEscalated"].astype(float) + 0.15 * df["IsTransferred"].astype(float))
    df["ComplianceRiskScore"] = risk.clip(0.01, 0.99).round(3)

    if "Case Type Code" in df.columns:
        df["CaseTypeLabel"] = df["Case Type Code"].map(CASE_TYPE_LABELS).fillna(df["Case Type Code"])
    if "Behalf Of" in df.columns:
        df["BehalfOfLabel"] = df["Behalf Of"].map(BEHALF_OF_LABELS).fillna(df["Behalf Of"])
    if "Pre Post Source Code" in df.columns:
        df["SourceLabel"] = df["Pre Post Source Code"].map(SOURCE_LABELS).fillna(df["Pre Post Source Code"])

    for src, dst in PROVIDER_RENAME.items():
        df[dst] = df[src] if src in df.columns else "Unknown"
    df["IsOutOfNetwork"] = df["Network Status"].astype(str).str.strip().str.title().eq("Out-Of-Network")

    if "Determination" in df.columns:
        det = df["Determination"].astype("string").str.strip().str.upper()
        det = det.mask(det.isin(["", "NAN", "NONE", "<NA>"]))
        df["Determination"] = det
        df["DeterminationLabel"] = det.fillna("Pending Decision").astype(str)
        df["IsOverturned"] = det.eq("OVERTURNED").fillna(False).astype(bool)
        df["IsPartiallyOverturned"] = det.eq("PARTIALLY OVERTURNED").fillna(False).astype(bool)
        df["IsUpheld"] = det.eq("UPHELD").fillna(False).astype(bool)
        df["IsAnyOverturn"] = det.isin(["OVERTURNED", "PARTIALLY OVERTURNED"]).fillna(False).astype(bool)
    else:
        df["DeterminationLabel"] = "Pending Decision"
        for col in ("IsOverturned", "IsPartiallyOverturned", "IsUpheld", "IsAnyOverturn"):
            df[col] = False

    if "Case Status" in df.columns:
        df["CaseStatusLabel"] = df["Case Status"].astype(str).str.strip().str.upper()
    else:
        df["CaseStatusLabel"] = np.where(df["IsClosed"], "CLOSED", "OPEN")
    return df


# ── KPIs ───────────────────────────────────────────────────────────────────────
def calculate_kpis(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {key: 0 for key, _ in KPI_META.values()}
    total = len(df)
    closed = int(df["IsClosed"].sum())
    compliant_closed = int((df["IsClosed"] & df["IsCompliant"]).sum())
    return {
        "total_appeals": total,
        "closed_count": closed,
        "open_count": total - closed,
        "compliance_rate": round(compliant_closed / closed * 100, 1) if closed else 0.0,
        "avg_turnaround": round(float(df.loc[df["IsClosed"], "CaseAgeDays"].mean()), 1) if closed else 0.0,
        "overdue_count": int(df["IsOverdue"].sum()),
    }


def fmt_kpi(label: str, value: Any) -> str:
    if value is None:
        return "N/A"
    if label == "Compliance Rate":
        return f"{value:.1f}%"
    if label == "Avg Turnaround (Days)":
        return f"{value:.1f}d"
    return f"{int(value):,}" if isinstance(value, (int, float, np.number)) else str(value)


def kpi_delta(label: str, current: float, baseline: dict) -> tuple[str, bool]:
    """Formatted delta vs. the baseline and whether the change is favourable."""
    diff = current - baseline.get(KPI_META[label][0], 0)
    if label == "Compliance Rate":
        return f"{diff:+.1f} pts", diff >= 0
    if label == "Avg Turnaround (Days)":
        return f"{diff:+.1f}d", diff <= 0
    if label == "Overdue Appeals":
        return f"{int(diff):+,}", diff <= 0
    return f"{int(diff):+,}", diff >= 0


# ── LLM grounding ──────────────────────────────────────────────────────────────
def value_context(df: pd.DataFrame | None, max_values_per_field: int = 25) -> dict:
    """Distinct categorical values and the date range, sent to the planner and SQL agent."""
    if df is None or df.empty:
        return {}
    context: dict[str, Any] = {}
    for name in VALUE_CONTEXT_FIELDS:
        if name not in df.columns:
            continue
        values = sorted({str(v) for v in df[name].dropna().unique()})
        context[name] = values if len(values) <= max_values_per_field else {
            "sample_values": values[:10], "distinct_count": len(values),
            "note": "High-cardinality field; sample is not an allow-list.",
        }
    if "Received Date" in df.columns and df["Received Date"].notna().any():
        context["available_date_range"] = [str(df["Received Date"].min().date()),
                                           str(df["Received Date"].max().date())]
    return context
