"""
feature_engineering.py
-----------------------
Derives business-meaningful features from raw CDR + CRM columns.

Key engineered features
───────────────────────
usage_trend_30d      — ratio of current-month vs prior-month usage
recharge_gap         — days since last recharge (alias, kept as-is)
data_burn_rate       — MB per day consumed this month
support_ticket_rate  — tickets per month of tenure
arpu_drop            — binary flag: ARPU lower than segment median
value_score          — composite engagement index (higher = more engaged)
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the merged CDR + CRM DataFrame and returns it with derived features.
    Original raw columns are preserved.
    """
    df = df.copy()

    # ── usage_trend_30d ───────────────────────────────────────────────────────
    # Ratio of current-month to prior-month usage (call minutes proxy).
    # Values < 1 indicate declining usage → churn signal.
    df["usage_trend_30d"] = df["usage_prev_month_ratio"].round(4)

    # ── recharge_gap ──────────────────────────────────────────────────────────
    # Days since last recharge — already present as last_recharge_days.
    df["recharge_gap"] = df["last_recharge_days"]

    # ── data_burn_rate ────────────────────────────────────────────────────────
    # Average MB consumed per day in the last 30 days.
    df["data_burn_rate"] = (df["data_mb_30d"] / 30).round(2)

    # ── support_ticket_rate ───────────────────────────────────────────────────
    # Support tickets per month of tenure (normalised complaint intensity).
    tenure_months = df["contract_months"].clip(lower=1)
    df["support_ticket_rate"] = (df["support_tickets_90d"] / 3 / tenure_months).round(4)

    # ── arpu_drop ─────────────────────────────────────────────────────────────
    # 1 if subscriber's ARPU is below their segment's median, else 0.
    segment_median = df.groupby("segment")["arpu_last_3m"].transform("median")
    df["arpu_drop"] = (df["arpu_last_3m"] < segment_median).astype(int)

    # ── recharge_intensity ────────────────────────────────────────────────────
    # Average BDT per recharge event (proxy for subscriber "richness").
    df["recharge_intensity"] = (
        df["recharge_amount_30d"] / df["recharge_count_30d"].clip(lower=1)
    ).round(2)

    # ── value_score ───────────────────────────────────────────────────────────
    # Composite engagement index: normalised sum of positive engagement signals.
    # Ranges roughly 0–4; higher = more engaged, lower = at-risk.
    def _norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    df["value_score"] = (
        _norm(df["call_minutes_30d"])
        + _norm(df["data_mb_30d"])
        + _norm(df["recharge_amount_30d"])
        + _norm(df["arpu_last_3m"])
    ).round(4)

    # ── segment_encoded ───────────────────────────────────────────────────────
    segment_map = {"Prepaid": 0, "Hybrid": 1, "Postpaid": 2}
    df["segment_encoded"] = df["segment"].map(segment_map).fillna(0).astype(int)

    return df


# ── Model feature list (for consistent column ordering) ───────────────────────
FEATURE_COLS = [
    "call_minutes_30d",
    "sms_count_30d",
    "data_mb_30d",
    "recharge_count_30d",
    "recharge_amount_30d",
    "last_recharge_days",
    "support_tickets_90d",
    "arpu_last_3m",
    "contract_months",
    "network_quality_score",
    # Derived
    "usage_trend_30d",
    "recharge_gap",
    "data_burn_rate",
    "support_ticket_rate",
    "arpu_drop",
    "recharge_intensity",
    "value_score",
    "segment_encoded",
]

TARGET_COL = "churn"
