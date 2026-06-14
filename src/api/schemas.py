"""
schemas.py
----------
Request / response schemas for the Flask Churn Prediction API.
Implemented as plain dataclasses with validation — no Pydantic required
(keeps dependency count minimal).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json


# ── Inbound ────────────────────────────────────────────────────────────────────
REQUIRED_FIELDS = [
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
    "segment",              # "Prepaid" | "Postpaid" | "Hybrid"
    "usage_prev_month_ratio",
]

OPTIONAL_FIELDS = {
    "subscriber_id": None,
}

VALID_SEGMENTS = {"Prepaid", "Postpaid", "Hybrid"}


def validate_predict_request(data: dict) -> tuple[dict, Optional[str]]:
    """
    Validate a single-subscriber predict request.
    Returns (cleaned_dict, error_message).  error_message is None on success.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return {}, f"Missing required fields: {missing}"

    segment = data.get("segment", "Prepaid")
    if segment not in VALID_SEGMENTS:
        return {}, f"segment must be one of {VALID_SEGMENTS}"

    # Type coercions
    try:
        cleaned = {
            "subscriber_id"          : str(data.get("subscriber_id", "UNKNOWN")),
            "call_minutes_30d"       : float(data["call_minutes_30d"]),
            "sms_count_30d"          : int(data["sms_count_30d"]),
            "data_mb_30d"            : float(data["data_mb_30d"]),
            "recharge_count_30d"     : int(data["recharge_count_30d"]),
            "recharge_amount_30d"    : float(data["recharge_amount_30d"]),
            "last_recharge_days"     : int(data["last_recharge_days"]),
            "support_tickets_90d"    : int(data["support_tickets_90d"]),
            "arpu_last_3m"           : float(data["arpu_last_3m"]),
            "contract_months"        : int(data["contract_months"]),
            "network_quality_score"  : float(data["network_quality_score"]),
            "segment"                : segment,
            "usage_prev_month_ratio" : float(data["usage_prev_month_ratio"]),
        }
    except (ValueError, TypeError) as e:
        return {}, f"Type error: {e}"

    return cleaned, None


# ── Outbound ───────────────────────────────────────────────────────────────────
RISK_TIERS = {"HIGH": 0.65, "MEDIUM": 0.35, "LOW": 0.0}

RETENTION_ACTIONS = {
    "HIGH"  : "Priority Retention Call + Bonus Data Offer (2GB free)",
    "MEDIUM": "SMS Loyalty Discount — 20% off next recharge",
    "LOW"   : "Standard Newsletter + Happy Hours Promotion",
}

REASON_LABELS = {
    "last_recharge_days"     : "recharge_gap",
    "recharge_gap"           : "recharge_gap",
    "support_tickets_90d"    : "support_tickets",
    "support_ticket_rate"    : "support_tickets",
    "arpu_last_3m"           : "low_arpu",
    "arpu_drop"              : "arpu_drop",
    "usage_trend_30d"        : "declining_usage",
    "usage_prev_month_ratio" : "declining_usage",
    "data_mb_30d"            : "low_data_usage",
    "data_burn_rate"         : "low_data_usage",
    "recharge_count_30d"     : "low_recharge_frequency",
    "recharge_intensity"     : "low_recharge_value",
    "network_quality_score"  : "network_quality_issues",
    "contract_months"        : "short_tenure",
    "value_score"            : "low_engagement",
}


def build_response(subscriber_id: str, churn_prob: float, reason_codes: List[str]) -> dict:
    prob = round(float(churn_prob), 4)
    if prob >= RISK_TIERS["HIGH"]:
        tier = "HIGH"
    elif prob >= RISK_TIERS["MEDIUM"]:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    # Map internal feature names to human-readable codes
    readable_reasons = list(dict.fromkeys(
        REASON_LABELS.get(r, r) for r in reason_codes
    ))[:3]

    return {
        "subscriber_id"     : subscriber_id,
        "churn_prob"        : prob,
        "risk_tier"         : tier,
        "reason_codes"      : readable_reasons,
        "recommended_action": RETENTION_ACTIONS[tier],
    }
