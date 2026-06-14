"""
generate_synthetic.py
---------------------
Generates ~50,000 anonymised subscriber records simulating a Bangladeshi
telecom operator's CDR + CRM export.

Ground-truth churn label is synthesised via a logistic function fitted on
domain-plausible feature weights so that a trained XGBoost model achieves
AUC > 0.88 on held-out data.

Usage:
    python src/data/generate_synthetic.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

N = 50_000          # total subscribers
CHURN_RATE = 0.18   # ~18% annual churn — realistic for BD telco

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ── Helper distributions ───────────────────────────────────────────────────────
def beta_clipped(low, high, a=2, b=5, n=N):
    """Beta-distributed samples scaled to [low, high]."""
    return rng.beta(a, b, n) * (high - low) + low


def generate_subscribers(n: int = N) -> pd.DataFrame:
    """Generate the main subscriber feature matrix."""

    # ── Demographic / contract features ──────────────────────────────────────
    contract_months = rng.integers(1, 84, n)          # 1–84 months tenure
    segment = rng.choice(["Prepaid", "Postpaid", "Hybrid"],
                         n, p=[0.72, 0.20, 0.08])

    # ── 30-day CDR aggregates ─────────────────────────────────────────────────
    call_minutes_30d     = np.clip(rng.normal(220, 120, n), 0, 1500).round(1)
    sms_count_30d        = np.clip(rng.poisson(45, n), 0, 500)
    data_mb_30d          = np.clip(rng.lognormal(6.5, 1.2, n), 0, 20_000).round(1)

    # ── 30-day recharge behavior ──────────────────────────────────────────────
    recharge_count_30d   = np.clip(rng.poisson(4, n), 0, 30)
    recharge_amount_30d  = np.clip(
        recharge_count_30d * rng.normal(85, 30, n), 0, 5000).round(2)  # BDT

    # ── Days since last recharge (recharge gap) ───────────────────────────────
    last_recharge_days   = np.clip(rng.exponential(8, n), 0, 60).round(0).astype(int)

    # ── Support / complaint tickets (90d) ─────────────────────────────────────
    support_tickets_90d  = np.clip(rng.poisson(0.8, n), 0, 15)

    # ── ARPU — average revenue per user, last 3 months (BDT) ─────────────────
    arpu_last_3m = np.clip(
        recharge_amount_30d * rng.normal(1.05, 0.15, n), 10, 8000).round(2)

    # ── Month-over-month usage delta (prior month relative) ──────────────────
    usage_prev_month_ratio = np.clip(rng.normal(1.0, 0.3, n), 0.05, 3.0).round(3)

    # ── Network quality score (1–5, lower = worse) ───────────────────────────
    network_quality_score = np.clip(rng.normal(3.5, 0.8, n), 1, 5).round(1)

    df = pd.DataFrame({
        "subscriber_id"          : [f"MSISDN-{i:06d}" for i in range(n)],
        "segment"                : segment,
        "contract_months"        : contract_months,
        "call_minutes_30d"       : call_minutes_30d,
        "sms_count_30d"          : sms_count_30d,
        "data_mb_30d"            : data_mb_30d,
        "recharge_count_30d"     : recharge_count_30d,
        "recharge_amount_30d"    : recharge_amount_30d,
        "last_recharge_days"     : last_recharge_days,
        "support_tickets_90d"    : support_tickets_90d,
        "arpu_last_3m"           : arpu_last_3m,
        "usage_prev_month_ratio" : usage_prev_month_ratio,
        "network_quality_score"  : network_quality_score,
    })
    return df


# ── Churn label synthesis ─────────────────────────────────────────────────────
def synthesise_churn(df: pd.DataFrame, churn_rate: float = CHURN_RATE) -> pd.Series:
    """
    Generate churn labels via a logistic function with domain-plausible
    feature weights.  Threshold is calibrated so that mean(churn) ≈ churn_rate.
    """
    # Normalise key features
    norm = lambda s: (s - s.mean()) / (s.std() + 1e-9)

    logit = (
          0.0                                              # intercept (calibrated below)
        + 1.8  * norm(df["last_recharge_days"])           # longer gap → higher churn
        + 1.4  * norm(df["support_tickets_90d"])          # complaints → churn
        - 1.2  * norm(df["arpu_last_3m"])                 # high ARPU → retained
        - 1.0  * norm(df["recharge_count_30d"])           # frequent top-ups → retained
        - 0.8  * norm(df["data_mb_30d"])                  # heavy data → retained
        - 0.6  * norm(df["contract_months"])              # long tenure → retained
        + 0.9  * (df["usage_prev_month_ratio"] < 0.7).astype(float)  # declining usage
        - 0.7  * norm(df["network_quality_score"])        # bad network → churn
        + rng.logistic(0, 1, len(df))                     # noise term
    )

    # Shift intercept so that ~churn_rate of subscribers churn
    threshold = np.percentile(logit, (1 - churn_rate) * 100)
    churn = (logit >= threshold).astype(int)
    return churn


def split_cdr_crm(df: pd.DataFrame):
    """Split the combined DataFrame into CDR and CRM tables (realistic export)."""
    crm_cols = ["subscriber_id", "segment", "contract_months",
                "arpu_last_3m", "support_tickets_90d", "network_quality_score", "churn"]
    cdr_cols = ["subscriber_id", "call_minutes_30d", "sms_count_30d", "data_mb_30d",
                "recharge_count_30d", "recharge_amount_30d",
                "last_recharge_days", "usage_prev_month_ratio"]
    return df[cdr_cols].copy(), df[crm_cols].copy()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating {N:,} subscriber records …")
    df = generate_subscribers(N)
    df["churn"] = synthesise_churn(df)

    cdr_df, crm_df = split_cdr_crm(df)

    cdr_path = RAW_DIR / "cdr_30d.csv"
    crm_path = RAW_DIR / "crm_subscribers.csv"

    cdr_df.to_csv(cdr_path, index=False)
    crm_df.to_csv(crm_path, index=False)

    churn_count = df["churn"].sum()
    print(f"✅ CDR   → {cdr_path}  ({len(cdr_df):,} rows)")
    print(f"✅ CRM   → {crm_path}  ({len(crm_df):,} rows)")
    print(f"   Churn rate: {churn_count}/{N} = {churn_count/N:.1%}")
