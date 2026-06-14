"""
etl_pipeline.py
---------------
Pandas ETL: reads raw CDR + CRM CSVs → merges → engineers features
→ writes feature-engineered Parquet to data/processed/.

Usage:
    python src/data/etl_pipeline.py
"""

import sys
from pathlib import Path

import pandas as pd

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_engineering import engineer_features, FEATURE_COLS, TARGET_COL

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_raw() -> pd.DataFrame:
    cdr = pd.read_csv(RAW_DIR / "cdr_30d.csv")
    crm = pd.read_csv(RAW_DIR / "crm_subscribers.csv")
    print(f"  CDR  rows: {len(cdr):,}")
    print(f"  CRM  rows: {len(crm):,}")

    df = cdr.merge(crm, on="subscriber_id", how="inner")
    print(f"  Merged rows: {len(df):,}   columns: {df.shape[1]}")
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data-quality checks."""
    before = len(df)

    # Drop duplicates
    df = df.drop_duplicates(subset="subscriber_id")

    # Drop rows with null subscriber_id or target
    df = df.dropna(subset=["subscriber_id", TARGET_COL])

    # Clip outliers on numerical columns
    df["data_mb_30d"] = df["data_mb_30d"].clip(0, 50_000)
    df["recharge_amount_30d"] = df["recharge_amount_30d"].clip(0, 10_000)

    after = len(df)
    if before != after:
        print(f"  ⚠  Dropped {before - after:,} rows during validation")
    return df


def run_etl():
    print("─── ETL Pipeline ───────────────────────────────────────")
    df = load_raw()
    df = validate(df)
    df = engineer_features(df)

    # Persist full feature matrix
    out_path = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Features written → {out_path}  ({len(df):,} rows, {len(df.columns)} cols)")

    # Summary
    churn_n = df[TARGET_COL].sum()
    print(f"   Churn rate: {churn_n}/{len(df)} = {churn_n/len(df):.1%}")
    return df


if __name__ == "__main__":
    run_etl()
