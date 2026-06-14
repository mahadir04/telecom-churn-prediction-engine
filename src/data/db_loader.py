"""
db_loader.py
------------
Loads the processed feature Parquet into a SQLite (default) or PostgreSQL
database using SQLAlchemy.

Usage:
    python src/data/db_loader.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROCESSED_DIR = Path("data/processed")
DB_ENGINE = os.getenv("DB_ENGINE", "sqlite")


def get_engine():
    if DB_ENGINE == "postgresql":
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db   = os.getenv("POSTGRES_DB", "telecom_churn")
        user = os.getenv("POSTGRES_USER", "postgres")
        pwd  = os.getenv("POSTGRES_PASSWORD", "")
        url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    else:
        db_path = os.getenv("DB_PATH", "data/churn.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"

    engine = create_engine(url, echo=False)
    print(f"  Connected: {url}")
    return engine


def create_tables(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS subscribers (
        subscriber_id           TEXT PRIMARY KEY,
        segment                 TEXT,
        contract_months         INTEGER,
        arpu_last_3m            REAL,
        support_tickets_90d     INTEGER,
        network_quality_score   REAL
    );

    CREATE TABLE IF NOT EXISTS subscriber_features (
        subscriber_id           TEXT PRIMARY KEY,
        call_minutes_30d        REAL,
        sms_count_30d           INTEGER,
        data_mb_30d             REAL,
        recharge_count_30d      INTEGER,
        recharge_amount_30d     REAL,
        last_recharge_days      INTEGER,
        usage_trend_30d         REAL,
        recharge_gap            INTEGER,
        data_burn_rate          REAL,
        support_ticket_rate     REAL,
        arpu_drop               INTEGER,
        recharge_intensity      REAL,
        value_score             REAL,
        segment_encoded         INTEGER,
        churn                   INTEGER
    );

    CREATE TABLE IF NOT EXISTS churn_predictions (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        subscriber_id       TEXT NOT NULL,
        churn_prob          REAL,
        risk_tier           TEXT,
        reason_codes        TEXT,    -- JSON array stored as text
        recommended_action  TEXT,
        scored_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS retention_offers (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        subscriber_id   TEXT NOT NULL,
        offer_code      TEXT,
        offer_desc      TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with engine.connect() as conn:
        for stmt in ddl.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()
    print("  Tables created / verified ✅")


def load_data(engine, df: pd.DataFrame):
    # subscribers (CRM columns)
    sub_cols = ["subscriber_id", "segment", "contract_months",
                "arpu_last_3m", "support_tickets_90d", "network_quality_score"]
    df[sub_cols].to_sql("subscribers", engine, if_exists="replace",
                        index=False, chunksize=5000)

    # subscriber_features
    feat_cols = [
        "subscriber_id", "call_minutes_30d", "sms_count_30d", "data_mb_30d",
        "recharge_count_30d", "recharge_amount_30d", "last_recharge_days",
        "usage_trend_30d", "recharge_gap", "data_burn_rate",
        "support_ticket_rate", "arpu_drop", "recharge_intensity",
        "value_score", "segment_encoded", "churn"
    ]
    df[feat_cols].to_sql("subscriber_features", engine, if_exists="replace",
                         index=False, chunksize=5000)
    print(f"  Loaded {len(df):,} subscriber records into DB ✅")


if __name__ == "__main__":
    print("─── DB Loader ──────────────────────────────────────────")
    feat_path = PROCESSED_DIR / "features.parquet"
    if not feat_path.exists():
        print("❌  features.parquet not found. Run etl_pipeline.py first.")
        sys.exit(1)

    df = pd.read_parquet(feat_path)
    engine = get_engine()
    create_tables(engine)
    load_data(engine, df)
    print("✅ Database loaded successfully.")
