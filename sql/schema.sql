-- =============================================================================
-- schema.sql — PostgreSQL DDL for Telecom Churn Prediction Engine
-- Compatible with PostgreSQL 14+
-- For SQLite dev use, the db_loader.py auto-creates equivalent tables.
-- =============================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- 1. Subscribers (CRM master data)
-- =============================================================================
CREATE TABLE IF NOT EXISTS subscribers (
    subscriber_id           VARCHAR(64)  PRIMARY KEY,
    segment                 VARCHAR(20)  NOT NULL CHECK (segment IN ('Prepaid','Postpaid','Hybrid')),
    contract_months         SMALLINT     NOT NULL DEFAULT 1,
    arpu_last_3m            NUMERIC(10,2),
    support_tickets_90d     SMALLINT     DEFAULT 0,
    network_quality_score   NUMERIC(3,1),
    created_at              TIMESTAMPTZ  DEFAULT NOW(),
    updated_at              TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_subscribers_segment ON subscribers (segment);

-- =============================================================================
-- 2. CDR Daily Aggregates
-- =============================================================================
CREATE TABLE IF NOT EXISTS cdr_daily (
    id                      BIGSERIAL    PRIMARY KEY,
    subscriber_id           VARCHAR(64)  NOT NULL REFERENCES subscribers(subscriber_id),
    agg_date                DATE         NOT NULL,
    call_minutes            NUMERIC(8,1) DEFAULT 0,
    sms_count               INTEGER      DEFAULT 0,
    data_mb                 NUMERIC(10,1) DEFAULT 0,
    recharge_count          SMALLINT     DEFAULT 0,
    recharge_amount         NUMERIC(10,2) DEFAULT 0,
    UNIQUE (subscriber_id, agg_date)
);

CREATE INDEX IF NOT EXISTS idx_cdr_daily_sub_date ON cdr_daily (subscriber_id, agg_date DESC);

-- =============================================================================
-- 3. Subscriber Features (engineered, updated by ETL)
-- =============================================================================
CREATE TABLE IF NOT EXISTS subscriber_features (
    subscriber_id           VARCHAR(64)  PRIMARY KEY REFERENCES subscribers(subscriber_id),
    call_minutes_30d        NUMERIC(8,1),
    sms_count_30d           INTEGER,
    data_mb_30d             NUMERIC(10,1),
    recharge_count_30d      SMALLINT,
    recharge_amount_30d     NUMERIC(10,2),
    last_recharge_days      SMALLINT,
    usage_trend_30d         NUMERIC(6,4),
    recharge_gap            SMALLINT,
    data_burn_rate          NUMERIC(8,2),
    support_ticket_rate     NUMERIC(8,4),
    arpu_drop               SMALLINT,
    recharge_intensity      NUMERIC(8,2),
    value_score             NUMERIC(6,4),
    segment_encoded         SMALLINT,
    churn                   SMALLINT,
    feature_date            DATE         DEFAULT CURRENT_DATE
);

-- =============================================================================
-- 4. Churn Predictions (model output store)
-- =============================================================================
CREATE TABLE IF NOT EXISTS churn_predictions (
    id                  BIGSERIAL    PRIMARY KEY,
    subscriber_id       VARCHAR(64)  NOT NULL,
    churn_prob          NUMERIC(5,4) NOT NULL,
    risk_tier           VARCHAR(10)  NOT NULL CHECK (risk_tier IN ('HIGH','MEDIUM','LOW')),
    reason_codes        JSONB,                   -- e.g. ["recharge_gap","arpu_drop"]
    recommended_action  VARCHAR(128),
    model_version       VARCHAR(32),
    scored_at           TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_sub      ON churn_predictions (subscriber_id);
CREATE INDEX IF NOT EXISTS idx_predictions_scored   ON churn_predictions (scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_risk     ON churn_predictions (risk_tier, churn_prob DESC);

-- =============================================================================
-- 5. Retention Offers (CVM campaign linkage)
-- =============================================================================
CREATE TABLE IF NOT EXISTS retention_offers (
    id              BIGSERIAL    PRIMARY KEY,
    subscriber_id   VARCHAR(64)  NOT NULL,
    offer_code      VARCHAR(32),
    offer_desc      TEXT,
    offer_value_bdt NUMERIC(10,2),
    status          VARCHAR(20)  DEFAULT 'SENT' CHECK (status IN ('SENT','ACCEPTED','REJECTED','EXPIRED')),
    created_at      TIMESTAMPTZ  DEFAULT NOW(),
    responded_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_offers_sub    ON retention_offers (subscriber_id);
CREATE INDEX IF NOT EXISTS idx_offers_status ON retention_offers (status);

-- =============================================================================
-- Useful views
-- =============================================================================
CREATE OR REPLACE VIEW v_top_risk_today AS
SELECT
    p.subscriber_id,
    p.churn_prob,
    p.risk_tier,
    p.reason_codes,
    p.recommended_action,
    s.segment,
    s.contract_months,
    s.arpu_last_3m
FROM churn_predictions p
JOIN subscribers s USING (subscriber_id)
WHERE DATE(p.scored_at) = CURRENT_DATE
ORDER BY p.churn_prob DESC
LIMIT 100;
