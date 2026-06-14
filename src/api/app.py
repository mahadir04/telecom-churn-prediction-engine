"""
app.py
------
Flask REST API for the Telecom Churn Prediction Engine.

Endpoints
─────────
  POST /predict          — single subscriber churn score
  POST /predict/batch    — batch scoring (JSON array)
  GET  /top-risk         — top-100 at-risk subscribers (from DB)
  GET  /health           — liveness check + model metadata
  GET  /metrics          — cached model performance metrics

Usage:
    python src/api/app.py
    curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \\
         -d @tests/sample_payload.json
"""

import os
import sys
import json
import time
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# ── Project imports ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.features.feature_engineering import engineer_features, FEATURE_COLS
from src.api.schemas import validate_predict_request, build_response
from src.utils import shap_patch

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Model loading ─────────────────────────────────────────────────────────────
ARTIFACT_DIR = BASE_DIR / "src/models/artifacts"
MODEL_PATH   = ARTIFACT_DIR / "churn_model.pkl"
FEAT_PATH    = ARTIFACT_DIR / "feature_names.json"
METRICS_PATH = BASE_DIR / "evaluation/metrics.json"

_model      = None
_feat_names = None
_metrics    = {}


def load_model():
    global _model, _feat_names, _metrics
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/models/train.py` first."
        )
    logger.info("Loading model from %s …", MODEL_PATH)
    _model = joblib.load(MODEL_PATH)
    with open(FEAT_PATH) as f:
        _feat_names = json.load(f)
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    logger.info("Model loaded. Features: %d", len(_feat_names))


# ── Helpers ───────────────────────────────────────────────────────────────────
def row_to_feature_vector(cleaned: dict) -> np.ndarray:
    """Convert a validated request dict → feature-engineered numpy row."""
    # Build single-row DataFrame in the expected raw format
    df = pd.DataFrame([{
        "subscriber_id"          : cleaned["subscriber_id"],
        "segment"                : cleaned["segment"],
        "contract_months"        : cleaned["contract_months"],
        "call_minutes_30d"       : cleaned["call_minutes_30d"],
        "sms_count_30d"          : cleaned["sms_count_30d"],
        "data_mb_30d"            : cleaned["data_mb_30d"],
        "recharge_count_30d"     : cleaned["recharge_count_30d"],
        "recharge_amount_30d"    : cleaned["recharge_amount_30d"],
        "last_recharge_days"     : cleaned["last_recharge_days"],
        "support_tickets_90d"    : cleaned["support_tickets_90d"],
        "arpu_last_3m"           : cleaned["arpu_last_3m"],
        "network_quality_score"  : cleaned["network_quality_score"],
        "usage_prev_month_ratio" : cleaned["usage_prev_month_ratio"],
    }])
    df = engineer_features(df)
    return df[FEATURE_COLS].values


def get_shap_top_features(X_row: np.ndarray, top_k: int = 3) -> list:
    """Compute SHAP for a single row and return top-k feature names."""
    try:
        import shap
        explainer = shap.TreeExplainer(_model)
        sv = explainer.shap_values(X_row)
        indices = np.argsort(np.abs(sv[0]))[::-1][:top_k]
        return [_feat_names[i] for i in indices]
    except Exception:
        # Fallback: use model feature importances
        importances = _model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]
        return [_feat_names[i] for i in indices]


# ── Request timing middleware ──────────────────────────────────────────────────
@app.before_request
def start_timer():
    g.start = time.time()


@app.after_request
def log_request(response):
    elapsed = (time.time() - g.start) * 1000
    logger.info("%s %s → %d  (%.1f ms)",
                request.method, request.path, response.status_code, elapsed)
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Liveness + readiness check."""
    return jsonify({
        "status"       : "healthy",
        "model_loaded" : _model is not None,
        "feature_count": len(_feat_names) if _feat_names else 0,
        "metrics"      : _metrics,
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return cached model performance metrics."""
    return jsonify(_metrics)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Score a single subscriber.

    Request body (JSON):
    {
      "subscriber_id": "MSISDN-001234",     // optional
      "call_minutes_30d": 310.5,
      "sms_count_30d": 42,
      "data_mb_30d": 2048.0,
      "recharge_count_30d": 5,
      "recharge_amount_30d": 420.0,
      "last_recharge_days": 3,
      "support_tickets_90d": 0,
      "arpu_last_3m": 140.0,
      "contract_months": 18,
      "network_quality_score": 4.0,
      "segment": "Prepaid",
      "usage_prev_month_ratio": 1.05
    }

    Response:
    {
      "subscriber_id": "MSISDN-001234",
      "churn_prob": 0.1234,
      "risk_tier": "LOW",
      "reason_codes": ["recharge_gap", "low_data_usage"],
      "recommended_action": "..."
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    cleaned, err = validate_predict_request(data)
    if err:
        return jsonify({"error": err}), 422

    X = row_to_feature_vector(cleaned)
    churn_prob = float(_model.predict_proba(X)[0, 1])
    reason_codes = get_shap_top_features(X)
    resp = build_response(cleaned["subscriber_id"], churn_prob, reason_codes)
    return jsonify(resp)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Score multiple subscribers in one call.

    Request body: JSON array of subscriber objects (same schema as /predict).
    Response: JSON array of prediction objects.
    Limit: 1000 subscribers per request.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, list):
        return jsonify({"error": "Request body must be a JSON array"}), 400
    if len(data) > 1000:
        return jsonify({"error": "Batch limit is 1000 subscribers"}), 413

    results, errors = [], []
    for i, record in enumerate(data):
        cleaned, err = validate_predict_request(record)
        if err:
            errors.append({"index": i, "error": err})
            continue
        X = row_to_feature_vector(cleaned)
        churn_prob = float(_model.predict_proba(X)[0, 1])
        importances = _model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:3]
        reason_codes = [_feat_names[i] for i in top_idx]
        results.append(build_response(cleaned["subscriber_id"], churn_prob, reason_codes))

    return jsonify({
        "total"   : len(data),
        "scored"  : len(results),
        "errors"  : len(errors),
        "results" : results,
        "error_details": errors if errors else None,
    })


@app.route("/top-risk", methods=["GET"])
def top_risk():
    """
    Return the top-100 at-risk subscribers from the database.
    Falls back to scoring the full feature Parquet if DB is not loaded.

    Query params:
      limit (int, default=100, max=500)
      tier  (str, optional: HIGH|MEDIUM|LOW)
    """
    limit = min(int(request.args.get("limit", 100)), 500)
    tier  = request.args.get("tier", "").upper() or None

    # Try DB first
    try:
        from src.data.db_loader import get_engine
        from sqlalchemy import text as sql_text

        engine = get_engine()
        query = """
            SELECT subscriber_id, churn_prob, risk_tier, reason_codes, recommended_action
            FROM churn_predictions
            ORDER BY churn_prob DESC
            LIMIT :limit
        """
        params = {"limit": limit}
        if tier:
            query = query.replace("ORDER BY", "WHERE risk_tier = :tier ORDER BY")
            params["tier"] = tier

        with engine.connect() as conn:
            rows = conn.execute(sql_text(query), params).fetchall()
        if rows:
            result = [dict(r._mapping) for r in rows]
            # Parse JSON stored as text in SQLite
            for r in result:
                if isinstance(r.get("reason_codes"), str):
                    try:
                        r["reason_codes"] = json.loads(r["reason_codes"])
                    except Exception:
                        pass
            return jsonify({"source": "database", "count": len(result), "data": result})
    except Exception as e:
        logger.warning("DB top-risk failed (%s), falling back to parquet scoring.", e)

    # Fallback: score from parquet
    feat_path = BASE_DIR / "data/processed/features.parquet"
    if not feat_path.exists():
        return jsonify({"error": "No scored data available. Run make train && make db"}), 503

    df = pd.read_parquet(feat_path)
    X_all = df[FEATURE_COLS].values
    probs = _model.predict_proba(X_all)[:, 1]
    df["churn_prob"] = probs

    if tier:
        tier_bounds = {"HIGH": 0.65, "MEDIUM": 0.35, "LOW": 0.0}
        tier_upper  = {"HIGH": 1.01, "MEDIUM": 0.65, "LOW": 0.35}
        low = tier_bounds.get(tier, 0.0)
        hi  = tier_upper.get(tier, 1.01)
        df = df[(df["churn_prob"] >= low) & (df["churn_prob"] < hi)]

    top = df.nlargest(limit, "churn_prob")[
        ["subscriber_id", "churn_prob", "segment", "arpu_last_3m",
         "last_recharge_days", "support_tickets_90d"]
    ].copy()

    importances = _model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:3]
    reason_codes = [_feat_names[i] for i in top_idx]

    result = []
    for _, row in top.iterrows():
        prob = float(row["churn_prob"])
        result.append(build_response(row["subscriber_id"], prob, reason_codes))

    return jsonify({"source": "parquet", "count": len(result), "data": result})


# ── Entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    logger.info("Starting Churn Prediction API on port %d …", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
