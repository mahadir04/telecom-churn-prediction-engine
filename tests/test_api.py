"""
test_api.py
-----------
Smoke tests for the Flask Churn Prediction API.

Requirements:
    - Flask server running on localhost:5000
      OR directly import and test the app in-process.

Run:
    pytest tests/ -v
"""

import json
import pytest
import sys
from pathlib import Path

# Allow project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Fixtures ──────────────────────────────────────────────────────────────────
VALID_PAYLOAD = {
    "subscriber_id"          : "TEST-001",
    "call_minutes_30d"       : 310.5,
    "sms_count_30d"          : 42,
    "data_mb_30d"            : 2048.0,
    "recharge_count_30d"     : 5,
    "recharge_amount_30d"    : 420.0,
    "last_recharge_days"     : 3,
    "support_tickets_90d"    : 0,
    "arpu_last_3m"           : 140.0,
    "contract_months"        : 18,
    "network_quality_score"  : 4.0,
    "segment"                : "Prepaid",
    "usage_prev_month_ratio" : 1.05,
}

HIGH_CHURN_PAYLOAD = {
    **VALID_PAYLOAD,
    "subscriber_id"          : "CHURN-999",
    "last_recharge_days"     : 55,    # very long gap
    "support_tickets_90d"    : 12,    # many complaints
    "arpu_last_3m"           : 10.0,  # very low ARPU
    "recharge_count_30d"     : 0,
    "usage_prev_month_ratio" : 0.10,  # declining usage
    "network_quality_score"  : 1.0,
}


@pytest.fixture(scope="session")
def client():
    """Create a Flask test client (in-process, no server needed)."""
    # Check if model exists; skip all tests gracefully if not
    model_path = Path("src/models/artifacts/churn_model.pkl")
    if not model_path.exists():
        pytest.skip(
            "Model not found. Run `make train` before running tests.",
            allow_module_level=True,
        )

    from src.api.app import app, load_model
    load_model()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ── Health check ──────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded(self, client):
        data = resp = client.get("/health").get_json()
        assert data["model_loaded"] is True

    def test_health_feature_count(self, client):
        data = client.get("/health").get_json()
        assert data["feature_count"] == 18   # FEATURE_COLS length


# ── Single prediction ──────────────────────────────────────────────────────────
class TestPredict:
    def test_predict_200(self, client):
        resp = client.post("/predict",
                           data=json.dumps(VALID_PAYLOAD),
                           content_type="application/json")
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict",
                           data=json.dumps(VALID_PAYLOAD),
                           content_type="application/json").get_json()
        assert "subscriber_id"      in data
        assert "churn_prob"         in data
        assert "risk_tier"          in data
        assert "reason_codes"       in data
        assert "recommended_action" in data

    def test_predict_prob_range(self, client):
        data = client.post("/predict",
                           data=json.dumps(VALID_PAYLOAD),
                           content_type="application/json").get_json()
        assert 0.0 <= data["churn_prob"] <= 1.0

    def test_predict_risk_tier_valid(self, client):
        data = client.post("/predict",
                           data=json.dumps(VALID_PAYLOAD),
                           content_type="application/json").get_json()
        assert data["risk_tier"] in {"HIGH", "MEDIUM", "LOW"}

    def test_predict_reason_codes_list(self, client):
        data = client.post("/predict",
                           data=json.dumps(VALID_PAYLOAD),
                           content_type="application/json").get_json()
        assert isinstance(data["reason_codes"], list)
        assert len(data["reason_codes"]) <= 3

    def test_predict_high_churn_signal(self, client):
        """A subscriber with multiple strong churn signals should score HIGH."""
        data = client.post("/predict",
                           data=json.dumps(HIGH_CHURN_PAYLOAD),
                           content_type="application/json").get_json()
        assert data["churn_prob"] > 0.5, (
            f"Expected high churn prob, got {data['churn_prob']}"
        )

    def test_predict_missing_field(self, client):
        bad_payload = {k: v for k, v in VALID_PAYLOAD.items()
                       if k != "arpu_last_3m"}
        resp = client.post("/predict",
                           data=json.dumps(bad_payload),
                           content_type="application/json")
        assert resp.status_code == 422

    def test_predict_invalid_segment(self, client):
        bad_payload = {**VALID_PAYLOAD, "segment": "Corporate"}
        resp = client.post("/predict",
                           data=json.dumps(bad_payload),
                           content_type="application/json")
        assert resp.status_code == 422

    def test_predict_no_body(self, client):
        resp = client.post("/predict", content_type="application/json")
        assert resp.status_code == 400


# ── Batch prediction ───────────────────────────────────────────────────────────
class TestPredictBatch:
    def test_batch_200(self, client):
        payload = [VALID_PAYLOAD, HIGH_CHURN_PAYLOAD]
        resp = client.post("/predict/batch",
                           data=json.dumps(payload),
                           content_type="application/json")
        assert resp.status_code == 200

    def test_batch_result_count(self, client):
        payload = [VALID_PAYLOAD, HIGH_CHURN_PAYLOAD]
        data = client.post("/predict/batch",
                           data=json.dumps(payload),
                           content_type="application/json").get_json()
        assert data["scored"] == 2
        assert data["total"]  == 2

    def test_batch_not_array(self, client):
        resp = client.post("/predict/batch",
                           data=json.dumps(VALID_PAYLOAD),
                           content_type="application/json")
        assert resp.status_code == 400


# ── Top-risk endpoint ──────────────────────────────────────────────────────────
class TestTopRisk:
    def test_top_risk_200(self, client):
        resp = client.get("/top-risk")
        assert resp.status_code in {200, 503}   # 503 if no parquet exists

    def test_top_risk_count_param(self, client):
        resp = client.get("/top-risk?limit=10")
        if resp.status_code == 200:
            data = resp.get_json()
            assert data["count"] <= 10
