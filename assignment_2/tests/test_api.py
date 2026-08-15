"""API integration tests (Objective 2.6 — integration tests).

Exercises the FastAPI app end-to-end through TestClient: request validation,
the wired-in Predictor dependency, response schema and status codes.

The app's own get_predictor() would load models/model.joblib from disk. That is
overridden with the session-trained test artifact so the suite passes on a
clean checkout and never depends on whatever happens to be in models/.
"""

import pytest
from fastapi.testclient import TestClient

from loan_default import service
from loan_default.service import app, get_predictor


@pytest.fixture
def client(predictor):
    """TestClient with the model dependency injected."""
    app.dependency_overrides[get_predictor] = lambda: predictor
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------
def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert {"decline_at", "refer_at", "model_artifact_present"} <= set(body)


def test_health_reports_the_configured_thresholds(client):
    """The UI reads policy from /health, so it must mirror config.yaml."""
    from loan_default.config import CONFIG

    body = client.get("/health").json()

    assert body["decline_at"] == CONFIG["decision"]["decline_at"]
    assert body["refer_at"] == CONFIG["decision"]["refer_at"]


# --------------------------------------------------------------------------
# /predict — happy path
# --------------------------------------------------------------------------
def test_predict_with_empty_body_uses_schema_defaults(client):
    """Every field has a documented default, so {} is a valid application."""
    response = client.post("/predict", json={})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"default_probability", "prediction", "decision"}


def test_predict_response_values_are_in_contract(client):
    body = client.post("/predict", json={}).json()

    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["decision"] in ("approve", "refer", "decline")


def test_predict_accepts_a_full_application(client, base_record):
    """A complete record, keyed by the real dataset column names."""
    payload = {
        key: value
        for key, value in base_record.items()
        if key
        not in (
            "ID",
            "Gender",
            "rate_of_interest",
            "Interest_rate_spread",
            "Upfront_charges",
            "credit_type",
        )
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200


def test_predict_is_directional_over_http(client):
    """The directional property must survive serialisation, not just the API."""
    poor = client.post("/predict", json={"Credit_Score": 520}).json()
    good = client.post("/predict", json={"Credit_Score": 830}).json()

    assert poor["default_probability"] > good["default_probability"]


# --------------------------------------------------------------------------
# /predict — input validation (Objective 2.9: reject malformed input)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload, reason",
    [
        ({"Credit_Score": "not-a-number"}, "wrong type"),
        ({"loan_amount": -5000}, "violates ge=0"),
        ({"term": -12}, "violates ge=0"),
    ],
)
def test_predict_rejects_invalid_input_with_422(client, payload, reason):
    """Pydantic must reject bad input at the edge, before it reaches the model.

    422 (not 500) is the point: a malformed request is the caller's error and
    should never be allowed to reach scikit-learn.
    """
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, reason


def test_validation_error_names_the_offending_field(client):
    """Useful errors: the response must say WHICH field failed."""
    detail = client.post("/predict", json={"Credit_Score": "abc"}).json()["detail"]
    assert any("Credit_Score" in str(item.get("loc", "")) for item in detail)


# --------------------------------------------------------------------------
# /predict — no trained model
# --------------------------------------------------------------------------
def test_predict_returns_503_when_model_is_missing(monkeypatch, tmp_path):
    """Before anyone runs train.py, /predict must 503, not crash with a 500."""

    def _no_artifact(*args, **kwargs):
        raise FileNotFoundError(tmp_path / "model.joblib")

    app.dependency_overrides.clear()
    monkeypatch.setattr(service, "Predictor", _no_artifact)
    monkeypatch.setattr(service, "_predictor", None)

    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={})

    assert response.status_code == 503
    assert "train" in response.json()["detail"].lower()


# --------------------------------------------------------------------------
# Documentation and UI surfaces
# --------------------------------------------------------------------------
def test_openapi_schema_documents_the_endpoints(client):
    schema = client.get("/openapi.json").json()

    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]


def test_docs_page_is_served(client):
    assert client.get("/docs").status_code == 200


def test_root_serves_the_ui(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
