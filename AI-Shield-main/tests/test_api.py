import pytest
from fastapi.testclient import TestClient

from api.main import app
from configs.settings import settings
from src.deployment.service import model_service
from src.security import rate_limit

client = TestClient(app)


def _token(username=None, password=None) -> dict:
    body = {
        "username": username or settings.admin_username,
        "password": password or settings.admin_password,
    }
    resp = client.post("/auth/token", json=body)
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _payload(row, explain=False):
    features = {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)}
    return {
        "time": float(row["Time"]),
        "amount": float(row["Amount"]),
        "features": features,
        "explain": explain,
    }


@pytest.fixture
def served(trained_env):
    assert model_service.load()
    return trained_env


def test_health_reports_model_state(served):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] and body["model_version"] == 1


def test_login_bad_credentials():
    assert (
        client.post("/auth/token", json={"username": "admin", "password": "wrong"}).status_code
        == 401
    )


def test_predict_requires_auth(served, drifted_raw):
    assert client.post("/predict", json=_payload(drifted_raw.iloc[0])).status_code == 401
    bad = {"Authorization": "Bearer not-a-token"}
    assert (
        client.post("/predict", json=_payload(drifted_raw.iloc[0]), headers=bad).status_code == 401
    )


def test_predict_503_without_model(drifted_raw):
    resp = client.post("/predict", json=_payload(drifted_raw.iloc[0]), headers=_token())
    assert resp.status_code == 503


def test_predict_flags_fraud_and_passes_legit(served):
    from src.ingestion.loader import generate_synthetic_data

    raw = generate_synthetic_data(n_rows=1500, seed=21)
    fraud, legit = raw[raw.Class == 1].iloc[0], raw[raw.Class == 0].iloc[0]
    headers = _token()
    fraud_out = client.post("/predict", json=_payload(fraud), headers=headers).json()
    legit_out = client.post("/predict", json=_payload(legit), headers=headers).json()
    assert fraud_out["is_fraud"] and fraud_out["fraud_probability"] > 0.8
    assert not legit_out["is_fraud"] and legit_out["model_version"] == 1
    assert "explanation" not in legit_out


def test_predict_with_shap_explanation(served, drifted_raw):
    resp = client.post(
        "/predict", json=_payload(drifted_raw.iloc[0], explain=True), headers=_token()
    )
    explanation = resp.json()["explanation"]
    assert len(explanation) == settings.explanation_top_k
    assert {"feature", "value", "contribution"} <= set(explanation[0])


def test_predict_validates_input(served):
    headers = _token()
    assert (
        client.post(
            "/predict", json={"time": 1, "amount": 1, "features": {"V1": 0}}, headers=headers
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/predict", json={"time": 1, "amount": -5, "features": {}}, headers=headers
        ).status_code
        == 422
    )


def test_rate_limit_returns_429(served, drifted_raw, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_requests", 2)
    rate_limit.reset_limiter()
    headers = _token()  # consumes 1
    assert (
        client.post("/predict", json=_payload(drifted_raw.iloc[0]), headers=headers).status_code
        == 200
    )
    resp = client.post("/predict", json=_payload(drifted_raw.iloc[0]), headers=headers)
    assert resp.status_code == 429 and "retry-after" in resp.headers


def test_admin_endpoints_forbidden_for_analyst(served):
    analyst = _token(settings.analyst_username, settings.analyst_password)
    for method, path in [
        ("get", "/admin/model"),
        ("get", "/admin/drift"),
        ("post", "/admin/heal"),
        ("post", "/admin/rollback"),
    ]:
        assert getattr(client, method)(path, headers=analyst).status_code == 403
        assert getattr(client, method)(path).status_code == 401


def test_drift_check_and_heal_flow_over_api(served, drifted_raw, monkeypatch):
    monkeypatch.setattr(settings, "drift_min_samples", 60)
    headers = _token()
    for _, row in drifted_raw.head(80).iterrows():
        assert client.post("/predict", json=_payload(row), headers=headers).status_code == 200

    drift = client.get("/admin/drift", headers=headers).json()
    assert drift["action"] == "retrain" and drift["report"]["dataset_drift"]

    fb = client.post(
        "/feedback",
        json={**_payload(drifted_raw.iloc[0]), "is_fraud": bool(drifted_raw.iloc[0]["Class"])},
        headers=headers,
    )
    assert fb.json() == {"stored": True, "feedback_rows": 1}

    heal = client.post("/admin/heal", headers=headers).json()
    assert heal["executed"] is False and "Insufficient" in heal["reason"]  # only 1 labeled row


def test_heal_is_noop_without_drift(served, monkeypatch):
    monkeypatch.setattr(settings, "drift_min_samples", 10)
    headers = _token()
    from src.ingestion.loader import generate_synthetic_data

    for _, row in generate_synthetic_data(n_rows=300, seed=33).head(40).iterrows():
        client.post("/predict", json=_payload(row), headers=headers)
    body = client.post("/admin/heal", headers=headers).json()
    assert body["executed"] is False and body["action"] in ("none", "monitor")


def test_model_info_and_rollback_conflict(served):
    headers = _token()
    info = client.get("/admin/model", headers=headers).json()
    assert info["production"] == 1 and len(info["versions"]) == 1
    assert client.post("/admin/rollback", headers=headers).status_code == 409


def test_metrics_endpoint_exposes_prometheus_series(served, drifted_raw):
    client.post("/predict", json=_payload(drifted_raw.iloc[0]), headers=_token())
    text = client.get("/metrics").text
    for name in (
        "aishield_predictions_total",
        "aishield_prediction_latency_seconds",
        "aishield_http_requests_total",
        "aishield_model_version",
    ):
        assert name in text


def test_copilot_endpoint(served):
    resp = client.post("/copilot/ask", json={"question": "which model is live?"}, headers=_token())
    assert (
        resp.status_code == 200
        and "v1" in resp.json()["answer"]
        and resp.json()["source"] == "rules"
    )


def test_full_self_healing_loop_over_api(served, drifted_raw, monkeypatch):
    """drift -> heal (promote + hot reload) -> serve v2 -> rollback to v1."""
    from src.feature_engineering.features import engineer_features
    from src.monitoring import feedback as feedback_store

    monkeypatch.setattr(settings, "drift_min_samples", 60)
    monkeypatch.setattr(settings, "min_feedback_rows", 100)
    headers = _token()
    for _, row in drifted_raw.head(80).iterrows():
        client.post("/predict", json=_payload(row), headers=headers)

    labeled = engineer_features(drifted_raw)
    settings.data_drift_dir.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(feedback_store._path(), index=False)  # pylint: disable=protected-access

    outcome = client.post("/admin/heal", headers=headers).json()
    assert outcome["executed"] and outcome["promoted"] and outcome["challenger_version"] == 2
    assert client.get("/health").json()["model_version"] == 2  # hot-reloaded
    assert feedback_store.load_feedback().empty  # consumed

    assert client.post("/admin/rollback", headers=headers).json() == {"production": 1}
    assert client.get("/health").json()["model_version"] == 1


def test_lifespan_loads_model_and_enforces_secure_config(trained_env, monkeypatch):
    with TestClient(app) as started:
        assert started.get("/health").json()["model_loaded"] is True
    monkeypatch.setattr(settings, "environment", "prod")
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass


def test_drift_endpoint_without_model_is_503():
    assert client.get("/admin/drift", headers=_token()).status_code == 503


def test_monitor_tick_updates_drift_report_and_optionally_heals(served, drifted_raw, monkeypatch):
    from api.main import monitor_tick
    from src.monitoring.drift import load_latest_report

    assert monitor_tick() == {"skipped": True}  # nothing buffered yet
    monkeypatch.setattr(settings, "drift_min_samples", 60)
    headers = _token()
    for _, row in drifted_raw.head(80).iterrows():
        client.post("/predict", json=_payload(row), headers=headers)
    assert monitor_tick() == {"action": "retrain", "executed": False}
    report = load_latest_report()
    assert report is not None
    assert report["dataset_drift"] is True

    monkeypatch.setattr(settings, "auto_heal", True)
    result = monitor_tick()
    assert result["executed"] is False and "Insufficient" in result["reason"]
