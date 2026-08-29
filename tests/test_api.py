from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.security import API_KEY_HEADER_NAME, rate_limiter

TEST_API_KEY = "test-secret"
AUTH_HEADERS = {API_KEY_HEADER_NAME: TEST_API_KEY}


def configure_api_key(monkeypatch) -> None:
    monkeypatch.setenv("GEORISK_API_KEY", TEST_API_KEY)
    monkeypatch.delenv("GEORISK_ALLOW_UNAUTHENTICATED", raising=False)


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-ms" in response.headers


def test_root_redirects_to_frontend():
    client = TestClient(app, follow_redirects=False)
    response = client.get("/")

    assert response.status_code == 307
    assert response.headers["location"].endswith(":8501")


def test_detailed_health_reports_dependencies():
    client = TestClient(app)
    response = client.get("/health/details")

    assert response.status_code == 200
    assert "checks" in response.json()
    assert response.json()["dependencies"]["python-multipart"] is True


def test_cors_rejects_unlisted_origin():
    client = TestClient(app)
    response = client.options(
        "/ml/predict",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 400
    assert response.headers.get("access-control-allow-origin") is None


def test_cors_allows_local_streamlit_origin():
    client = TestClient(app)
    response = client.options(
        "/ml/predict",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"


def test_protected_endpoint_requires_configured_api_key(monkeypatch):
    monkeypatch.delenv("GEORISK_API_KEY", raising=False)
    monkeypatch.delenv("GEORISK_ALLOW_UNAUTHENTICATED", raising=False)
    client = TestClient(app)

    response = client.post("/ml/predict", json={})

    assert response.status_code == 503
    assert "GEORISK_API_KEY" in response.json()["detail"]


def test_protected_endpoint_rejects_missing_or_invalid_api_key(monkeypatch):
    configure_api_key(monkeypatch)
    client = TestClient(app)

    missing = client.post("/ml/predict", json={})
    invalid = client.post("/ml/predict", json={}, headers={API_KEY_HEADER_NAME: "wrong"})
    authorized = client.post(
        "/ml/predict",
        json={},
        headers=AUTH_HEADERS,
    )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert authorized.status_code == 200


def test_rate_limit_returns_429(monkeypatch):
    configure_api_key(monkeypatch)
    client = TestClient(app)
    original_default = rate_limiter.default_limit
    rate_limiter.default_limit = 1
    try:
        first = client.post(
            "/ml/scenarios", json={"baseline": {}, "scenarios": []}, headers=AUTH_HEADERS
        )
        second = client.post(
            "/ml/scenarios", json={"baseline": {}, "scenarios": []}, headers=AUTH_HEADERS
        )
    finally:
        rate_limiter.default_limit = original_default

    assert first.status_code == 200
    assert second.status_code == 429


def test_advanced_predict_endpoint(monkeypatch):
    configure_api_key(monkeypatch)

    def fake_advanced_prediction_for(features):
        return {
            "dose_rate_usv_h": 0.42,
            "risk_level": "Elevated",
            "advisory": "Check advanced model assumptions.",
            "model_version": "classic-real-test",
            "data_mode": "real",
            "model_name": "extra_trees",
            "feature_set": "env_plus_no_ratio",
            "features_used": features.to_feature_dict(),
        }

    monkeypatch.setattr(main_module, "advanced_prediction_for", fake_advanced_prediction_for)
    client = TestClient(app)
    response = client.post("/ml/predict/advanced", json={}, headers=AUTH_HEADERS)

    assert response.status_code == 200
    assert response.json()["data_mode"] == "real"


def test_batch_prediction_endpoint(monkeypatch):
    configure_api_key(monkeypatch)

    def fake_batch(records):
        predictions = [
            {
                "dose_rate_usv_h": 0.12,
                "risk_level": "Low",
                "advisory": "Continue routine monitoring.",
                "model_version": "test-model",
                "features_used": item.to_feature_dict(),
            }
            for item in records
        ]
        return {"predictions": predictions, "count": len(predictions)}

    monkeypatch.setattr(main_module, "batch_predictions_for", fake_batch)
    client = TestClient(app)
    response = client.post(
        "/ml/predict/batch",
        json={"records": [{}, {}]},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert len(response.json()["predictions"]) == 2


def test_advanced_train_endpoint_reports_missing_real_data(monkeypatch):
    configure_api_key(monkeypatch)

    def fake_train_advanced_model(**kwargs):
        raise FileNotFoundError("Real training data file not found: missing.csv")

    monkeypatch.setattr(main_module, "train_advanced_model", fake_train_advanced_model)
    client = TestClient(app)
    response = client.post("/ml/train/advanced", json={}, headers=AUTH_HEADERS)

    assert response.status_code == 404
    assert "Real training data file not found" in response.json()["detail"]


def test_scenario_endpoint_rejects_unknown_override(monkeypatch):
    configure_api_key(monkeypatch)
    client = TestClient(app)
    payload = {
        "baseline": {},
        "scenarios": [{"name": "Typo", "overrides": {"contamination_typo": 123}}],
    }

    response = client.post("/ml/scenarios", json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 422
    assert "Unsupported scenario override" in response.text


def test_scenario_endpoint_validates_merged_values(monkeypatch):
    configure_api_key(monkeypatch)
    client = TestClient(app)
    payload = {
        "baseline": {},
        "scenarios": [{"name": "Impossible soil", "overrides": {"soil_clay_pct": 140}}],
    }

    response = client.post("/ml/scenarios", json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 422
    assert "less than or equal to 100" in response.text


def test_pdf_upload_rejects_invalid_pdf(monkeypatch):
    configure_api_key(monkeypatch)
    client = TestClient(app)
    response = client.post(
        "/rag/upload",
        files={"file": ("bad.pdf", b"not a real pdf", "application/pdf")},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    assert "Could not read PDF" in response.json()["detail"]


def test_pdf_upload_uses_generated_storage_name(monkeypatch):
    configure_api_key(monkeypatch)
    captured = {}

    def fake_ingest_pdf(path, source_name=None):
        captured["path"] = path
        captured["source_name"] = source_name
        return {
            "message": "PDF ingested",
            "source": source_name,
            "chunks_added": 1,
            "total_chunks": 1,
            "index_path": "test",
        }

    monkeypatch.setattr(main_module, "ingest_pdf", fake_ingest_pdf)
    client = TestClient(app)

    response = client.post(
        "/rag/upload",
        files={"file": ("../../etc/passwd.pdf", b"%PDF-1.4 test", "application/pdf")},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert captured["path"].name.startswith("upload_")
    assert captured["path"].name.endswith(".pdf")
    assert captured["source_name"] == "passwd.pdf"
