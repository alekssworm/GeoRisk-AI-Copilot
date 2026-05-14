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
