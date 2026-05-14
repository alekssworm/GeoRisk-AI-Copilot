from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-ms" in response.headers


def test_scenario_endpoint_rejects_unknown_override():
    client = TestClient(app)
    payload = {
        "baseline": {},
        "scenarios": [{"name": "Typo", "overrides": {"contamination_typo": 123}}],
    }

    response = client.post("/ml/scenarios", json=payload)

    assert response.status_code == 422
    assert "Unsupported scenario override" in response.text


def test_pdf_upload_rejects_invalid_pdf():
    client = TestClient(app)
    response = client.post(
        "/rag/upload",
        files={"file": ("bad.pdf", b"not a real pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert "Could not read PDF" in response.json()["detail"]
