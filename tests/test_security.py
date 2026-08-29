from starlette.requests import Request

from app.security import unauthenticated_access_allowed


def _request_from(host: str) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/ml/predict",
            "raw_path": b"/ml/predict",
            "query_string": b"",
            "headers": [],
            "client": (host, 50000),
            "server": ("127.0.0.1", 8000),
        }
    )


def test_development_mode_allows_only_loopback(monkeypatch):
    monkeypatch.setenv("GEORISK_ENV", "development")
    monkeypatch.delenv("GEORISK_ALLOW_UNAUTHENTICATED", raising=False)

    assert unauthenticated_access_allowed(_request_from("127.0.0.1")) is True
    assert unauthenticated_access_allowed(_request_from("203.0.113.5")) is False


def test_production_mode_requires_auth_even_on_loopback(monkeypatch):
    monkeypatch.setenv("GEORISK_ENV", "production")
    monkeypatch.delenv("GEORISK_ALLOW_UNAUTHENTICATED", raising=False)

    assert unauthenticated_access_allowed(_request_from("127.0.0.1")) is False
