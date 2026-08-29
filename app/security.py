import os
import secrets
import time
from collections import defaultdict, deque
from ipaddress import ip_address

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.config import (
    RAG_RATE_LIMIT_REQUESTS,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    TRAIN_RATE_LIMIT_REQUESTS,
    UPLOAD_RATE_LIMIT_REQUESTS,
)

API_KEY_HEADER_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def configured_api_key() -> str | None:
    value = os.getenv("GEORISK_API_KEY", "").strip()
    return value or None


def configured_environment() -> str:
    return os.getenv("GEORISK_ENV", "development").strip().lower()


def _is_loopback_client(request: Request | None) -> bool:
    if request is None or request.client is None:
        return False
    host = request.client.host.strip().lower()
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def unauthenticated_access_allowed(request: Request | None = None) -> bool:
    value = os.getenv("GEORISK_ALLOW_UNAUTHENTICATED", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    return configured_environment() in {"development", "dev", "local"} and _is_loopback_client(
        request
    )


def api_key_required(request: Request | None = None) -> bool:
    return not unauthenticated_access_allowed(request)


def require_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> None:
    if unauthenticated_access_allowed(request):
        return

    expected_api_key = configured_api_key()
    if expected_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key authentication is required but GEORISK_API_KEY is not configured.",
        )

    if not api_key or not secrets.compare_digest(api_key, expected_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )


class InMemoryRateLimiter:
    def __init__(
        self,
        default_limit: int = RATE_LIMIT_REQUESTS,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
    ):
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.rules = {
            "/ml/train": TRAIN_RATE_LIMIT_REQUESTS,
            "/ml/train/advanced": TRAIN_RATE_LIMIT_REQUESTS,
            "/rag/upload": UPLOAD_RATE_LIMIT_REQUESTS,
            "/rag/ask": RAG_RATE_LIMIT_REQUESTS,
            "/reports/risk": RAG_RATE_LIMIT_REQUESTS,
            "/reports/risk.pdf": RAG_RATE_LIMIT_REQUESTS,
            "/reports/risk.docx": RAG_RATE_LIMIT_REQUESTS,
        }
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def reset(self) -> None:
        self._hits.clear()

    def limit_for_path(self, path: str) -> int:
        return self.rules.get(path, self.default_limit)

    def check(self, request: Request) -> None:
        if request.method == "OPTIONS":
            return

        path = request.url.path
        if path == "/health":
            return

        client_host = request.client.host if request.client else "unknown"
        key = (client_host, path)
        now = time.monotonic()
        window_start = now - self.window_seconds
        hits = self._hits[key]

        while hits and hits[0] < window_start:
            hits.popleft()

        limit = self.limit_for_path(path)
        if len(hits) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later.",
            )

        hits.append(now)


rate_limiter = InMemoryRateLimiter()
