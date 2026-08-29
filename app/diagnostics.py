from __future__ import annotations

import os
import sys
from importlib.util import find_spec

from app.config import ENVIRONMENT, FRONTEND_URL, UPLOAD_DIR
from app.security import configured_api_key
from ml.config import (
    ADVANCED_MODEL_PATH,
    DATA_MODE,
    MODEL_PATH,
    REAL_ENV_DATA_PATH,
    REAL_NUCLIDE_DATA_PATH,
)
from rag.config import RAG_INDEX_PATH


def _dependency_status() -> dict[str, bool]:
    return {
        package: find_spec(module) is not None
        for package, module in {
            "fastapi": "fastapi",
            "python-multipart": "multipart",
            "scikit-learn": "sklearn",
            "streamlit": "streamlit",
            "reportlab": "reportlab",
            "python-docx": "docx",
        }.items()
    }


def system_diagnostics() -> dict:
    dependencies = _dependency_status()
    checks = {
        "model_ready": MODEL_PATH.is_file(),
        "advanced_model_ready": ADVANCED_MODEL_PATH.is_file(),
        "rag_index_ready": RAG_INDEX_PATH.is_file(),
        "real_environment_data_ready": REAL_ENV_DATA_PATH.is_file(),
        "real_nuclide_data_ready": REAL_NUCLIDE_DATA_PATH.is_file(),
        "upload_directory_ready": UPLOAD_DIR.is_dir() and os.access(UPLOAD_DIR, os.W_OK),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "api_key_configured": configured_api_key() is not None,
        "dependencies_ready": all(dependencies.values()),
    }
    required_checks = ["upload_directory_ready", "dependencies_ready"]
    if DATA_MODE == "real":
        required_checks.extend(
            [
                "advanced_model_ready",
                "real_environment_data_ready",
                "real_nuclide_data_ready",
            ]
        )
    else:
        required_checks.append("model_ready")
    status = "ok" if all(checks[name] for name in required_checks) else "degraded"
    auth_mode = (
        "explicitly-unauthenticated"
        if os.getenv("GEORISK_ALLOW_UNAUTHENTICATED", "").strip().lower()
        in {"1", "true", "yes", "on"}
        else "loopback-development"
        if ENVIRONMENT in {"development", "dev", "local"}
        else "api-key"
    )
    return {
        "status": status,
        "service": "GeoRisk AI Copilot",
        "environment": ENVIRONMENT,
        "data_mode": DATA_MODE,
        "auth_mode": auth_mode,
        "frontend_url": FRONTEND_URL,
        "python_version": sys.version.split()[0],
        "checks": checks,
        "dependencies": dependencies,
    }
