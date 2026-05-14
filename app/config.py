from project_config import bool_from_env, int_from_env, list_from_env, project_path_from_env


MODEL_PATH = project_path_from_env("GEORISK_MODEL_PATH", "models/georisk_model.joblib")
RAG_INDEX_PATH = project_path_from_env("GEORISK_RAG_INDEX_PATH", "storage/rag_index.joblib")
UPLOAD_DIR = project_path_from_env("GEORISK_UPLOAD_DIR", "storage/uploads")
MAX_UPLOAD_MB = int_from_env("GEORISK_MAX_UPLOAD_MB", 25, min_value=1)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
PDF_PROCESSING_TIMEOUT_SECONDS = int_from_env(
    "GEORISK_PDF_PROCESSING_TIMEOUT_SECONDS", 30, min_value=1
)

CORS_ALLOW_ORIGINS = list_from_env(
    "GEORISK_CORS_ALLOW_ORIGINS",
    ["http://localhost:8501", "http://127.0.0.1:8501"],
)
CORS_ALLOW_CREDENTIALS = bool_from_env("GEORISK_CORS_ALLOW_CREDENTIALS", False)

RATE_LIMIT_WINDOW_SECONDS = int_from_env("GEORISK_RATE_LIMIT_WINDOW_SECONDS", 60, min_value=1)
RATE_LIMIT_REQUESTS = int_from_env("GEORISK_RATE_LIMIT_REQUESTS", 120, min_value=1)
TRAIN_RATE_LIMIT_REQUESTS = int_from_env("GEORISK_TRAIN_RATE_LIMIT_REQUESTS", 5, min_value=1)
UPLOAD_RATE_LIMIT_REQUESTS = int_from_env("GEORISK_UPLOAD_RATE_LIMIT_REQUESTS", 20, min_value=1)
RAG_RATE_LIMIT_REQUESTS = int_from_env("GEORISK_RAG_RATE_LIMIT_REQUESTS", 60, min_value=1)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
RAG_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
