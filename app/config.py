from project_config import int_from_env, project_path_from_env


MODEL_PATH = project_path_from_env("GEORISK_MODEL_PATH", "models/georisk_model.joblib")
RAG_INDEX_PATH = project_path_from_env("GEORISK_RAG_INDEX_PATH", "storage/rag_index.joblib")
UPLOAD_DIR = project_path_from_env("GEORISK_UPLOAD_DIR", "storage/uploads")
MAX_UPLOAD_MB = int_from_env("GEORISK_MAX_UPLOAD_MB", 25, min_value=1)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
RAG_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
