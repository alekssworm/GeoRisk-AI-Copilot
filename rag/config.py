from project_config import int_from_env, project_path_from_env


RAG_INDEX_PATH = project_path_from_env("GEORISK_RAG_INDEX_PATH", "storage/rag_index.joblib")
CHUNK_SIZE = int_from_env("GEORISK_RAG_CHUNK_SIZE", 1200, min_value=1)
CHUNK_OVERLAP = int_from_env("GEORISK_RAG_CHUNK_OVERLAP", 180, min_value=0)
LLM_MAX_PROMPT_CHARS = int_from_env("GEORISK_LLM_MAX_PROMPT_CHARS", 16000, min_value=1000)
