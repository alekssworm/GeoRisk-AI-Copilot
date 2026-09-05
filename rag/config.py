from project_config import int_from_env, project_path_from_env

RAG_INDEX_PATH = project_path_from_env("GEORISK_RAG_INDEX_PATH", "storage/rag_index.joblib")
CHUNK_SIZE = int_from_env("GEORISK_RAG_CHUNK_SIZE", 1200, min_value=1)
CHUNK_OVERLAP = int_from_env("GEORISK_RAG_CHUNK_OVERLAP", 180, min_value=0)
LLM_MAX_PROMPT_CHARS = int_from_env("GEORISK_LLM_MAX_PROMPT_CHARS", 16000, min_value=1000)

# Shelf 1 stays small; shelf 2 holds active overflow; shelf 3 is the archive.
SHELF_TOP_CAPACITY = int_from_env("GEORISK_RAG_TOP_CAPACITY", 64, min_value=1)
SHELF_MIDDLE_CAPACITY = int_from_env("GEORISK_RAG_MIDDLE_CAPACITY", 256, min_value=1)
SHELF_RECENT_SLOTS = int_from_env("GEORISK_RAG_RECENT_SLOTS", 16, min_value=0)
SHELF_HEAT_HALF_LIFE = int_from_env("GEORISK_RAG_HEAT_HALF_LIFE", 50, min_value=1)
# A percentage makes invalid environment values fall back through int_from_env.
SHELF_MIN_SCORE = min(100, int_from_env("GEORISK_RAG_MIN_SCORE_PCT", 35, min_value=1)) / 100
