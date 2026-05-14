from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def project_path_from_env(env_name: str, default_relative_path: str | Path) -> Path:
    raw_value = os.getenv(env_name)
    path = Path(raw_value) if raw_value else PROJECT_ROOT / default_relative_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def int_from_env(env_name: str, default: int, min_value: int | None = None) -> int:
    raw_value = os.getenv(env_name)
    if raw_value in (None, ""):
        value = default
    else:
        try:
            value = int(raw_value)
        except ValueError:
            value = default

    if min_value is not None:
        value = max(min_value, value)
    return value


def bool_from_env(env_name: str, default: bool = False) -> bool:
    raw_value = os.getenv(env_name)
    if raw_value in (None, ""):
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def list_from_env(env_name: str, default: list[str]) -> list[str]:
    raw_value = os.getenv(env_name)
    if raw_value in (None, ""):
        return default
    return [item.strip() for item in raw_value.split(",") if item.strip()]
