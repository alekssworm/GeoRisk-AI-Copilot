import pytest

from app.security import rate_limiter
from ml.classic.predict import clear_advanced_model_cache
from ml.predict import clear_prediction_cache


@pytest.fixture(autouse=True)
def clear_runtime_caches():
    clear_prediction_cache()
    clear_advanced_model_cache()
    rate_limiter.reset()
    yield
    clear_prediction_cache()
    clear_advanced_model_cache()
    rate_limiter.reset()
