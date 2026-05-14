import pytest

from ml.predict import load_model


@pytest.fixture(autouse=True)
def clear_model_cache():
    load_model.cache_clear()
    yield
    load_model.cache_clear()
