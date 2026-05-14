import pytest

from ml.config import DEFAULT_BASE_FEATURES
from ml.explain import explain_prediction
from ml.predict import _predict_dose_cached, clear_prediction_cache, compare_scenarios, predict_dose
from ml.train import train_model


def test_training_prediction_and_explanation(tmp_path):
    model_path = tmp_path / "georisk_model.joblib"
    artifact = train_model(n_samples=350, n_estimators=20, model_path=model_path)

    assert artifact.__class__.__module__ == "ml.artifact"
    assert artifact.metrics["r2"] > 0.75

    prediction = predict_dose(DEFAULT_BASE_FEATURES, model_path=model_path)
    assert prediction["dose_rate_usv_h"] > 0
    assert prediction["risk_level"] in {"Low", "Moderate", "Elevated", "High", "Critical"}

    explanation = explain_prediction(DEFAULT_BASE_FEATURES, model_path=model_path, top_n=5)
    assert len(explanation["top_features"]) == 5


def test_scenario_comparison_reacts_to_contamination(tmp_path):
    model_path = tmp_path / "georisk_model.joblib"
    train_model(n_samples=350, n_estimators=20, model_path=model_path)

    scenarios = [
        {"name": "Reduced contamination", "overrides": {"contamination_bq_m2": 10000}},
        {"name": "Increased contamination", "overrides": {"contamination_bq_m2": 90000}},
    ]
    rows = compare_scenarios(DEFAULT_BASE_FEATURES, scenarios, model_path=model_path)
    assert len(rows) == 3
    assert rows[2]["dose_rate_usv_h"] > rows[1]["dose_rate_usv_h"]


def test_scenario_comparison_rejects_unknown_overrides(tmp_path):
    model_path = tmp_path / "georisk_model.joblib"
    train_model(n_samples=350, n_estimators=20, model_path=model_path)

    scenarios = [{"name": "Typo", "overrides": {"rainfall_typo": 1000}}]

    with pytest.raises(ValueError, match="Unsupported scenario override"):
        compare_scenarios(DEFAULT_BASE_FEATURES, scenarios, model_path=model_path)


def test_prediction_cache_reuses_identical_inputs(tmp_path):
    model_path = tmp_path / "georisk_model.joblib"
    train_model(n_samples=350, n_estimators=20, model_path=model_path)
    clear_prediction_cache()

    predict_dose(DEFAULT_BASE_FEATURES, model_path=model_path)
    first_info = _predict_dose_cached.cache_info()
    predict_dose(DEFAULT_BASE_FEATURES, model_path=model_path)
    second_info = _predict_dose_cached.cache_info()

    assert first_info.misses == 1
    assert second_info.hits == 1
