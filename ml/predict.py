from functools import lru_cache
from pathlib import Path
from copy import deepcopy

import joblib

from ml.artifact import ModelArtifact
from ml.config import BASE_FEATURE_COLUMNS, MODEL_PATH, RISK_THRESHOLDS
from ml.features import build_feature_frame
from ml.train import train_model


ALLOWED_FEATURE_OVERRIDES = set(BASE_FEATURE_COLUMNS)
PREDICTION_CACHE_SIZE = 512


@lru_cache(maxsize=4)
def load_model(model_path: str | Path = MODEL_PATH) -> ModelArtifact:
    path = Path(model_path)
    if not path.exists():
        train_model(model_path=path)
    try:
        return joblib.load(path)
    except Exception:
        return train_model(model_path=path)


def clear_prediction_cache() -> None:
    load_model.cache_clear()
    _predict_dose_cached.cache_clear()


def classify_risk(dose_rate_usv_h: float) -> tuple[str, str]:
    for threshold, label, advisory in RISK_THRESHOLDS:
        if dose_rate_usv_h < threshold:
            return label, advisory
    return RISK_THRESHOLDS[-1][1], RISK_THRESHOLDS[-1][2]


def _feature_cache_key(features: dict) -> tuple[tuple[str, float], ...]:
    feature_frame = build_feature_frame(features)
    return tuple(
        (column, round(float(feature_frame.iloc[0][column]), 8)) for column in feature_frame.columns
    )


@lru_cache(maxsize=PREDICTION_CACHE_SIZE)
def _predict_dose_cached(model_path: str, feature_key: tuple[tuple[str, float], ...]) -> dict:
    artifact = load_model(model_path)
    features = {column: value for column, value in feature_key}
    feature_frame = build_feature_frame(features)
    prediction = float(artifact.pipeline.predict(feature_frame)[0])
    prediction = max(0.0, prediction)
    risk_level, advisory = classify_risk(prediction)
    return {
        "dose_rate_usv_h": round(prediction, 4),
        "risk_level": risk_level,
        "advisory": advisory,
        "model_version": artifact.model_version,
        "features_used": feature_frame.iloc[0].to_dict(),
    }


def predict_dose(features: dict, model_path: str | Path = MODEL_PATH) -> dict:
    return deepcopy(_predict_dose_cached(str(Path(model_path)), _feature_cache_key(features)))


def _validate_scenario_overrides(overrides: dict) -> None:
    if not isinstance(overrides, dict):
        raise ValueError(
            "Scenario overrides must be an object of feature names and numeric values."
        )

    unknown = sorted(set(overrides) - ALLOWED_FEATURE_OVERRIDES)
    if unknown:
        allowed = ", ".join(BASE_FEATURE_COLUMNS)
        raise ValueError(
            f"Unsupported scenario override(s): {', '.join(unknown)}. Allowed fields: {allowed}"
        )


def compare_scenarios(
    baseline: dict,
    scenarios: list[dict],
    model_path: str | Path = MODEL_PATH,
) -> list[dict]:
    baseline_prediction = predict_dose(baseline, model_path=model_path)
    rows = [
        {
            "name": "Baseline",
            "dose_rate_usv_h": baseline_prediction["dose_rate_usv_h"],
            "risk_level": baseline_prediction["risk_level"],
            "delta_vs_baseline_usv_h": 0.0,
            "input": baseline,
        }
    ]

    for scenario in scenarios:
        overrides = scenario.get("overrides", {})
        _validate_scenario_overrides(overrides)
        updated = dict(baseline)
        updated.update(overrides)
        prediction = predict_dose(updated, model_path=model_path)
        rows.append(
            {
                "name": scenario.get("name", "Scenario"),
                "dose_rate_usv_h": prediction["dose_rate_usv_h"],
                "risk_level": prediction["risk_level"],
                "delta_vs_baseline_usv_h": round(
                    prediction["dose_rate_usv_h"] - baseline_prediction["dose_rate_usv_h"], 4
                ),
                "input": updated,
            }
        )
    return rows
