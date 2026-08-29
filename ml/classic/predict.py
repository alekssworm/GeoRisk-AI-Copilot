from functools import lru_cache
from pathlib import Path

import numpy as np

from ml.classic.feature_sets import ALLOWED_ADVANCED_OVERRIDES
from ml.classic.model_registry import ClassicModelArtifact, load_artifact
from ml.classic.train import train_advanced_model
from ml.config import (
    ADVANCED_MODEL_PATH,
    REAL_ENV_FEATURE_COLUMNS,
    REAL_NUCLIDE_FEATURE_COLUMNS,
    REAL_RATIO_FEATURE_COLUMNS,
    REAL_SPATIAL_COLUMNS,
)
from ml.features import build_real_feature_frame
from ml.predict import classify_risk


@lru_cache(maxsize=2)
def load_advanced_model(model_path: str | Path = ADVANCED_MODEL_PATH) -> ClassicModelArtifact:
    path = Path(model_path)
    if not path.exists():
        train_advanced_model(model_path=path)
    return load_artifact(path)


def clear_advanced_model_cache() -> None:
    load_advanced_model.cache_clear()


def _advanced_uncertainty(artifact: ClassicModelArtifact, feature_frame, prediction: float) -> dict:
    model = artifact.pipeline.named_steps["model"]
    estimators = getattr(model, "estimators_", None)
    if not estimators:
        return {
            "method": "not_available",
            "lower_usv_h": None,
            "upper_usv_h": None,
            "std_usv_h": None,
            "confidence_label": "not_available",
            "calibrated": False,
        }

    transformed = artifact.pipeline[:-1].transform(feature_frame)
    predictions = np.asarray(
        [float(estimator.predict(transformed)[0]) for estimator in estimators], dtype=float
    )
    lower = max(0.0, float(np.quantile(predictions, 0.10)))
    upper = max(0.0, float(np.quantile(predictions, 0.90)))
    relative_width = (upper - lower) / max(abs(prediction), 0.05)
    confidence = "high" if relative_width <= 0.25 else "medium" if relative_width <= 0.50 else "low"
    return {
        "method": "tree_ensemble_p10_p90",
        "lower_usv_h": round(lower, 4),
        "upper_usv_h": round(upper, 4),
        "std_usv_h": round(float(np.std(predictions, ddof=0)), 4),
        "confidence_label": confidence,
        "calibrated": False,
    }


def predict_advanced_dose(features: dict, model_path: str | Path = ADVANCED_MODEL_PATH) -> dict:
    artifact = load_advanced_model(model_path)
    feature_frame = build_real_feature_frame(features, feature_columns=artifact.feature_names)
    prediction = max(0.0, float(artifact.pipeline.predict(feature_frame)[0]))
    risk_level, advisory = classify_risk(prediction)
    return {
        "dose_rate_usv_h": round(prediction, 4),
        "risk_level": risk_level,
        "advisory": advisory,
        "model_version": artifact.model_version,
        "data_mode": artifact.data_mode,
        "model_name": artifact.model_name,
        "feature_set": artifact.feature_set,
        "features_used": feature_frame.iloc[0].to_dict(),
        "uncertainty": _advanced_uncertainty(artifact, feature_frame, prediction),
        "distribution_check": None,
    }


def _validate_advanced_overrides(overrides: dict) -> None:
    unknown = sorted(set(overrides) - ALLOWED_ADVANCED_OVERRIDES)
    if unknown:
        allowed = ", ".join(
            sorted(
                REAL_NUCLIDE_FEATURE_COLUMNS
                + REAL_ENV_FEATURE_COLUMNS
                + REAL_RATIO_FEATURE_COLUMNS
                + REAL_SPATIAL_COLUMNS
            )
        )
        raise ValueError(
            f"Unsupported advanced scenario override(s): {', '.join(unknown)}. "
            f"Allowed fields: {allowed}"
        )


def compare_advanced_scenarios(
    baseline: dict,
    scenarios: list[dict],
    model_path: str | Path = ADVANCED_MODEL_PATH,
) -> list[dict]:
    baseline_prediction = predict_advanced_dose(baseline, model_path=model_path)
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
        _validate_advanced_overrides(overrides)
        updated = dict(baseline)
        updated.update(overrides)
        prediction = predict_advanced_dose(updated, model_path=model_path)
        rows.append(
            {
                "name": scenario.get("name", "Scenario"),
                "dose_rate_usv_h": prediction["dose_rate_usv_h"],
                "risk_level": prediction["risk_level"],
                "delta_vs_baseline_usv_h": round(
                    prediction["dose_rate_usv_h"] - baseline_prediction["dose_rate_usv_h"],
                    4,
                ),
                "input": updated,
            }
        )
    return rows
