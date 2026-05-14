from pathlib import Path

import numpy as np

from ml.config import MODEL_PATH
from ml.features import build_feature_frame
from ml.predict import load_model, predict_dose


def _fallback_contributions(artifact, feature_frame) -> tuple[list[float], str]:
    model = artifact.pipeline.named_steps["model"]
    importances = getattr(model, "feature_importances_", np.ones(len(artifact.feature_names)))
    contributions = []
    for feature, importance in zip(artifact.feature_names, importances, strict=False):
        value = float(feature_frame.iloc[0][feature])
        stats = artifact.feature_stats.get(feature, {"mean": 0.0, "std": 1.0})
        z_score = (value - stats["mean"]) / (stats["std"] or 1.0)
        contributions.append(float(z_score * importance))
    return contributions, "feature_importance_approximation"


def explain_prediction(
    features: dict,
    model_path: str | Path = MODEL_PATH,
    top_n: int = 8,
) -> dict:
    artifact = load_model(model_path)
    feature_frame = build_feature_frame(features)
    transformed = artifact.pipeline.named_steps["imputer"].transform(feature_frame)
    model = artifact.pipeline.named_steps["model"]
    method = "shap"
    base_value = None

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        contributions = [float(value) for value in np.asarray(shap_values)[0]]
        raw_base = explainer.expected_value
        if isinstance(raw_base, (list, np.ndarray)):
            raw_base = raw_base[0]
        base_value = float(raw_base)
    except Exception:
        contributions, method = _fallback_contributions(artifact, feature_frame)

    rows = []
    for feature, contribution in zip(artifact.feature_names, contributions, strict=False):
        value = float(feature_frame.iloc[0][feature])
        rows.append(
            {
                "feature": feature,
                "value": value,
                "contribution": contribution,
                "absolute_contribution": abs(contribution),
                "direction": "increases risk" if contribution >= 0 else "decreases risk",
            }
        )

    rows = sorted(rows, key=lambda item: item["absolute_contribution"], reverse=True)
    prediction = predict_dose(features, model_path=model_path)
    importances = getattr(model, "feature_importances_", np.zeros(len(artifact.feature_names)))
    global_importance = sorted(
        [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(artifact.feature_names, importances, strict=False)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )

    return {
        "method": method,
        "prediction": prediction,
        "base_value": base_value,
        "top_features": rows[:top_n],
        "global_feature_importance": global_importance,
    }
