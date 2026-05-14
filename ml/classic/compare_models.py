from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.classic.feature_sets import DEFAULT_FEATURE_SET, get_feature_set
from ml.classic.model_registry import build_model, get_default_params
from ml.classic.spatial_cv import spatial_cv_splits
from ml.classic.train import load_real_training_frame
from ml.config import REAL_ENV_DATA_PATH, REAL_NUCLIDE_DATA_PATH, REAL_TARGET_COLUMN
from ml.features import build_real_feature_frame
from ml.train import train_model


@dataclass(frozen=True)
class CandidateResult:
    name: str
    model_name: str
    feature_set: str
    rows: int
    feature_count: int
    cv_mae_mean: float
    cv_rmse_mean: float
    cv_r2_mean: float
    cv_r2_std: float


def _fold_metrics(y_true, predictions) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def evaluate_real_candidate(
    name: str,
    frame: pd.DataFrame,
    model_name: str,
    feature_set: str,
    random_state: int = 42,
    cv_splits: int = 5,
    block_size_deg: float = 0.02,
) -> CandidateResult:
    spec = get_feature_set(feature_set)
    params = get_default_params(model_name)
    if "random_state" in params:
        params["random_state"] = random_state
    x = build_real_feature_frame(frame, feature_columns=spec.feature_names)
    y = pd.to_numeric(frame[REAL_TARGET_COLUMN], errors="coerce")

    fold_scores = []
    for train_index, test_index in spatial_cv_splits(
        frame,
        n_splits=cv_splits,
        random_state=random_state,
        block_size_deg=block_size_deg,
    ):
        pipeline = build_model(model_name=model_name, params=params)
        pipeline.fit(x.iloc[train_index], y.iloc[train_index])
        predictions = pipeline.predict(x.iloc[test_index])
        fold_scores.append(_fold_metrics(y.iloc[test_index], predictions))

    return CandidateResult(
        name=name,
        model_name=model_name,
        feature_set=feature_set,
        rows=len(frame),
        feature_count=len(spec.feature_names),
        cv_mae_mean=float(np.mean([fold["mae"] for fold in fold_scores])),
        cv_rmse_mean=float(np.mean([fold["rmse"] for fold in fold_scores])),
        cv_r2_mean=float(np.mean([fold["r2"] for fold in fold_scores])),
        cv_r2_std=float(np.std([fold["r2"] for fold in fold_scores], ddof=0)),
    )


def compare_real_models(
    env_path: str | Path = REAL_ENV_DATA_PATH,
    nuclide_path: str | Path = REAL_NUCLIDE_DATA_PATH,
    random_state: int = 42,
    cv_splits: int = 5,
    block_size_deg: float = 0.02,
) -> list[CandidateResult]:
    frame = load_real_training_frame(
        env_path=env_path,
        nuclide_path=nuclide_path,
        feature_set=DEFAULT_FEATURE_SET,
        target_column=REAL_TARGET_COLUMN,
    )
    return [
        evaluate_real_candidate(
            name="MVP-B primary from Radiation_Dose_Rate_Prediction",
            frame=frame,
            model_name="extra_trees",
            feature_set="env_plus_no_ratio",
            random_state=random_state,
            cv_splits=cv_splits,
            block_size_deg=block_size_deg,
        ),
        evaluate_real_candidate(
            name="Current real-data reference without nuclides",
            frame=frame,
            model_name="random_forest",
            feature_set="env_only",
            random_state=random_state,
            cv_splits=cv_splits,
            block_size_deg=block_size_deg,
        ),
    ]


def compare_synthetic_demo(random_state: int = 42) -> dict[str, float | str]:
    artifact = train_model(n_samples=5000, random_state=random_state)
    return {
        "name": "Current synthetic demo model",
        "model_version": artifact.model_version,
        **artifact.metrics,
        "note": "Synthetic split only; not comparable to real-data spatial CV.",
    }


def write_comparison_report(
    output_path: str | Path = "docs/model_comparison.md",
    random_state: int = 42,
    cv_splits: int = 5,
    block_size_deg: float = 0.02,
) -> Path:
    output_path = Path(output_path)
    real_results = compare_real_models(
        random_state=random_state,
        cv_splits=cv_splits,
        block_size_deg=block_size_deg,
    )
    synthetic_result = compare_synthetic_demo(random_state=random_state)

    rows = "\n".join(
        "| {name} | {model_name} | {feature_set} | {rows} | {feature_count} | "
        "{cv_r2_mean:.4f} | {cv_rmse_mean:.5f} | {cv_mae_mean:.5f} |".format(**result.__dict__)
        for result in real_results
    )
    synthetic_line = (
        f"- Current synthetic demo model: test R2 `{synthetic_result['r2']:.4f}`, "
        f"RMSE `{synthetic_result['rmse']:.5f}`, MAE `{synthetic_result['mae']:.5f}`. "
        "This is measured on generated synthetic data, so it is not directly comparable "
        "to the real-data spatial CV table."
    )

    content = f"""# Model Comparison

Generated with `ml.classic.compare_models` using spatial block CV on the original
`Radiation_Dose_Rate_Prediction` nuclide training table.

| Candidate | Model | Feature set | Rows | Features | CV R2 mean | CV RMSE mean | CV MAE mean |
|---|---|---|---:|---:|---:|---:|---:|
{rows}

{synthetic_line}

Conclusion: for the original real dataset, the MVP-B primary model is the
production candidate because it uses the original radionuclide and environmental
feature set. The synthetic demo model remains useful for local UI/API demos, but
its score is not evidence of real-world accuracy.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    path = write_comparison_report()
    print(path)
