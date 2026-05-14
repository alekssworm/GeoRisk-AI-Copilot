from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.classic.feature_sets import (
    DEFAULT_FEATURE_SET,
    get_feature_set,
    required_columns_for_feature_set,
)
from ml.classic.model_registry import (
    DEFAULT_MODEL_NAME,
    ClassicModelArtifact,
    build_model,
    get_default_params,
    save_artifact,
)
from ml.classic.spatial_cv import spatial_cv_splits
from ml.config import (
    ADVANCED_FEATURE_IMPORTANCE_PATH,
    ADVANCED_METRICS_PATH,
    ADVANCED_MODEL_PATH,
    REAL_ENV_DATA_PATH,
    REAL_NUCLIDE_DATA_PATH,
    REAL_RECORD_ID_COLUMN,
    REAL_TARGET_COLUMN,
)
from ml.features import build_real_feature_frame


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Real training data file not found: {path}")
    return pd.read_csv(path)


def _has_required_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    target_column: str,
) -> bool:
    required = set(required_columns)
    required.discard("target_dose_rate")
    required.add(target_column)
    return required.issubset(frame.columns)


def load_real_training_frame(
    env_path: str | Path = REAL_ENV_DATA_PATH,
    nuclide_path: str | Path = REAL_NUCLIDE_DATA_PATH,
    feature_set: str = DEFAULT_FEATURE_SET,
    target_column: str = REAL_TARGET_COLUMN,
) -> pd.DataFrame:
    env = _read_csv(env_path)
    nuclides = _read_csv(nuclide_path)
    required = required_columns_for_feature_set(feature_set)

    if _has_required_columns(nuclides, required, target_column):
        frame = nuclides
    elif _has_required_columns(env, required, target_column):
        frame = env
    elif len(env) == len(nuclides):
        duplicate_columns = [column for column in nuclides.columns if column in env.columns]
        frame = pd.concat(
            [
                env.reset_index(drop=True),
                nuclides.drop(columns=duplicate_columns).reset_index(drop=True),
            ],
            axis=1,
        )
    elif (
        REAL_RECORD_ID_COLUMN in env.columns
        and REAL_RECORD_ID_COLUMN in nuclides.columns
        and not env[REAL_RECORD_ID_COLUMN].duplicated().any()
        and not nuclides[REAL_RECORD_ID_COLUMN].duplicated().any()
    ):
        frame = env.merge(
            nuclides, on=REAL_RECORD_ID_COLUMN, how="inner", suffixes=("", "_nuclide")
        )
    else:
        raise ValueError(
            "Real data files must either contain all requested feature columns, "
            "share row order, or contain a unique GEORISK_REAL_RECORD_ID_COLUMN."
        )

    required_for_runtime = set(required)
    required_for_runtime.discard("target_dose_rate")
    required_for_runtime.add(target_column)
    missing = sorted(required_for_runtime - set(frame.columns))
    if missing:
        raise ValueError(f"Real training data is missing required columns: {', '.join(missing)}")
    return frame.copy()


def _metrics(y_true, predictions) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def _model_params(
    model_name: str,
    random_state: int,
    n_estimators: int | None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = get_default_params(model_name)
    if "random_state" in params:
        params["random_state"] = random_state
    if n_estimators is not None and "n_estimators" in params:
        params["n_estimators"] = n_estimators
    if overrides:
        params.update(overrides)
    return params


def _cv_metrics(
    frame: pd.DataFrame,
    feature_names: list[str],
    target_column: str,
    model_name: str,
    params: dict[str, Any],
    random_state: int,
    cv_splits: int,
    block_size_deg: float,
) -> dict[str, float]:
    fold_metrics = []
    x = build_real_feature_frame(frame, feature_columns=feature_names)
    y = pd.to_numeric(frame[target_column], errors="coerce")

    for train_index, test_index in spatial_cv_splits(
        frame,
        n_splits=cv_splits,
        random_state=random_state,
        block_size_deg=block_size_deg,
    ):
        pipeline = build_model(model_name=model_name, params=params)
        pipeline.fit(x.iloc[train_index], y.iloc[train_index])
        predictions = pipeline.predict(x.iloc[test_index])
        fold_metrics.append(_metrics(y.iloc[test_index], predictions))

    return {
        f"cv_{metric}": float(np.mean([fold[metric] for fold in fold_metrics]))
        for metric in ("mae", "rmse", "r2")
    }


def _write_feature_importance(
    artifact: ClassicModelArtifact,
    path: Path,
) -> None:
    model = artifact.pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = np.abs(np.ravel(model.coef_))
    else:
        return

    importance = pd.DataFrame(
        {"feature": artifact.feature_names, "importance": importance_values}
    ).sort_values("importance", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(path, index=False)


def train_advanced_model(
    env_path: str | Path = REAL_ENV_DATA_PATH,
    nuclide_path: str | Path = REAL_NUCLIDE_DATA_PATH,
    model_path: str | Path = ADVANCED_MODEL_PATH,
    n_estimators: int | None = 500,
    random_state: int = 42,
    cv_splits: int = 5,
    model_name: str = DEFAULT_MODEL_NAME,
    feature_set: str = DEFAULT_FEATURE_SET,
    block_size_deg: float = 0.02,
    model_params: dict[str, Any] | None = None,
) -> ClassicModelArtifact:
    spec = get_feature_set(feature_set)
    frame = load_real_training_frame(
        env_path=env_path,
        nuclide_path=nuclide_path,
        feature_set=feature_set,
        target_column=REAL_TARGET_COLUMN,
    )
    y = pd.to_numeric(frame[REAL_TARGET_COLUMN], errors="coerce")
    valid_rows = y.notna()
    if not valid_rows.any():
        raise ValueError(f"Target column '{REAL_TARGET_COLUMN}' contains no numeric values.")
    frame = frame.loc[valid_rows].copy()
    y = y.loc[valid_rows]

    params = _model_params(
        model_name=model_name,
        random_state=random_state,
        n_estimators=n_estimators,
        overrides=model_params,
    )
    x = build_real_feature_frame(frame, feature_columns=spec.feature_names)

    pipeline = build_model(model_name=model_name, params=params)
    pipeline.fit(x, y)
    predictions = pipeline.predict(x)
    metrics = _metrics(y, predictions)
    cv_metrics = _cv_metrics(
        frame,
        feature_names=spec.feature_names,
        target_column=REAL_TARGET_COLUMN,
        model_name=model_name,
        params=params,
        random_state=random_state,
        cv_splits=cv_splits,
        block_size_deg=block_size_deg,
    )

    trained_at = datetime.now(timezone.utc).isoformat()
    artifact = ClassicModelArtifact(
        pipeline=pipeline,
        feature_names=spec.feature_names,
        metrics=metrics,
        cv_metrics=cv_metrics,
        target_name=REAL_TARGET_COLUMN,
        trained_at=trained_at,
        model_version=f"mvp-b-{model_name}-{feature_set}-{trained_at[:10]}",
        data_mode="real",
        training_rows=len(frame),
        model_name=model_name,
        feature_set=feature_set,
        model_params=params,
    )

    save_artifact(artifact, model_path)

    metrics_path = (
        ADVANCED_METRICS_PATH
        if Path(model_path) == ADVANCED_MODEL_PATH
        else Path(model_path).with_suffix(".metrics.json")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "feature_set": feature_set,
                "feature_names": spec.feature_names,
                "model_params": params,
                "metrics": metrics,
                "cv_metrics": cv_metrics,
                "training_rows": len(frame),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    importance_path = (
        ADVANCED_FEATURE_IMPORTANCE_PATH
        if Path(model_path) == ADVANCED_MODEL_PATH
        else Path(model_path).with_suffix(".feature_importance.csv")
    )
    _write_feature_importance(artifact, importance_path)

    return artifact


if __name__ == "__main__":
    trained = train_advanced_model()
    print(
        json.dumps(
            {
                "model_version": trained.model_version,
                "model_name": trained.model_name,
                "feature_set": trained.feature_set,
                "metrics": trained.metrics,
                "cv_metrics": trained.cv_metrics,
            },
            indent=2,
        )
    )
