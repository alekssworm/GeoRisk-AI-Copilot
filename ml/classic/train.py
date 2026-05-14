from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from ml.classic.feature_sets import REQUIRED_REAL_COLUMNS
from ml.classic.model_registry import ClassicModelArtifact, save_artifact
from ml.classic.spatial_cv import spatial_cv_splits
from ml.config import (
    ADVANCED_FEATURE_IMPORTANCE_PATH,
    ADVANCED_METRICS_PATH,
    ADVANCED_MODEL_PATH,
    REAL_ENV_DATA_PATH,
    REAL_MODEL_FEATURE_COLUMNS,
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


def load_real_training_frame(
    env_path: str | Path = REAL_ENV_DATA_PATH,
    nuclide_path: str | Path = REAL_NUCLIDE_DATA_PATH,
) -> pd.DataFrame:
    env = _read_csv(env_path)
    nuclides = _read_csv(nuclide_path)

    if REAL_RECORD_ID_COLUMN in env.columns and REAL_RECORD_ID_COLUMN in nuclides.columns:
        frame = env.merge(
            nuclides, on=REAL_RECORD_ID_COLUMN, how="inner", suffixes=("", "_nuclide")
        )
    elif len(env) == len(nuclides):
        duplicate_columns = [column for column in nuclides.columns if column in env.columns]
        frame = pd.concat(
            [
                env.reset_index(drop=True),
                nuclides.drop(columns=duplicate_columns).reset_index(drop=True),
            ],
            axis=1,
        )
    else:
        raise ValueError(
            "Real data files must share GEORISK_REAL_RECORD_ID_COLUMN "
            "or contain the same number of rows."
        )

    missing = sorted(REQUIRED_REAL_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Real training data is missing required columns: {', '.join(missing)}")
    return frame


def _metrics(y_true, predictions) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def _cv_metrics(
    frame: pd.DataFrame,
    n_estimators: int,
    random_state: int,
    cv_splits: int,
) -> dict[str, float]:
    fold_metrics = []
    x = build_real_feature_frame(frame)
    y = frame[REAL_TARGET_COLUMN]
    for train_index, test_index in spatial_cv_splits(
        frame, n_splits=cv_splits, random_state=random_state
    ):
        pipeline = _build_pipeline(n_estimators=n_estimators, random_state=random_state)
        pipeline.fit(x.iloc[train_index], y.iloc[train_index])
        predictions = pipeline.predict(x.iloc[test_index])
        fold_metrics.append(_metrics(y.iloc[test_index], predictions))

    return {
        f"cv_{metric}": float(np.mean([fold[metric] for fold in fold_metrics]))
        for metric in ("mae", "rmse", "r2")
    }


def _build_pipeline(n_estimators: int, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_advanced_model(
    env_path: str | Path = REAL_ENV_DATA_PATH,
    nuclide_path: str | Path = REAL_NUCLIDE_DATA_PATH,
    model_path: str | Path = ADVANCED_MODEL_PATH,
    n_estimators: int = 300,
    random_state: int = 42,
    cv_splits: int = 5,
) -> ClassicModelArtifact:
    frame = load_real_training_frame(env_path=env_path, nuclide_path=nuclide_path)
    x = build_real_feature_frame(frame)
    y = frame[REAL_TARGET_COLUMN]

    pipeline = _build_pipeline(n_estimators=n_estimators, random_state=random_state)
    pipeline.fit(x, y)
    predictions = pipeline.predict(x)
    metrics = _metrics(y, predictions)
    cv_metrics = _cv_metrics(
        frame,
        n_estimators=n_estimators,
        random_state=random_state,
        cv_splits=cv_splits,
    )

    trained_at = datetime.now(timezone.utc).isoformat()
    artifact = ClassicModelArtifact(
        pipeline=pipeline,
        feature_names=REAL_MODEL_FEATURE_COLUMNS,
        metrics=metrics,
        cv_metrics=cv_metrics,
        target_name=REAL_TARGET_COLUMN,
        trained_at=trained_at,
        model_version=f"classic-real-{trained_at[:10]}",
        data_mode="real",
        training_rows=len(frame),
    )

    save_artifact(artifact, model_path)

    metrics_path = (
        ADVANCED_METRICS_PATH
        if Path(model_path) == ADVANCED_MODEL_PATH
        else Path(model_path).with_suffix(".metrics.json")
    )
    metrics_path.write_text(
        json.dumps({"metrics": metrics, "cv_metrics": cv_metrics}, indent=2),
        encoding="utf-8",
    )

    model = pipeline.named_steps["model"]
    importance = pd.DataFrame(
        {"feature": REAL_MODEL_FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_path = (
        ADVANCED_FEATURE_IMPORTANCE_PATH
        if Path(model_path) == ADVANCED_MODEL_PATH
        else Path(model_path).with_suffix(".feature_importance.csv")
    )
    importance.to_csv(importance_path, index=False)

    return artifact


if __name__ == "__main__":
    trained = train_advanced_model()
    print(
        json.dumps(
            {
                "model_version": trained.model_version,
                "metrics": trained.metrics,
                "cv_metrics": trained.cv_metrics,
            },
            indent=2,
        )
    )
