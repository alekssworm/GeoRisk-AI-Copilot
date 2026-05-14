from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.artifact import ModelArtifact
from ml.config import (
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from ml.data import generate_synthetic_radiation_dataset
from ml.features import build_feature_frame


def _feature_stats(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        column: {
            "mean": float(frame[column].mean()),
            "std": float(frame[column].std() or 1.0),
        }
        for column in frame.columns
    }


def train_model(
    n_samples: int = 5000,
    random_state: int = 42,
    model_path: str | Path = MODEL_PATH,
    n_estimators: int = 250,
) -> ModelArtifact:
    raw = generate_synthetic_radiation_dataset(n_samples=n_samples, random_state=random_state)
    x = build_feature_frame(raw)
    y = raw[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=3,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    trained_at = datetime.now(timezone.utc).isoformat()
    artifact = ModelArtifact(
        pipeline=pipeline,
        feature_names=MODEL_FEATURE_COLUMNS,
        metrics=metrics,
        feature_stats=_feature_stats(x_train),
        target_name=TARGET_COLUMN,
        trained_at=trained_at,
        model_version=f"rf-synthetic-{trained_at[:10]}",
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    metrics_path = (
        METRICS_PATH if model_path == MODEL_PATH else model_path.with_suffix(".metrics.json")
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model = pipeline.named_steps["model"]
    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_path = (
        FEATURE_IMPORTANCE_PATH
        if model_path == MODEL_PATH
        else model_path.with_suffix(".feature_importance.csv")
    )
    importance.to_csv(importance_path, index=False)

    return artifact


if __name__ == "__main__":
    trained = train_model()
    print(
        json.dumps({"model_version": trained.model_version, "metrics": trained.metrics}, indent=2)
    )
