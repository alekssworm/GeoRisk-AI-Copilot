from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MODEL_NAME = "extra_trees"

DEFAULT_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "ridge_cv": {"alphas": (0.1, 1.0, 10.0)},
    "random_forest": {
        "n_estimators": 500,
        "max_depth": 8,
        "min_samples_leaf": 10,
        "max_features": 0.5,
        "random_state": 42,
        "n_jobs": -1,
    },
    "extra_trees": {
        "n_estimators": 500,
        "max_depth": 6,
        "min_samples_leaf": 2,
        "max_features": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    },
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    default_params: dict[str, Any]


MODEL_REGISTRY = {
    "ridge_cv": ModelSpec(
        name="ridge_cv",
        description="Median-imputed, standardized RidgeCV baseline from the MVP-B registry.",
        default_params=DEFAULT_MODEL_PARAMS["ridge_cv"],
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        description="MVP-B environmental reference model family.",
        default_params=DEFAULT_MODEL_PARAMS["random_forest"],
    ),
    "extra_trees": ModelSpec(
        name="extra_trees",
        description="MVP-B primary model family.",
        default_params=DEFAULT_MODEL_PARAMS["extra_trees"],
    ),
}


@dataclass
class ClassicModelArtifact:
    pipeline: Pipeline
    feature_names: list[str]
    metrics: dict[str, float]
    cv_metrics: dict[str, float]
    target_name: str
    trained_at: str
    model_version: str
    data_mode: str
    training_rows: int
    model_name: str = DEFAULT_MODEL_NAME
    feature_set: str = "env_plus_no_ratio"
    model_params: dict[str, Any] = field(default_factory=dict)


def list_model_names() -> list[str]:
    return sorted(MODEL_REGISTRY)


def get_default_params(model_name: str = DEFAULT_MODEL_NAME) -> dict[str, Any]:
    try:
        return dict(MODEL_REGISTRY[model_name].default_params)
    except KeyError as exc:
        available = ", ".join(list_model_names())
        raise ValueError(f"Unknown classic model '{model_name}'. Available: {available}") from exc


def build_model(
    model_name: str = DEFAULT_MODEL_NAME,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    merged_params = get_default_params(model_name)
    if params:
        merged_params.update(params)

    if model_name == "ridge_cv":
        model = RidgeCV(**merged_params)
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
    if model_name == "random_forest":
        model = RandomForestRegressor(**merged_params)
    elif model_name == "extra_trees":
        model = ExtraTreesRegressor(**merged_params)
    else:
        available = ", ".join(list_model_names())
        raise ValueError(f"Unknown classic model '{model_name}'. Available: {available}")

    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("model", model)])


def save_artifact(artifact: ClassicModelArtifact, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: str | Path) -> ClassicModelArtifact:
    return joblib.load(Path(path))
