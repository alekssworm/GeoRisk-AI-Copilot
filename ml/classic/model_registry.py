from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


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


def save_artifact(artifact: ClassicModelArtifact, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: str | Path) -> ClassicModelArtifact:
    return joblib.load(Path(path))
