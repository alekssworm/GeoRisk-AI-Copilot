from dataclasses import dataclass

from sklearn.pipeline import Pipeline


@dataclass
class ModelArtifact:
    pipeline: Pipeline
    feature_names: list[str]
    metrics: dict[str, float]
    feature_stats: dict[str, dict[str, float]]
    target_name: str
    trained_at: str
    model_version: str
