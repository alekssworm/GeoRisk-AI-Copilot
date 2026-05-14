from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


@dataclass(frozen=True)
class SpatialCVSpec:
    latitude_col: str = "latitude"
    longitude_col: str = "longitude"
    block_size_deg: float = 0.02
    n_splits: int = 5


def infer_block_origin(
    frame: pd.DataFrame,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
) -> tuple[float, float]:
    return float(frame[latitude_col].min()), float(frame[longitude_col].min())


def make_spatial_group_labels(
    frame: pd.DataFrame,
    spec: SpatialCVSpec = SpatialCVSpec(),
    origin: tuple[float, float] | None = None,
) -> np.ndarray:
    if spec.latitude_col not in frame or spec.longitude_col not in frame:
        return np.arange(len(frame))
    if spec.block_size_deg <= 0:
        raise ValueError("block_size_deg must be positive.")

    lat0, lon0 = origin or infer_block_origin(frame, spec.latitude_col, spec.longitude_col)
    lat_blocks = np.floor((frame[spec.latitude_col].astype(float) - lat0) / spec.block_size_deg)
    lon_blocks = np.floor((frame[spec.longitude_col].astype(float) - lon0) / spec.block_size_deg)
    labels = lat_blocks.astype(int).astype(str) + "_" + lon_blocks.astype(int).astype(str)
    return labels.to_numpy()


def spatial_group_labels(
    frame: pd.DataFrame,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    block_size_deg: float = 0.02,
) -> np.ndarray:
    spec = SpatialCVSpec(
        latitude_col=latitude_col,
        longitude_col=longitude_col,
        block_size_deg=block_size_deg,
    )
    return make_spatial_group_labels(frame, spec=spec)


def spatial_cv_splits(
    frame: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    block_size_deg: float = 0.02,
):
    n_rows = len(frame)
    if n_rows < 2:
        raise ValueError("At least two rows are required for cross-validation.")

    spec = SpatialCVSpec(n_splits=n_splits, block_size_deg=block_size_deg)
    groups = make_spatial_group_labels(frame, spec=spec)
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        split_count = min(n_splits, len(unique_groups))
        yield from GroupKFold(n_splits=split_count).split(frame, groups=groups)
        return

    split_count = min(n_splits, n_rows)
    yield from KFold(n_splits=split_count, shuffle=True, random_state=random_state).split(frame)
