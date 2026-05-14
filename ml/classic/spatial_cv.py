import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


def spatial_group_labels(
    frame: pd.DataFrame,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    n_bins: int = 5,
) -> np.ndarray:
    if latitude_col not in frame or longitude_col not in frame:
        return np.arange(len(frame))

    lat_bins = pd.cut(
        frame[latitude_col], bins=min(n_bins, max(1, frame[latitude_col].nunique())), labels=False
    )
    lon_bins = pd.cut(
        frame[longitude_col], bins=min(n_bins, max(1, frame[longitude_col].nunique())), labels=False
    )
    lat_bins = pd.Series(lat_bins, index=frame.index).fillna(0).astype(int)
    lon_bins = pd.Series(lon_bins, index=frame.index).fillna(0).astype(int)
    return (lat_bins.astype(str) + "_" + lon_bins.astype(str)).to_numpy()


def spatial_cv_splits(
    frame: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
):
    n_rows = len(frame)
    if n_rows < 2:
        raise ValueError("At least two rows are required for cross-validation.")

    groups = spatial_group_labels(frame)
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        split_count = min(n_splits, len(unique_groups))
        yield from GroupKFold(n_splits=split_count).split(frame, groups=groups)
        return

    split_count = min(n_splits, n_rows)
    yield from KFold(n_splits=split_count, shuffle=True, random_state=random_state).split(frame)
