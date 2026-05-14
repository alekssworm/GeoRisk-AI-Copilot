import numpy as np
import pandas as pd

from ml.config import (
    BASE_FEATURE_COLUMNS,
    DEFAULT_BASE_FEATURES,
    DEFAULT_REAL_FEATURES,
    MODEL_FEATURE_COLUMNS,
    REAL_ENV_FEATURE_COLUMNS,
    REAL_MODEL_FEATURE_COLUMNS,
    REAL_NUCLIDE_FEATURE_COLUMNS,
)


def _as_dataframe(records: dict | list[dict] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    elif isinstance(records, dict):
        frame = pd.DataFrame([records])
    else:
        frame = pd.DataFrame(records)
    return frame


def build_feature_frame(records: dict | list[dict] | pd.DataFrame) -> pd.DataFrame:
    frame = _as_dataframe(records)

    for column, default_value in DEFAULT_BASE_FEATURES.items():
        if column not in frame:
            frame[column] = default_value

    frame = frame[BASE_FEATURE_COLUMNS].copy()

    frame["contamination_bq_m2"] = frame["contamination_bq_m2"].clip(lower=0)
    frame["soil_clay_pct"] = frame["soil_clay_pct"].clip(0, 100)
    frame["soil_organic_pct"] = frame["soil_organic_pct"].clip(0, 100)
    frame["rainfall_mm_year"] = frame["rainfall_mm_year"].clip(lower=0)
    frame["slope_deg"] = frame["slope_deg"].clip(0, 90)
    frame["distance_to_water_km"] = frame["distance_to_water_km"].clip(lower=0)
    frame["population_density_km2"] = frame["population_density_km2"].clip(lower=0)
    frame["land_cover_urban_pct"] = frame["land_cover_urban_pct"].clip(0, 100)

    frame["soil_retention_index"] = (
        0.55 * frame["soil_clay_pct"] + 0.45 * frame["soil_organic_pct"]
    ) / 100.0
    frame["runoff_potential"] = (
        np.log1p(frame["rainfall_mm_year"]) * np.log1p(frame["slope_deg"])
    ) / 20.0
    frame["water_proximity_index"] = np.exp(-frame["distance_to_water_km"] / 3.0)
    frame["exposure_pressure"] = np.log1p(frame["population_density_km2"]) * (
        1.0 + frame["land_cover_urban_pct"] / 100.0
    )
    frame["spatial_lat_lon_interaction"] = np.sin(np.radians(frame["latitude"])) * np.cos(
        np.radians(frame["longitude"])
    )

    return frame[MODEL_FEATURE_COLUMNS]


def build_real_feature_frame(records: dict | list[dict] | pd.DataFrame) -> pd.DataFrame:
    frame = _as_dataframe(records)

    for column, default_value in DEFAULT_REAL_FEATURES.items():
        if column not in frame:
            frame[column] = default_value

    feature_columns = REAL_NUCLIDE_FEATURE_COLUMNS + REAL_ENV_FEATURE_COLUMNS
    frame = frame[feature_columns].copy()

    for column in REAL_NUCLIDE_FEATURE_COLUMNS:
        frame[column] = frame[column].clip(lower=0)

    frame["latitude"] = frame["latitude"].clip(-90, 90)
    frame["longitude"] = frame["longitude"].clip(-180, 180)
    frame["slope_deg"] = frame["slope_deg"].clip(0, 90)
    frame["distance_to_water_km"] = frame["distance_to_water_km"].clip(lower=0)
    frame["rainfall_mm_year"] = frame["rainfall_mm_year"].clip(lower=0)
    frame["soil_clay_pct"] = frame["soil_clay_pct"].clip(0, 100)
    frame["soil_organic_pct"] = frame["soil_organic_pct"].clip(0, 100)
    frame["population_density_km2"] = frame["population_density_km2"].clip(lower=0)
    frame["land_cover_urban_pct"] = frame["land_cover_urban_pct"].clip(0, 100)

    frame["total_fallout_kBq_m2"] = frame["cs137_kBq_m2"] + frame["sr90_kBq_m2"]
    frame["natural_activity_index"] = (
        frame["k40_Bq_kg"] / 481.0 + frame["ra226_Bq_kg"] / 370.0 + frame["th232_Bq_kg"] / 259.0
    )
    frame["cs137_sr90_ratio"] = frame["cs137_kBq_m2"] / (frame["sr90_kBq_m2"] + 1.0)
    frame["log_cs137"] = np.log1p(frame["cs137_kBq_m2"])
    frame["spatial_lat_lon_interaction"] = np.sin(np.radians(frame["latitude"])) * np.cos(
        np.radians(frame["longitude"])
    )

    return frame[REAL_MODEL_FEATURE_COLUMNS]
