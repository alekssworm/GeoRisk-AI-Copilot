import numpy as np
import pandas as pd

from ml.config import (
    BASE_FEATURE_COLUMNS,
    DEFAULT_BASE_FEATURES,
    DEFAULT_REAL_FEATURES,
    MODEL_FEATURE_COLUMNS,
    REAL_MODEL_FEATURE_COLUMNS,
    REAL_NUCLIDE_FEATURE_COLUMNS,
    REAL_RATIO_FEATURE_COLUMNS,
    REAL_SPATIAL_COLUMNS,
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


def build_real_feature_frame(
    records: dict | list[dict] | pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    frame = _as_dataframe(records)
    feature_columns = list(feature_columns or REAL_MODEL_FEATURE_COLUMNS)
    required_input_columns = list(
        dict.fromkeys(feature_columns + REAL_SPATIAL_COLUMNS + REAL_RATIO_FEATURE_COLUMNS)
    )
    if "ratio_cs_sr" in feature_columns:
        required_input_columns = list(
            dict.fromkeys(required_input_columns + ["cs137_kBq_m2", "sr90_kBq_m2"])
        )
    ratio_missing = "ratio_cs_sr" not in frame.columns

    for column, default_value in DEFAULT_REAL_FEATURES.items():
        if column in required_input_columns and column not in frame:
            frame[column] = default_value

    if "ratio_cs_sr" in feature_columns and ratio_missing:
        frame["ratio_cs_sr"] = frame["cs137_kBq_m2"] / (frame["sr90_kBq_m2"] + 1e-9)

    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Real feature frame is missing columns: {', '.join(missing)}")

    for column in set(feature_columns + REAL_SPATIAL_COLUMNS):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in REAL_NUCLIDE_FEATURE_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].clip(lower=0)

    bounded_0_100 = [
        "organic_carbon_b0",
        "organic_carbon_b10",
        "clay_fraction_0_30",
        "clay_fraction_30_60",
        "sand_fraction_b0",
        "sand_fraction_b10",
    ]
    for column in bounded_0_100:
        if column in frame.columns:
            frame[column] = frame[column].clip(0, 100)

    if "latitude" in frame.columns:
        frame["latitude"] = frame["latitude"].clip(-90, 90)
    if "longitude" in frame.columns:
        frame["longitude"] = frame["longitude"].clip(-180, 180)
    if "slope_deg_final" in frame.columns:
        frame["slope_deg_final"] = frame["slope_deg_final"].clip(0, 90)

    for column in ("bulk_density_b0", "bulk_density_b10", "elevation_m", "twi_scaled"):
        if column in frame.columns:
            frame[column] = frame[column].clip(lower=0)

    if "ratio_cs_sr" in feature_columns:
        frame["ratio_cs_sr"] = frame["ratio_cs_sr"].clip(lower=0)

    return frame[feature_columns].copy()
