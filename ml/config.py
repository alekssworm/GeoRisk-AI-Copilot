import os

from project_config import list_from_env, project_path_from_env


DATA_MODE = os.getenv("GEORISK_DATA_MODE", "synthetic").strip().lower()

MODEL_PATH = project_path_from_env("GEORISK_MODEL_PATH", "models/georisk_model.joblib")
ADVANCED_MODEL_PATH = project_path_from_env(
    "GEORISK_ADVANCED_MODEL_PATH", "models/georisk_advanced_model.joblib"
)
METRICS_PATH = MODEL_PATH.with_suffix(".metrics.json")
FEATURE_IMPORTANCE_PATH = MODEL_PATH.with_suffix(".feature_importance.csv")
ADVANCED_METRICS_PATH = ADVANCED_MODEL_PATH.with_suffix(".metrics.json")
ADVANCED_FEATURE_IMPORTANCE_PATH = ADVANCED_MODEL_PATH.with_suffix(".feature_importance.csv")

REAL_ENV_DATA_PATH = project_path_from_env(
    "GEORISK_REAL_ENV_DATA_PATH", "data/processed/train_env_v1.csv"
)
REAL_NUCLIDE_DATA_PATH = project_path_from_env(
    "GEORISK_REAL_NUCLIDE_DATA_PATH", "data/processed/train_nuclide_v1.csv"
)

TARGET_COLUMN = "dose_rate_usv_h"

BASE_FEATURE_COLUMNS = [
    "contamination_bq_m2",
    "soil_clay_pct",
    "soil_organic_pct",
    "rainfall_mm_year",
    "elevation_m",
    "slope_deg",
    "distance_to_water_km",
    "population_density_km2",
    "latitude",
    "longitude",
    "land_cover_urban_pct",
]

ENGINEERED_FEATURE_COLUMNS = [
    "soil_retention_index",
    "runoff_potential",
    "water_proximity_index",
    "exposure_pressure",
    "spatial_lat_lon_interaction",
]

MODEL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS

REAL_TARGET_COLUMN = os.getenv("GEORISK_REAL_TARGET_COLUMN", "target_dose_rate")
REAL_RECORD_ID_COLUMN = os.getenv("GEORISK_REAL_RECORD_ID_COLUMN", "Code")

REAL_NUCLIDE_FEATURE_COLUMNS = list_from_env(
    "GEORISK_REAL_NUCLIDE_FEATURE_COLUMNS",
    [
        "cs137_kBq_m2",
        "sr90_kBq_m2",
        "k40_Bq_kg",
        "ra226_Bq_kg",
        "th232_Bq_kg",
    ],
)

REAL_ENV_FEATURE_COLUMNS = list_from_env(
    "GEORISK_REAL_ENV_FEATURE_COLUMNS",
    [
        "organic_carbon_b0",
        "organic_carbon_b10",
        "clay_fraction_0_30",
        "clay_fraction_30_60",
        "sand_fraction_b0",
        "sand_fraction_b10",
        "bulk_density_b0",
        "bulk_density_b10",
        "soil_pH_b0",
        "soil_pH_b10",
        "elevation_m",
        "slope_deg_final",
        "twi_scaled",
    ],
)

REAL_SPATIAL_COLUMNS = ["latitude", "longitude"]
REAL_RATIO_FEATURE_COLUMNS = ["ratio_cs_sr"]
REAL_ENGINEERED_FEATURE_COLUMNS = []

REAL_MODEL_FEATURE_COLUMNS = REAL_ENV_FEATURE_COLUMNS + REAL_NUCLIDE_FEATURE_COLUMNS

DEFAULT_BASE_FEATURES = {
    "contamination_bq_m2": 35000.0,
    "soil_clay_pct": 28.0,
    "soil_organic_pct": 6.0,
    "rainfall_mm_year": 750.0,
    "elevation_m": 120.0,
    "slope_deg": 4.0,
    "distance_to_water_km": 2.5,
    "population_density_km2": 180.0,
    "latitude": 59.91,
    "longitude": 10.75,
    "land_cover_urban_pct": 35.0,
}

DEFAULT_REAL_FEATURES = {
    "latitude": 50.9712,
    "longitude": 29.8660,
    "organic_carbon_b0": 4.0,
    "organic_carbon_b10": 4.0,
    "clay_fraction_0_30": 16.0,
    "clay_fraction_30_60": 17.0,
    "sand_fraction_b0": 59.0,
    "sand_fraction_b10": 59.0,
    "bulk_density_b0": 132.0,
    "bulk_density_b10": 131.0,
    "soil_pH_b0": 62.0,
    "soil_pH_b10": 62.0,
    "elevation_m": 136.0,
    "slope_deg_final": 10.0,
    "twi_scaled": 5.68,
    "cs137_kBq_m2": 27.4,
    "sr90_kBq_m2": 4.2,
    "ratio_cs_sr": 6.52,
    "k40_Bq_kg": 120.0,
    "ra226_Bq_kg": 11.0,
    "th232_Bq_kg": 8.0,
}

RISK_THRESHOLDS = [
    (0.10, "Low", "Routine monitoring is sufficient for this modeled condition."),
    (0.30, "Moderate", "Review monitoring cadence and verify key field assumptions."),
    (0.70, "Elevated", "Prioritize field validation and exposure pathway review."),
    (1.50, "High", "Escalate controls and prepare site-specific mitigation options."),
    (float("inf"), "Critical", "Immediate expert review and protective actions are recommended."),
]
