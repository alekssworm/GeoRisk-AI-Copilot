from project_config import project_path_from_env


MODEL_PATH = project_path_from_env("GEORISK_MODEL_PATH", "models/georisk_model.joblib")
METRICS_PATH = MODEL_PATH.with_suffix(".metrics.json")
FEATURE_IMPORTANCE_PATH = MODEL_PATH.with_suffix(".feature_importance.csv")

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

RISK_THRESHOLDS = [
    (0.10, "Low", "Routine monitoring is sufficient for this modeled condition."),
    (0.30, "Moderate", "Review monitoring cadence and verify key field assumptions."),
    (0.70, "Elevated", "Prioritize field validation and exposure pathway review."),
    (1.50, "High", "Escalate controls and prepare site-specific mitigation options."),
    (float("inf"), "Critical", "Immediate expert review and protective actions are recommended."),
]
