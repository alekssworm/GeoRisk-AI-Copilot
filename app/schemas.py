from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ml.config import (
    BASE_FEATURE_COLUMNS,
    DEFAULT_BASE_FEATURES,
    DEFAULT_REAL_FEATURES,
    REAL_ENV_FEATURE_COLUMNS,
    REAL_NUCLIDE_FEATURE_COLUMNS,
    REAL_RATIO_FEATURE_COLUMNS,
    REAL_SPATIAL_COLUMNS,
)
from ml.classic.feature_sets import DEFAULT_FEATURE_SET, list_feature_sets
from ml.classic.model_registry import DEFAULT_MODEL_NAME, list_model_names


ALLOWED_FEATURE_OVERRIDES = set(BASE_FEATURE_COLUMNS)
ALLOWED_ADVANCED_FEATURE_OVERRIDES = set(
    REAL_NUCLIDE_FEATURE_COLUMNS
    + REAL_ENV_FEATURE_COLUMNS
    + REAL_RATIO_FEATURE_COLUMNS
    + REAL_SPATIAL_COLUMNS
)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RadiationFeatures(StrictBaseModel):
    contamination_bq_m2: float = Field(
        DEFAULT_BASE_FEATURES["contamination_bq_m2"], ge=0, description="Surface contamination"
    )
    soil_clay_pct: float = Field(DEFAULT_BASE_FEATURES["soil_clay_pct"], ge=0, le=100)
    soil_organic_pct: float = Field(DEFAULT_BASE_FEATURES["soil_organic_pct"], ge=0, le=100)
    rainfall_mm_year: float = Field(DEFAULT_BASE_FEATURES["rainfall_mm_year"], ge=0)
    elevation_m: float = Field(DEFAULT_BASE_FEATURES["elevation_m"])
    slope_deg: float = Field(DEFAULT_BASE_FEATURES["slope_deg"], ge=0, le=90)
    distance_to_water_km: float = Field(DEFAULT_BASE_FEATURES["distance_to_water_km"], ge=0)
    population_density_km2: float = Field(DEFAULT_BASE_FEATURES["population_density_km2"], ge=0)
    latitude: float = Field(DEFAULT_BASE_FEATURES["latitude"], ge=-90, le=90)
    longitude: float = Field(DEFAULT_BASE_FEATURES["longitude"], ge=-180, le=180)
    land_cover_urban_pct: float = Field(DEFAULT_BASE_FEATURES["land_cover_urban_pct"], ge=0, le=100)

    def to_feature_dict(self) -> dict[str, float]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


class PredictionResponse(StrictBaseModel):
    dose_rate_usv_h: float
    risk_level: str
    advisory: str
    model_version: str
    features_used: dict[str, float]


class AdvancedRadiationFeatures(StrictBaseModel):
    latitude: float = Field(DEFAULT_REAL_FEATURES["latitude"], ge=-90, le=90)
    longitude: float = Field(DEFAULT_REAL_FEATURES["longitude"], ge=-180, le=180)
    organic_carbon_b0: float = Field(DEFAULT_REAL_FEATURES["organic_carbon_b0"], ge=0, le=100)
    organic_carbon_b10: float = Field(DEFAULT_REAL_FEATURES["organic_carbon_b10"], ge=0, le=100)
    clay_fraction_0_30: float = Field(DEFAULT_REAL_FEATURES["clay_fraction_0_30"], ge=0, le=100)
    clay_fraction_30_60: float = Field(DEFAULT_REAL_FEATURES["clay_fraction_30_60"], ge=0, le=100)
    sand_fraction_b0: float = Field(DEFAULT_REAL_FEATURES["sand_fraction_b0"], ge=0, le=100)
    sand_fraction_b10: float = Field(DEFAULT_REAL_FEATURES["sand_fraction_b10"], ge=0, le=100)
    bulk_density_b0: float = Field(DEFAULT_REAL_FEATURES["bulk_density_b0"], ge=0)
    bulk_density_b10: float = Field(DEFAULT_REAL_FEATURES["bulk_density_b10"], ge=0)
    soil_pH_b0: float = Field(DEFAULT_REAL_FEATURES["soil_pH_b0"], ge=0)
    soil_pH_b10: float = Field(DEFAULT_REAL_FEATURES["soil_pH_b10"], ge=0)
    elevation_m: float = Field(DEFAULT_REAL_FEATURES["elevation_m"], ge=0)
    slope_deg_final: float = Field(DEFAULT_REAL_FEATURES["slope_deg_final"], ge=0, le=90)
    twi_scaled: float = Field(DEFAULT_REAL_FEATURES["twi_scaled"], ge=0)
    cs137_kBq_m2: float = Field(DEFAULT_REAL_FEATURES["cs137_kBq_m2"], ge=0)
    sr90_kBq_m2: float = Field(DEFAULT_REAL_FEATURES["sr90_kBq_m2"], ge=0)
    ratio_cs_sr: float = Field(DEFAULT_REAL_FEATURES["ratio_cs_sr"], ge=0)
    k40_Bq_kg: float = Field(DEFAULT_REAL_FEATURES["k40_Bq_kg"], ge=0)
    ra226_Bq_kg: float = Field(DEFAULT_REAL_FEATURES["ra226_Bq_kg"], ge=0)
    th232_Bq_kg: float = Field(DEFAULT_REAL_FEATURES["th232_Bq_kg"], ge=0)

    def to_feature_dict(self) -> dict[str, float]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


class AdvancedPredictionResponse(PredictionResponse):
    data_mode: str
    model_name: str
    feature_set: str


class ScenarioInput(StrictBaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    overrides: dict[str, float] = Field(default_factory=dict)

    @field_validator("overrides")
    @classmethod
    def validate_override_keys(cls, overrides: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(overrides) - ALLOWED_FEATURE_OVERRIDES)
        if unknown:
            allowed = ", ".join(BASE_FEATURE_COLUMNS)
            raise ValueError(
                f"Unsupported scenario override(s): {', '.join(unknown)}. Allowed fields: {allowed}"
            )
        return overrides


class AdvancedScenarioInput(StrictBaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    overrides: dict[str, float] = Field(default_factory=dict)

    @field_validator("overrides")
    @classmethod
    def validate_override_keys(cls, overrides: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(overrides) - ALLOWED_ADVANCED_FEATURE_OVERRIDES)
        if unknown:
            allowed = ", ".join(sorted(ALLOWED_ADVANCED_FEATURE_OVERRIDES))
            raise ValueError(
                f"Unsupported advanced scenario override(s): {', '.join(unknown)}. "
                f"Allowed fields: {allowed}"
            )
        return overrides


class ScenarioComparisonRequest(StrictBaseModel):
    baseline: RadiationFeatures
    scenarios: list[ScenarioInput] = Field(default_factory=list)


class AdvancedScenarioComparisonRequest(StrictBaseModel):
    baseline: AdvancedRadiationFeatures
    scenarios: list[AdvancedScenarioInput] = Field(default_factory=list)


class TrainRequest(StrictBaseModel):
    n_samples: int = Field(5000, ge=250, le=100000)
    random_state: int = 42


class AdvancedTrainRequest(StrictBaseModel):
    n_estimators: int = Field(500, ge=20, le=2000)
    random_state: int = 42
    cv_splits: int = Field(5, ge=2, le=10)
    model_name: str = DEFAULT_MODEL_NAME
    feature_set: str = DEFAULT_FEATURE_SET
    block_size_deg: float = Field(0.02, gt=0, le=1)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, model_name: str) -> str:
        if model_name not in list_model_names():
            raise ValueError(f"Unsupported model_name. Available: {', '.join(list_model_names())}")
        return model_name

    @field_validator("feature_set")
    @classmethod
    def validate_feature_set(cls, feature_set: str) -> str:
        if feature_set not in list_feature_sets():
            raise ValueError(
                f"Unsupported feature_set. Available: {', '.join(list_feature_sets())}"
            )
        return feature_set


class RAGQuestionRequest(StrictBaseModel):
    question: str = Field(..., min_length=3, max_length=4000)
    top_k: int = Field(4, ge=1, le=10)


class RAGAnswerResponse(StrictBaseModel):
    answer: str
    citations: list[dict[str, Any]]
    retrieved_context: list[dict[str, Any]]


class RiskReportRequest(StrictBaseModel):
    baseline: RadiationFeatures
    scenarios: list[ScenarioInput] = Field(default_factory=list)
    rag_question: str | None = Field(default=None, max_length=4000)


class RiskReportResponse(StrictBaseModel):
    report_markdown: str
    prediction: PredictionResponse
    scenario_comparison: list[dict[str, Any]]
    explanation: dict[str, Any]
    rag_answer: RAGAnswerResponse | None = None
