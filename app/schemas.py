from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ml.config import (
    BASE_FEATURE_COLUMNS,
    DEFAULT_BASE_FEATURES,
    DEFAULT_REAL_FEATURES,
    REAL_ENV_FEATURE_COLUMNS,
    REAL_NUCLIDE_FEATURE_COLUMNS,
)


ALLOWED_FEATURE_OVERRIDES = set(BASE_FEATURE_COLUMNS)
ALLOWED_ADVANCED_FEATURE_OVERRIDES = set(REAL_NUCLIDE_FEATURE_COLUMNS + REAL_ENV_FEATURE_COLUMNS)


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
    cs137_kBq_m2: float = Field(DEFAULT_REAL_FEATURES["cs137_kBq_m2"], ge=0)
    sr90_kBq_m2: float = Field(DEFAULT_REAL_FEATURES["sr90_kBq_m2"], ge=0)
    k40_Bq_kg: float = Field(DEFAULT_REAL_FEATURES["k40_Bq_kg"], ge=0)
    ra226_Bq_kg: float = Field(DEFAULT_REAL_FEATURES["ra226_Bq_kg"], ge=0)
    th232_Bq_kg: float = Field(DEFAULT_REAL_FEATURES["th232_Bq_kg"], ge=0)
    latitude: float = Field(DEFAULT_REAL_FEATURES["latitude"], ge=-90, le=90)
    longitude: float = Field(DEFAULT_REAL_FEATURES["longitude"], ge=-180, le=180)
    elevation_m: float = Field(DEFAULT_REAL_FEATURES["elevation_m"])
    slope_deg: float = Field(DEFAULT_REAL_FEATURES["slope_deg"], ge=0, le=90)
    distance_to_water_km: float = Field(DEFAULT_REAL_FEATURES["distance_to_water_km"], ge=0)
    rainfall_mm_year: float = Field(DEFAULT_REAL_FEATURES["rainfall_mm_year"], ge=0)
    soil_clay_pct: float = Field(DEFAULT_REAL_FEATURES["soil_clay_pct"], ge=0, le=100)
    soil_organic_pct: float = Field(DEFAULT_REAL_FEATURES["soil_organic_pct"], ge=0, le=100)
    population_density_km2: float = Field(DEFAULT_REAL_FEATURES["population_density_km2"], ge=0)
    land_cover_urban_pct: float = Field(DEFAULT_REAL_FEATURES["land_cover_urban_pct"], ge=0, le=100)

    def to_feature_dict(self) -> dict[str, float]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


class AdvancedPredictionResponse(PredictionResponse):
    data_mode: str


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
            allowed = ", ".join(REAL_NUCLIDE_FEATURE_COLUMNS + REAL_ENV_FEATURE_COLUMNS)
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
    n_estimators: int = Field(300, ge=20, le=2000)
    random_state: int = 42
    cv_splits: int = Field(5, ge=2, le=10)


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
