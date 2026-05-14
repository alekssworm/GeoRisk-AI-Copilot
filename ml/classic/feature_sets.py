from __future__ import annotations

from dataclasses import dataclass


TARGET_COLUMN = "target_dose_rate"
SECONDARY_TARGET_COLUMN = "target_dose_rate_0_1m"
ID_COLUMNS = ("Code",)
COORD_COLUMNS = ("latitude", "longitude")

ENV_BASE_FEATURES = [
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
]

CONTAMINATION_FEATURES = ["cs137_kBq_m2", "sr90_kBq_m2", "ratio_cs_sr"]
CONTAMINATION_NO_RATIO_FEATURES = ["cs137_kBq_m2", "sr90_kBq_m2"]
NATURAL_NUCLIDE_FEATURES = ["k40_Bq_kg", "ra226_Bq_kg", "th232_Bq_kg"]
FULL_NUCLIDE_FEATURES = CONTAMINATION_FEATURES + NATURAL_NUCLIDE_FEATURES
FULL_NUCLIDE_NO_RATIO_FEATURES = CONTAMINATION_NO_RATIO_FEATURES + NATURAL_NUCLIDE_FEATURES

ENV_MODE = "env_mode"
NUCLIDE_MODE = "nuclide_mode"
DEFAULT_FEATURE_SET = "env_plus_no_ratio"


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    mode: str
    feature_names: list[str]
    description: str


FEATURE_SET_REGISTRY = {
    "env_only": FeatureSetSpec(
        name="env_only",
        mode=ENV_MODE,
        feature_names=ENV_BASE_FEATURES,
        description="Environmental and terrain predictors only.",
    ),
    "nuclide_only": FeatureSetSpec(
        name="nuclide_only",
        mode=NUCLIDE_MODE,
        feature_names=FULL_NUCLIDE_FEATURES,
        description="Artificial and natural radionuclide predictors, including Cs/Sr ratio.",
    ),
    "nuclide_no_ratio": FeatureSetSpec(
        name="nuclide_no_ratio",
        mode=NUCLIDE_MODE,
        feature_names=FULL_NUCLIDE_NO_RATIO_FEATURES,
        description="Radionuclide predictors without the engineered Cs/Sr ratio.",
    ),
    "env_plus": FeatureSetSpec(
        name="env_plus",
        mode=NUCLIDE_MODE,
        feature_names=ENV_BASE_FEATURES + FULL_NUCLIDE_FEATURES,
        description="MVP-B full environmental plus radionuclide feature set.",
    ),
    "env_plus_no_ratio": FeatureSetSpec(
        name="env_plus_no_ratio",
        mode=NUCLIDE_MODE,
        feature_names=ENV_BASE_FEATURES + FULL_NUCLIDE_NO_RATIO_FEATURES,
        description="MVP-B primary feature set without the Cs/Sr ratio.",
    ),
}

REQUIRED_REAL_COLUMNS = set(
    FEATURE_SET_REGISTRY[DEFAULT_FEATURE_SET].feature_names + [TARGET_COLUMN]
)
OPTIONAL_REAL_COLUMNS = set(COORD_COLUMNS + ID_COLUMNS)
ALLOWED_ADVANCED_OVERRIDES = set(ENV_BASE_FEATURES + FULL_NUCLIDE_FEATURES + list(COORD_COLUMNS))


def get_feature_set(name: str = DEFAULT_FEATURE_SET) -> FeatureSetSpec:
    try:
        return FEATURE_SET_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(FEATURE_SET_REGISTRY))
        raise ValueError(f"Unknown feature set '{name}'. Available: {available}") from exc


def list_feature_sets() -> list[str]:
    return sorted(FEATURE_SET_REGISTRY)


def required_columns_for_feature_set(name: str = DEFAULT_FEATURE_SET) -> set[str]:
    spec = get_feature_set(name)
    return set(spec.feature_names + [TARGET_COLUMN])
