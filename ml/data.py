import numpy as np
import pandas as pd

from ml.config import BASE_FEATURE_COLUMNS, TARGET_COLUMN


def generate_synthetic_radiation_dataset(
    n_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a realistic synthetic dataset for portfolio demos and tests.

    The target is intentionally grounded in transparent assumptions: contamination is
    the dominant driver, soil retention and hydrology modulate transport, and
    population/urban land cover serve as exposure-pressure features.
    """

    rng = np.random.default_rng(random_state)

    contamination = np.clip(rng.lognormal(mean=10.25, sigma=0.65, size=n_samples), 500, 150000)
    soil_clay = rng.beta(2.2, 4.0, size=n_samples) * 65
    soil_organic = rng.beta(1.8, 7.0, size=n_samples) * 18
    rainfall = rng.normal(780, 260, size=n_samples).clip(120, 1800)
    elevation = rng.normal(160, 220, size=n_samples).clip(-5, 1400)
    slope = rng.gamma(2.1, 2.2, size=n_samples).clip(0, 38)
    distance_to_water = rng.gamma(2.4, 1.8, size=n_samples).clip(0.05, 22)
    population_density = rng.lognormal(mean=5.0, sigma=1.15, size=n_samples).clip(1, 6500)
    latitude = rng.uniform(45, 66, size=n_samples)
    longitude = rng.uniform(-8, 32, size=n_samples)
    urban = rng.beta(1.4, 3.2, size=n_samples) * 100

    soil_retention = (0.55 * soil_clay + 0.45 * soil_organic) / 100.0
    runoff = (np.log1p(rainfall) * np.log1p(slope)) / 20.0
    water_proximity = np.exp(-distance_to_water / 3.0)
    exposure = np.log1p(population_density) * (1 + urban / 100)
    spatial = np.sin(np.radians(latitude)) * np.cos(np.radians(longitude))

    dose = (
        0.025
        + contamination * 0.0000062 * (1 + 0.8 * soil_retention)
        + 0.035 * runoff
        + 0.08 * water_proximity
        + 0.006 * exposure
        + 0.025 * spatial
        - 0.000015 * elevation
    )
    noise = rng.normal(0, 0.035, size=n_samples)
    dose = np.clip(dose + noise, 0.01, None)

    frame = pd.DataFrame(
        {
            "contamination_bq_m2": contamination,
            "soil_clay_pct": soil_clay,
            "soil_organic_pct": soil_organic,
            "rainfall_mm_year": rainfall,
            "elevation_m": elevation,
            "slope_deg": slope,
            "distance_to_water_km": distance_to_water,
            "population_density_km2": population_density,
            "latitude": latitude,
            "longitude": longitude,
            "land_cover_urban_pct": urban,
            TARGET_COLUMN: dose,
        }
    )
    return frame[BASE_FEATURE_COLUMNS + [TARGET_COLUMN]]
