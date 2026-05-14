import numpy as np
import pandas as pd

from ml.classic.predict import compare_advanced_scenarios, predict_advanced_dose
from ml.classic.train import load_real_training_frame, train_advanced_model
from ml.config import DEFAULT_REAL_FEATURES
from ml.features import build_real_feature_frame


def write_real_training_csvs(tmp_path, n_rows: int = 80):
    rng = np.random.default_rng(42)
    sample_id = np.arange(n_rows)
    cs137 = rng.uniform(5, 95, size=n_rows)
    sr90 = rng.uniform(0.5, 15, size=n_rows)
    k40 = rng.uniform(250, 650, size=n_rows)
    ra226 = rng.uniform(15, 80, size=n_rows)
    th232 = rng.uniform(10, 70, size=n_rows)

    env = pd.DataFrame(
        {
            "sample_id": sample_id,
            "latitude": rng.uniform(58, 61, size=n_rows),
            "longitude": rng.uniform(9, 12, size=n_rows),
            "elevation_m": rng.uniform(20, 500, size=n_rows),
            "slope_deg": rng.uniform(0, 18, size=n_rows),
            "distance_to_water_km": rng.uniform(0.2, 8, size=n_rows),
            "rainfall_mm_year": rng.uniform(500, 1200, size=n_rows),
            "soil_clay_pct": rng.uniform(8, 55, size=n_rows),
            "soil_organic_pct": rng.uniform(1, 16, size=n_rows),
            "population_density_km2": rng.uniform(5, 1200, size=n_rows),
            "land_cover_urban_pct": rng.uniform(0, 85, size=n_rows),
        }
    )
    dose = (
        0.04
        + 0.006 * cs137
        + 0.018 * sr90
        + 0.00008 * k40
        + 0.0009 * ra226
        + 0.0007 * th232
        - 0.00002 * env["elevation_m"].to_numpy()
    )
    nuclides = pd.DataFrame(
        {
            "sample_id": sample_id,
            "cs137_kBq_m2": cs137,
            "sr90_kBq_m2": sr90,
            "k40_Bq_kg": k40,
            "ra226_Bq_kg": ra226,
            "th232_Bq_kg": th232,
            "dose_rate_usv_h": dose,
        }
    )

    env_path = tmp_path / "train_env_v1.csv"
    nuclide_path = tmp_path / "train_nuclide_v1.csv"
    env.to_csv(env_path, index=False)
    nuclides.to_csv(nuclide_path, index=False)
    return env_path, nuclide_path


def test_build_real_feature_frame_uses_real_nuclides():
    frame = build_real_feature_frame(DEFAULT_REAL_FEATURES)

    assert frame.iloc[0]["cs137_kBq_m2"] == DEFAULT_REAL_FEATURES["cs137_kBq_m2"]
    assert frame.iloc[0]["total_fallout_kBq_m2"] > 0
    assert frame.iloc[0]["natural_activity_index"] > 0


def test_load_real_training_frame_merges_env_and_nuclides(tmp_path):
    env_path, nuclide_path = write_real_training_csvs(tmp_path)
    frame = load_real_training_frame(env_path=env_path, nuclide_path=nuclide_path)

    assert len(frame) == 80
    assert {"cs137_kBq_m2", "latitude", "dose_rate_usv_h"}.issubset(frame.columns)


def test_advanced_training_prediction_and_scenarios(tmp_path):
    env_path, nuclide_path = write_real_training_csvs(tmp_path)
    model_path = tmp_path / "advanced_model.joblib"
    artifact = train_advanced_model(
        env_path=env_path,
        nuclide_path=nuclide_path,
        model_path=model_path,
        n_estimators=30,
        cv_splits=3,
    )

    assert artifact.data_mode == "real"
    assert artifact.training_rows == 80
    assert artifact.metrics["r2"] > 0.80
    assert "cv_rmse" in artifact.cv_metrics

    prediction = predict_advanced_dose(DEFAULT_REAL_FEATURES, model_path=model_path)
    assert prediction["dose_rate_usv_h"] > 0
    assert prediction["data_mode"] == "real"

    scenarios = [
        {"name": "Lower Cs-137", "overrides": {"cs137_kBq_m2": 10}},
        {"name": "Higher Cs-137", "overrides": {"cs137_kBq_m2": 80}},
    ]
    rows = compare_advanced_scenarios(DEFAULT_REAL_FEATURES, scenarios, model_path=model_path)
    assert len(rows) == 3
    assert rows[2]["dose_rate_usv_h"] > rows[1]["dose_rate_usv_h"]
