import json
import os
from pathlib import Path
import sys

import pandas as pd
import requests
import streamlit as st
from requests import RequestException


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import DEFAULT_BASE_FEATURES  # noqa: E402


API_URL = os.getenv("GEORISK_API_URL", "http://localhost:8000")


st.set_page_config(page_title="GeoRisk AI Copilot", layout="wide")
st.title("GeoRisk AI Copilot")
st.caption(
    "Radiation dose prediction, scenario analysis, explainability, and document-grounded RAG."
)


def post_api(base_url: str, path: str, **kwargs):
    try:
        response = requests.post(f"{base_url.rstrip('/')}{path}", **kwargs)
    except RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None

    if not response.ok:
        st.error(response.text)
        return None

    try:
        return response.json()
    except ValueError:
        st.error("API returned an invalid JSON response.")
        return None


def location_frame(features: dict, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "label": label,
                "lat": features["latitude"],
                "lon": features["longitude"],
            }
        ]
    )


with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("FastAPI URL", API_URL)
    st.divider()
    if st.button("Train / refresh model"):
        with st.spinner("Training model..."):
            result = post_api(
                api_url,
                "/ml/train",
                json={"n_samples": 5000, "random_state": 42},
                timeout=120,
            )
        if result is not None:
            st.success("Model trained")
            st.json(result["metrics"])


def feature_inputs(prefix: str = "baseline") -> dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        contamination = st.number_input(
            "Contamination (Bq/m2)",
            min_value=0.0,
            value=float(DEFAULT_BASE_FEATURES["contamination_bq_m2"]),
            step=1000.0,
            key=f"{prefix}_contamination",
        )
        clay = st.slider(
            "Soil clay (%)",
            0.0,
            100.0,
            float(DEFAULT_BASE_FEATURES["soil_clay_pct"]),
            key=f"{prefix}_clay",
        )
        organic = st.slider(
            "Soil organic (%)",
            0.0,
            100.0,
            float(DEFAULT_BASE_FEATURES["soil_organic_pct"]),
            key=f"{prefix}_organic",
        )
        rainfall = st.number_input(
            "Rainfall (mm/year)",
            min_value=0.0,
            value=float(DEFAULT_BASE_FEATURES["rainfall_mm_year"]),
            step=25.0,
            key=f"{prefix}_rainfall",
        )
    with c2:
        elevation = st.number_input(
            "Elevation (m)",
            value=float(DEFAULT_BASE_FEATURES["elevation_m"]),
            step=10.0,
            key=f"{prefix}_elevation",
        )
        slope = st.slider(
            "Slope (deg)",
            0.0,
            90.0,
            float(DEFAULT_BASE_FEATURES["slope_deg"]),
            key=f"{prefix}_slope",
        )
        water = st.number_input(
            "Distance to water (km)",
            min_value=0.0,
            value=float(DEFAULT_BASE_FEATURES["distance_to_water_km"]),
            step=0.25,
            key=f"{prefix}_water",
        )
        population = st.number_input(
            "Population density (/km2)",
            min_value=0.0,
            value=float(DEFAULT_BASE_FEATURES["population_density_km2"]),
            step=25.0,
            key=f"{prefix}_population",
        )
    with c3:
        latitude = st.number_input(
            "Latitude",
            value=float(DEFAULT_BASE_FEATURES["latitude"]),
            step=0.1,
            key=f"{prefix}_lat",
        )
        longitude = st.number_input(
            "Longitude",
            value=float(DEFAULT_BASE_FEATURES["longitude"]),
            step=0.1,
            key=f"{prefix}_lon",
        )
        urban = st.slider(
            "Urban land cover (%)",
            0.0,
            100.0,
            float(DEFAULT_BASE_FEATURES["land_cover_urban_pct"]),
            key=f"{prefix}_urban",
        )

    return {
        "contamination_bq_m2": contamination,
        "soil_clay_pct": clay,
        "soil_organic_pct": organic,
        "rainfall_mm_year": rainfall,
        "elevation_m": elevation,
        "slope_deg": slope,
        "distance_to_water_km": water,
        "population_density_km2": population,
        "latitude": latitude,
        "longitude": longitude,
        "land_cover_urban_pct": urban,
    }


tabs = st.tabs(["Prediction", "Scenarios", "PDF Assistant", "Risk Report"])

with tabs[0]:
    st.subheader("Dose Rate Prediction")
    baseline = feature_inputs("predict")
    st.map(location_frame(baseline, "Prediction point"), latitude="lat", longitude="lon", zoom=7)
    if st.button("Predict risk", type="primary"):
        with st.spinner("Running prediction..."):
            result = post_api(api_url, "/ml/predict", json=baseline, timeout=60)
        if result is not None:
            m1, m2 = st.columns(2)
            m1.metric("Predicted dose rate", f"{result['dose_rate_usv_h']:.3f} uSv/h")
            m2.metric("Risk level", result["risk_level"])
            st.info(result["advisory"])

            with st.spinner("Explaining main drivers..."):
                explanation = post_api(api_url, "/ml/explain", json=baseline, timeout=60)
            if explanation is not None:
                importance = pd.DataFrame(explanation["top_features"])
                st.bar_chart(importance.set_index("feature")["absolute_contribution"])
                st.download_button(
                    "Download prediction JSON",
                    data=json.dumps(result, indent=2),
                    file_name="prediction.json",
                    mime="application/json",
                )

with tabs[1]:
    st.subheader("Scenario Comparison")
    scenario_baseline = feature_inputs("scenario_base")
    c1, c2, c3 = st.columns(3)
    with c1:
        high_rain = st.number_input(
            "Wet scenario rainfall", value=scenario_baseline["rainfall_mm_year"] * 1.4, step=25.0
        )
    with c2:
        remediation = st.number_input(
            "Remediated contamination",
            value=scenario_baseline["contamination_bq_m2"] * 0.55,
            step=1000.0,
        )
    with c3:
        new_distance = st.number_input(
            "Water proximity scenario (km)",
            value=max(0.2, scenario_baseline["distance_to_water_km"] * 0.5),
            step=0.25,
        )

    scenarios = [
        {"name": "Wet year", "overrides": {"rainfall_mm_year": high_rain}},
        {"name": "Remediation", "overrides": {"contamination_bq_m2": remediation}},
        {"name": "Closer water pathway", "overrides": {"distance_to_water_km": new_distance}},
    ]
    if st.button("Compare scenarios", type="primary"):
        payload = {"baseline": scenario_baseline, "scenarios": scenarios}
        with st.spinner("Comparing scenarios..."):
            result = post_api(api_url, "/ml/scenarios", json=payload, timeout=60)
        if result is not None:
            frame = pd.DataFrame(result)
            st.dataframe(
                frame[["name", "dose_rate_usv_h", "risk_level", "delta_vs_baseline_usv_h"]],
                use_container_width=True,
            )
            st.line_chart(frame.set_index("name")["dose_rate_usv_h"])
            st.download_button(
                "Download scenario CSV",
                data=frame.to_csv(index=False),
                file_name="scenario_comparison.csv",
                mime="text/csv",
            )

with tabs[2]:
    st.subheader("Technical PDF Assistant")
    uploaded = st.file_uploader("Upload a technical PDF", type=["pdf"])
    if uploaded and st.button("Ingest PDF"):
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        with st.spinner("Indexing PDF..."):
            result = post_api(api_url, "/rag/upload", files=files, timeout=120)
        if result is not None:
            st.success(f"Ingested {result['chunks_added']} chunks")

    question = st.text_input("Ask a document-grounded question")
    if st.button("Ask assistant", type="primary") and question:
        with st.spinner("Retrieving context..."):
            result = post_api(
                api_url, "/rag/ask", json={"question": question, "top_k": 4}, timeout=120
            )
        if result is not None:
            st.write(result["answer"])
            st.dataframe(pd.DataFrame(result["citations"]), use_container_width=True)

with tabs[3]:
    st.subheader("Risk Analysis Report")
    report_baseline = feature_inputs("report")
    report_question = st.text_input(
        "Optional document question for report context", key="report_question"
    )
    report_scenarios = [
        {
            "name": "Higher rainfall",
            "overrides": {"rainfall_mm_year": report_baseline["rainfall_mm_year"] * 1.25},
        },
        {
            "name": "50% contamination reduction",
            "overrides": {"contamination_bq_m2": report_baseline["contamination_bq_m2"] * 0.5},
        },
    ]
    if st.button("Generate report", type="primary"):
        payload = {
            "baseline": report_baseline,
            "scenarios": report_scenarios,
            "rag_question": report_question or None,
        }
        with st.spinner("Generating report..."):
            result = post_api(api_url, "/reports/risk", json=payload, timeout=120)
        if result is not None:
            st.markdown(result["report_markdown"])
            st.download_button(
                "Download report Markdown",
                data=result["report_markdown"],
                file_name="risk_report.md",
                mime="text/markdown",
            )
