import json
import os
import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from requests import RequestException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import BASE_FEATURE_COLUMNS, DEFAULT_BASE_FEATURES

API_URL = os.getenv("GEORISK_API_URL", "http://localhost:8000")
API_KEY = os.getenv("GEORISK_API_KEY", "")
RISK_COLORS = {
    "low": [34, 197, 94, 190],
    "moderate": [234, 179, 8, 200],
    "elevated": [249, 115, 22, 210],
    "high": [249, 115, 22, 210],
    "critical": [220, 38, 38, 220],
}


st.set_page_config(page_title="GeoRisk AI Copilot", page_icon="🌍", layout="wide")
st.title("GeoRisk AI Copilot")
st.caption(
    "Radiation dose prediction, scenario analysis, uncertainty, explainability, and "
    "document-grounded RAG."
)


def api_headers() -> dict[str, str]:
    api_key = st.session_state.get("api_key", "").strip()
    return {"X-API-Key": api_key} if api_key else {}


def request_api(method: str, base_url: str, path: str, *, expect_json: bool = True, **kwargs):
    headers = dict(kwargs.pop("headers", {}) or {}) | api_headers()
    if headers:
        kwargs["headers"] = headers
    try:
        response = requests.request(method, f"{base_url.rstrip('/')}{path}", **kwargs)
    except RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"API {response.status_code}: {detail}")
        return None
    if not expect_json:
        return response.content
    try:
        return response.json()
    except ValueError:
        st.error("API returned an invalid JSON response.")
        return None


def post_api(base_url: str, path: str, **kwargs):
    return request_api("POST", base_url, path, **kwargs)


def risk_map(frame: pd.DataFrame, *, zoom: float = 5.0) -> None:
    if frame.empty or not {"lat", "lon"}.issubset(frame.columns):
        return
    points = frame.copy()
    points["risk_level"] = points.get("risk_level", "unknown").astype(str)
    points["color"] = points["risk_level"].str.lower().map(RISK_COLORS)
    points["color"] = points["color"].apply(
        lambda value: value if isinstance(value, list) else [59, 130, 246, 190]
    )
    points["dose_rate_usv_h"] = pd.to_numeric(
        points.get("dose_rate_usv_h", 0.0), errors="coerce"
    ).fillna(0.0)
    center = {
        "latitude": float(points["lat"].mean()),
        "longitude": float(points["lon"].mean()),
    }
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=1800 if len(points) > 1 else 4500,
        radius_min_pixels=5,
        radius_max_pixels=24,
        pickable=True,
        stroked=True,
        get_line_color=[15, 23, 42, 220],
        line_width_min_pixels=1,
    )
    tooltip = {
        "html": "<b>{label}</b><br/>Risk: {risk_level}<br/>Dose: {dose_rate_usv_h} uSv/h",
        "style": {"backgroundColor": "#0f172a", "color": "white"},
    }
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(**center, zoom=zoom, pitch=0),
            tooltip=tooltip,
            map_style=None,
        ),
        width="stretch",
    )


def feature_inputs(prefix: str = "baseline") -> dict[str, float]:
    c1, c2, c3 = st.columns(3)
    with c1:
        contamination = st.number_input(
            "Contamination (Bq/m²)",
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
            "Population density (/km²)",
            min_value=0.0,
            value=float(DEFAULT_BASE_FEATURES["population_density_km2"]),
            step=25.0,
            key=f"{prefix}_population",
        )
    with c3:
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(DEFAULT_BASE_FEATURES["latitude"]),
            step=0.1,
            key=f"{prefix}_lat",
        )
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
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


def prediction_summary(result: dict) -> None:
    uncertainty = result.get("uncertainty") or {}
    columns = st.columns(3)
    columns[0].metric("Predicted dose rate", f"{result['dose_rate_usv_h']:.3f} µSv/h")
    columns[1].metric("Risk level", result["risk_level"])
    columns[2].metric("Model confidence", uncertainty.get("confidence_label", "n/a"))
    if uncertainty.get("lower_usv_h") is not None:
        st.caption(
            f"Uncalibrated P10–P90 model spread: {uncertainty['lower_usv_h']:.3f}–"
            f"{uncertainty['upper_usv_h']:.3f} µSv/h."
        )
    distribution = result.get("distribution_check") or {}
    if distribution.get("warning"):
        st.warning(distribution["warning"])
    st.info(result["advisory"])


def scenario_rows_to_payload(frame: pd.DataFrame, baseline: dict[str, float]) -> list[dict]:
    scenarios: dict[str, dict[str, float]] = {}
    for row in frame.fillna("").to_dict("records"):
        name = str(row.get("name", "")).strip()
        feature = str(row.get("feature", "")).strip()
        if not name or feature not in baseline:
            continue
        percent = float(row.get("percent_delta") or 0)
        scenarios.setdefault(name, {})[feature] = baseline[feature] * (1 + percent / 100)
    return [{"name": name, "overrides": overrides} for name, overrides in scenarios.items()]


def batch_payload(frame: pd.DataFrame) -> list[dict[str, float]]:
    records = []
    for source in frame.to_dict("records"):
        record = dict(DEFAULT_BASE_FEATURES)
        for feature in BASE_FEATURE_COLUMNS:
            value = source.get(feature)
            if pd.notna(value):
                record[feature] = float(value)
        records.append(record)
    return records


def geojson_bytes(frame: pd.DataFrame) -> bytes:
    features = []
    for row in frame.where(pd.notna(frame), None).to_dict("records"):
        properties = {
            key: value for key, value in row.items() if key not in {"latitude", "longitude"}
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["longitude"], row["latitude"]],
                },
                "properties": properties,
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2).encode()


with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("FastAPI URL", API_URL)
    st.text_input("API key", value=API_KEY, type="password", key="api_key")
    diagnostics = request_api("GET", api_url, "/health/details", timeout=3)
    if diagnostics:
        state = diagnostics.get("status", "unknown")
        (st.success if state == "ok" else st.warning)(f"System status: {state}")
        with st.expander("Diagnostics"):
            st.json(diagnostics)
    else:
        st.error("Backend is unavailable")
    st.divider()
    if st.button("Train / refresh model"):
        with st.spinner("Training model..."):
            trained = post_api(
                api_url,
                "/ml/train",
                json={"n_samples": 5000, "random_state": 42},
                timeout=120,
            )
        if trained is not None:
            st.success("Model trained")
            st.json(trained["metrics"])


tabs = st.tabs(["Prediction", "Batch & Map", "Scenarios", "PDF Assistant", "Risk Report"])

with tabs[0]:
    st.subheader("Dose Rate Prediction")
    baseline = feature_inputs("predict")
    if st.button("Predict risk", type="primary"):
        with st.spinner("Running prediction..."):
            result = post_api(api_url, "/ml/predict", json=baseline, timeout=60)
        if result is not None:
            prediction_summary(result)
            risk_map(
                pd.DataFrame(
                    [
                        {
                            "label": "Prediction point",
                            "lat": baseline["latitude"],
                            "lon": baseline["longitude"],
                            "risk_level": result["risk_level"],
                            "dose_rate_usv_h": result["dose_rate_usv_h"],
                        }
                    ]
                ),
                zoom=7,
            )
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
    st.subheader("Batch Prediction and Interactive Risk Map")
    st.write(
        "Upload up to 1,000 CSV rows. Missing model columns use the documented default values; "
        "an optional `region` column enables regional comparison."
    )
    template = pd.DataFrame([{**DEFAULT_BASE_FEATURES, "region": "Region A"}])
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False),
        "georisk_batch_template.csv",
        "text/csv",
    )
    batch_file = st.file_uploader("Batch CSV", type=["csv"], key="batch_csv")
    if batch_file is not None:
        try:
            source_frame = pd.read_csv(batch_file)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            st.error(f"Could not read CSV: {exc}")
            source_frame = pd.DataFrame()
        unknown = sorted(
            set(source_frame.columns) - set(BASE_FEATURE_COLUMNS) - {"region", "label"}
        )
        if unknown:
            st.warning(f"Ignored columns: {', '.join(unknown)}")
        if len(source_frame) > 1000:
            st.error("The batch limit is 1,000 rows.")
        elif not source_frame.empty and st.button("Run batch prediction", type="primary"):
            records = batch_payload(source_frame)
            with st.spinner(f"Predicting {len(records)} locations..."):
                batch_result = post_api(
                    api_url,
                    "/ml/predict/batch",
                    json={"records": records},
                    timeout=180,
                )
            if batch_result is not None:
                output = source_frame.copy().reset_index(drop=True)
                predictions = batch_result["predictions"]
                output["dose_rate_usv_h"] = [item["dose_rate_usv_h"] for item in predictions]
                output["risk_level"] = [item["risk_level"] for item in predictions]
                output["confidence"] = [
                    (item.get("uncertainty") or {}).get("confidence_label", "n/a")
                    for item in predictions
                ]
                output["ood_warning"] = [
                    (item.get("distribution_check") or {}).get("warning") or ""
                    for item in predictions
                ]
                defaults = pd.DataFrame(records)
                for coordinate in ("latitude", "longitude"):
                    output[coordinate] = defaults[coordinate]
                map_frame = output.rename(columns={"latitude": "lat", "longitude": "lon"})
                map_frame["label"] = output.get("label", output.index.astype(str))
                risk_map(map_frame)
                st.dataframe(output, width="stretch")
                if "region" in output.columns:
                    regional = (
                        output.groupby("region", dropna=False)
                        .agg(
                            locations=("dose_rate_usv_h", "size"),
                            mean_dose_usv_h=("dose_rate_usv_h", "mean"),
                            max_dose_usv_h=("dose_rate_usv_h", "max"),
                        )
                        .reset_index()
                    )
                    st.markdown("#### Regional comparison")
                    st.dataframe(regional, width="stretch")
                left, right = st.columns(2)
                left.download_button(
                    "Download results CSV",
                    output.to_csv(index=False),
                    "georisk_batch_results.csv",
                    "text/csv",
                )
                right.download_button(
                    "Download results GeoJSON",
                    geojson_bytes(output),
                    "georisk_batch_results.geojson",
                    "application/geo+json",
                )

with tabs[2]:
    st.subheader("Dynamic Scenario Editor")
    scenario_baseline = feature_inputs("scenario_base")
    default_scenarios = pd.DataFrame(
        [
            {"name": "Wet year", "feature": "rainfall_mm_year", "percent_delta": 40.0},
            {
                "name": "Remediation",
                "feature": "contamination_bq_m2",
                "percent_delta": -50.0,
            },
            {
                "name": "Closer water pathway",
                "feature": "distance_to_water_km",
                "percent_delta": -50.0,
            },
        ]
    )
    if "scenario_editor" not in st.session_state:
        st.session_state.scenario_editor = default_scenarios
    scenario_file = st.file_uploader("Load scenario JSON", type=["json"], key="scenario_json")
    if scenario_file is not None and st.button("Load scenarios"):
        try:
            loaded = json.loads(scenario_file.getvalue())
            st.session_state.scenario_editor = pd.DataFrame(loaded)
            st.rerun()
        except (ValueError, TypeError) as exc:
            st.error(f"Invalid scenario JSON: {exc}")
    scenario_frame = st.data_editor(
        st.session_state.scenario_editor,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "name": st.column_config.TextColumn("Scenario", required=True),
            "feature": st.column_config.SelectboxColumn(
                "Feature", options=list(BASE_FEATURE_COLUMNS), required=True
            ),
            "percent_delta": st.column_config.NumberColumn("Change (%)", format="%.1f"),
        },
        key="scenario_table",
    )
    st.session_state.scenario_editor = scenario_frame
    st.download_button(
        "Save scenario JSON",
        json.dumps(scenario_frame.fillna("").to_dict("records"), indent=2),
        "georisk_scenarios.json",
        "application/json",
    )
    scenarios = scenario_rows_to_payload(scenario_frame, scenario_baseline)
    if st.button("Compare scenarios", type="primary"):
        payload = {"baseline": scenario_baseline, "scenarios": scenarios}
        with st.spinner("Comparing scenarios..."):
            comparison = post_api(api_url, "/ml/scenarios", json=payload, timeout=60)
        if comparison is not None:
            st.session_state.latest_scenarios = scenarios
            frame = pd.DataFrame(comparison)
            if not frame.empty:
                baseline_dose = float(frame.iloc[0]["dose_rate_usv_h"])
                frame["delta_vs_baseline_pct"] = frame["delta_vs_baseline_usv_h"].apply(
                    lambda value: value / baseline_dose * 100 if baseline_dose else 0.0
                )
                st.dataframe(frame, width="stretch")
                st.bar_chart(frame.set_index("name")["dose_rate_usv_h"])
                st.download_button(
                    "Download scenario CSV",
                    frame.to_csv(index=False),
                    "scenario_comparison.csv",
                    "text/csv",
                )

with tabs[3]:
    st.subheader("Technical PDF Assistant")
    uploaded = st.file_uploader("Upload a technical PDF", type=["pdf"])
    stable_document = st.checkbox(
        "Keep as stable reference material",
        help="Keep this document on the bottom shelf. It remains available when search expands.",
    )
    if uploaded and st.button("Ingest PDF"):
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        with st.spinner("Indexing PDF..."):
            ingested = post_api(
                api_url,
                "/rag/upload",
                files=files,
                data={"stable": str(stable_document).lower()},
                timeout=120,
            )
        if ingested is not None:
            st.success(f"Ingested {ingested['chunks_added']} chunks")

    question = st.text_input("Ask a document-grounded question")
    full_search = st.checkbox(
        "Search all documents",
        help="Include every shelf, even when the upper shelves already have relevant passages.",
    )
    if st.button("Ask assistant", type="primary") and question:
        with st.spinner("Retrieving context..."):
            answer = post_api(
                api_url,
                "/rag/ask",
                json={"question": question, "top_k": 4, "full_search": full_search},
                timeout=120,
            )
        if answer is not None:
            st.write(answer["answer"])
            st.dataframe(pd.DataFrame(answer["citations"]), width="stretch")
            retrieval = answer.get("retrieval", {})
            if retrieval:
                st.caption(
                    f"Searched {retrieval['chunks_scored']} of {retrieval['total_chunks']} passages "
                    f"across {len(retrieval['shelves_searched'])} shelves."
                )

with tabs[4]:
    st.subheader("Professional Risk Analysis Report")
    report_baseline = feature_inputs("report")
    report_question = st.text_input(
        "Optional document question for report context", key="report_question"
    )
    report_scenarios = st.session_state.get(
        "latest_scenarios",
        [
            {
                "name": "Higher rainfall",
                "overrides": {"rainfall_mm_year": report_baseline["rainfall_mm_year"] * 1.25},
            },
            {
                "name": "50% contamination reduction",
                "overrides": {"contamination_bq_m2": report_baseline["contamination_bq_m2"] * 0.5},
            },
        ],
    )
    if st.button("Generate report", type="primary"):
        payload = {
            "baseline": report_baseline,
            "scenarios": report_scenarios,
            "rag_question": report_question or None,
        }
        with st.spinner("Generating Markdown, PDF, and DOCX reports..."):
            report_json = post_api(api_url, "/reports/risk", json=payload, timeout=120)
            report_pdf = request_api(
                "POST",
                api_url,
                "/reports/risk.pdf",
                expect_json=False,
                json=payload,
                timeout=120,
            )
            report_docx = request_api(
                "POST",
                api_url,
                "/reports/risk.docx",
                expect_json=False,
                json=payload,
                timeout=120,
            )
        if report_json is not None:
            st.session_state.report_bundle = {
                "json": report_json,
                "pdf": report_pdf,
                "docx": report_docx,
            }
    bundle = st.session_state.get("report_bundle")
    if bundle:
        st.markdown(bundle["json"]["report_markdown"])
        download_columns = st.columns(3)
        download_columns[0].download_button(
            "Download Markdown",
            bundle["json"]["report_markdown"],
            "georisk-risk-report.md",
            "text/markdown",
        )
        if bundle.get("pdf"):
            download_columns[1].download_button(
                "Download PDF",
                bundle["pdf"],
                "georisk-risk-report.pdf",
                "application/pdf",
            )
        if bundle.get("docx"):
            download_columns[2].download_button(
                "Download DOCX",
                bundle["docx"],
                "georisk-risk-report.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
