# GeoRisk AI Copilot

GeoRisk AI Copilot is an end-to-end AI Engineer portfolio project for environmental
and radiation risk analysis. It combines a traditional machine learning model for
radiation dose-rate prediction with a document-grounded RAG assistant for technical
PDF question answering.

The project is intentionally structured like an application, not a notebook:
FastAPI backend, Streamlit frontend, modular ML and RAG packages, Docker support,
tests, and documentation.

## Core Features

- Train a geospatial/tabular ML model to predict radiation dose rate from
  contamination, soil, terrain, hydrology, exposure, and location features.
- Compare scenarios by changing input parameters and measuring predicted risk deltas.
- Explain predictions using SHAP when available, with a feature-importance fallback.
- Upload technical PDFs, retrieve relevant chunks, answer questions, and cite sources.
- Generate a concise risk analysis report combining ML results, scenarios,
  explainability, and document-grounded answers.
- Cache repeated predictions and expose request IDs/timing headers for API diagnostics.
- Expose the system through FastAPI and a simple Streamlit operator UI.
- Visualize the selected location in the Streamlit workflow and export results.
- Run locally or with Docker Compose.

## Project Structure

```text
app/          FastAPI backend, schemas, orchestration services
ml/           data generation, geospatial helpers, feature engineering, training, prediction
rag/          PDF ingestion, retrieval store, LLM client, Q&A assistant
frontend/     Streamlit UI
tests/        unit tests for ML, RAG, and API health
docs/         architecture notes, API examples, screenshot folder
models/       generated model artifacts
storage/      uploaded PDFs and retrieval index
```

## Architecture

```mermaid
flowchart LR
    UI[Streamlit] --> API[FastAPI]
    API --> ML[ML model]
    API --> RAG[RAG assistant]
    API --> Cache[Prediction cache]
    ML --> Model[(models/georisk_model.joblib)]
    RAG --> Index[(storage/rag_index.joblib)]
    API --> Report[Risk report]
```

The default retrieval index uses a local TF-IDF vector store so the demo runs
offline and inside tests. The RAG layer is isolated behind a small interface and
can be swapped for Chroma or FAISS. If `OPENAI_API_KEY` is set, the assistant uses
an LLM for grounded synthesis; otherwise it returns evidence-focused fallback
answers with citations.

API responses include `X-Request-ID` and `X-Process-Time-ms` headers. Repeated
ML predictions use an in-process LRU cache, which is cleared automatically after
model retraining.

## Quickstart

Recommended with `uv`:

```powershell
uv run --with-requirements requirements-dev.txt python -m ml.train
uv run --with-requirements requirements-dev.txt uvicorn app.main:app --reload
```

In another terminal:

```powershell
uv run --with-requirements requirements-dev.txt streamlit run frontend/streamlit_app.py
```

If you already have Python installed and prefer a classic virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m ml.train
uvicorn app.main:app --reload
```

Open:

- API docs: http://localhost:8000/docs
- Frontend: http://localhost:8501

## Environment

Copy `.env.example` to `.env` and fill optional values:

```powershell
Copy-Item .env.example .env
```

Important variables:

- `GEORISK_API_URL`: frontend-to-backend URL.
- `GEORISK_MODEL_PATH`: model artifact path.
- `GEORISK_RAG_INDEX_PATH`: retrieval index path.
- `GEORISK_UPLOAD_DIR`: uploaded PDF storage path.
- `GEORISK_MAX_UPLOAD_MB`: max PDF upload size.
- `GEORISK_RAG_CHUNK_SIZE` and `GEORISK_RAG_CHUNK_OVERLAP`: retrieval chunking controls.
- `OPENAI_API_KEY`: optional key for LLM-based answer synthesis.
- `GEORISK_LLM_MODEL`: optional LLM model name.

## Model Training and Evaluation

Train the model:

```bash
python -m ml.train
```

This creates:

- `models/georisk_model.joblib`
- `models/georisk_model.metrics.json`
- `models/georisk_model.feature_importance.csv`

The synthetic dataset is generated from transparent environmental assumptions:
contamination is the dominant driver, while soil retention, runoff potential,
water proximity, exposure pressure, elevation, and spatial position modulate the
predicted dose rate.

For real GIS layers, `ml/geospatial.py` includes optional GeoPandas helpers for
converting latitude/longitude records into point geometries and enriching records
with nearest-layer distances.

## API Endpoints

- `GET /health`
- `POST /ml/train`
- `POST /ml/predict`
- `POST /ml/scenarios`
- `POST /ml/explain`
- `POST /rag/upload`
- `POST /rag/ask`
- `POST /reports/risk`

Example payloads are in `docs/api_examples.md`.

## Docker

```bash
docker compose up --build
```

Services:

- FastAPI: http://localhost:8000
- Streamlit: http://localhost:8501

## Tests

```powershell
uv run --with-requirements requirements-dev.txt pytest
uv run --with-requirements requirements-dev.txt ruff check .
uv run --with-requirements requirements-dev.txt ruff format --check .
```

The tests cover model training/prediction/explanation, scenario sensitivity,
retrieval with citations, API health, request observability, and prediction
caching. The same checks run in GitHub Actions through `.github/workflows/ci.yml`.

## Troubleshooting

See `docs/troubleshooting.md` for common local setup issues, including Streamlit
imports, stale model artifacts, `uv` cache permissions, API connectivity, and
text extraction from scanned PDFs.

## Demo Flow

1. Start the API and frontend.
2. Train or refresh the model from the sidebar.
3. Run a baseline prediction.
4. Compare wet-year, remediation, or water-pathway scenarios.
5. Upload a technical PDF and ask a question.
6. Generate a report that combines the prediction, scenario deltas, drivers, and
   document-grounded context.

## Portfolio Notes

This project demonstrates the core responsibilities of an AI Engineer:

- building reusable ML pipelines,
- exposing model behavior through APIs,
- integrating explainability,
- grounding LLM answers in retrieved sources,
- designing a human-facing workflow,
- packaging with Docker,
- and adding tests that protect the main behavior.
