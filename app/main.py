from pathlib import Path
import logging
import time
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from app.config import MAX_UPLOAD_BYTES, MAX_UPLOAD_MB, UPLOAD_DIR
from app.schemas import (
    PredictionResponse,
    RAGAnswerResponse,
    RAGQuestionRequest,
    RiskReportRequest,
    RiskReportResponse,
    ScenarioComparisonRequest,
    TrainRequest,
    RadiationFeatures,
)
from app.services import (
    compare_for,
    explanation_for,
    generate_report,
    prediction_for,
    rag_answer_for,
)
from ml.predict import clear_prediction_cache
from ml.train import train_model
from rag.ingest import ingest_pdf


logger = logging.getLogger("georisk.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(
    title="GeoRisk AI Copilot",
    description="Environmental and radiation risk analysis with ML, explainability, and RAG.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_observability(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request_failed method=%s path=%s duration_ms=%.2f request_id=%s",
            request.method,
            request.url.path,
            duration_ms,
            request_id,
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{duration_ms:.2f}"
    logger.info(
        "request_complete method=%s path=%s status=%s duration_ms=%.2f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
    )
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "GeoRisk AI Copilot"}


@app.post("/ml/train")
def train(request: TrainRequest) -> dict:
    artifact = train_model(n_samples=request.n_samples, random_state=request.random_state)
    clear_prediction_cache()
    return {
        "message": "Model trained",
        "model_version": artifact.model_version,
        "metrics": artifact.metrics,
    }


@app.post("/ml/predict", response_model=PredictionResponse)
def predict(features: RadiationFeatures) -> dict:
    return prediction_for(features)


@app.post("/ml/scenarios")
def scenarios(request: ScenarioComparisonRequest) -> list[dict]:
    return compare_for(request.baseline, request.scenarios)


@app.post("/ml/explain")
def explain(features: RadiationFeatures) -> dict:
    return explanation_for(features)


@app.post("/rag/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")

    safe_name = Path(file.filename).name
    destination = UPLOAD_DIR / safe_name
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Upload a non-empty PDF file.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF is too large. Maximum upload size is {MAX_UPLOAD_MB} MB.",
        )
    destination.write_bytes(content)

    try:
        result = ingest_pdf(destination, source_name=safe_name)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc

    if result["chunks_added"] == 0:
        result["message"] = "PDF ingested, but no extractable text was found."
    return result


@app.post("/rag/ask", response_model=RAGAnswerResponse)
def ask_rag(request: RAGQuestionRequest) -> RAGAnswerResponse:
    return rag_answer_for(request.question, top_k=request.top_k)


@app.post("/reports/risk", response_model=RiskReportResponse)
def risk_report(request: RiskReportRequest) -> dict:
    return generate_report(
        baseline=request.baseline,
        scenarios=request.scenarios,
        rag_question=request.rag_question,
    )
