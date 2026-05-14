from pathlib import Path
import logging
import re
import time
from uuid import uuid4

from anyio import fail_after, to_thread
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.responses import Response

from app.config import (
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_ORIGINS,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    PDF_PROCESSING_TIMEOUT_SECONDS,
    UPLOAD_DIR,
)
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
from app.security import require_api_key, rate_limiter
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
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
)


def _clean_source_name(filename: str) -> str:
    name = Path(filename).name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return cleaned[:120] or "uploaded.pdf"


@app.middleware("http")
async def add_request_observability(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    start = time.perf_counter()
    try:
        try:
            rate_limiter.check(request)
        except HTTPException as exc:
            response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        else:
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


@app.post("/ml/train", dependencies=[Depends(require_api_key)])
def train(request: TrainRequest) -> dict:
    artifact = train_model(n_samples=request.n_samples, random_state=request.random_state)
    clear_prediction_cache()
    return {
        "message": "Model trained",
        "model_version": artifact.model_version,
        "metrics": artifact.metrics,
    }


@app.post("/ml/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
def predict(features: RadiationFeatures) -> dict:
    return prediction_for(features)


@app.post("/ml/scenarios", dependencies=[Depends(require_api_key)])
def scenarios(request: ScenarioComparisonRequest) -> list[dict]:
    return compare_for(request.baseline, request.scenarios)


@app.post("/ml/explain", dependencies=[Depends(require_api_key)])
def explain(features: RadiationFeatures) -> dict:
    return explanation_for(features)


@app.post("/rag/upload", dependencies=[Depends(require_api_key)])
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")

    source_name = _clean_source_name(file.filename)
    safe_name = f"upload_{uuid4().hex}.pdf"
    destination = (UPLOAD_DIR / safe_name).resolve()
    if not destination.is_relative_to(UPLOAD_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid upload destination.")

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
        with fail_after(PDF_PROCESSING_TIMEOUT_SECONDS):
            result = await to_thread.run_sync(
                ingest_pdf,
                destination,
                source_name,
                abandon_on_cancel=True,
            )
    except TimeoutError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=408,
            detail="PDF processing timed out. Try a smaller or text-searchable PDF.",
        ) from exc
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc

    if result["chunks_added"] == 0:
        result["message"] = "PDF ingested, but no extractable text was found."
    return result


@app.post("/rag/ask", response_model=RAGAnswerResponse, dependencies=[Depends(require_api_key)])
def ask_rag(request: RAGQuestionRequest) -> RAGAnswerResponse:
    return rag_answer_for(request.question, top_k=request.top_k)


@app.post(
    "/reports/risk", response_model=RiskReportResponse, dependencies=[Depends(require_api_key)]
)
def risk_report(request: RiskReportRequest) -> dict:
    return generate_report(
        baseline=request.baseline,
        scenarios=request.scenarios,
        rag_question=request.rag_question,
    )
