"""FastAPI application exposing default-risk predictions.

Run with:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
GET  /health         -> service + model status
GET  /features        -> list of expected input features
POST /predict         -> single application -> {score, decision}
POST /predict_batch   -> list of applications -> list of results
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.config import DEFAULT_CONFIG
from src.serving.ensemble import ModelEnsemble
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.5"))

ensemble: ModelEnsemble | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ensemble
    models_dir = Path(
        os.environ.get("MODELS_DIR", str(DEFAULT_CONFIG.models_dir))
    )
    logger.info("Loading model ensemble from %s", models_dir)
    ensemble = ModelEnsemble(models_dir).load()
    logger.info("Ensemble ready (%d models, threshold=%.3f)",
                ensemble.n_models, DECISION_THRESHOLD)
    yield
    ensemble = None  # type: ignore[assignment]


app = FastAPI(
    title="Housing Loans Approval API",
    description="Predict home-credit default risk from application features.",
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
async def ui() -> HTMLResponse:
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found</h1>", status_code=404)


class PredictionResult(BaseModel):
    score: float = Field(..., description="Predicted default probability [0, 1]")
    decision: str = Field(..., description="'approve' or 'reject'")
    threshold: float


class BatchRequest(BaseModel):
    applications: list[dict[str, Any]]


class BatchResponse(BaseModel):
    results: list[PredictionResult]
    n: int


def _decide(score: float) -> str:
    return "reject" if score >= DECISION_THRESHOLD else "approve"


def _result(score: float) -> PredictionResult:
    return PredictionResult(
        score=float(score),
        decision=_decide(score),
        threshold=DECISION_THRESHOLD,
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "models_loaded": ensemble is not None and ensemble.n_models > 0,
        "n_models": ensemble.n_models if ensemble else 0,
        "threshold": DECISION_THRESHOLD,
    }


@app.get("/features")
def features() -> dict:
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "n_features": len(ensemble.columns),
        "n_categorical": len(ensemble.cat_features),
        "columns": ensemble.columns,
        "categorical": ensemble.cat_features,
    }


@app.post("/predict", response_model=PredictionResult)
def predict(application: dict[str, Any]) -> PredictionResult:
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        score = ensemble.predict([application])[0]
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
    return _result(float(score))


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest) -> BatchResponse:
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if not req.applications:
        raise HTTPException(status_code=400, detail="applications list is empty")
    try:
        scores = ensemble.predict(req.applications)
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
    results = [_result(float(s)) for s in scores]
    return BatchResponse(results=results, n=len(results))
