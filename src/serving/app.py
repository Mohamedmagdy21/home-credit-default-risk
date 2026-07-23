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
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.config import DEFAULT_CONFIG
from src.data.loader import load_all
from src.features.builder import build_train_dataset, build_test_dataset
from src.serving.ensemble import ModelEnsemble
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.08"))

ensemble: ModelEnsemble | None = None
_test_cache: pd.DataFrame | None = None


def _get_test_data() -> pd.DataFrame:
    """Lazily load and cache the full merged test dataset (all 197 features)."""
    global _test_cache
    if _test_cache is not None:
        return _test_cache
    logger.info("Loading test data for /sample endpoint (first call)...")
    frames = load_all(DEFAULT_CONFIG.data_dir, DEFAULT_CONFIG.data_files)
    X, _, _ = build_train_dataset(
        frames, target=DEFAULT_CONFIG.target, id_column=DEFAULT_CONFIG.id_column
    )
    X_test, _ = build_test_dataset(
        frames,
        id_column=DEFAULT_CONFIG.id_column,
        train_columns=list(X.columns),
    )
    _test_cache = X_test
    logger.info("Test data cached: %s", X_test.shape)
    return _test_cache


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
    decision: str = Field(..., description="'approve', 'review', or 'reject'")
    threshold: float


class BatchRequest(BaseModel):
    applications: list[dict[str, Any]]


class BatchResponse(BaseModel):
    results: list[PredictionResult]
    n: int


def _decide(score: float) -> str:
    if score >= DECISION_THRESHOLD:
        return "reject"
    if score >= DECISION_THRESHOLD * 0.6:
        return "review"
    return "approve"


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


def _row_to_dict(row: pd.Series) -> dict:
    """Convert a pandas Series to a JSON-serializable dict (native types)."""
    result = {}
    for k, v in row.items():
        if pd.isna(v):
            result[k] = None
        elif hasattr(v, "item"):
            result[k] = v.item()
        else:
            result[k] = v
    return result


@app.get("/sample")
def sample() -> dict:
    """Return a random complete application with all 197 features populated."""
    test_df = _get_test_data()
    idx = random.randint(0, len(test_df) - 1)
    row = test_df.iloc[idx]
    return {
        "index": idx,
        "total": len(test_df),
        "application": _row_to_dict(row),
    }


@app.get("/sample/{index}")
def sample_by_index(index: int) -> dict:
    """Return a specific application by its row index."""
    test_df = _get_test_data()
    if index < 0 or index >= len(test_df):
        raise HTTPException(status_code=404, detail=f"Index out of range (0-{len(test_df)-1})")
    row = test_df.iloc[index]
    return {
        "index": index,
        "total": len(test_df),
        "application": _row_to_dict(row),
    }
