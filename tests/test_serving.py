"""Tests for the serving layer (FastAPI app + model ensemble).

Run with:  python -m pytest tests/test_serving.py -v

Trains a tiny CatBoost model, saves it + the feature schema, then exercises
the API endpoints via FastAPI's TestClient (no network needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import CatBoostParams, ID_COLUMN, TARGET
from src.features.builder import build_train_dataset, build_test_dataset
from src.features.schema import save_schema
from src.models.catboost_model import CatBoostModel


@pytest.fixture
def trained_app(tmp_path):
    """Train a tiny model, persist artifacts, and yield a TestClient."""
    rng = np.random.default_rng(42)
    n = 200

    df = pd.DataFrame(
        {
            ID_COLUMN: np.arange(n),
            "num_feat": rng.normal(size=n),
            "cat_feat": rng.choice(["a", "b", "c"], size=n),
            TARGET: rng.integers(0, 2, size=n),
        }
    )
    frames = {"train": df, "test": df.drop(columns=[TARGET]).head(10).copy()}
    X, y, cat_features = build_train_dataset(frames, target=TARGET, id_column=ID_COLUMN)

    params = CatBoostParams(iterations=30, verbose=0, early_stopping_rounds=10)
    model = CatBoostModel(params)
    model.fit(X.iloc[:150], y.iloc[:150], X.iloc[150:], y.iloc[150:], cat_features=cat_features)
    model.save(tmp_path / "model_fold1.cbm")

    save_schema(list(X.columns), cat_features, tmp_path / "feature_schema.json")

    import os
    os.environ["MODELS_DIR"] = str(tmp_path)
    os.environ["DECISION_THRESHOLD"] = "0.5"

    # Import after env is set so lifespan picks up the right dir
    from src.serving.app import app

    with TestClient(app) as client:
        yield client, X


def test_health(trained_app):
    client, _ = trained_app
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert body["n_models"] == 1


def test_features_endpoint(trained_app):
    client, _ = trained_app
    resp = client.get("/features")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_features"] == 2  # num_feat + cat_feat (id/target dropped)
    assert body["n_categorical"] == 1
    assert "cat_feat" in body["categorical"]


def test_predict(trained_app):
    client, X = trained_app
    row = X.iloc[0]
    payload = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in ("approve", "reject")
    assert body["threshold"] == 0.5


def test_predict_sparse_input(trained_app):
    """Only one feature provided; rest should default to NaN."""
    client, _ = trained_app
    resp = client.post("/predict", json={"num_feat": 1.5})
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["score"] <= 1.0


def test_predict_batch(trained_app):
    client, X = trained_app
    apps = [
        {k: (None if pd.isna(v) else v) for k, v in X.iloc[i].items()}
        for i in range(3)
    ]
    resp = client.post("/predict_batch", json={"applications": apps})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n"] == 3
    assert len(body["results"]) == 3
    for r in body["results"]:
        assert 0.0 <= r["score"] <= 1.0


def test_predict_batch_empty_rejected(trained_app):
    client, _ = trained_app
    resp = client.post("/predict_batch", json={"applications": []})
    assert resp.status_code == 400
