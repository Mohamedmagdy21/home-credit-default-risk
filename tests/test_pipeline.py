"""Smoke tests for the housing-loans pipeline.

Run with:  python -m pytest tests/ -v
These tests use a tiny synthetic fixture so they don't depend on the full
~70MB parquet dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import Config, CatBoostParams, CVConfig, DATA_FILES, TARGET, ID_COLUMN
from src.features.builder import (
    build_test_dataset,
    build_train_dataset,
    merge_auxiliary,
    prepare_categoricals,
)
from src.models.catboost_model import CatBoostModel
from src.pipeline.predict import write_submission


@pytest.fixture
def synthetic_frames() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 60

    train = pd.DataFrame(
        {
            ID_COLUMN: np.arange(n),
            "num_feat": rng.normal(size=n),
            "cat_feat": rng.choice(["a", "b", "c"], size=n),
            TARGET: rng.integers(0, 2, size=n),
        }
    )
    test = pd.DataFrame(
        {
            ID_COLUMN: np.arange(n, n + 20),
            "num_feat": rng.normal(size=20),
            "cat_feat": rng.choice(["a", "b", "c"], size=20),
        }
    )
    bureau = pd.DataFrame(
        {
            ID_COLUMN: np.arange(n + 20),
            "bureau_loan_count": rng.integers(0, 5, size=n + 20),
        }
    )
    return {"train": train, "test": test, "bureau": bureau}


def test_merge_auxiliary_keeps_rows(synthetic_frames):
    merged = merge_auxiliary(synthetic_frames["train"], synthetic_frames, on=ID_COLUMN)
    assert len(merged) == len(synthetic_frames["train"])
    assert "bureau_loan_count" in merged.columns


def test_prepare_categoricals(synthetic_frames):
    df = synthetic_frames["train"].copy()
    df.loc[0, "cat_feat"] = None
    df, cats = prepare_categoricals(df)
    assert cats == ["cat_feat"]
    assert df["cat_feat"].iloc[0] == "missing"
    assert df["cat_feat"].dtype == object


def test_build_train_test_datasets(synthetic_frames):
    X, y, cats = build_train_dataset(
        synthetic_frames, target=TARGET, id_column=ID_COLUMN
    )
    assert len(X) == len(y) == 60
    assert TARGET not in X.columns and ID_COLUMN not in X.columns
    assert cats == ["cat_feat"]

    X_test, ids = build_test_dataset(
        synthetic_frames, id_column=ID_COLUMN, train_columns=list(X.columns)
    )
    assert len(X_test) == 20
    assert list(X_test.columns) == list(X.columns)
    assert len(ids) == 20


def test_catboost_model_fit_predict(tmp_path, synthetic_frames):
    X, y, cats = build_train_dataset(
        synthetic_frames, target=TARGET, id_column=ID_COLUMN
    )
    params = CatBoostParams(iterations=20, verbose=0, early_stopping_rounds=5)
    model = CatBoostModel(params)
    model.fit(X.iloc[:50], y.iloc[:50], X.iloc[50:], y.iloc[50:], cat_features=cats)
    preds = model.predict_proba(X.iloc[50:])
    assert preds.shape == (10,)
    assert preds.min() >= 0 and preds.max() <= 1

    fi = model.feature_importance()
    assert set(fi.columns) == {"feature", "importance"}

    path = tmp_path / "m.cbm"
    model.save(path)
    loaded = CatBoostModel(params).load(path)
    assert np.allclose(loaded.predict_proba(X.iloc[50:]), preds)


def test_write_submission(tmp_path, synthetic_frames):
    X, _, _ = build_train_dataset(
        synthetic_frames, target=TARGET, id_column=ID_COLUMN
    )
    X_test, ids = build_test_dataset(
        synthetic_frames, id_column=ID_COLUMN, train_columns=list(X.columns)
    )

    cfg = Config()
    cfg.submissions_dir = tmp_path
    cfg.reports_dir = tmp_path
    cfg.models_dir = tmp_path

    preds = np.linspace(0, 1, len(ids))
    path = write_submission(ids, preds, cfg, filename="sub.csv")
    df = pd.read_csv(path)
    assert list(df.columns) == [ID_COLUMN, TARGET]
    assert len(df) == len(ids)
    assert np.allclose(df[TARGET].values, preds)
