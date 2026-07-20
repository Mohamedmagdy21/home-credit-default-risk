"""Inference pipeline: build a Kaggle-style submission CSV.

Reuses the averaged test predictions produced during CV training. If no
predictions are supplied, a stored ensemble of fold models can be loaded
and run instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import Config
from src.data.loader import load_all
from src.features.builder import build_train_dataset, build_test_dataset
from src.models.catboost_model import CatBoostModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def write_submission(
    test_ids: pd.Series,
    test_preds: np.ndarray,
    config: Config,
    filename: str = "submission.csv",
) -> Path:
    submission = pd.DataFrame(
        {
            config.id_column: test_ids.values,
            config.target: test_preds,
        }
    )
    path = config.submissions_dir / filename
    submission.to_csv(path, index=False)
    logger.info("Saved submission (%d rows) -> %s", len(submission), path)
    return path


def predict_from_saved_models(config: Config) -> tuple[pd.Series, np.ndarray]:
    """Reload fold models and average their test predictions."""
    logger.info("=== Loading data for inference ===")
    frames = load_all(config.data_dir, config.data_files)

    logger.info("=== Rebuilding train schema ===")
    X, _, cat_features = build_train_dataset(
        frames,
        target=config.target,
        id_column=config.id_column,
        fill_value="missing",
    )

    X_test, test_ids = build_test_dataset(
        frames,
        id_column=config.id_column,
        train_columns=list(X.columns),
        fill_value="missing",
    )

    model_files = sorted(config.models_dir.glob("model_fold*.cbm"))
    if not model_files:
        raise FileNotFoundError(
            f"No trained models found in {config.models_dir}. Train first."
        )
    logger.info("Found %d fold models", len(model_files))

    test_preds = np.zeros(len(X_test))
    for mf in model_files:
        model = CatBoostModel(config.catboost_params).load(mf)
        test_preds += model.predict_proba(X_test) / len(model_files)

    return test_ids, test_preds
