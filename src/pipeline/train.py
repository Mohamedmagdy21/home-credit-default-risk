"""Training pipeline: stratified k-fold CV with out-of-fold predictions.

Produces:
  * OOF AUC score
  * Averaged test predictions
  * Trained per-fold model artifacts
  * A feature-importance report
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config.config import Config
from src.data.loader import load_all, check_id_uniqueness
from src.features.builder import build_train_dataset, build_test_dataset
from src.features.schema import save_schema
from src.models.catboost_model import CatBoostModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def run_training(config: Config) -> dict:
    """Execute the full CV training pipeline.

    Returns a dict with ``oof_score``, ``oof_preds``, ``test_preds``,
    ``feature_importance`` and ``ids``.
    """
    logger.info("=== Loading data ===")
    frames = load_all(config.data_dir, config.data_files)
    check_id_uniqueness(frames, config.id_column)

    logger.info("=== Building train dataset ===")
    X, y, cat_features = build_train_dataset(
        frames,
        target=config.target,
        id_column=config.id_column,
        fill_value="missing",
    )

    save_schema(list(X.columns), cat_features, config.models_dir / "feature_schema.json")

    logger.info("=== Building test dataset ===")
    X_test, test_ids = build_test_dataset(
        frames,
        id_column=config.id_column,
        train_columns=list(X.columns),
        fill_value="missing",
    )

    cv = config.cv_config
    kf = StratifiedKFold(
        n_splits=cv.n_splits,
        shuffle=cv.shuffle,
        random_state=cv.random_state,
    )

    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores: list[float] = []
    fold_importances: list[pd.DataFrame] = []

    logger.info("=== Cross-validation (%d folds) ===", cv.n_splits)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
        logger.info("--- Fold %d ---", fold)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostModel(config.catboost_params)
        model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            cat_features=cat_features,
            early_stopping_rounds=config.catboost_params.early_stopping_rounds,
        )

        oof[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(X_test) / cv.n_splits

        fold_score = roc_auc_score(y_val, oof[val_idx])
        fold_scores.append(fold_score)
        logger.info("Fold %d ROC-AUC: %.5f", fold, fold_score)

        model.save(config.models_dir / f"model_fold{fold}.cbm")
        fold_importances.append(model.feature_importance())

    oof_score = float(roc_auc_score(y, oof))
    logger.info("=== OOF ROC-AUC: %.5f ===", oof_score)
    logger.info("Mean fold AUC: %.5f (+/- %.5f)",
                np.mean(fold_scores), np.std(fold_scores))

    feature_importance = (
        pd.concat(fold_importances)
        .groupby("feature", as_index=False)["importance"].mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    report_path = config.reports_dir / "feature_importance.csv"
    feature_importance.to_csv(report_path, index=False)
    logger.info("Saved feature importance -> %s", report_path)

    return {
        "oof_score": oof_score,
        "fold_scores": fold_scores,
        "oof_preds": oof,
        "test_preds": test_preds,
        "test_ids": test_ids,
        "feature_importance": feature_importance,
    }
