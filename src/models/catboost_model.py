"""CatBoost model wrapper.

Wraps :class:`CatBoostClassifier` so training/inference details (categorical
features, early stopping, seed) live in one place and the pipeline stays
agnostic of the underlying estimator.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CatBoostModel:
    """Thin wrapper around :class:`CatBoostClassifier`."""

    def __init__(self, params) -> None:
        self.params = params
        self.model: Optional[CatBoostClassifier] = None
        self.cat_features: list[str] = []

    def _build(self) -> CatBoostClassifier:
        return CatBoostClassifier(**self.params.to_dict())

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cat_features: Optional[list[str]] = None,
        early_stopping_rounds: Optional[int] = None,
    ) -> "CatBoostModel":
        self.cat_features = cat_features or []
        self.model = self._build()

        fit_kwargs: dict = {"cat_features": self.cat_features}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = (X_val, y_val)
            if early_stopping_rounds is not None:
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        return self.model.predict_proba(X)[:, 1]

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        preds = self.predict_proba(X)
        return float(roc_auc_score(y, preds))

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        importance = self.model.get_feature_importance()
        return pd.DataFrame(
            {"feature": self.model.feature_names_, "importance": importance}
        ).sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info("Saved model -> %s", path)

    def load(self, path: Path) -> "CatBoostModel":
        self.model = self._build()
        self.model.load_model(str(path))
        logger.info("Loaded model <- %s", path)
        return self
