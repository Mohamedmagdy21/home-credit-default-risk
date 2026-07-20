"""Model ensemble loader for the serving layer.

Loads the saved fold models + feature schema once and exposes a predict
helper that aligns incoming rows to the training schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.features.schema import load_schema
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelEnsemble:
    """Holds the fold models and the training feature schema."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.models: list[CatBoostClassifier] = []
        self.columns: list[str] = []
        self.cat_features: list[str] = []
        self._loaded = False

    def load(self) -> "ModelEnsemble":
        schema_path = self.models_dir / "feature_schema.json"
        self.columns, self.cat_features = load_schema(schema_path)

        model_files = sorted(self.models_dir.glob("model_fold*.cbm"))
        if not model_files:
            raise FileNotFoundError(
                f"No trained models found in {self.models_dir}. Train first."
            )
        self.models = []
        for mf in model_files:
            m = CatBoostClassifier()
            m.load_model(str(mf))
            self.models.append(m)
        logger.info("Loaded ensemble of %d fold models", len(self.models))
        self._loaded = True
        return self

    @property
    def n_models(self) -> int:
        return len(self.models)

    def _align(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        missing = set(self.columns) - set(df.columns)
        if missing:
            logger.info("Filling %d missing features with NaN", len(missing))
            for col in missing:
                df[col] = np.nan
        df = df[self.columns]
        for col in self.cat_features:
            if col in df.columns:
                df[col] = df[col].fillna("missing").astype(str)
        return df

    def predict(self, rows: list[dict]) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Ensemble not loaded. Call load() first.")
        df = self._align(rows)
        preds = np.zeros(len(df))
        for m in self.models:
            preds += m.predict_proba(df)[:, 1] / len(self.models)
        return preds
