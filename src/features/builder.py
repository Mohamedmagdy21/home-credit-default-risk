"""Feature engineering layer.

Merges the auxiliary bureau/credit/installment/pos/previous tables onto the
main application table and prepares categorical columns for CatBoost.
"""
from __future__ import annotations

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

AUXILIARY_TABLES = (
    "bureau",
    "previous_application",
    "credit_balance",
    "installment_payment",
    "pos_cash",
)


def merge_auxiliary(
    base: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    on: str,
    how: str = "left",
) -> pd.DataFrame:
    """Left-merge all auxiliary tables onto ``base`` keeping one row per id."""
    merged = base.copy()
    for name in AUXILIARY_TABLES:
        if name not in frames:
            logger.warning("Auxiliary table '%s' missing from frames; skipping", name)
            continue
        before = merged.shape
        merged = merged.merge(frames[name], on=on, how=how)
        logger.info(
            "Merged %-22s -> %s (was %s)", name, merged.shape, before
        )
    return merged


def prepare_categoricals(
    df: pd.DataFrame,
    fill_value: str = "missing",
) -> tuple[pd.DataFrame, list[str]]:
    """Fill nulls in object columns and coerce to string for CatBoost.

    Returns the (possibly modified) dataframe and the list of categorical
    column names.
    """
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna(fill_value).astype(str)
    logger.info("Prepared %d categorical columns", len(cat_cols))
    return df, cat_cols


def build_train_dataset(
    frames: dict[str, pd.DataFrame],
    target: str,
    id_column: str,
    fill_value: str = "missing",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, cat_features) for training."""
    train = frames["train"]
    train_merged = merge_auxiliary(train, frames, on=id_column)

    y = train_merged[target].astype(int)
    X = train_merged.drop(columns=[target, id_column], errors="ignore")
    X, cat_features = prepare_categoricals(X, fill_value=fill_value)
    logger.info("Final training matrix: %s", X.shape)
    return X, y, cat_features


def build_test_dataset(
    frames: dict[str, pd.DataFrame],
    id_column: str,
    train_columns: list[str],
    fill_value: str = "missing",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X_test, ids) aligned to ``train_columns``."""
    test = frames["test"]
    test_merged = merge_auxiliary(test, frames, on=id_column)

    ids = test_merged[id_column].copy()
    X_test = test_merged.drop(columns=[id_column], errors="ignore")
    X_test, _ = prepare_categoricals(X_test, fill_value=fill_value)

    missing = set(train_columns) - set(X_test.columns)
    if missing:
        import numpy as np

        logger.warning("Adding %d missing columns to test set", len(missing))
        for col in missing:
            X_test[col] = np.nan
    X_test = X_test[train_columns]
    logger.info("Final test matrix: %s", X_test.shape)
    return X_test, ids
