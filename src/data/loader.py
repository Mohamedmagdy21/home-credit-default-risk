"""Data loading layer.

Reads the raw parquet files produced by the upstream feature-engineering
notebook and exposes them as a typed dictionary.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_parquet(name: str, path: Path) -> pd.DataFrame:
    logger.info("Loading %s from %s", name, path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path)
    logger.info("  %s shape=%s", name, df.shape)
    return df


def load_all(data_dir: Path, data_files: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Load every parquet file declared in ``data_files``.

    Parameters
    ----------
    data_dir:
        Directory containing the parquet files.
    data_files:
        Mapping of logical name -> filename.
    """
    frames: dict[str, pd.DataFrame] = {}
    for name, filename in data_files.items():
        frames[name] = load_parquet(name, data_dir / filename)
    return frames


def check_id_uniqueness(frames: dict[str, pd.DataFrame], id_column: str) -> None:
    """Warn if any auxiliary frame has duplicate ids (would corrupt a left merge)."""
    for name, df in frames.items():
        if id_column not in df.columns:
            logger.warning("%s has no %s column", name, id_column)
            continue
        dup = int(df[id_column].duplicated().sum())
        if dup:
            logger.warning("%s has %d duplicate %s values", name, dup, id_column)
        else:
            logger.info("%s: %s is unique (%d rows)", name, id_column, len(df))
