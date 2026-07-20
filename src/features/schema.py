"""Feature schema persistence.

Saves the exact column order and categorical-feature names from training so
the serving layer can align incoming requests to the model without rebuilding
the full feature matrix.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def save_schema(
    columns: list[str],
    cat_features: list[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"columns": columns, "cat_features": cat_features}
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved feature schema (%d cols, %d cat) -> %s",
                len(columns), len(cat_features), path)


def load_schema(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text())
    columns = payload["columns"]
    cat_features = payload["cat_features"]
    logger.info("Loaded feature schema (%d cols, %d cat) <- %s",
                len(columns), len(cat_features), path)
    return columns, cat_features
