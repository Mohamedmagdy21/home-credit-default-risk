"""Centralized logging setup used across modules.

All module loggers are children of a single root project logger
(``housing_loans``) so handlers (stdout + optional file) only need to be
attached once on the root, and child loggers propagate to it.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT_LOGGER_NAME = "housing_loans"
_CONFIGURED = False


def _configure_root(level: int) -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if _CONFIGURED:
        return root

    root.setLevel(level)
    root.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    _CONFIGURED = True
    return root


def setup_logger(name: str = ROOT_LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    """Return a logger under the project root.

    Passing ``__name__`` from a module yields ``housing_loans.src.data.loader``
    etc., which propagates to the configured root.
    """
    _configure_root(level)
    if name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def add_file_handler(logger: logging.Logger, path: Path) -> None:
    """Attach a file handler to the project root logger."""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
