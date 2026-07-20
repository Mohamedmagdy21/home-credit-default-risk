"""Command-line entrypoint for the Housing Loans Approval project.

Usage examples
--------------
Train + predict (default):
    python main.py train

Predict only, reusing saved fold models:
    python main.py predict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config.config import DEFAULT_CONFIG, Config
from src.pipeline.train import run_training
from src.pipeline.predict import write_submission, predict_from_saved_models
from src.utils.logger import setup_logger, add_file_handler

logger = setup_logger("housing_loans")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Housing Loans Approval pipeline")
    parser.add_argument(
        "command",
        choices=["train", "predict", "all"],
        default="all",
        nargs="?",
        help="train: CV train + write submission; predict: load saved models; all: both",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_CONFIG.data_dir,
        help="Directory containing the parquet files",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_CONFIG.cv_config.n_splits,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_CONFIG.catboost_params.iterations,
        help="CatBoost iterations",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    cfg = DEFAULT_CONFIG
    cfg.data_dir = args.data_dir
    cfg.cv_config.n_splits = args.n_splits
    cfg.catboost_params.iterations = args.iterations
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)

    add_file_handler(logger, config.reports_dir / "run.log")

    logger.info("Command: %s", args.command)
    logger.info("Data dir: %s", config.data_dir)
    logger.info("CV splits: %d | iterations: %d",
                config.cv_config.n_splits, config.catboost_params.iterations)

    if args.command in ("train", "all"):
        results = run_training(config)
        sub_path = write_submission(
            results["test_ids"],
            results["test_preds"],
            config,
            filename="submission.csv",
        )
        logger.info("OOF ROC-AUC: %.5f", results["oof_score"])
        logger.info("Submission: %s", sub_path)

    if args.command == "predict":
        test_ids, test_preds = predict_from_saved_models(config)
        sub_path = write_submission(
            test_ids, test_preds, config, filename="submission.csv"
        )
        logger.info("Submission: %s", sub_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
