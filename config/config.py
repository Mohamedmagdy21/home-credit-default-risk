"""Project configuration: paths, data file mapping, and model hyperparameters.

All tunable knobs and filesystem locations live here so the rest of the
codebase stays declarative and easy to override.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path("/home/mohamed/Desktop/project")
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
REPORTS_DIR = OUTPUTS_DIR / "reports"

for _d in (OUTPUTS_DIR, MODELS_DIR, SUBMISSIONS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


DATA_FILES: dict[str, str] = {
    "train": "train(1).parquet",
    "test": "test(1).parquet",
    "bureau": "bureau_agg_curr.parquet",
    "previous_application": "previous_application(1).parquet",
    "credit_balance": "credit_balance.parquet",
    "installment_payment": "installments(1).parquet",
    "pos_cash": "pos_cash(1).parquet",
}

TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"

CATEGORICAL_FILL_VALUE = "missing"


@dataclass
class CatBoostParams:
    iterations: int = 1000
    learning_rate: float = 0.03
    depth: int = 6
    eval_metric: str = "AUC"
    random_seed: int = 42
    verbose: int = 200
    early_stopping_rounds: int = 100
    allow_writing_files: bool = False

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "eval_metric": self.eval_metric,
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "allow_writing_files": self.allow_writing_files,
        }


@dataclass
class CVConfig:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42


@dataclass
class Config:
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    submissions_dir: Path = SUBMISSIONS_DIR
    reports_dir: Path = REPORTS_DIR
    target: str = TARGET
    id_column: str = ID_COLUMN
    data_files: dict = field(default_factory=lambda: dict(DATA_FILES))
    catboost_params: CatBoostParams = field(default_factory=CatBoostParams)
    cv_config: CVConfig = field(default_factory=CVConfig)


DEFAULT_CONFIG = Config()
