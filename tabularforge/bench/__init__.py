"""TabArena-style benchmark harness."""
from .base import accuracy, rmse
from .imputation import run_imputation
from .classification import run_classification
from .regression import run_regression
from .generation import run_generation
from .speed import run_speed
from .tabarena import run_all, summarize

__all__ = [
    "accuracy", "rmse",
    "run_imputation", "run_classification", "run_regression",
    "run_generation", "run_speed", "run_all", "summarize",
]
