"""repvar: how much of a benchmark score is run-to-run noise."""

from .betabinom import BetaBinomialFit, fit_betabinomial, fit_scores
from .bootstrap import Interval, bootstrap
from .compare import Comparison, compare, detectable_difference
from .components import Components, NotIdentifiableError, variance_components
from .data import Scores, read_long
from .design import DesignSE, best_split, budget_splits, mean_se, repeats_to_plateau
from .icc import ICC, icc, reliability, solve_shares
from .simulate import simulate_binary, simulate_normal

__version__ = "0.1.0"

__all__ = [
    "ICC",
    "BetaBinomialFit",
    "Comparison",
    "Components",
    "DesignSE",
    "Interval",
    "NotIdentifiableError",
    "Scores",
    "__version__",
    "best_split",
    "bootstrap",
    "budget_splits",
    "compare",
    "detectable_difference",
    "fit_betabinomial",
    "fit_scores",
    "icc",
    "mean_se",
    "read_long",
    "reliability",
    "repeats_to_plateau",
    "simulate_binary",
    "simulate_normal",
    "solve_shares",
    "variance_components",
]
