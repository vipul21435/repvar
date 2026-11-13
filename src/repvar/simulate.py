"""Simulated repeated-run scores with planted variance components."""

from __future__ import annotations

import numpy as np

from .bootstrap import ITEMS, bootstrap
from .components import nested_components
from .data import Scores
from .icc import icc, icc_value


def simulate_binary(
    n_items: int,
    n_runs: int,
    *,
    mean: float = 0.5,
    icc: float = 0.5,
    seed: int | np.random.Generator = 0,
) -> Scores:
    """0/1 scores from the beta-binomial model, runs independent within items.

    Each item gets a pass probability from the beta distribution with mean ``mean`` and
    ``alpha + beta = 1 / icc - 1``; item variance is ``icc * mean * (1 - mean)`` and rerun
    variance ``(1 - icc) * mean * (1 - mean)``. ``icc=0`` gives every item the same
    probability and ``icc=1`` makes every item always pass or always fail.
    """
    _check(n_items, n_runs)
    if not 0.0 < mean < 1.0:
        raise ValueError("mean must be strictly between 0 and 1")
    if not 0.0 <= icc <= 1.0:
        raise ValueError("icc must be between 0 and 1")
    rng = np.random.default_rng(seed)
    if icc == 0.0:
        p = np.full(n_items, mean)
    elif icc == 1.0:
        p = (rng.random(n_items) < mean).astype(float)
    else:
        size = 1.0 / icc - 1.0
        p = rng.beta(mean * size, (1.0 - mean) * size, size=n_items)
    table = (rng.random((n_items, n_runs)) < p[:, None]).astype(float)
    return Scores.from_matrix(table, labelled=False)


def simulate_normal(
    n_items: int,
    n_runs: int,
    *,
    mean: float = 0.0,
    item_sd: float = 1.0,
    run_sd: float = 0.0,
    residual_sd: float = 1.0,
    seed: int | np.random.Generator = 0,
) -> Scores:
    """Real scores ``mean + item + run + residual`` with normal effects.

    With ``run_sd > 0`` the runs are crossed with the items and keep their labels;
    otherwise the runs of each item are unlabelled.
    """
    _check(n_items, n_runs)
    if min(item_sd, run_sd, residual_sd) < 0:
        raise ValueError("standard deviations cannot be negative")
    rng = np.random.default_rng(seed)
    table = (
        mean
        + rng.normal(0.0, 1.0, size=(n_items, 1)) * item_sd
        + rng.normal(0.0, 1.0, size=(1, n_runs)) * run_sd
        + rng.normal(0.0, 1.0, size=(n_items, n_runs)) * residual_sd
    )
    return Scores.from_matrix(table, labelled=run_sd > 0)


def _check(n_items: int, n_runs: int) -> None:
    if n_items < 1 or n_runs < 1:
        raise ValueError("n_items and n_runs must be at least 1")


def interval_coverage(
    n_items: int,
    n_runs: int,
    *,
    mean: float = 0.5,
    icc_true: float = 0.5,
    reps: int = 200,
    n_boot: int = 200,
    level: float = 0.95,
    resample: str = ITEMS,
    seed: int = 0,
) -> dict[str, float]:
    """Share of simulated data sets whose intervals contain the planted value.

    Draws ``reps`` beta-binomial data sets and checks the bootstrap intervals for the
    mean, the item variance and the intraclass correlation, and the F interval for the
    intraclass correlation, against the values the data were generated with.
    """
    if reps < 1:
        raise ValueError("reps must be at least 1")
    truth = {"mean": mean, "item": icc_true * mean * (1.0 - mean), "icc": icc_true}

    def statistic(scores: Scores) -> dict[str, float]:
        comp = nested_components(scores)
        return {"mean": comp.mean, "item": comp.item, "icc": icc_value(comp)}

    rng = np.random.default_rng(seed)
    hits = {"bootstrap: mean": 0, "bootstrap: item variance": 0, "bootstrap: icc": 0, "F: icc": 0}
    for _ in range(reps):
        scores = simulate_binary(n_items, n_runs, mean=mean, icc=icc_true, seed=rng)
        boot = bootstrap(
            scores,
            design="nested",
            resample=resample,
            n_boot=n_boot,
            level=level,
            seed=rng,
            statistic=statistic,
        )
        for key, label in (("mean", "mean"), ("item", "item variance"), ("icc", "icc")):
            hits[f"bootstrap: {label}"] += boot[key].low <= truth[key] <= boot[key].high
        exact = icc(nested_components(scores), level)
        hits["F: icc"] += exact.low <= icc_true <= exact.high
    return {name: count / reps for name, count in hits.items()}
