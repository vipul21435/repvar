"""Percentile bootstrap intervals over items and, optionally, over runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

import numpy as np

from .components import CROSSED, NotIdentifiableError, choose_design, variance_components
from .data import Scores
from .design import mean_se
from .icc import icc_value, reliability, solve_shares

ITEMS = "items"
ITEMS_RUNS = "items+runs"
RUNS = "runs"


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def point_estimates(scores: Scores, design: str = "auto") -> dict[str, float]:
    """The quantities that :func:`bootstrap` puts intervals on by default."""
    comp = variance_components(scores, design)
    k = max(1, round(comp.runs_per_item))
    return {
        "mean": comp.mean,
        "item": comp.item,
        "run": comp.run,
        "residual": comp.residual,
        "icc": icc_value(comp),
        "reliability_one_run": reliability(comp, 1),
        "se_one_run": mean_se(comp, comp.n_items, 1).se,
        "se_observed": mean_se(comp, comp.n_items, k).se,
        "sometimes": solve_shares(scores).sometimes,
    }


def resampler(
    scores: Scores, design: str, resample: str, rng: np.random.Generator
) -> Callable[[], Scores]:
    """A function that returns one bootstrap replicate of ``scores`` per call.

    ``resample`` is ``"items"`` (draw items with replacement, keep each item's runs),
    ``"items+runs"`` (also redraw the runs) or ``"runs"`` (keep the items, redraw the
    runs). In a nested design runs are redrawn within each item; in a crossed design
    whole runs (columns) are redrawn.
    """
    if resample not in (ITEMS, ITEMS_RUNS, RUNS):
        raise ValueError(f"unknown resample {resample!r}: use 'items', 'items+runs' or 'runs'")
    n = scores.n_items
    if design == CROSSED:
        table, _ = scores.matrix()
        n_runs = table.shape[1]

        def draw_crossed() -> Scores:
            out = table if resample == RUNS else table[rng.integers(0, n, size=n)]
            if resample != ITEMS:
                out = out[:, rng.integers(0, n_runs, size=n_runs)]
            return Scores.from_matrix(out)

        return draw_crossed

    order = np.argsort(scores.item_index, kind="stable")
    values = scores.values[order]
    counts = scores.counts
    starts = np.cumsum(counts) - counts
    ids = tuple(str(i) for i in range(n))
    everything = np.arange(n)

    def draw_nested() -> Scores:
        idx = everything if resample == RUNS else rng.integers(0, n, size=n)
        c = counts[idx]
        total = int(c.sum())
        if resample == ITEMS:
            within = np.arange(total) - np.repeat(np.cumsum(c) - c, c)
        else:
            within = (rng.random(total) * np.repeat(c, c)).astype(np.int64)
        return Scores(
            item_ids=ids,
            item_index=np.repeat(everything, c),
            run_labels=(),
            values=values[np.repeat(starts[idx], c) + within],
            has_run_labels=False,
        )

    return draw_nested


def bootstrap(
    scores: Scores,
    *,
    design: str = "auto",
    resample: str = ITEMS,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int | np.random.Generator = 0,
    statistic: Callable[[Scores], Mapping[str, float]] | None = None,
) -> dict[str, Interval]:
    """Percentile intervals for ``statistic`` (default :func:`point_estimates`).

    Replicates on which the statistic cannot be computed (for example a draw in which no
    item has two runs) are left out; if more than half fail the call raises.
    """
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    if design == "auto":
        design = choose_design(scores)
    if statistic is None:

        def statistic(replicate: Scores) -> Mapping[str, float]:
            return point_estimates(replicate, design)

    estimate = dict(statistic(scores))
    draw = resampler(scores, design, resample, np.random.default_rng(seed))
    rows = []
    for _ in range(n_boot):
        try:
            rows.append([float(v) for v in statistic(draw()).values()])
        except NotIdentifiableError:
            continue
    if len(rows) < n_boot / 2:
        raise NotIdentifiableError("the statistic failed on more than half of the replicates")
    draws = np.asarray(rows, dtype=float)
    tail = 100.0 * (1.0 - level) / 2.0
    out = {}
    for column, (name, value) in enumerate(estimate.items()):
        col = draws[:, column]
        col = col[~np.isnan(col)]
        if col.size == 0:
            out[name] = Interval(float(value), float("nan"), float("nan"))
        else:
            low, high = np.percentile(col, [tail, 100.0 - tail])
            out[name] = Interval(float(value), float(low), float(high))
    return out
