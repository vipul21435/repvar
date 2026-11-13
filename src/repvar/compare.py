"""Difference between two systems scored on the same items with independent runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np

from .components import NotIdentifiableError, variance_components
from .data import Scores
from .design import DesignSE

NAN = float("nan")


@dataclass(frozen=True)
class Comparison:
    """Mean difference ``a - b`` over the shared items and where its error comes from.

    ``se`` is the standard error of the difference: the item part (the systems' true
    per-item differences vary from item to item) plus the rerun part (each system's item
    means are noisy). ``se_items_only`` is the paired standard error computed from the
    item means alone; in a nested design the two agree, in a crossed design ``se`` also
    carries each system's run-to-run variance. ``se_unpaired`` ignores that the items are
    shared. ``item_diff_variance``, ``rerun_a`` and ``rerun_b`` are the inputs for
    planning another design; they are NaN when no item has two runs.
    """

    system_a: str
    system_b: str
    n_items: int
    runs_a: float
    runs_b: float
    mean_a: float
    mean_b: float
    difference: float
    se: float
    item_part: float
    rerun_part: float
    se_items_only: float
    se_unpaired: float
    item_diff_variance: float
    residual_a: float
    residual_b: float
    run_a: float
    run_b: float
    low: float
    high: float
    z: float
    p_value: float
    level: float

    def plan(self, n_items: int, runs_a: int, runs_b: int | None = None) -> DesignSE:
        """Error variance of the difference for ``n_items`` shared items and given runs."""
        runs_b = runs_a if runs_b is None else runs_b
        if n_items < 1 or runs_a < 1 or runs_b < 1:
            raise ValueError("n_items and the numbers of runs must be at least 1")
        rerun = (self.residual_a / runs_a + self.residual_b / runs_b) / n_items
        rerun += self.run_a / runs_a + self.run_b / runs_b
        return DesignSE(n_items, runs_a, self.item_diff_variance / n_items, rerun)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detectable_difference(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true difference a two-sided z-test at ``alpha`` finds with ``power``."""
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must be between 0 and 1")
    normal = NormalDist()
    return (normal.inv_cdf(1.0 - alpha / 2.0) + normal.inv_cdf(power)) * se


def _restrict(scores: Scores, keep: dict[str, int]) -> Scores:
    """The observations of ``scores`` on the items in ``keep``, re-indexed in its order."""
    lookup = np.full(scores.n_items, -1, dtype=np.int64)
    for position, item in enumerate(scores.item_ids):
        if item in keep:
            lookup[position] = keep[item]
    index = lookup[scores.item_index]
    mask = index >= 0
    labels = scores.run_labels
    if scores.has_run_labels:
        labels = tuple(label for label, ok in zip(labels, mask.tolist(), strict=True) if ok)
    return Scores(
        item_ids=tuple(keep),
        item_index=index[mask],
        run_labels=labels,
        values=scores.values[mask],
        system=scores.system,
        has_run_labels=scores.has_run_labels,
    )


def compare(
    a: Scores,
    b: Scores,
    *,
    design: str = "auto",
    level: float = 0.95,
    names: tuple[str, str] | None = None,
) -> Comparison:
    """Compare two systems on the items both were run on.

    The runs of ``a`` are taken to be independent of the runs of ``b``. Under that
    assumption the covariance of the two systems' item means estimates the covariance of
    their true per-item scores without bias, so the variance of the per-item differences
    splits into a true-difference part and each system's rerun noise.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    in_b = set(b.item_ids)
    keep = {item: pos for pos, item in enumerate(i for i in a.item_ids if i in in_b)}
    n = len(keep)
    if n < 2:
        raise NotIdentifiableError("the two systems share fewer than two items")
    a, b = _restrict(a, keep), _restrict(b, keep)
    diffs = a.item_means - b.item_means
    difference = float(diffs.mean())
    var_items_only = float(diffs.var(ddof=1)) / n
    var_unpaired = (float(a.item_means.var(ddof=1)) + float(b.item_means.var(ddof=1))) / n

    parts = []
    for scores in (a, b):
        try:
            comp = variance_components(scores, design)
        except NotIdentifiableError:
            parts.append((NAN, NAN, NAN))
            continue
        inverse_runs = float(np.mean(1.0 / scores.counts))
        run_part = comp.run / comp.n_runs if comp.run > 0.0 else 0.0
        parts.append((comp.residual, comp.run, comp.residual * inverse_runs / n + run_part))
    (res_a, run_a, rerun_a), (res_b, run_b, rerun_b) = parts
    if math.isnan(rerun_a) or math.isnan(rerun_b):
        item_part = rerun_part = item_diff = NAN
        variance = var_items_only
    else:
        residual_noise = (
            res_a * float(np.mean(1.0 / a.counts)) + res_b * float(np.mean(1.0 / b.counts))
        ) / n
        item_part = max(var_items_only - residual_noise, 0.0)
        rerun_part = rerun_a + rerun_b
        item_diff = item_part * n
        variance = item_part + rerun_part
    se = math.sqrt(variance)
    normal = NormalDist()
    crit = normal.inv_cdf(1.0 - (1.0 - level) / 2.0)
    if se > 0.0:
        z = difference / se
        p_value = 2.0 * (1.0 - normal.cdf(abs(z)))
    else:
        z, p_value = NAN, NAN
    label_a, label_b = names or (a.system or "a", b.system or "b")
    return Comparison(
        system_a=label_a,
        system_b=label_b,
        n_items=n,
        runs_a=a.n_obs / n,
        runs_b=b.n_obs / n,
        mean_a=a.mean,
        mean_b=b.mean,
        difference=difference,
        se=se,
        item_part=item_part,
        rerun_part=rerun_part,
        se_items_only=math.sqrt(var_items_only),
        se_unpaired=math.sqrt(var_unpaired),
        item_diff_variance=item_diff,
        residual_a=res_a,
        residual_b=res_b,
        run_a=run_a,
        run_b=run_b,
        low=difference - crit * se,
        high=difference + crit * se,
        z=z,
        p_value=p_value,
        level=level,
    )
