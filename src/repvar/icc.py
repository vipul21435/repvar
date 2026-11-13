"""Intraclass correlation with an interval, and the reliability of a k-run score."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import f as f_dist

from .components import CROSSED, Components
from .data import Scores

NAN = float("nan")


@dataclass(frozen=True)
class ICC:
    """Intraclass correlation of single scores and its confidence interval.

    ``kind`` is ``ICC(1,1)`` for a nested design (one-way table) and ``ICC(2,1)`` for a
    crossed design (two-way table, absolute agreement), in the numbering of Shrout and
    Fleiss. ``value`` is not clamped and can be negative.
    """

    kind: str
    value: float
    low: float
    high: float
    level: float
    method: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def icc_value(comp: Components) -> float:
    """Point estimate: item variance over total variance, from the unclamped estimates."""
    total = comp.item_raw + comp.run_raw + comp.residual_raw
    if total <= 0.0:
        return NAN
    return comp.item_raw / total


def icc(comp: Components, level: float = 0.95) -> ICC:
    """Intraclass correlation with an F-distribution interval.

    Nested designs use the exact interval of the one-way random-effects table; with
    unequal runs per item ``k0`` takes the place of ``k``, which is an approximation.
    Complete crossed designs use the Satterthwaite-type interval for absolute agreement.
    Crossed designs with missing cells get the point estimate only. The intervals assume
    normal effects, so on 0/1 scores they are approximate; a bootstrap over items is the
    alternative.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    tail = 1.0 - (1.0 - level) / 2.0
    value = icc_value(comp)
    if comp.design == CROSSED:
        kind = "ICC(2,1)"
        if not comp.balanced or comp.ms_item is None:
            return ICC(kind, value, NAN, NAN, level, "no interval: crossed design with gaps")
        n, k = comp.n_items, comp.n_runs
        bms, jms, ems = comp.ms_item, comp.ms_run, comp.ms_residual
        if ems <= 0.0 or math.isnan(value):
            return ICC(kind, value, NAN, NAN, level, "no interval: zero residual mean square")
        fj = jms / ems
        a = k * value * fj + n * (1.0 + (k - 1.0) * value) - k * value
        b = (n - 1.0) * k**2 * value**2 * fj**2 + (n * (1.0 + (k - 1.0) * value) - k * value) ** 2
        v = (k - 1.0) * (n - 1.0) * a**2 / b
        f_low = f_dist.ppf(tail, n - 1.0, v)
        f_high = f_dist.ppf(tail, v, n - 1.0)
        spread = k * jms + (k * n - k - n) * ems
        low = n * (bms - f_low * ems) / (f_low * spread + n * bms)
        high = n * (f_high * bms - ems) / (spread + n * f_high * bms)
        return ICC(kind, value, float(low), float(high), level, "F, Satterthwaite df")

    kind = "ICC(1,1)"
    k = comp.runs_per_item
    msb, msw = comp.ms_item, comp.ms_residual
    if msw <= 0.0 or math.isnan(value):
        return ICC(kind, value, NAN, NAN, level, "no interval: zero within-item mean square")
    f0 = msb / msw
    f_low = f0 / f_dist.ppf(tail, comp.df_item, comp.df_residual)
    f_high = f0 * f_dist.ppf(tail, comp.df_residual, comp.df_item)
    low = (f_low - 1.0) / (f_low + k - 1.0)
    high = (f_high - 1.0) / (f_high + k - 1.0)
    method = "F, exact under normality" if comp.balanced else "F with k0 (approximate)"
    return ICC(kind, value, float(low), float(high), level, method)


def reliability(comp: Components, n_runs: int = 1, absolute: bool = True) -> float:
    """Generalizability coefficient of an item's mean over ``n_runs`` runs.

    Items are the objects of measurement: this is the expected squared correlation between
    an item's ``n_runs``-run mean and its long-run mean. ``absolute=True`` counts the run
    effect as error (dependability, phi); ``absolute=False`` leaves it out (relative
    coefficient), which only differs in a crossed design.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    error = comp.residual / n_runs + (comp.run / n_runs if absolute else 0.0)
    total = comp.item + error
    return comp.item / total if total > 0.0 else NAN


@dataclass(frozen=True)
class SolveShares:
    """Shares of the items with two or more runs, by how their runs came out.

    ``sometimes`` counts items whose runs did not all give the same score. ``always`` and
    ``never`` (every run 1, every run 0) are defined for 0/1 scores only.
    """

    n_items: int
    always: float
    never: float
    sometimes: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def solve_shares(scores: Scores) -> SolveShares:
    """Share of repeated items solved always, never and sometimes."""
    counts = scores.counts
    multi = counts >= 2
    n = int(multi.sum())
    if n == 0:
        return SolveShares(0, NAN, NAN, NAN)
    low = np.full(scores.n_items, np.inf)
    high = np.full(scores.n_items, -np.inf)
    np.minimum.at(low, scores.item_index, scores.values)
    np.maximum.at(high, scores.item_index, scores.values)
    sometimes = float(np.mean(low[multi] != high[multi]))
    if not scores.is_binary:
        return SolveShares(n, NAN, NAN, sometimes)
    always = float(np.mean(low[multi] == 1.0))
    never = float(np.mean(high[multi] == 0.0))
    return SolveShares(n, always, never, sometimes)
