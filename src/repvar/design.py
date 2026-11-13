"""Standard error of a mean score for a design of n items with k runs each."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .components import Components


@dataclass(frozen=True)
class DesignSE:
    """Error variance of the mean over ``n_items`` items and ``n_runs`` runs of each.

    ``item_part`` is the variance that comes from which items were drawn and
    ``rerun_part`` the variance that a fresh set of runs on the same items would show;
    ``se`` is the square root of their sum. ``se_rerun`` is the standard error when the
    item set is taken as fixed.
    """

    n_items: int
    n_runs: int
    item_part: float
    rerun_part: float

    @property
    def variance(self) -> float:
        return self.item_part + self.rerun_part

    @property
    def se(self) -> float:
        return math.sqrt(self.variance)

    @property
    def se_item(self) -> float:
        return math.sqrt(self.item_part)

    @property
    def se_rerun(self) -> float:
        return math.sqrt(self.rerun_part)

    @property
    def rerun_share(self) -> float:
        """Fraction of the error variance that reruns account for."""
        return self.rerun_part / self.variance if self.variance > 0 else float("nan")

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = asdict(self)
        out.update(
            se=self.se, se_item=self.se_item, se_rerun=self.se_rerun, rerun_share=self.rerun_share
        )
        return out


def mean_se(components: Components, n_items: int, n_runs: int) -> DesignSE:
    """Error variance of a mean score over ``n_items`` items, each run ``n_runs`` times.

    ``Var = item / n + run / k + residual / (n k)``. The first term is the item part; the
    other two are the rerun part (the run term is zero in a nested design).
    """
    if n_items < 1 or n_runs < 1:
        raise ValueError("n_items and n_runs must be at least 1")
    item_part = components.item / n_items
    rerun_part = components.run / n_runs + components.residual / (n_items * n_runs)
    return DesignSE(n_items=n_items, n_runs=n_runs, item_part=item_part, rerun_part=rerun_part)


def observed_se(components: Components) -> DesignSE:
    """:func:`mean_se` for the design the components were estimated from.

    With unequal runs per item the number of runs is ``components.runs_per_item`` rounded
    to the nearest whole run.
    """
    k = max(1, round(components.runs_per_item))
    return mean_se(components, components.n_items, k)


@dataclass(frozen=True)
class Split:
    """One way to spend a budget: ``n_items`` items with ``n_runs`` runs each."""

    n_items: int
    n_runs: int
    cost: float
    design: DesignSE

    def to_dict(self) -> dict[str, object]:
        return {"n_items": self.n_items, "n_runs": self.n_runs, "cost": self.cost} | (
            self.design.to_dict()
        )


def budget_splits(
    components: Components,
    budget: float,
    *,
    max_items: int | None = None,
    max_runs: int | None = None,
    item_cost: float = 0.0,
) -> list[Split]:
    """Every split of ``budget`` item-runs into ``k`` runs of as many items as fit.

    A design with ``n`` items and ``k`` runs costs ``n * (k + item_cost)``; ``item_cost``
    is the one-off cost of adding an item (writing it, building its environment) in units
    of one run. ``max_items`` caps the number of items, for a benchmark of fixed size.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if item_cost < 0:
        raise ValueError("item_cost cannot be negative")
    top = max_runs if max_runs is not None else 64
    splits = []
    for k in range(1, top + 1):
        n = int(budget // (k + item_cost))
        if max_items is not None:
            n = min(n, max_items)
        if n < 1:
            break
        cost = n * (k + item_cost)
        splits.append(Split(n, k, cost, mean_se(components, n, k)))
    if not splits:
        raise ValueError("the budget does not pay for a single run of one item")
    return splits


def best_split(splits: list[Split], fixed_items: bool = False) -> Split:
    """The split with the smallest standard error (fewest runs on a tie).

    ``fixed_items=True`` minimises the rerun part alone, for conclusions about the items
    at hand rather than about the population they stand for.
    """

    def key(split: Split) -> tuple[float, int]:
        design = split.design
        return (design.rerun_part if fixed_items else design.variance, split.n_runs)

    return min(splits, key=key)


def repeats_to_plateau(components: Components, n_items: int, tolerance: float = 0.05) -> float:
    """Smallest number of runs whose standard error is within ``tolerance`` of the floor.

    With ``n_items`` items the standard error cannot go below ``sqrt(item / n_items)``
    however many runs are added. Returns the smallest ``k`` with
    ``SE(n, k) <= (1 + tolerance) * floor``; ``inf`` when the item variance is zero while
    reruns still vary (then every extra run helps at the same rate).
    """
    if n_items < 1:
        raise ValueError("n_items must be at least 1")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    floor = components.item / n_items
    rerun_one = components.run + components.residual / n_items
    if rerun_one <= 0.0:
        return 1.0
    if floor <= 0.0:
        return math.inf
    allowed = ((1.0 + tolerance) ** 2 - 1.0) * floor
    return float(max(1, math.ceil(rerun_one / allowed - 1e-12)))
