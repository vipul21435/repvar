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

    With unequal runs per item the rerun part uses the mean of ``1 / k_i`` through the
    effective number of runs stored in ``components.runs_per_item``, rounded to the
    nearest whole run.
    """
    k = max(1, round(components.runs_per_item))
    return mean_se(components, components.n_items, k)
