"""Variance components of an items-by-runs design, by the analysis of variance.

Two designs are covered.

``nested``
    The runs of one item have nothing to do with the runs of another (each item was simply
    rerun some number of times). Model ``x_ij = mu + a_i + e_ij`` with components
    ``item`` and ``residual``. One-way random-effects ANOVA; with unequal numbers of runs
    the item mean square is divided by ``k0 = (N - sum(k_i^2) / N) / (n - 1)``.

``crossed``
    Run ``j`` is the same pass over every item (same seed, same checkpoint, same day), so
    a run can be good or bad as a whole. Model ``x_ij = mu + a_i + b_j + e_ij`` with
    components ``item``, ``run`` and ``residual``. The estimator equates the uncorrected
    sums of squares of the item totals, run totals and single scores to their
    expectations and solves the three linear equations. On a complete table this is the
    textbook expected-mean-squares solution; with missing cells it is the same method
    applied to the cell counts that are there.

Negative estimates are reported as they come in the ``*_raw`` fields and set to zero in
the main fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .data import Scores

NESTED = "nested"
CROSSED = "crossed"


class NotIdentifiableError(ValueError):
    """The data cannot separate the components that were asked for."""


@dataclass(frozen=True)
class Components:
    """Estimated variance components of one system's scores."""

    design: str
    method: str
    n_items: int
    n_runs: int
    n_obs: int
    balanced: bool
    runs_per_item: float
    mean: float
    item: float
    run: float
    residual: float
    item_raw: float
    run_raw: float
    residual_raw: float
    ms_item: float | None = None
    ms_run: float | None = None
    ms_residual: float | None = None
    df_item: float | None = None
    df_run: float | None = None
    df_residual: float | None = None

    @property
    def total(self) -> float:
        """Variance of a single score of a random item in a random run."""
        return self.item + self.run + self.residual

    @property
    def rerun(self) -> float:
        """Variance that rerunning changes: run plus residual."""
        return self.run + self.residual

    def shares(self) -> dict[str, float]:
        """Each component as a fraction of the total (NaN when the total is zero)."""
        total = self.total
        parts = {"item": self.item, "run": self.run, "residual": self.residual}
        if total <= 0.0:
            return {name: float("nan") for name in parts}
        return {name: value / total for name, value in parts.items()}

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["total"] = self.total
        out["shares"] = self.shares()
        return out


def nested_components(scores: Scores) -> Components:
    """One-way random-effects ANOVA: items as groups, runs as replicates."""
    counts = scores.counts.astype(float)
    n_items = scores.n_items
    n_obs = float(scores.n_obs)
    if n_items < 2:
        raise NotIdentifiableError("need at least two items to estimate the item variance")
    if scores.n_obs <= n_items:
        raise NotIdentifiableError(
            "every item has a single run: item variance and rerun variance cannot be "
            "separated without at least one item that was run twice"
        )
    means = scores.item_means
    grand = float(scores.values.sum() / n_obs)
    ss_item = float(np.sum(counts * (means - grand) ** 2))
    ss_res = float(np.sum((scores.values - means[scores.item_index]) ** 2))
    df_item = n_items - 1.0
    df_res = n_obs - n_items
    ms_item = ss_item / df_item
    ms_res = ss_res / df_res
    k0 = (n_obs - float(np.sum(counts**2)) / n_obs) / df_item
    item_raw = (ms_item - ms_res) / k0
    balanced = scores.balanced
    return Components(
        design=NESTED,
        method="one-way ANOVA" if balanced else "one-way ANOVA, unequal runs (k0)",
        n_items=n_items,
        n_runs=int(counts.max()),
        n_obs=scores.n_obs,
        balanced=balanced,
        runs_per_item=k0,
        mean=scores.mean,
        item=max(item_raw, 0.0),
        run=0.0,
        residual=ms_res,
        item_raw=item_raw,
        run_raw=0.0,
        residual_raw=ms_res,
        ms_item=ms_item,
        ms_residual=ms_res,
        df_item=df_item,
        df_residual=df_res,
    )


def crossed_components(scores: Scores) -> Components:
    """Two-way random-effects ANOVA without replication: items by runs, cells may be missing."""
    table, run_ids = scores.matrix()
    present = ~np.isnan(table)
    filled = np.where(present, table, 0.0)
    n_items, n_runs = table.shape
    n_obs = float(present.sum())
    if n_items < 2 or n_runs < 2:
        raise NotIdentifiableError("a crossed design needs at least two items and two runs")
    per_item = present.sum(axis=1).astype(float)
    per_run = present.sum(axis=0).astype(float)
    df_res = n_obs - n_items - n_runs + 1.0
    if df_res <= 0:
        raise NotIdentifiableError("too few scores for an items-by-runs decomposition")

    # Uncorrected sums of squares and the coefficients of their expectations.
    t_mean = filled.sum() ** 2 / n_obs
    t_item = float(np.sum(filled.sum(axis=1) ** 2 / per_item))
    t_run = float(np.sum(filled.sum(axis=0) ** 2 / per_run))
    t_all = float(np.sum(filled**2))
    k_item = float(np.sum(per_item**2)) / n_obs
    k_run = float(np.sum(per_run**2)) / n_obs
    k_item_run = float(np.sum(present / per_item[:, None]))
    k_run_item = float(np.sum(present / per_run[None, :]))
    lhs = np.array(
        [
            [n_obs - k_item, k_item_run - k_run, n_items - 1.0],
            [k_run_item - k_item, n_obs - k_run, n_runs - 1.0],
            [k_item - k_run_item, k_run - k_item_run, df_res],
        ]
    )
    rhs = np.array([t_item - t_mean, t_run - t_mean, t_all - t_item - t_run + t_mean])
    try:
        item_raw, run_raw, res_raw = (float(v) for v in np.linalg.solve(lhs, rhs))
    except np.linalg.LinAlgError as exc:
        raise NotIdentifiableError("the items-by-runs table is too sparse to solve") from exc
    balanced = bool(present.all())
    extra: dict[str, float] = {}
    if balanced:
        extra = {
            "ms_item": float(rhs[0] / (n_items - 1.0)),
            "ms_run": float(rhs[1] / (n_runs - 1.0)),
            "ms_residual": float(rhs[2] / df_res),
            "df_item": n_items - 1.0,
            "df_run": n_runs - 1.0,
            "df_residual": df_res,
        }
    return Components(
        design=CROSSED,
        method="two-way ANOVA" if balanced else "two-way ANOVA on the observed cells",
        n_items=n_items,
        n_runs=len(run_ids),
        n_obs=int(n_obs),
        balanced=balanced,
        runs_per_item=n_obs / n_items,
        mean=scores.mean,
        item=max(item_raw, 0.0),
        run=max(run_raw, 0.0),
        residual=max(res_raw, 0.0),
        item_raw=item_raw,
        run_raw=run_raw,
        residual_raw=res_raw,
        **extra,
    )


def choose_design(scores: Scores) -> str:
    """``crossed`` when the file names its runs and every run label covers two or more items."""
    if not scores.has_run_labels:
        return NESTED
    labels, counts = np.unique(np.asarray(scores.run_labels), return_counts=True)
    if len(labels) < 2 or np.any(counts < 2):
        return NESTED
    pairs = set(zip(scores.item_index.tolist(), scores.run_labels, strict=True))
    if len(pairs) < scores.n_obs:
        return NESTED
    return CROSSED


def variance_components(scores: Scores, design: str = "auto") -> Components:
    """Estimate the variance components of ``scores`` under the chosen design.

    ``design`` is ``"nested"``, ``"crossed"`` or ``"auto"`` (see :func:`choose_design`).
    """
    if design == "auto":
        design = choose_design(scores)
    if design == NESTED:
        return nested_components(scores)
    if design == CROSSED:
        return crossed_components(scores)
    raise ValueError(f"unknown design {design!r}: use 'nested', 'crossed' or 'auto'")
