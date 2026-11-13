"""Markdown and JSON rendering of results."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .betabinom import BetaBinomialFit, fit_scores
from .bootstrap import Interval, bootstrap
from .compare import Comparison, detectable_difference
from .components import CROSSED, Components, NotIdentifiableError, variance_components
from .data import Scores
from .design import DesignSE, Split, best_split, mean_se, repeats_to_plateau
from .icc import ICC, SolveShares, icc, reliability, solve_shares


def fmt(value: float | None, digits: int = 4) -> str:
    """Fixed-point number, or ``n/a`` for NaN and ``inf`` for infinity."""
    if value is None or math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def pct(value: float | None, digits: int = 1) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{100.0 * value:.{digits}f} %"


def p_text(value: float) -> str:
    """``p 0.0123``, or ``p < 1e-300`` when the value underflows to zero."""
    if math.isnan(value):
        return "p n/a"
    return "p < 1e-300" if value < 1e-300 else f"p {value:.3g}"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def to_json(payload: object) -> str:
    """JSON text with NaN and infinity written as null (strict JSON has neither)."""

    def clean(obj: object) -> object:
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        if isinstance(obj, Mapping):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        return obj

    return json.dumps(clean(payload), indent=2)


def design_line(comp: Components) -> str:
    if comp.design == CROSSED:
        kind = "crossed (each run is one pass over the items)"
    else:
        kind = "nested (runs independent within items)"
    return f"design: {kind}; method: {comp.method}"


SE_HEADER = ["design", "standard error", "from items", "from reruns", "rerun share of variance"]


def se_row(d: DesignSE) -> list[str]:
    label = f"{d.n_items} items x {d.n_runs} run(s)"
    return [label, fmt(d.se), fmt(d.se_item), fmt(d.se_rerun), pct(d.rerun_share)]


@dataclass(frozen=True)
class Analysis:
    """Everything ``repvar components`` reports for one system."""

    components: Components
    icc: ICC
    shares: SolveShares
    betabinomial: BetaBinomialFit | None
    intervals: dict[str, Interval] | None
    binary: bool = False

    @property
    def runs(self) -> int:
        return max(1, round(self.components.runs_per_item))

    def to_dict(self) -> dict[str, object]:
        comp = self.components
        out: dict[str, object] = comp.to_dict()
        out["se_one_run"] = mean_se(comp, comp.n_items, 1).to_dict()
        out["se_observed"] = mean_se(comp, comp.n_items, self.runs).to_dict()
        out["binary"] = self.binary
        out["icc"] = self.icc.to_dict()
        out["reliability_one_run"] = reliability(comp, 1)
        out["reliability_observed_runs"] = reliability(comp, self.runs)
        out["solve_shares"] = self.shares.to_dict()
        out["betabinomial"] = self.betabinomial.to_dict() if self.betabinomial else None
        out["bootstrap"] = (
            {k: v.to_dict() for k, v in self.intervals.items()} if self.intervals else None
        )
        return out


def analyse(
    scores: Scores,
    design: str = "auto",
    *,
    level: float = 0.95,
    n_boot: int = 0,
    resample: str = "items",
    seed: int = 0,
) -> Analysis:
    """Variance components and the summaries built on them, for one system."""
    comp = variance_components(scores, design)
    fit = None
    if scores.is_binary:
        try:
            fit = fit_scores(scores)
        except NotIdentifiableError:
            fit = None
    intervals = None
    if n_boot:
        intervals = bootstrap(
            scores, design=comp.design, resample=resample, n_boot=n_boot, level=level, seed=seed
        )
    return Analysis(
        comp, icc(comp, level), solve_shares(scores), fit, intervals, binary=scores.is_binary
    )


def components_markdown(name: str, result: Analysis) -> str:
    """The variance components of one system and what follows from them."""
    comp = result.components
    shares = comp.shares()
    rows = [["item", fmt(comp.item, 5), pct(shares["item"])]]
    if comp.design == CROSSED:
        rows.append(["run", fmt(comp.run, 5), pct(shares["run"])])
    rows.append(["residual", fmt(comp.residual, 5), pct(shares["residual"])])
    k = result.runs
    one = mean_se(comp, comp.n_items, 1)
    obs = mean_se(comp, comp.n_items, k)
    shape = "balanced" if comp.balanced else "unbalanced"
    level = pct(result.icc.level, 0)
    lines = [
        f"## {name}",
        "",
        design_line(comp),
        f"items: {comp.n_items}; scores: {comp.n_obs}; runs per item: "
        f"{fmt(comp.runs_per_item, 2)} ({shape}); mean score: {fmt(comp.mean)}",
        "",
        table(["component", "variance", "share"], rows),
        "",
        table(SE_HEADER, [se_row(d) for d in ([one] if k == 1 else [one, obs])]),
        "",
        f"- intraclass correlation {result.icc.kind}: {fmt(result.icc.value)} "
        f"({level} interval {fmt(result.icc.low)} to {fmt(result.icc.high)}; "
        f"{result.icc.method})",
    ]
    if result.binary and not math.isnan(result.icc.low):
        lines.append(
            "  (this F interval assumes normal effects and can under-cover on 0/1 scores: "
            "prefer the --bootstrap interval)"
        )
    lines.append(
        f"- reliability of an item's score: {fmt(reliability(comp, 1))} from 1 run, "
        f"{fmt(reliability(comp, k))} from the mean of {k} runs"
    )
    sh = result.shares
    if sh.n_items and math.isnan(sh.always):
        lines.append(
            f"- items whose runs disagree: {pct(sh.sometimes)} of the {sh.n_items} items "
            "with 2+ runs"
        )
    elif sh.n_items:
        lines.append(
            f"- of the {sh.n_items} items with 2+ runs: always solved {pct(sh.always)}, "
            f"sometimes {pct(sh.sometimes)}, never {pct(sh.never)}"
        )
    if result.betabinomial is not None:
        lines += ["", betabinomial_markdown(result.betabinomial)]
    negative = [
        label for label, raw in (("item", comp.item_raw), ("run", comp.run_raw)) if raw < 0.0
    ]
    if negative:
        lines += ["", f"negative estimate set to zero: {', '.join(negative)}"]
    if result.intervals:
        rows = [
            [key, fmt(iv.estimate, 5), fmt(iv.low, 5), fmt(iv.high, 5)]
            for key, iv in result.intervals.items()
        ]
        lines += [
            "",
            f"bootstrap intervals ({level}, percentile):",
            "",
            table(["quantity", "estimate", "low", "high"], rows),
        ]
    return "\n".join(lines)


def betabinomial_markdown(fit: BetaBinomialFit) -> str:
    if fit.boundary == "all scores equal":
        return "beta-binomial fit: every score is the same, nothing to fit"
    expected = ", ".join(f"{k} runs {pct(fit.sometimes_share(k))}" for k in (2, 5, 10))
    lines = [
        f"beta-binomial fit: mean {fmt(fit.mean)}, rho {fmt(fit.rho)} "
        f"(alpha {fmt(fit.alpha, 3)}, beta {fmt(fit.beta, 3)})",
        f"- log-likelihood {fmt(fit.loglik, 2)} against {fmt(fit.loglik_binomial, 2)} for one "
        f"shared pass probability (likelihood ratio {fmt(fit.lr_stat, 2)}, {p_text(fit.p_value)})",
        f"- item variance {fmt(fit.item_variance, 5)}, rerun variance {fmt(fit.rerun_variance, 5)}",
        f"- expected share of items solved sometimes: {expected}",
    ]
    if fit.boundary:
        lines.append(f"- note: {fit.boundary}")
    return "\n".join(lines)


def design_markdown(
    name: str,
    comp: Components,
    splits: list[Split],
    *,
    budget: float,
    n_items: int,
    tolerance: float,
    fixed_items: bool,
) -> str:
    """Decision study: standard error for every split of a budget, and the plateau."""
    best = best_split(splits, fixed_items)
    rows = []
    for split in splits:
        d = split.design
        runs = f"{split.n_runs} *" if split is best else str(split.n_runs)
        rows.append(
            [runs, str(split.n_items), fmt(split.cost, 0), fmt(d.se), fmt(d.se_item)]
            + [fmt(d.se_rerun)]
        )
    plateau = repeats_to_plateau(comp, n_items, tolerance)
    floor = math.sqrt(comp.item / n_items)
    target = "rerun part" if fixed_items else "standard error"
    if math.isinf(plateau):
        plateau_line = (
            f"with {n_items} items there is no plateau: the item variance is zero, so the "
            "standard error keeps falling as 1 / sqrt(runs)"
        )
    else:
        plateau_line = (
            f"with {n_items} items the standard error cannot go below {fmt(floor)}; "
            f"{int(plateau)} run(s) per item bring it within {pct(tolerance, 0)} of that"
        )
    header = ["runs per item", "items", "cost", "standard error", "from items", "from reruns"]
    return "\n".join(
        [
            f"## {name}",
            "",
            design_line(comp),
            f"variance: item {fmt(comp.item, 5)}, "
            + (f"run {fmt(comp.run, 5)}, " if comp.design == CROSSED else "")
            + f"residual {fmt(comp.residual, 5)}",
            "",
            f"budget: {fmt(budget, 0)} item-runs",
            "",
            table(header, rows),
            "",
            f"best split (*, smallest {target}): {best.n_items} items x {best.n_runs} run(s), "
            f"standard error {fmt(best.design.se)}",
            plateau_line,
        ]
    )


def compare_markdown(
    cmp: Comparison,
    *,
    alpha: float,
    power: float,
    n_items: int,
    runs: Sequence[int] = (1, 2, 3, 5, 10),
) -> str:
    """Difference of two systems, its error split, and what a design could detect."""
    level = pct(cmp.level, 0)
    lines = [
        f"# Comparison: {cmp.system_a} minus {cmp.system_b}",
        "",
        f"shared items: {cmp.n_items}; runs per item: {fmt(cmp.runs_a, 2)} and "
        f"{fmt(cmp.runs_b, 2)}",
        f"mean scores: {fmt(cmp.mean_a)} and {fmt(cmp.mean_b)}",
        f"difference: {fmt(cmp.difference)} (standard error {fmt(cmp.se)}, {level} interval "
        f"{fmt(cmp.low)} to {fmt(cmp.high)}, z {fmt(cmp.z, 2)}, {p_text(cmp.p_value)})",
        "",
        table(
            ["standard error of the difference", "value"],
            [
                ["from items (true per-item differences vary)", fmt(_root(cmp.item_part))],
                ["from reruns (both systems)", fmt(_root(cmp.rerun_part))],
                ["total", fmt(cmp.se)],
                ["paired over item means only", fmt(cmp.se_items_only)],
                ["unpaired (ignores that items are shared)", fmt(cmp.se_unpaired)],
            ],
        ),
    ]
    if math.isnan(cmp.item_diff_variance):
        lines += ["", "no item has two runs, so the error cannot be split or planned"]
        return "\n".join(lines)
    rows = []
    for k in runs:
        plan = cmp.plan(n_items, k)
        detect = detectable_difference(plan.se, alpha, power)
        rows.append([str(k), fmt(plan.se), fmt(detect), pct(plan.rerun_share)])
    lines += [
        "",
        f"smallest detectable difference with {n_items} shared items "
        f"(two-sided alpha {alpha:g}, power {power:g}):",
        "",
        table(["runs per system", "standard error", "detectable difference", "rerun share"], rows),
    ]
    return "\n".join(lines)


def _root(value: float) -> float:
    return math.sqrt(value) if value >= 0.0 else float("nan")
