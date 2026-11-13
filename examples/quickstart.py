"""Variance components, a decision study and a comparison through the Python API.

Run from the repository root:

    python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

from repvar.betabinom import fit_scores
from repvar.bootstrap import bootstrap
from repvar.compare import compare, detectable_difference
from repvar.components import variance_components
from repvar.data import read_long
from repvar.design import best_split, budget_splits, mean_se, repeats_to_plateau
from repvar.icc import icc, reliability, solve_shares
from repvar.simulate import simulate_binary

HERE = Path(__file__).resolve().parent


def main() -> None:
    scores = read_long(HERE / "runs.csv")["all"]
    # The file names its runs 1 to 4, but they are plain repeats: analyse them as nested.
    comp = variance_components(scores, design="nested")
    print(f"items {comp.n_items}, runs per item {comp.runs_per_item:.0f}, mean {comp.mean:.4f}")
    print(f"item variance {comp.item:.4f}, rerun variance {comp.residual:.4f}")

    for runs in (1, 4, 16):
        d = mean_se(comp, comp.n_items, runs)
        print(f"  {runs:>2} run(s): SE {d.se:.4f} (items {d.se_item:.4f}, reruns {d.se_rerun:.4f})")

    corr = icc(comp)
    print(f"{corr.kind} {corr.value:.3f}; reliability of a 4-run mean {reliability(comp, 4):.3f}")
    shares = solve_shares(scores)
    print(f"solved always {shares.always:.1%}, sometimes {shares.sometimes:.1%}")
    fit = fit_scores(scores)
    print(f"beta-binomial: mean {fit.mean:.3f}, rho {fit.rho:.3f}, loglik {fit.loglik:.2f}")

    interval = bootstrap(scores, design="nested", n_boot=500, seed=1)["mean"]
    print(
        f"mean {interval.estimate:.3f}, 95 % bootstrap interval {interval.low:.3f} to "
        f"{interval.high:.3f}"
    )

    splits = budget_splits(comp, budget=64, max_items=8)
    best = best_split(splits)
    print(
        f"64 item-runs on at most 8 items: best is {best.n_items} x {best.n_runs} runs, "
        f"SE {best.design.se:.4f}"
    )
    print(f"repeats to come within 5 % of the floor: {repeats_to_plateau(comp, 8):.0f}")

    # Two simulated systems on the same 200 items, 3 runs each.
    a = simulate_binary(200, 3, mean=0.55, icc=0.5, seed=1)
    b = simulate_binary(200, 3, mean=0.45, icc=0.5, seed=2)
    result = compare(a, b, names=("a", "b"))
    print(
        f"difference {result.difference:.3f}, SE {result.se:.4f}, "
        f"detectable at 80 % power {detectable_difference(result.se):.3f}"
    )


if __name__ == "__main__":
    main()
