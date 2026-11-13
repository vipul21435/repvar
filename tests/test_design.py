import math

import pytest

from repvar.components import Components, crossed_components, nested_components
from repvar.data import Scores
from repvar.design import (
    best_split,
    budget_splits,
    mean_se,
    observed_se,
    repeats_to_plateau,
)


def planted(item, run, residual, design="nested"):
    return Components(
        design=design,
        method="planted",
        n_items=100,
        n_runs=5,
        n_obs=500,
        balanced=True,
        runs_per_item=5.0,
        mean=0.5,
        item=item,
        run=run,
        residual=residual,
        item_raw=item,
        run_raw=run,
        residual_raw=residual,
    )


def test_mean_se_formula_and_parts():
    comp = planted(0.04, 0.01, 0.2, design="crossed")
    d = mean_se(comp, 50, 4)
    assert d.item_part == pytest.approx(0.04 / 50)
    assert d.rerun_part == pytest.approx(0.01 / 4 + 0.2 / 200)
    assert d.se == pytest.approx(math.sqrt(0.04 / 50 + 0.01 / 4 + 0.2 / 200))
    assert d.se_item**2 + d.se_rerun**2 == pytest.approx(d.se**2)
    assert d.rerun_share == pytest.approx(d.rerun_part / d.variance)
    assert d.to_dict()["se"] == pytest.approx(d.se)
    assert math.isnan(mean_se(planted(0.0, 0.0, 0.0), 10, 2).rerun_share)
    for n, k in ((0, 1), (1, 0)):
        with pytest.raises(ValueError):
            mean_se(comp, n, k)


def test_observed_se_uses_the_observed_design():
    scores = Scores.from_lists([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    d = observed_se(nested_components(scores))
    assert (d.n_items, d.n_runs) == (3, 3)
    assert d.variance == pytest.approx(27.0 / 9.0)


def test_nested_budget_without_a_cap_prefers_one_run_per_item():
    # item / n + residual / (n k) with n k = B is item * k / B + residual / B: rising in k.
    comp = planted(0.05, 0.0, 0.15)
    splits = budget_splits(comp, 1200, max_runs=6)
    assert [s.n_runs for s in splits] == [1, 2, 3, 4, 5, 6]
    assert [s.n_items for s in splits] == [1200, 600, 400, 300, 240, 200]
    assert best_split(splits).n_runs == 1
    ses = [s.design.se for s in splits]
    assert ses == sorted(ses)
    assert splits[2].design.variance == pytest.approx(0.05 / 400 + 0.15 / 1200)
    assert splits[2].to_dict()["cost"] == pytest.approx(1200)


def test_capped_items_push_the_budget_into_repeats():
    comp = planted(0.05, 0.0, 0.15)
    splits = budget_splits(comp, 1200, max_items=300, max_runs=8)
    best = best_split(splits)
    assert (best.n_items, best.n_runs) == (300, 4)
    assert splits[0].cost == pytest.approx(300)  # one run of 300 items leaves budget unused


def test_item_cost_gives_the_classical_optimum():
    # With cost n (k + c), the optimum is k = sqrt(c * residual / item) = sqrt(8 * .2 / .1) = 4.
    comp = planted(0.1, 0.0, 0.2)
    splits = budget_splits(comp, 12000, item_cost=8.0, max_runs=12)
    assert best_split(splits).n_runs == 4


def test_crossed_run_variance_makes_repeats_worthwhile():
    # item k / B + run / k + residual / B is smallest near k = sqrt(B run / item) = 6.
    comp = planted(0.05, 0.002, 0.15, design="crossed")
    splits = budget_splits(comp, 900, max_runs=20)
    assert best_split(splits).n_runs in (5, 6, 7)


def test_fixed_items_choose_by_the_rerun_part():
    comp = planted(0.05, 0.0, 0.15)
    splits = budget_splits(comp, 1000, max_items=100, max_runs=12)
    assert best_split(splits, fixed_items=True).n_runs == 10


def test_budget_errors():
    comp = planted(0.05, 0.0, 0.15)
    with pytest.raises(ValueError, match="positive"):
        budget_splits(comp, 0)
    with pytest.raises(ValueError, match="negative"):
        budget_splits(comp, 10, item_cost=-1)
    with pytest.raises(ValueError, match="single run"):
        budget_splits(comp, 0.5)
    assert len(budget_splits(comp, 3)) == 3


def test_repeats_to_plateau():
    comp = planted(0.05, 0.0, 0.15)
    k = repeats_to_plateau(comp, 200, tolerance=0.05)
    floor = math.sqrt(0.05 / 200)
    assert mean_se(comp, 200, int(k)).se <= 1.05 * floor
    assert mean_se(comp, 200, int(k) - 1).se > 1.05 * floor
    # residual / item / ((1.05)^2 - 1) = 3 / 0.1025 = 29.27
    assert k == 30
    assert repeats_to_plateau(comp, 200, tolerance=0.5) < k
    assert math.isinf(repeats_to_plateau(planted(0.0, 0.0, 0.15), 200))
    assert repeats_to_plateau(planted(0.05, 0.0, 0.0), 200) == 1
    crossed = planted(0.05, 0.01, 0.15, design="crossed")
    assert repeats_to_plateau(crossed, 200) > k
    with pytest.raises(ValueError):
        repeats_to_plateau(comp, 0)
    with pytest.raises(ValueError):
        repeats_to_plateau(comp, 10, tolerance=0)


def test_crossed_components_feed_the_design():
    scores = Scores.from_matrix([[1, 3], [2, 6], [6, 6]])
    d = mean_se(crossed_components(scores), 3, 2)
    assert d.variance == pytest.approx(3.0 / 3 + (4.0 / 3) / 2 + 2.0 / 6)
