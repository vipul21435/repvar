import math

import numpy as np
import pytest

from repvar.bootstrap import bootstrap, point_estimates, resampler
from repvar.components import NotIdentifiableError, nested_components
from repvar.data import Scores
from repvar.design import mean_se
from repvar.simulate import simulate_binary, simulate_normal


def mean_only(scores):
    return {"mean": scores.mean}


def test_point_estimates_and_default_intervals():
    scores = simulate_binary(150, 4, mean=0.4, icc=0.5, seed=3)
    estimates = point_estimates(scores, "nested")
    comp = nested_components(scores)
    assert estimates["mean"] == pytest.approx(comp.mean)
    assert estimates["se_observed"] == pytest.approx(mean_se(comp, 150, 4).se)
    intervals = bootstrap(scores, n_boot=200, seed=1)
    assert set(intervals) == set(estimates)
    for name, interval in intervals.items():
        assert interval.estimate == pytest.approx(estimates[name])
        assert interval.low <= interval.high
    assert intervals["mean"].low < estimates["mean"] < intervals["mean"].high
    assert intervals["icc"].low < 0.5 < intervals["icc"].high
    assert intervals["mean"].to_dict()["low"] == intervals["mean"].low


def test_same_seed_same_interval():
    scores = simulate_binary(60, 3, seed=5)
    first = bootstrap(scores, n_boot=100, seed=9, statistic=mean_only)["mean"]
    again = bootstrap(scores, n_boot=100, seed=9, statistic=mean_only)["mean"]
    other = bootstrap(scores, n_boot=100, seed=10, statistic=mean_only)["mean"]
    assert (first.low, first.high) == (again.low, again.high)
    assert (first.low, first.high) != (other.low, other.high)


def test_item_bootstrap_interval_for_the_mean_covers_near_nominal():
    rng = np.random.default_rng(12)
    hits = 0
    reps = 250
    for _ in range(reps):
        scores = simulate_binary(60, 3, mean=0.35, icc=0.5, seed=rng)
        interval = bootstrap(scores, n_boot=200, level=0.9, seed=rng, statistic=mean_only)["mean"]
        hits += interval.low <= 0.35 <= interval.high
    # Nominal 90 %; the percentile bootstrap with 60 items runs slightly under. Accept
    # 84 % to 95 % (binomial sd over 250 draws is 1.9 %).
    assert 0.84 <= hits / reps <= 0.95


def test_resampling_spreads_match_the_error_parts():
    scores = simulate_normal(400, 4, item_sd=1.0, residual_sd=1.0, seed=2)
    comp = nested_components(scores)
    design = mean_se(comp, 400, 4)
    rng = np.random.default_rng(0)
    spread = {}
    for how in ("items", "runs", "items+runs"):
        draw = resampler(scores, "nested", how, rng)
        spread[how] = float(np.std([draw().mean for _ in range(1500)], ddof=1))
    # Items only reproduces the full standard error (item part plus rerun part).
    assert spread["items"] == pytest.approx(design.se, rel=0.08)
    # Runs only: resampling k runs with replacement gives (k - 1) / k of the rerun variance.
    assert spread["runs"] == pytest.approx(design.se_rerun * math.sqrt(3 / 4), rel=0.1)
    # Both levels count the rerun noise twice over, so the spread is a little wider.
    assert design.se < spread["items+runs"] < 1.25 * design.se


def test_replicates_keep_the_shape_of_the_data():
    scores = Scores.from_lists([[1, 0, 1], [0, 0], [1, 1, 1, 0], [0, 1]])
    rng = np.random.default_rng(1)
    kept = resampler(scores, "nested", "runs", rng)()
    assert kept.counts.tolist() == [3, 2, 4, 2]
    assert not kept.has_run_labels
    drawn = resampler(scores, "nested", "items", rng)()
    assert drawn.n_items == 4
    assert set(drawn.counts.tolist()) <= {2, 3, 4}
    sums = sorted(float(x.sum()) for x in drawn.per_item())
    assert set(sums) <= {0.0, 1.0, 2.0, 3.0}
    with pytest.raises(ValueError, match="unknown resample"):
        resampler(scores, "nested", "rows", rng)


def test_crossed_bootstrap_resamples_rows_and_columns():
    scores = simulate_normal(40, 6, item_sd=1.0, run_sd=0.8, residual_sd=0.5, seed=4)
    items = bootstrap(scores, n_boot=150, seed=2)
    both = bootstrap(scores, n_boot=150, seed=2, resample="items+runs")
    runs = bootstrap(scores, n_boot=150, seed=2, resample="runs")
    assert items["run"].estimate > 0.1
    # Redrawing whole runs moves the mean by the run effect; redrawing items does not.
    assert (both["mean"].high - both["mean"].low) > (items["mean"].high - items["mean"].low)
    assert runs["mean"].low < runs["mean"].estimate < runs["mean"].high


def test_failed_replicates_and_bad_arguments():
    # Only one item has two runs: about a third of the item draws miss it and are skipped.
    sparse = Scores.from_lists([[1, 0]] + [[1]] * 30)
    partial = bootstrap(sparse, n_boot=200, design="nested", seed=3)
    assert partial["mean"].low <= partial["mean"].estimate <= partial["mean"].high

    calls = {"n": 0}

    def fails_after_first(_):
        calls["n"] += 1
        if calls["n"] > 1:
            raise NotIdentifiableError("nothing to estimate")
        return {"x": 1.0}

    with pytest.raises(NotIdentifiableError, match="more than half"):
        bootstrap(sparse, n_boot=20, design="nested", statistic=fails_after_first)
    scores = simulate_binary(20, 2, seed=1)
    with pytest.raises(ValueError):
        bootstrap(scores, n_boot=1)
    with pytest.raises(ValueError):
        bootstrap(scores, level=0)

    def always_nan(_):
        return {"x": float("nan")}

    assert math.isnan(bootstrap(scores, n_boot=10, statistic=always_nan)["x"].low)
