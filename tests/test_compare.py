import math

import numpy as np
import pytest

from repvar.compare import compare, detectable_difference
from repvar.components import NotIdentifiableError
from repvar.data import Scores


def two_systems(rng, n=200, ka=4, kb=3, shift=0.3, diff_sd=0.5, sd_a=1.0, sd_b=0.8):
    """Shared item effects, a per-item true difference, independent runs."""
    base = rng.normal(0.0, 1.0, size=(n, 1))
    delta = rng.normal(shift, diff_sd, size=(n, 1))
    a = base + delta + rng.normal(0.0, sd_a, size=(n, ka))
    b = base + rng.normal(0.0, sd_b, size=(n, kb))
    return Scores.from_matrix(a, labelled=False), Scores.from_matrix(b, labelled=False)


def test_difference_and_paired_standard_error():
    rng = np.random.default_rng(1)
    a, b = two_systems(rng)
    result = compare(a, b, names=("new", "old"))
    diffs = a.item_means - b.item_means
    assert (result.system_a, result.system_b) == ("new", "old")
    assert result.n_items == 200
    assert result.runs_a == pytest.approx(4.0) and result.runs_b == pytest.approx(3.0)
    assert result.difference == pytest.approx(float(diffs.mean()))
    assert result.difference == pytest.approx(result.mean_a - result.mean_b)
    assert result.se_items_only == pytest.approx(float(diffs.std(ddof=1)) / math.sqrt(200))
    # Nested runs: the split adds back up to the paired standard error.
    assert result.se == pytest.approx(result.se_items_only)
    assert result.item_part + result.rerun_part == pytest.approx(result.se**2)
    assert result.se < result.se_unpaired  # shared items remove the common item effect
    assert result.low < result.difference < result.high
    assert result.high - result.low == pytest.approx(2 * 1.959964 * result.se, rel=1e-5)
    assert result.z == pytest.approx(result.difference / result.se)
    assert 0 <= result.p_value < 0.05
    assert result.to_dict()["n_items"] == 200


def test_split_recovers_planted_parts():
    rng = np.random.default_rng(2)
    a, b = two_systems(rng, n=6000, ka=4, kb=3, diff_sd=0.5, sd_a=1.0, sd_b=0.8)
    result = compare(a, b)
    assert result.item_diff_variance == pytest.approx(0.25, rel=0.12)
    assert result.residual_a == pytest.approx(1.0, rel=0.05)
    assert result.residual_b == pytest.approx(0.64, rel=0.05)
    assert result.rerun_part == pytest.approx((1.0 / 4 + 0.64 / 3) / 6000, rel=0.05)
    plan = result.plan(500, 2, 5)
    expected = (
        result.item_diff_variance / 500 + (result.residual_a / 2 + result.residual_b / 5) / 500
    )
    assert plan.variance == pytest.approx(expected)
    assert result.plan(500, 2).rerun_part == pytest.approx(
        (result.residual_a + result.residual_b) / 2 / 500
    )
    with pytest.raises(ValueError):
        result.plan(0, 1)


def test_standard_error_matches_the_spread_over_simulations():
    rng = np.random.default_rng(3)
    differences, errors = [], []
    for _ in range(400):
        a, b = two_systems(rng, n=60, ka=3, kb=2)
        result = compare(a, b)
        differences.append(result.difference)
        errors.append(result.se)
    assert np.mean(errors) == pytest.approx(np.std(differences, ddof=1), rel=0.1)


def test_identical_true_scores_leave_only_rerun_noise():
    rng = np.random.default_rng(4)
    a, b = two_systems(rng, n=3000, shift=0.0, diff_sd=0.0)
    result = compare(a, b)
    assert result.item_part < 0.1 * result.rerun_part
    assert abs(result.z) < 3.5


def test_only_shared_items_are_compared():
    a = Scores.from_lists({"x": [1, 1], "y": [0, 1], "z": [1, 0], "only-a": [1, 1]}, system="A")
    b = Scores.from_lists({"z": [0, 0], "y": [0, 0], "x": [1, 0], "only-b": [1, 1]}, system="B")
    result = compare(a, b)
    assert result.n_items == 3
    assert (result.system_a, result.system_b) == ("A", "B")
    assert result.mean_a == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert result.mean_b == pytest.approx((0.5 + 0.0 + 0.0) / 3)
    assert result.difference == pytest.approx(0.5)
    with pytest.raises(NotIdentifiableError, match="fewer than two"):
        compare(a, Scores.from_lists({"x": [1, 0], "other": [1, 1]}))


def test_single_runs_give_the_paired_error_without_a_split():
    a = Scores.from_lists({"x": [1], "y": [0], "z": [1], "w": [1]})
    b = Scores.from_lists({"x": [0], "y": [0], "z": [1], "w": [0]})
    result = compare(a, b)
    assert result.difference == pytest.approx(0.5)
    assert result.se == pytest.approx(result.se_items_only)
    assert math.isnan(result.item_part) and math.isnan(result.item_diff_variance)


def test_crossed_runs_add_their_run_variance():
    rng = np.random.default_rng(6)
    n, k = 300, 6
    base = rng.normal(0.0, 1.0, size=(n, 1))
    a = base + rng.normal(0.0, 0.6, size=(1, k)) + rng.normal(0.0, 0.5, size=(n, k))
    b = base + rng.normal(0.0, 0.6, size=(1, k)) + rng.normal(0.0, 0.5, size=(n, k))
    result = compare(Scores.from_matrix(a), Scores.from_matrix(b), design="crossed")
    assert result.run_a > 0.05 and result.run_b > 0.05
    assert result.se > 2 * result.se_items_only
    assert result.rerun_part == pytest.approx(
        result.run_a / k + result.run_b / k + (result.residual_a + result.residual_b) / k / n
    )
    labelled = compare(Scores.from_matrix(a), Scores.from_matrix(b))
    assert labelled.se == pytest.approx(result.se)


def test_identical_scores_have_zero_error():
    a = Scores.from_lists([[1, 1], [0, 0], [1, 1]])
    result = compare(a, a)
    assert result.difference == 0.0 and result.se == 0.0
    assert math.isnan(result.z) and math.isnan(result.p_value)


def test_detectable_difference():
    assert detectable_difference(0.01) == pytest.approx((1.959964 + 0.841621) * 0.01, rel=1e-5)
    assert detectable_difference(0.01, alpha=0.01, power=0.9) > detectable_difference(0.01)
    for alpha, power in ((0.0, 0.8), (0.05, 1.0)):
        with pytest.raises(ValueError):
            detectable_difference(0.01, alpha, power)
    with pytest.raises(ValueError):
        compare(Scores.from_lists([[1, 0], [1, 1]]), Scores.from_lists([[1, 0], [1, 1]]), level=2)
