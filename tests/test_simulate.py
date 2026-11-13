import numpy as np
import pytest

from repvar.components import CROSSED, NESTED, variance_components
from repvar.simulate import interval_coverage, simulate_binary, simulate_normal


def test_binary_simulation_has_the_planted_components():
    scores = simulate_binary(5000, 6, mean=0.3, icc=0.4, seed=1)
    assert scores.is_binary and scores.balanced and not scores.has_run_labels
    comp = variance_components(scores)
    assert comp.design == NESTED
    assert comp.mean == pytest.approx(0.3, abs=0.015)
    assert comp.item == pytest.approx(0.4 * 0.21, rel=0.08)
    assert comp.residual == pytest.approx(0.6 * 0.21, rel=0.05)


def test_binary_edge_values_of_icc():
    same = variance_components(simulate_binary(3000, 5, mean=0.4, icc=0.0, seed=2))
    assert same.item == pytest.approx(0.0, abs=0.004)
    fixed = simulate_binary(500, 5, mean=0.4, icc=1.0, seed=2)
    assert set(np.unique(fixed.item_means)) <= {0.0, 1.0}
    assert variance_components(fixed).residual == 0.0


def test_normal_simulation_is_crossed_only_with_a_run_effect():
    crossed = simulate_normal(800, 30, mean=2.0, item_sd=1.0, run_sd=0.5, residual_sd=0.7, seed=3)
    assert crossed.has_run_labels
    comp = variance_components(crossed)
    assert comp.design == CROSSED
    assert comp.item == pytest.approx(1.0, rel=0.12)
    assert comp.residual == pytest.approx(0.49, rel=0.05)
    assert comp.run == pytest.approx(0.25, rel=0.6)
    nested = simulate_normal(50, 3, seed=3)
    assert not nested.has_run_labels
    assert variance_components(nested).design == NESTED


def test_same_seed_same_data():
    a = simulate_binary(30, 4, seed=7)
    b = simulate_binary(30, 4, seed=7)
    np.testing.assert_array_equal(a.values, b.values)
    assert not np.array_equal(a.values, simulate_binary(30, 4, seed=8).values)


def test_interval_coverage_is_near_nominal():
    rates = interval_coverage(
        50, 4, mean=0.4, icc_true=0.4, reps=150, n_boot=150, level=0.9, seed=5
    )
    assert set(rates) == {"bootstrap: mean", "bootstrap: item variance", "bootstrap: icc", "F: icc"}
    # 150 draws: binomial sd about 2.5 %. The bootstrap for the mean should sit near 90 %;
    # the variance-type intervals are known to run under and get a wider band.
    assert 0.82 <= rates["bootstrap: mean"] <= 0.97
    assert 0.70 <= rates["bootstrap: item variance"] <= 0.97
    assert 0.70 <= rates["bootstrap: icc"] <= 0.97
    assert 0.70 <= rates["F: icc"] <= 0.99


def test_argument_checks():
    for kwargs in ({"mean": 0.0}, {"mean": 1.0}, {"icc": -0.1}, {"icc": 1.5}):
        with pytest.raises(ValueError):
            simulate_binary(10, 2, **kwargs)
    with pytest.raises(ValueError):
        simulate_binary(0, 2)
    with pytest.raises(ValueError):
        simulate_normal(10, 0)
    with pytest.raises(ValueError):
        simulate_normal(10, 2, item_sd=-1.0)
    with pytest.raises(ValueError):
        interval_coverage(10, 2, reps=0)
