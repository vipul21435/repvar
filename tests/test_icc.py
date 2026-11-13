import numpy as np
import pytest

from repvar.components import crossed_components, nested_components
from repvar.data import Scores
from repvar.icc import icc, icc_value, reliability, solve_shares
from repvar.simulate import simulate_normal

# Six targets (rows) rated by four judges (columns).
RATINGS = np.array(
    [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]],
    dtype=float,
)


def test_one_way_icc_on_the_ratings_table():
    # Row means 6, 3, 6.5, 4, 7.5, 4.75, grand mean 5.2917: between-rows mean square
    # 11.2417 on 5 df, within-rows mean square 6.2639 on 18 df, so
    # ICC(1,1) = (11.2417 - 6.2639) / (11.2417 + 3 * 6.2639) = 0.1657.
    comp = nested_components(Scores.from_matrix(RATINGS, labelled=False))
    assert comp.ms_item == pytest.approx(11.2417, abs=1e-3)
    assert comp.ms_residual == pytest.approx(6.2639, abs=1e-3)
    result = icc(comp)
    assert result.kind == "ICC(1,1)"
    assert result.value == pytest.approx(0.1657, abs=1e-3)
    assert result.value == pytest.approx(icc_value(comp))
    assert result.low < 0 < result.value < result.high < 1
    assert result.low == pytest.approx(-0.133, abs=2e-3)
    assert result.high == pytest.approx(0.723, abs=2e-3)


def test_two_way_icc_on_the_ratings_table():
    # Column means 7.667, 2.5, 4.333, 6.667: between-columns mean square 32.4861 on 3 df,
    # residual mean square 1.0194 on 15 df, so
    # ICC(2,1) = (11.2417 - 1.0194) / (11.2417 + 3 * 1.0194 + 4 * (32.4861 - 1.0194) / 6).
    comp = crossed_components(Scores.from_matrix(RATINGS))
    assert comp.ms_run == pytest.approx(32.4861, abs=1e-3)
    assert comp.ms_residual == pytest.approx(1.0194, abs=1e-3)
    result = icc(comp)
    assert result.kind == "ICC(2,1)"
    assert result.value == pytest.approx(0.2898, abs=1e-3)
    assert result.low == pytest.approx(0.019, abs=3e-3)
    assert result.high == pytest.approx(0.761, abs=3e-3)


def test_one_way_interval_covers_at_the_nominal_rate():
    rng = np.random.default_rng(17)
    truth = 0.5
    hits = 0
    reps = 600
    for _ in range(reps):
        scores = simulate_normal(30, 4, item_sd=1.0, residual_sd=1.0, seed=rng)
        result = icc(nested_components(scores), level=0.9)
        hits += result.low <= truth <= result.high
    # Exact interval under normality: 90 % nominal, binomial sd over 600 draws is 1.2 %.
    assert 0.86 <= hits / reps <= 0.94


def test_two_way_interval_covers_about_the_nominal_rate():
    rng = np.random.default_rng(23)
    truth = 1.0 / (1.0 + 0.25 + 0.75)
    hits = 0
    reps = 500
    for _ in range(reps):
        scores = simulate_normal(25, 6, item_sd=1.0, run_sd=0.5, residual_sd=0.75**0.5, seed=rng)
        result = icc(crossed_components(scores), level=0.9)
        hits += result.low <= truth <= result.high
    # Approximate (Satterthwaite) interval: allow 85 % to 96 % for a 90 % nominal level.
    assert 0.85 <= hits / reps <= 0.96


def test_unbalanced_and_degenerate_intervals():
    lists = [[1.0, 3.0], [2.0, 2.0, 5.0], [7.0, 6.5], [4.0, 6.0, 8.0, 6.0]]
    result = icc(nested_components(Scores.from_lists(lists)))
    assert "k0" in result.method
    assert result.low <= result.value <= result.high

    flat = icc(nested_components(Scores.from_lists([[1, 1], [1, 1], [1, 1]])))
    assert np.isnan(flat.value) and np.isnan(flat.low)

    perfect = icc(nested_components(Scores.from_lists([[1, 1], [0, 0], [1, 1]])))
    assert perfect.value == pytest.approx(1.0)
    assert np.isnan(perfect.low) and "zero within-item" in perfect.method

    table = RATINGS.copy()
    table[0, 0] = np.nan
    gaps = icc(crossed_components(Scores.from_matrix(table)))
    assert 0 < gaps.value < 1 and np.isnan(gaps.low) and "gaps" in gaps.method

    no_noise = np.add.outer(np.arange(4.0), np.arange(3.0))
    exact = icc(crossed_components(Scores.from_matrix(no_noise)))
    assert np.isnan(exact.low)

    with pytest.raises(ValueError):
        icc(nested_components(Scores.from_lists(lists)), level=1.5)


def test_reliability_follows_spearman_brown():
    comp = nested_components(Scores.from_lists([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    one = reliability(comp, 1)
    assert one == pytest.approx((26 / 3) / (26 / 3 + 1))
    for k in (2, 5, 20):
        assert reliability(comp, k) == pytest.approx(k * one / (1 + (k - 1) * one))
    crossed = crossed_components(Scores.from_matrix(RATINGS))
    assert reliability(crossed, 1, absolute=False) > reliability(crossed, 1, absolute=True)
    assert reliability(crossed, 4, absolute=False) == pytest.approx(
        crossed.item / (crossed.item + crossed.residual / 4)
    )
    flat = nested_components(Scores.from_lists([[1, 1], [1, 1]]))
    assert np.isnan(reliability(flat))
    with pytest.raises(ValueError):
        reliability(comp, 0)


def test_solve_shares():
    scores = Scores.from_lists([[1, 1, 1], [0, 0], [1, 0, 1], [0, 1], [1], [0, 0, 0, 0]])
    shares = solve_shares(scores)
    assert shares.n_items == 5  # the single-run item is left out
    assert shares.always == pytest.approx(1 / 5)
    assert shares.never == pytest.approx(2 / 5)
    assert shares.sometimes == pytest.approx(2 / 5)
    assert shares.always + shares.never + shares.sometimes == pytest.approx(1.0)

    real = solve_shares(Scores.from_lists([[0.5, 0.5], [0.2, 0.9], [1.0, 1.0]]))
    assert real.sometimes == pytest.approx(1 / 3)
    assert np.isnan(real.always) and np.isnan(real.never)

    single = solve_shares(Scores.from_lists([[1], [0]]))
    assert single.n_items == 0 and np.isnan(single.sometimes)
