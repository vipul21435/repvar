import numpy as np
import pytest
from scipy import optimize, stats

from repvar.betabinom import (
    betabinom_loglik,
    binomial_loglik,
    fit_betabinomial,
    fit_scores,
)
from repvar.components import NotIdentifiableError
from repvar.data import Scores
from repvar.simulate import simulate_binary


def test_loglik_matches_scipy():
    y = np.array([0, 1, 3, 5, 2, 7])
    k = np.array([4, 4, 5, 5, 2, 9])
    for alpha, beta in ((0.7, 2.1), (3.0, 3.0), (25.0, 0.4)):
        assert betabinom_loglik(y, k, alpha, beta) == pytest.approx(
            float(np.sum(stats.betabinom.logpmf(y, k, alpha, beta))), rel=1e-10
        )
    assert binomial_loglik(y, k, 0.3) == pytest.approx(
        float(np.sum(stats.binom.logpmf(y, k, 0.3))), rel=1e-10
    )
    weights = np.array([1, 2, 1, 3, 1, 1])
    assert betabinom_loglik(y, k, 1.5, 2.5, weights) == pytest.approx(
        float(np.sum(weights * stats.betabinom.logpmf(y, k, 1.5, 2.5))), rel=1e-10
    )


def test_fit_matches_a_direct_scipy_optimisation():
    rng = np.random.default_rng(21)
    k = rng.integers(2, 9, size=300)
    y = stats.betabinom.rvs(k, 0.8, 1.9, random_state=rng)
    fit = fit_betabinomial(y, k)

    def nll(log_params):
        a, b = np.exp(log_params)
        return -float(np.sum(stats.betabinom.logpmf(y, k, a, b)))

    ref = optimize.minimize(nll, x0=[0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-7})
    assert fit.loglik == pytest.approx(-ref.fun, abs=1e-4)
    assert fit.alpha == pytest.approx(float(np.exp(ref.x[0])), rel=2e-3)
    assert fit.beta == pytest.approx(float(np.exp(ref.x[1])), rel=2e-3)
    assert fit.mean == pytest.approx(fit.alpha / (fit.alpha + fit.beta))
    assert fit.rho == pytest.approx(1.0 / (fit.alpha + fit.beta + 1.0))
    assert fit.n_items == 300 and fit.n_obs == int(k.sum())
    assert fit.boundary is None


def test_fit_recovers_planted_parameters():
    scores = simulate_binary(6000, 6, mean=0.3, icc=0.4, seed=4)
    fit = fit_scores(scores)
    assert fit.mean == pytest.approx(0.3, abs=0.015)
    assert fit.rho == pytest.approx(0.4, abs=0.02)
    assert fit.item_variance == pytest.approx(0.4 * 0.21, abs=0.006)
    assert fit.item_variance + fit.rerun_variance == pytest.approx(fit.mean * (1 - fit.mean))
    assert fit.lr_stat > 100 and fit.p_value < 1e-10


def test_sometimes_share_closed_form_and_simulation():
    fit = fit_betabinomial([1, 2, 0, 3, 3, 1, 0, 2], [3] * 8)
    a, b = fit.alpha, fit.beta
    # Two runs disagree with probability 2 E[p (1 - p)] = 2 a b / ((a + b) (a + b + 1)).
    assert fit.sometimes_share(2) == pytest.approx(2 * a * b / ((a + b) * (a + b + 1)))
    assert fit.sometimes_share(1) == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(8)
    p = rng.beta(a, b, size=200_000)
    passes = rng.binomial(5, p)
    assert fit.sometimes_share(5) == pytest.approx(np.mean((passes > 0) & (passes < 5)), abs=0.005)
    assert fit.sometimes_share(5) < fit.sometimes_share(50)
    with pytest.raises(ValueError):
        fit.sometimes_share(0)


def test_plain_binomial_data_shows_no_extra_spread():
    rng = np.random.default_rng(2)
    y = rng.binomial(8, 0.4, size=400)
    fit = fit_betabinomial(y, np.full(400, 8))
    assert fit.rho < 0.02
    assert fit.lr_stat < 4.0
    assert fit.p_value > 0.02
    assert fit.loglik >= fit.loglik_binomial - 1e-6


def test_boundary_cases():
    same = fit_betabinomial([0, 0, 0], [3, 3, 3])
    assert same.boundary == "all scores equal"
    assert same.mean == 0.0 and np.isnan(same.rho)
    assert same.sometimes_share(4) == 0.0
    assert "item_variance" in same.to_dict()

    all_or_nothing = fit_betabinomial([4, 0, 4, 0, 4, 0], [4] * 6)
    assert all_or_nothing.rho > 0.99
    assert all_or_nothing.boundary is not None
    assert all_or_nothing.mean == pytest.approx(0.5, abs=1e-3)
    assert all_or_nothing.sometimes_share(10) < 1e-3

    no_spread = fit_betabinomial([2, 2, 2, 2, 2, 2], [4] * 6)
    assert no_spread.rho < 1e-4
    assert "lower bound" in no_spread.boundary


def test_input_validation():
    with pytest.raises(NotIdentifiableError, match="single run"):
        fit_betabinomial([1, 0, 1], [1, 1, 1])
    for y, k in (([1, 5], [3, 3]), ([-1, 1], [3, 3]), ([0.5, 1], [3, 3]), ([1], [0]), ([], [])):
        with pytest.raises(ValueError):
            fit_betabinomial(y, k)
    with pytest.raises(ValueError):
        fit_betabinomial([1, 2], [3])
    with pytest.raises(ValueError, match="0/1"):
        fit_scores(Scores.from_lists([[0.5, 1.0], [0.0, 1.0]]))
