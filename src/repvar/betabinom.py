"""Beta-binomial model for 0/1 scores: each item has its own pass probability.

Item ``i`` has pass probability ``p_i`` drawn from ``Beta(alpha, beta)`` and its ``k_i``
runs are independent Bernoulli trials, so its number of passes is beta-binomial. In terms
of the mean ``mu = alpha / (alpha + beta)`` and ``rho = 1 / (alpha + beta + 1)``:

* variance of ``p_i`` across items (item variance): ``rho * mu * (1 - mu)``
* mean Bernoulli variance within an item (rerun variance): ``(1 - rho) * mu * (1 - mu)``

so ``rho`` is the intraclass correlation of single runs under this model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, expit, gammaln, logit, xlogy
from scipy.stats import chi2

from .components import NotIdentifiableError
from .data import Scores

_BOUND = 12.0


def betabinom_loglik(
    successes: object, trials: object, alpha: float, beta: float, weights: object = None
) -> float:
    """Log-likelihood of pass counts ``successes`` out of ``trials`` under Beta(alpha, beta)."""
    y = np.asarray(successes, dtype=float)
    k = np.asarray(trials, dtype=float)
    terms = (
        gammaln(k + 1.0)
        - gammaln(y + 1.0)
        - gammaln(k - y + 1.0)
        + betaln(y + alpha, k - y + beta)
        - betaln(alpha, beta)
    )
    if weights is not None:
        terms = terms * np.asarray(weights, dtype=float)
    return float(np.sum(terms))


def binomial_loglik(successes: object, trials: object, p: float, weights: object = None) -> float:
    """Log-likelihood when every item shares the pass probability ``p`` (no item spread)."""
    y = np.asarray(successes, dtype=float)
    k = np.asarray(trials, dtype=float)
    terms = (
        gammaln(k + 1.0)
        - gammaln(y + 1.0)
        - gammaln(k - y + 1.0)
        + xlogy(y, p)
        + xlogy(k - y, 1.0 - p)
    )
    if weights is not None:
        terms = terms * np.asarray(weights, dtype=float)
    return float(np.sum(terms))


@dataclass(frozen=True)
class BetaBinomialFit:
    """Maximum-likelihood beta-binomial fit.

    ``lr_stat`` is twice the log-likelihood gain over the plain binomial (``rho = 0``).
    Because ``rho = 0`` is on the edge of the parameter space, ``p_value`` uses the
    half-and-half mixture of a point mass at zero and a chi-square with one degree of
    freedom. ``boundary`` is set when the optimum sits on a bound of the search.
    """

    alpha: float
    beta: float
    mean: float
    rho: float
    loglik: float
    loglik_binomial: float
    lr_stat: float
    p_value: float
    n_items: int
    n_obs: int
    boundary: str | None = None

    @property
    def item_variance(self) -> float:
        return self.rho * self.mean * (1.0 - self.mean)

    @property
    def rerun_variance(self) -> float:
        return (1.0 - self.rho) * self.mean * (1.0 - self.mean)

    def sometimes_share(self, n_runs: int) -> float:
        """Expected share of items with at least one pass and one failure in ``n_runs`` runs."""
        if n_runs < 1:
            raise ValueError("n_runs must be at least 1")
        if math.isnan(self.alpha) or math.isnan(self.beta):
            return 0.0 if self.mean in (0.0, 1.0) else float("nan")
        base = betaln(self.alpha, self.beta)
        always = math.exp(betaln(self.alpha + n_runs, self.beta) - base)
        never = math.exp(betaln(self.alpha, self.beta + n_runs) - base)
        return float(min(1.0, max(0.0, 1.0 - always - never)))

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["item_variance"] = self.item_variance
        out["rerun_variance"] = self.rerun_variance
        return out


def fit_betabinomial(successes: object, trials: object) -> BetaBinomialFit:
    """Fit ``Beta(alpha, beta)`` to per-item pass counts by maximum likelihood."""
    y = np.asarray(successes, dtype=float).ravel()
    k = np.asarray(trials, dtype=float).ravel()
    if y.shape != k.shape or y.size == 0:
        raise ValueError("successes and trials must be non-empty and of equal length")
    if np.any(k < 1) or np.any(y < 0) or np.any(y > k) or np.any(y != np.round(y)):
        raise ValueError("need whole numbers with 0 <= successes <= trials and trials >= 1")
    if not np.any(k >= 2):
        raise NotIdentifiableError(
            "every item has a single run: the spread of pass probabilities is not identifiable"
        )
    pairs, weights = np.unique(np.stack([y, k], axis=1), axis=0, return_counts=True)
    yy, kk, ww = pairs[:, 0], pairs[:, 1], weights.astype(float)
    total_y, total_k = float(y.sum()), float(k.sum())
    pooled = total_y / total_k
    if total_y == 0.0 or total_y == total_k:
        return BetaBinomialFit(
            alpha=float("nan"),
            beta=float("nan"),
            mean=pooled,
            rho=float("nan"),
            loglik=0.0,
            loglik_binomial=0.0,
            lr_stat=0.0,
            p_value=1.0,
            n_items=int(y.size),
            n_obs=int(total_k),
            boundary="all scores equal",
        )
    ll_binomial = binomial_loglik(yy, kk, pooled, ww)

    def negloglik(theta: np.ndarray) -> float:
        mu, rho = expit(theta[0]), expit(theta[1])
        size = (1.0 - rho) / rho
        return -betabinom_loglik(yy, kk, mu * size, (1.0 - mu) * size, ww)

    start_mu = float(logit(min(max(pooled, 1e-4), 1.0 - 1e-4)))
    best_x, best_fun = None, math.inf
    for start_rho in (_moment_rho(y, k), 0.1, 0.6):
        result = minimize(
            negloglik,
            x0=np.array([start_mu, float(logit(start_rho))]),
            method="L-BFGS-B",
            bounds=[(-_BOUND, _BOUND), (-_BOUND, _BOUND)],
        )
        if result.fun < best_fun:
            best_x, best_fun = np.asarray(result.x, dtype=float), float(result.fun)
    # The likelihood can keep rising towards rho = 0 or rho = 1 with a gradient too small
    # for the optimiser to follow; take the edge whenever it is at least as good.
    for edge in (-_BOUND, _BOUND):
        candidate = np.array([best_x[0], edge])
        value = negloglik(candidate)
        if value <= best_fun + 1e-9:
            best_x, best_fun = candidate, value
    mu, rho = float(expit(best_x[0])), float(expit(best_x[1]))
    size = (1.0 - rho) / rho
    loglik = -best_fun
    boundary = None
    if best_x[1] <= -_BOUND + 1e-3 or loglik <= ll_binomial + 1e-9:
        boundary = "rho at its lower bound: no spread beyond binomial"
    elif best_x[1] >= _BOUND - 1e-3:
        boundary = "rho at its upper bound: items are all-or-nothing"
    lr_stat = max(0.0, 2.0 * (loglik - ll_binomial))
    return BetaBinomialFit(
        alpha=mu * size,
        beta=(1.0 - mu) * size,
        mean=mu,
        rho=rho,
        loglik=loglik,
        loglik_binomial=ll_binomial,
        lr_stat=lr_stat,
        p_value=float(0.5 * chi2.sf(lr_stat, 1)) if lr_stat > 0.0 else 1.0,
        n_items=int(y.size),
        n_obs=int(total_k),
        boundary=boundary,
    )


def _moment_rho(y: np.ndarray, k: np.ndarray) -> float:
    """ANOVA estimate of the intraclass correlation, clipped away from 0 and 1."""
    n, total = y.size, float(k.sum())
    if n < 2 or total <= n:
        return 0.2
    pooled = float(y.sum()) / total
    ms_within = float(np.sum(y * (1.0 - y / k))) / (total - n)
    ms_between = float(np.sum(k * (y / k - pooled) ** 2)) / (n - 1)
    k0 = (total - float(np.sum(k**2)) / total) / (n - 1)
    denom = ms_between + (k0 - 1.0) * ms_within
    rho = (ms_between - ms_within) / denom if denom > 0 else 0.2
    return float(min(max(rho, 0.02), 0.95))


def fit_scores(scores: Scores) -> BetaBinomialFit:
    """:func:`fit_betabinomial` on the per-item pass counts of 0/1 ``scores``."""
    if not scores.is_binary:
        raise ValueError("the beta-binomial model needs 0/1 scores")
    return fit_betabinomial(scores.sums, scores.counts)
