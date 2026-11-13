"""ANOVA components against restricted maximum likelihood fits from statsmodels.

On balanced data with non-negative ANOVA estimates the two methods give the same
variance components, so the mixed-model fit is an independent check of the ANOVA code.
"""

import warnings

import numpy as np
import pytest

from repvar.components import crossed_components, nested_components
from repvar.data import Scores

sm = pytest.importorskip("statsmodels.api")
pd = pytest.importorskip("pandas")


def long_frame(table):
    rows, cols = np.indices(table.shape)
    return pd.DataFrame({"y": table.ravel(), "item": rows.ravel(), "run": cols.ravel()})


def test_nested_components_match_a_random_intercept_model():
    rng = np.random.default_rng(31)
    table = 1.0 + rng.normal(0, 1.2, size=(60, 1)) + rng.normal(0, 0.8, size=(60, 5))
    comp = nested_components(Scores.from_matrix(table, labelled=False))
    frame = long_frame(table)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.MixedLM.from_formula("y ~ 1", groups="item", data=frame).fit(reml=True)
    assert comp.item == pytest.approx(float(fit.cov_re.iloc[0, 0]), rel=2e-3)
    assert comp.residual == pytest.approx(float(fit.scale), rel=2e-3)
    assert comp.mean == pytest.approx(float(fit.fe_params.iloc[0]), rel=1e-6)


def test_crossed_components_match_a_crossed_random_effects_model():
    rng = np.random.default_rng(32)
    table = (
        rng.normal(0, 1.0, size=(24, 1))
        + rng.normal(0, 0.9, size=(1, 6))
        + rng.normal(0, 0.6, size=(24, 6))
    )
    comp = crossed_components(Scores.from_matrix(table))
    frame = long_frame(table)
    frame["whole"] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.MixedLM.from_formula(
            "y ~ 1",
            groups="whole",
            re_formula="0",
            vc_formula={"item": "0 + C(item)", "run": "0 + C(run)"},
            data=frame,
        )
        fit = model.fit(reml=True)
    estimates = dict(zip(model.exog_vc.names, fit.vcomp, strict=True))
    assert comp.item == pytest.approx(float(estimates["item"]), rel=2e-2)
    assert comp.run == pytest.approx(float(estimates["run"]), rel=2e-2)
    assert comp.residual == pytest.approx(float(fit.scale), rel=2e-2)
