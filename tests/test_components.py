import numpy as np
import pytest

from repvar.components import (
    CROSSED,
    NESTED,
    NotIdentifiableError,
    choose_design,
    crossed_components,
    nested_components,
    variance_components,
)
from repvar.data import Scores
from repvar.design import mean_se


def test_nested_hand_worked_balanced_table():
    # Item means 2, 5, 8; grand mean 5. SS between = 3 * (9 + 0 + 9) = 54, MS = 27.
    # SS within = 3 * 2 = 6 on 6 df, MS = 1. Item variance = (27 - 1) / 3.
    scores = Scores.from_lists([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    comp = nested_components(scores)
    assert comp.design == NESTED
    assert comp.balanced
    assert comp.ms_item == pytest.approx(27.0)
    assert comp.ms_residual == pytest.approx(1.0)
    assert comp.item == pytest.approx(26.0 / 3.0)
    assert comp.residual == pytest.approx(1.0)
    assert comp.run == 0.0
    assert comp.runs_per_item == pytest.approx(3.0)
    assert comp.mean == pytest.approx(5.0)
    assert comp.shares()["item"] == pytest.approx((26 / 3) / (26 / 3 + 1))


def test_crossed_hand_worked_balanced_table():
    # Rows are items, columns are runs. Item means 2, 4, 6; run means 3, 5; grand mean 4.
    # SS item = 2 * 8 = 16 (MS 8), SS run = 3 * 2 = 6 (MS 6), SS total = 26, SS residual = 4
    # on 2 df (MS 2). item = (8 - 2) / 2, run = (6 - 2) / 3, residual = 2.
    scores = Scores.from_matrix([[1, 3], [2, 6], [6, 6]])
    comp = crossed_components(scores)
    assert comp.design == CROSSED
    assert comp.ms_item == pytest.approx(8.0)
    assert comp.ms_run == pytest.approx(6.0)
    assert comp.ms_residual == pytest.approx(2.0)
    assert comp.item == pytest.approx(3.0)
    assert comp.run == pytest.approx(4.0 / 3.0)
    assert comp.residual == pytest.approx(2.0)


def test_variance_of_the_grand_mean_matches_the_item_mean_square():
    # In a balanced nested design, item / n + residual / (n k) equals MS item / (n k).
    scores = Scores.from_lists([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    comp = nested_components(scores)
    se = mean_se(comp, 3, 3)
    assert se.variance == pytest.approx(27.0 / 9.0)
    assert se.item_part == pytest.approx(26.0 / 9.0)
    assert se.rerun_part == pytest.approx(1.0 / 9.0)
    assert se.rerun_share == pytest.approx(1.0 / 27.0)


def test_unbalanced_nested_uses_k0():
    lists = [[1.0, 3.0], [2.0, 2.0, 5.0], [7.0], [4.0, 6.0, 8.0, 6.0]]
    comp = nested_components(Scores.from_lists(lists))
    counts = np.array([len(x) for x in lists], dtype=float)
    total = counts.sum()
    allv = np.concatenate(lists)
    grand = allv.mean()
    ss_between = sum(len(x) * (np.mean(x) - grand) ** 2 for x in lists)
    ss_within = sum(float(np.sum((np.asarray(x) - np.mean(x)) ** 2)) for x in lists)
    ms_between = ss_between / 3
    ms_within = ss_within / (total - 4)
    k0 = (total - np.sum(counts**2) / total) / 3
    assert not comp.balanced
    assert comp.runs_per_item == pytest.approx(k0)
    assert comp.residual == pytest.approx(ms_within)
    assert comp.item_raw == pytest.approx((ms_between - ms_within) / k0)
    # The reported mean gives every item the same weight.
    assert comp.mean == pytest.approx(np.mean([np.mean(x) for x in lists]))


def test_recovers_planted_nested_components():
    rng = np.random.default_rng(11)
    n, k = 4000, 5
    item = rng.normal(0.0, np.sqrt(2.0), size=(n, 1))
    data = 1.5 + item + rng.normal(0.0, np.sqrt(0.5), size=(n, k))
    comp = nested_components(Scores.from_matrix(data))
    assert comp.item == pytest.approx(2.0, rel=0.08)
    assert comp.residual == pytest.approx(0.5, rel=0.05)
    assert comp.mean == pytest.approx(1.5, abs=0.08)


def test_recovers_planted_crossed_components_with_missing_cells():
    rng = np.random.default_rng(5)
    n, k = 600, 40
    data = (
        rng.normal(0.0, 1.0, size=(n, 1))
        + rng.normal(0.0, np.sqrt(0.3), size=(1, k))
        + rng.normal(0.0, np.sqrt(0.6), size=(n, k))
    )
    full = crossed_components(Scores.from_matrix(data))
    assert full.balanced
    assert full.item == pytest.approx(1.0, rel=0.15)
    assert full.residual == pytest.approx(0.6, rel=0.05)
    assert full.run == pytest.approx(0.3, rel=0.6)  # only 40 runs: wide

    holes = data.copy()
    holes[rng.random(size=data.shape) < 0.3] = np.nan
    part = crossed_components(Scores.from_matrix(holes))
    assert not part.balanced
    assert part.item == pytest.approx(full.item, rel=0.1)
    assert part.residual == pytest.approx(full.residual, rel=0.1)
    assert part.run == pytest.approx(full.run, rel=0.15)


def test_crossed_estimator_is_unbiased_with_missing_cells():
    rng = np.random.default_rng(3)
    estimates = []
    for _ in range(300):
        data = (
            rng.normal(0.0, 1.0, size=(30, 1))
            + rng.normal(0.0, 0.7, size=(1, 8))
            + rng.normal(0.0, 0.5, size=(30, 8))
        )
        data[rng.random(size=data.shape) < 0.25] = np.nan
        comp = crossed_components(Scores.from_matrix(data))
        estimates.append([comp.item_raw, comp.run_raw, comp.residual_raw])
    mean = np.mean(estimates, axis=0)
    assert mean[0] == pytest.approx(1.0, abs=0.06)
    assert mean[1] == pytest.approx(0.49, abs=0.08)
    assert mean[2] == pytest.approx(0.25, abs=0.02)


def test_negative_item_estimate_is_clamped_and_kept_raw():
    # Item means are identical while runs differ, so MS item < MS residual.
    comp = nested_components(Scores.from_lists([[0, 2], [2, 0], [0, 2]]))
    assert comp.item_raw < 0
    assert comp.item == 0.0
    assert comp.residual == pytest.approx(2.0)


def test_all_equal_scores_give_zero_components():
    for comp in (
        nested_components(Scores.from_lists([[1, 1, 1], [1, 1, 1]])),
        crossed_components(Scores.from_matrix(np.ones((3, 4)))),
    ):
        assert comp.item == pytest.approx(0.0, abs=1e-12)
        assert comp.run == pytest.approx(0.0, abs=1e-12)
        assert comp.residual == pytest.approx(0.0, abs=1e-12)
        assert np.isnan(comp.shares()["item"]) or comp.total == 0.0


def test_one_run_per_item_is_not_identifiable():
    with pytest.raises(NotIdentifiableError, match="single run"):
        nested_components(Scores.from_lists([[1], [0], [1]]))
    with pytest.raises(NotIdentifiableError):
        nested_components(Scores.from_lists([[1, 0, 1]]))
    with pytest.raises(NotIdentifiableError):
        crossed_components(Scores.from_matrix([[1.0], [0.0], [1.0]]))


def test_design_is_chosen_from_the_run_labels():
    labelled = Scores.from_records([("a", "s1", 1), ("a", "s2", 0), ("b", "s1", 1), ("b", "s2", 1)])
    assert choose_design(labelled) == CROSSED
    assert variance_components(labelled).design == CROSSED
    unlabelled = Scores.from_lists([[1, 0], [1, 1]])
    assert choose_design(unlabelled) == NESTED
    # Run labels that never repeat across items say nothing about a shared run effect.
    unique = Scores.from_records([("a", "r1", 1), ("a", "r2", 0), ("b", "r3", 1), ("b", "r4", 1)])
    assert choose_design(unique) == NESTED
    with pytest.raises(ValueError, match="unknown design"):
        variance_components(labelled, design="split-plot")
