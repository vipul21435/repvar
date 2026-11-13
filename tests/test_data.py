import json

import numpy as np
import pytest

from repvar.data import ALL, Scores, parse_score, read_long


def test_parse_score_accepts_numbers_booleans_and_words():
    assert parse_score(True) == 1.0
    assert parse_score(False) == 0.0
    assert parse_score("0.25") == pytest.approx(0.25)
    assert parse_score(" Resolved ") == 1.0
    assert parse_score("fail") == 0.0
    assert parse_score(3) == 3.0
    for bad in ("maybe", "nan", float("inf"), [1]):
        with pytest.raises(ValueError):
            parse_score(bad)


def test_scores_summaries():
    scores = Scores.from_lists({"a": [1, 0, 1], "b": [0, 0], "c": [1]})
    assert scores.item_ids == ("a", "b", "c")
    assert scores.n_items == 3
    assert scores.n_obs == 6
    assert scores.counts.tolist() == [3, 2, 1]
    assert scores.sums.tolist() == [2.0, 0.0, 1.0]
    assert scores.mean == pytest.approx((2 / 3 + 0 + 1) / 3)
    assert scores.is_binary
    assert not scores.balanced
    assert not scores.has_run_labels
    assert [x.tolist() for x in scores.per_item()] == [[1.0, 0.0, 1.0], [0.0, 0.0], [1.0]]


def test_matrix_round_trip_with_missing_cells():
    table = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
    scores = Scores.from_matrix(table)
    back, runs = scores.matrix()
    assert runs == ("0", "2", "1")
    order = [runs.index(str(j)) for j in range(3)]
    np.testing.assert_array_equal(np.isnan(back[:, order]), np.isnan(table))
    np.testing.assert_allclose(back[:, order][~np.isnan(table)], table[~np.isnan(table)])
    with pytest.raises(ValueError, match="two-dimensional"):
        Scores.from_matrix([1.0, 2.0])


def test_matrix_rejects_duplicate_cells():
    scores = Scores.from_records([("a", "r1", 1), ("a", "r1", 0), ("b", "r1", 1)])
    with pytest.raises(ValueError, match="same run label"):
        scores.matrix()


def test_read_jsonl_with_systems(tmp_path):
    rows = [
        {"instance": "i1", "seed": 1, "resolved": True, "model": "m1"},
        {"instance": "i1", "seed": 2, "resolved": False, "model": "m1"},
        {"instance": "i2", "seed": 1, "resolved": True, "model": "m1"},
        {"instance": "i1", "seed": 1, "resolved": False, "model": "m2"},
        {"instance": "i2", "seed": 1, "resolved": None, "model": "m2"},
    ]
    path = tmp_path / "runs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    groups = read_long(path, item="instance", run="seed", score="resolved", system="model")
    assert set(groups) == {"m1", "m2"}
    assert groups["m1"].n_obs == 3
    assert groups["m1"].has_run_labels
    assert groups["m1"].system == "m1"
    assert groups["m2"].n_obs == 1  # the null score is skipped


def test_read_csv_without_run_column(tmp_path):
    path = tmp_path / "runs.csv"
    path.write_text("item,score\na,1\na,0\nb,0.5\nb,\n")
    groups = read_long(path)
    assert set(groups) == {ALL}
    scores = groups[ALL]
    assert scores.n_obs == 3
    assert not scores.has_run_labels
    assert scores.run_labels == ("1", "2", "1")
    assert not scores.is_binary


def test_read_errors(tmp_path):
    path = tmp_path / "runs.csv"
    path.write_text("item,score\na,1\n")
    with pytest.raises(ValueError, match="no column 'seed'"):
        read_long(path, run="seed")
    with pytest.raises(ValueError, match="no column 'value'"):
        read_long(path, score="value")
    with pytest.raises(ValueError, match="unknown format"):
        read_long(tmp_path / "runs.parquet")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    with pytest.raises(ValueError, match="line 1"):
        read_long(bad)
    bad.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match="JSON object"):
        read_long(bad)
    empty = tmp_path / "empty.csv"
    empty.write_text("item,score\n")
    with pytest.raises(ValueError, match="no scored rows"):
        read_long(empty)
    with pytest.raises(ValueError, match="no observations"):
        Scores.from_records([])
