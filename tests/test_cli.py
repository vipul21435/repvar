import json
from pathlib import Path

import pytest

from repvar.cli import main

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "runs.csv"


def run_cli(capsys, *args):
    code = main([str(a) for a in args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_components_markdown_on_the_example(capsys):
    code, out, _ = run_cli(capsys, "components", EXAMPLE)
    assert code == 0
    assert "# Variance components" in out
    assert "crossed" in out
    assert "| item |" in out and "| run |" in out and "| residual |" in out
    assert "8 items x 4 run(s)" in out


def test_components_json_matches_the_library(capsys):
    code, out, _ = run_cli(capsys, "components", EXAMPLE, "--design", "nested", "--format", "json")
    assert code == 0
    payload = json.loads(out)["all"]
    assert payload["design"] == "nested"
    assert payload["n_items"] == 8
    assert payload["n_obs"] == 32
    # Item sums are 4, 3, 1, 0, 3, 2, 4, 1 out of 4 runs.
    assert payload["mean"] == pytest.approx(18 / 32)
    assert payload["residual"] == pytest.approx((0.75 * 4 + 1.0) / 24)
    assert payload["se_observed"]["n_runs"] == 4
    assert payload["se_one_run"]["se"] > payload["se_observed"]["se"]


def test_output_file_and_errors(tmp_path, capsys):
    target = tmp_path / "out.md"
    code, out, _ = run_cli(capsys, "components", EXAMPLE, "--output", target)
    assert code == 0 and out == ""
    assert target.read_text().startswith("# Variance components")

    single = tmp_path / "single.csv"
    single.write_text("item,score\na,1\nb,0\n")
    code, _, err = run_cli(capsys, "components", single)
    assert code == 2
    assert "single run" in err

    code, _, err = run_cli(capsys, "components", tmp_path / "missing.csv")
    assert code == 2
    assert "repvar:" in err
