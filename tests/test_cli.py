import json
from pathlib import Path

import pytest

from repvar.cli import main
from repvar.data import read_long

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "runs.csv"


def run_cli(capsys, *args):
    code = main([str(a) for a in args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture()
def two_systems(tmp_path):
    """Two systems on 30 shared items, 4 runs each, written as a long CSV."""
    paths = []
    for name, seed, mean in (("alpha", 1, 0.6), ("beta", 2, 0.4)):
        path = tmp_path / f"{name}.csv"
        assert (
            main(
                ["simulate", "--items", "30", "--runs", "4", "--mean", str(mean)]
                + ["--icc", "0.4", "--seed", str(seed), "--output", str(path)]
            )
            == 0
        )
        paths.append((name, path))
    combined = tmp_path / "both.csv"
    lines = ["item,system,score"]
    for name, path in paths:
        for row in path.read_text().splitlines()[1:]:
            item, score = row.split(",")
            lines.append(f"{item},{name},{score}")
    combined.write_text("\n".join(lines) + "\n")
    return combined


def test_components_markdown_on_the_example(capsys):
    code, out, _ = run_cli(capsys, "components", EXAMPLE)
    assert code == 0
    assert "# Variance components" in out
    assert "crossed" in out
    assert "| item |" in out and "| run |" in out and "| residual |" in out
    assert "8 items x 4 run(s)" in out
    assert "ICC(2,1)" in out
    assert "always solved 25.0 %, sometimes 62.5 %, never 12.5 %" in out
    assert "beta-binomial fit" in out


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
    assert payload["icc"]["kind"] == "ICC(1,1)"
    assert payload["solve_shares"]["sometimes"] == pytest.approx(5 / 8)
    assert payload["betabinomial"]["mean"] == pytest.approx(18 / 32, abs=0.02)
    assert payload["bootstrap"] is None


def test_components_with_bootstrap(capsys):
    code, out, _ = run_cli(
        capsys, "components", EXAMPLE, "--design", "nested", "--bootstrap", "50", "--seed", "4"
    )
    assert code == 0
    assert "bootstrap intervals (95 %, percentile)" in out
    assert "| se_observed |" in out
    code, out, _ = run_cli(
        capsys, "components", EXAMPLE, "--design", "nested", "--bootstrap", "50", "--format", "json"
    )
    low, high = (json.loads(out)["all"]["bootstrap"]["mean"][k] for k in ("low", "high"))
    assert low < 18 / 32 < high


def test_real_valued_scores_skip_the_binary_parts(tmp_path, capsys):
    path = tmp_path / "real.csv"
    assert (
        main(
            ["simulate", "--kind", "normal", "--items", "12", "--runs", "3", "--run-sd", "0.5"]
            + ["--output", str(path)]
        )
        == 0
    )
    assert path.read_text().splitlines()[0] == "item,run,score"
    code, out, _ = run_cli(capsys, "components", path)
    assert code == 0
    assert "beta-binomial" not in out
    assert "items whose runs disagree: 100.0 %" in out


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


def test_design_table_best_split_and_plateau(capsys):
    code, out, _ = run_cli(
        capsys, "design", EXAMPLE, "--design", "nested", "--budget", "64", "--max-items", "8"
    )
    assert code == 0
    assert "# Decision study" in out
    assert "budget: 64 item-runs" in out
    assert "| 8 * | 8 | 64 |" in out
    assert "best split (*, smallest standard error): 8 items x 8 run(s)" in out
    assert "with 8 items the standard error cannot go below" in out

    code, out, _ = run_cli(capsys, "design", EXAMPLE, "--design", "nested", "--format", "json")
    payload = json.loads(out)["all"]
    assert payload["budget"] == 32
    assert payload["best"]["n_runs"] == 1 and payload["best"]["n_items"] == 32
    assert len(payload["splits"]) == 10
    assert payload["plateau_runs"] >= 1

    code, out, _ = run_cli(
        capsys, "design", EXAMPLE, "--design", "nested", "--fixed-items", "--max-items", "8"
    )
    assert "smallest rerun part" in out


def test_design_without_item_variance_has_no_plateau(tmp_path, capsys):
    path = tmp_path / "flat.csv"
    path.write_text("item,score\na,1\na,0\nb,0\nb,1\nc,1\nc,0\n")
    code, out, _ = run_cli(capsys, "design", path)
    assert code == 0
    assert "there is no plateau" in out


def test_design_chart(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    chart = tmp_path / "se.png"
    code, _, _ = run_cli(capsys, "design", EXAMPLE, "--design", "nested", "--chart", chart)
    assert code == 0
    assert chart.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    code, _, err = run_cli(
        capsys, "design", EXAMPLE, "--chart", tmp_path / "x.png", "--max-runs", "1"
    )
    assert code == 0  # a one-run table still draws two points on the chart


def test_compare(two_systems, capsys):
    code, out, _ = run_cli(capsys, "compare", two_systems, "--system-col", "system")
    assert code == 0
    assert "# Comparison: alpha minus beta" in out
    assert "shared items: 30; runs per item: 4.00 and 4.00" in out
    assert "smallest detectable difference with 30 shared items" in out
    assert "| 10 |" in out

    code, out, _ = run_cli(
        capsys, "compare", two_systems, "--system-col", "system", "--a", "beta", "--b", "alpha"
    )
    assert "# Comparison: beta minus alpha" in out

    code, out, _ = run_cli(
        capsys, "compare", two_systems, "--system-col", "system", "--format", "json"
    )
    payload = json.loads(out)
    groups = read_long(two_systems, system="system")
    assert payload["difference"] == pytest.approx(groups["alpha"].mean - groups["beta"].mean)
    assert payload["detectable"]["observed_design"] == pytest.approx(
        (1.959964 + 0.841621) * payload["se"], rel=1e-5
    )


def test_compare_errors(two_systems, tmp_path, capsys):
    code, _, err = run_cli(capsys, "compare", two_systems)
    assert code == 2 and "--system-col" in err
    code, _, err = run_cli(capsys, "compare", two_systems, "--system-col", "system", "--a", "alpha")
    assert code == 2 and "--a and --b" in err
    code, _, err = run_cli(
        capsys, "compare", two_systems, "--system-col", "system", "--a", "alpha", "--b", "gamma"
    )
    assert code == 2 and "no system 'gamma'" in err

    single = tmp_path / "single.csv"
    single.write_text("item,system,score\na,x,1\nb,x,0\nc,x,1\na,y,0\nb,y,0\nc,y,1\n")
    code, out, _ = run_cli(capsys, "compare", single, "--system-col", "system")
    assert code == 0
    assert "cannot be split or planned" in out


def test_simulate_writes_a_readable_file(tmp_path, capsys):
    code, out, _ = run_cli(capsys, "simulate", "--items", "5", "--runs", "3", "--seed", "2")
    assert code == 0
    lines = out.strip().splitlines()
    assert lines[0] == "item,score"
    assert len(lines) == 16
    assert set(line.split(",")[1] for line in lines[1:]) <= {"0", "1"}
    path = tmp_path / "sim.csv"
    path.write_text(out)
    scores = read_long(path)["all"]
    assert scores.n_items == 5 and scores.n_obs == 15 and not scores.has_run_labels


def test_simulate_coverage(capsys):
    args = ["simulate", "--items", "30", "--runs", "3", "--coverage", "20", "--bootstrap", "40"]
    code, out, _ = run_cli(capsys, *args)
    assert code == 0
    assert "# Interval coverage" in out and "| bootstrap: mean |" in out
    code, out, _ = run_cli(capsys, *args, "--format", "json")
    rates = json.loads(out)["coverage"]
    assert all(0.0 <= rate <= 1.0 for rate in rates.values())
    code, _, err = run_cli(capsys, *args, "--kind", "normal")
    assert code == 2 and "0/1" in err
