"""Command line interface: ``repvar <command> ...``."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__, report
from .compare import compare, detectable_difference
from .components import variance_components
from .data import Scores, read_long
from .design import best_split, budget_splits, repeats_to_plateau
from .simulate import interval_coverage, simulate_binary, simulate_normal


def _add_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("file", type=Path, help="long JSONL or CSV file, one row per (item, run)")
    parser.add_argument("--item-col", default="item", help="item column (default: item)")
    parser.add_argument(
        "--run-col",
        default=None,
        help="run column (default: 'run' when present, otherwise runs are unlabelled)",
    )
    parser.add_argument("--score-col", default="score", help="score column (default: score)")
    parser.add_argument("--system-col", default=None, help="column naming the system, if any")
    parser.add_argument(
        "--input-format", choices=["jsonl", "csv", "tsv"], default=None, help="override the suffix"
    )
    parser.add_argument(
        "--design",
        choices=["auto", "nested", "crossed"],
        default="auto",
        help="nested: runs independent within items; crossed: run j is one pass over all "
        "items; auto: crossed when run labels are shared by items (default)",
    )
    parser.add_argument(
        "--first-runs",
        type=int,
        default=None,
        metavar="K",
        help="balance the design: keep the first K scores of every item that has K or more",
    )
    _add_output(parser)


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=["md", "json"], default="md", help="output format")
    parser.add_argument("--output", type=Path, default=None, help="write here instead of stdout")


def _load(args: argparse.Namespace) -> dict[str, Scores]:
    groups = read_long(
        args.file,
        item=args.item_col,
        run=args.run_col,
        score=args.score_col,
        system=args.system_col,
        fmt=args.input_format,
    )
    if args.first_runs is not None:
        groups = {name: scores.first_runs(args.first_runs) for name, scores in groups.items()}
    return groups


def _emit(args: argparse.Namespace, text: str) -> None:
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def cmd_components(args: argparse.Namespace) -> int:
    results = {
        name: report.analyse(
            scores,
            args.design,
            level=args.level,
            n_boot=args.bootstrap,
            resample=args.resample,
            seed=args.seed,
        )
        for name, scores in _load(args).items()
    }
    if args.format == "json":
        _emit(args, report.to_json({name: res.to_dict() for name, res in results.items()}))
    else:
        parts = ["# Variance components"]
        parts += [report.components_markdown(name, res) for name, res in results.items()]
        _emit(args, "\n\n".join(parts))
    return 0


def cmd_design(args: argparse.Namespace) -> int:
    comps = {name: variance_components(s, args.design) for name, s in _load(args).items()}
    payload = {}
    parts = ["# Decision study"]
    for name, comp in comps.items():
        budget = args.budget if args.budget is not None else float(comp.n_obs)
        n_items = args.items if args.items is not None else comp.n_items
        splits = budget_splits(
            comp,
            budget,
            max_items=args.max_items,
            max_runs=args.max_runs,
            item_cost=args.item_cost,
        )
        best = best_split(splits, args.fixed_items)
        payload[name] = {
            "components": comp.to_dict(),
            "budget": budget,
            "splits": [s.to_dict() for s in splits],
            "best": best.to_dict(),
            "plateau_items": n_items,
            "plateau_runs": repeats_to_plateau(comp, n_items, args.tolerance),
            "tolerance": args.tolerance,
        }
        parts.append(
            report.design_markdown(
                name,
                comp,
                splits,
                budget=budget,
                n_items=n_items,
                tolerance=args.tolerance,
                fixed_items=args.fixed_items,
            )
        )
    if args.chart is not None:
        from .plot import se_chart

        se_chart(comps, args.chart, n_items=args.items, max_runs=max(2, args.max_runs))
    _emit(args, report.to_json(payload) if args.format == "json" else "\n\n".join(parts))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    if args.system_col is None:
        raise ValueError("compare needs --system-col")
    groups = _load(args)
    if args.a is None and args.b is None and len(groups) == 2:
        name_a, name_b = list(groups)
    elif args.a is not None and args.b is not None:
        name_a, name_b = args.a, args.b
    else:
        raise ValueError(f"name two systems with --a and --b; found: {', '.join(groups)}")
    for name in (name_a, name_b):
        if name not in groups:
            raise ValueError(f"no system {name!r}; found: {', '.join(groups)}")
    result = compare(
        groups[name_a],
        groups[name_b],
        design=args.design,
        level=args.level,
        names=(name_a, name_b),
    )
    n_items = args.items if args.items is not None else result.n_items
    if args.format == "json":
        payload = result.to_dict()
        payload["detectable"] = {
            "alpha": args.alpha,
            "power": args.power,
            "n_items": n_items,
            "observed_design": detectable_difference(result.se, args.alpha, args.power),
        }
        _emit(args, report.to_json(payload))
    else:
        _emit(
            args,
            report.compare_markdown(result, alpha=args.alpha, power=args.power, n_items=n_items),
        )
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    if args.coverage:
        if args.kind != "binary":
            raise ValueError("--coverage simulates 0/1 scores; drop --kind normal")
        rates = interval_coverage(
            args.items,
            args.runs,
            mean=args.mean,
            icc_true=args.icc,
            reps=args.coverage,
            n_boot=args.bootstrap,
            level=args.level,
            resample=args.resample,
            seed=args.seed,
        )
        if args.format == "json":
            _emit(args, report.to_json({"level": args.level, "coverage": rates}))
        else:
            head = (
                f"# Interval coverage\n\n{args.coverage} simulated data sets of {args.items} "
                f"items x {args.runs} runs (mean {args.mean:g}, icc {args.icc:g}); nominal "
                f"level {report.pct(args.level, 0)}; bootstrap over {args.resample}, "
                f"{args.bootstrap} replicates\n\n"
            )
            rows = [[name, report.pct(rate)] for name, rate in rates.items()]
            _emit(args, head + report.table(["interval", "covered the planted value"], rows))
        return 0
    if args.kind == "binary":
        scores = simulate_binary(
            args.items, args.runs, mean=args.mean, icc=args.icc, seed=args.seed
        )
    else:
        scores = simulate_normal(
            args.items,
            args.runs,
            mean=args.mean,
            item_sd=args.item_sd,
            run_sd=args.run_sd,
            residual_sd=args.residual_sd,
            seed=args.seed,
        )
    table, runs = scores.matrix()
    handle = args.output.open("w", encoding="utf-8", newline="") if args.output else sys.stdout
    try:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["item", "run", "score"] if scores.has_run_labels else ["item", "score"])
        for i, row in enumerate(table):
            for run, value in zip(runs, row, strict=True):
                cell = int(value) if args.kind == "binary" else repr(float(value))
                writer.writerow(
                    [f"item-{i + 1}"] + ([run] if scores.has_run_labels else []) + [cell]
                )
    finally:
        if args.output:
            handle.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repvar",
        description="Separate item variance from rerun variance in repeated benchmark runs.",
    )
    parser.add_argument("--version", action="version", version=f"repvar {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    comp = sub.add_parser("components", help="variance components and the standard error")
    _add_input(comp)
    comp.add_argument("--level", type=float, default=0.95, help="interval level (default 0.95)")
    comp.add_argument(
        "--bootstrap", type=int, default=0, metavar="N", help="add N-replicate bootstrap intervals"
    )
    comp.add_argument(
        "--resample",
        choices=["items", "items+runs", "runs"],
        default="items",
        help="what the bootstrap redraws (default: items)",
    )
    comp.add_argument("--seed", type=int, default=0, help="bootstrap seed (default 0)")
    comp.set_defaults(func=cmd_components)

    des = sub.add_parser("design", help="standard error for every split of a budget")
    _add_input(des)
    des.add_argument(
        "--budget", type=float, default=None, help="item-runs to spend (default: as observed)"
    )
    des.add_argument("--max-items", type=int, default=None, help="cap on the number of items")
    des.add_argument("--max-runs", type=int, default=10, help="most runs per item (default 10)")
    des.add_argument(
        "--item-cost", type=float, default=0.0, help="one-off cost of an item, in runs (default 0)"
    )
    des.add_argument(
        "--fixed-items", action="store_true", help="pick the split by its rerun part alone"
    )
    des.add_argument(
        "--items", type=int, default=None, help="items for the plateau (default: as observed)"
    )
    des.add_argument(
        "--tolerance", type=float, default=0.05, help="plateau: within this of the floor (0.05)"
    )
    des.add_argument("--chart", type=Path, default=None, help="save a PNG of SE against runs")
    des.set_defaults(func=cmd_design)

    cmp_ = sub.add_parser("compare", help="difference of two systems on shared items")
    _add_input(cmp_)
    cmp_.add_argument("--a", default=None, help="first system")
    cmp_.add_argument("--b", default=None, help="second system")
    cmp_.add_argument("--level", type=float, default=0.95, help="interval level (default 0.95)")
    cmp_.add_argument("--alpha", type=float, default=0.05, help="two-sided test size (0.05)")
    cmp_.add_argument("--power", type=float, default=0.8, help="power for detection (0.8)")
    cmp_.add_argument(
        "--items", type=int, default=None, help="shared items to plan for (default: as observed)"
    )
    cmp_.set_defaults(func=cmd_compare)

    sim = sub.add_parser("simulate", help="write simulated scores, or check interval coverage")
    sim.add_argument("--items", type=int, default=200, help="number of items (default 200)")
    sim.add_argument("--runs", type=int, default=5, help="runs per item (default 5)")
    sim.add_argument("--kind", choices=["binary", "normal"], default="binary")
    sim.add_argument("--mean", type=float, default=0.5, help="mean score (default 0.5)")
    sim.add_argument("--icc", type=float, default=0.5, help="binary: intraclass correlation")
    sim.add_argument("--item-sd", type=float, default=1.0, help="normal: item sd")
    sim.add_argument("--run-sd", type=float, default=0.0, help="normal: run sd (crossed if > 0)")
    sim.add_argument("--residual-sd", type=float, default=1.0, help="normal: residual sd")
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument(
        "--coverage",
        type=int,
        default=0,
        metavar="REPS",
        help="instead of writing data, simulate REPS data sets and report interval coverage",
    )
    sim.add_argument("--bootstrap", type=int, default=200, help="coverage: bootstrap replicates")
    sim.add_argument("--resample", choices=["items", "items+runs", "runs"], default="items")
    sim.add_argument("--level", type=float, default=0.95, help="coverage: interval level")
    _add_output(sim)
    sim.set_defaults(func=cmd_simulate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"repvar: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
