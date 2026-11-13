"""Command line interface: ``repvar <command> FILE``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__, report
from .components import NotIdentifiableError, variance_components
from .data import Scores, read_long
from .design import mean_se


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
    parser.add_argument("--format", choices=["md", "json"], default="md", help="output format")
    parser.add_argument("--output", type=Path, default=None, help="write here instead of stdout")


def _load(args: argparse.Namespace) -> dict[str, Scores]:
    return read_long(
        args.file,
        item=args.item_col,
        run=args.run_col,
        score=args.score_col,
        system=args.system_col,
        fmt=args.input_format,
    )


def _emit(args: argparse.Namespace, text: str) -> None:
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def cmd_components(args: argparse.Namespace) -> int:
    groups = _load(args)
    results = {name: variance_components(scores, args.design) for name, scores in groups.items()}
    if args.format == "json":
        payload = {}
        for name, comp in results.items():
            entry = comp.to_dict()
            k = max(1, round(comp.runs_per_item))
            entry["se_one_run"] = mean_se(comp, comp.n_items, 1).to_dict()
            entry["se_observed"] = mean_se(comp, comp.n_items, k).to_dict()
            payload[name] = entry
        _emit(args, report.to_json(payload))
    else:
        parts = ["# Variance components"]
        parts += [report.components_markdown(name, comp) for name, comp in results.items()]
        _emit(args, "\n\n".join(parts))
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
    comp.set_defaults(func=cmd_components)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (NotIdentifiableError, ValueError, OSError) as exc:
        print(f"repvar: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
