"""Markdown and JSON rendering of results."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence

from .components import CROSSED, Components
from .design import DesignSE, mean_se


def fmt(value: float, digits: int = 4) -> str:
    """Fixed-point number, or ``n/a`` for NaN and ``inf`` for infinity."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{100.0 * value:.{digits}f} %"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def to_json(payload: object) -> str:
    """JSON text with NaN and infinity written as null (strict JSON has neither)."""

    def clean(obj: object) -> object:
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        if isinstance(obj, Mapping):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        return obj

    return json.dumps(clean(payload), indent=2)


def design_line(comp: Components) -> str:
    if comp.design == CROSSED:
        kind = "crossed (each run is one pass over the items)"
    else:
        kind = "nested (runs independent within items)"
    return f"design: {kind}; method: {comp.method}"


def se_row(d: DesignSE) -> list[str]:
    label = f"{d.n_items} items x {d.n_runs} run(s)"
    return [label, fmt(d.se), fmt(d.se_item), fmt(d.se_rerun)]


def components_markdown(name: str, comp: Components) -> str:
    """The variance components of one system and the standard error they imply."""
    shares = comp.shares()
    rows = [["item", fmt(comp.item, 5), pct(shares["item"])]]
    if comp.design == CROSSED:
        rows.append(["run", fmt(comp.run, 5), pct(shares["run"])])
    rows.append(["residual", fmt(comp.residual, 5), pct(shares["residual"])])
    k = max(1, round(comp.runs_per_item))
    one = mean_se(comp, comp.n_items, 1)
    obs = mean_se(comp, comp.n_items, k)
    shape = "balanced" if comp.balanced else "unbalanced"
    lines = [
        f"## {name}",
        "",
        design_line(comp),
        f"items: {comp.n_items}; scores: {comp.n_obs}; runs per item: "
        f"{fmt(comp.runs_per_item, 2)} ({shape}); mean score: {fmt(comp.mean)}",
        "",
        table(["component", "variance", "share"], rows),
        "",
        table(
            ["design", "standard error", "item part", "rerun part"],
            [se_row(d) for d in ([one] if k == 1 else [one, obs])],
        ),
    ]
    negative = [
        label for label, raw in (("item", comp.item_raw), ("run", comp.run_raw)) if raw < 0.0
    ]
    if negative:
        lines += ["", f"negative estimate set to zero: {', '.join(negative)}"]
    return "\n".join(lines)
