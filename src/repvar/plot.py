"""Chart of the standard error of a mean score against the number of runs per item."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from .components import Components
from .design import mean_se


def se_chart(
    curves: Mapping[str, Components],
    path: str | Path,
    *,
    n_items: int | None = None,
    max_runs: int = 10,
) -> Path:
    """Draw standard error against runs per item for each system and save it to ``path``.

    Solid lines are the standard error of the mean over ``n_items`` items (default: the
    number each system was estimated from); dashed lines are the floor ``sqrt(item / n)``
    that more runs cannot get under. Needs matplotlib (``pip install repvar[plot]``).
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("the chart needs matplotlib: pip install matplotlib") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if max_runs < 2:
        raise ValueError("max_runs must be at least 2")
    runs = list(range(1, max_runs + 1))
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e5484d", "#7a5af8"]
    for position, (name, comp) in enumerate(curves.items()):
        n = n_items if n_items is not None else comp.n_items
        color = colors[position % len(colors)]
        ses = [mean_se(comp, n, k).se for k in runs]
        ax.plot(
            runs,
            ses,
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"{name} ({n} items)",
        )
        floor = math.sqrt(comp.item / n)
        ax.axhline(floor, color=color, linewidth=1, linestyle="--", alpha=0.8)
    ax.set_xlabel("runs per item")
    ax.set_ylabel("standard error of the mean score")
    ax.set_title("Standard error against repeats (dashed: floor set by the items)", fontsize=10)
    ax.set_xticks(runs)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path
