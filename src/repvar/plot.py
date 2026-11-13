"""Chart of the standard error of a mean score against the number of runs per item."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from .components import Components
from .design import mean_se

#: Series colours in a fixed order; a system keeps its colour whatever else is drawn.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"


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
    if max_runs < 2:
        raise ValueError("max_runs must be at least 2")
    if len(curves) > len(SERIES):
        raise ValueError(f"the chart takes at most {len(SERIES)} systems")
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("the chart needs matplotlib: pip install matplotlib") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = list(range(1, max_runs + 1))
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for color, (name, comp) in zip(SERIES, curves.items(), strict=False):
        n = n_items if n_items is not None else comp.n_items
        ses = [mean_se(comp, n, k).se for k in runs]
        ax.plot(runs, ses, color=color, linewidth=2, marker="o", markersize=5, label=name)
        ax.axhline(math.sqrt(comp.item / n), color=color, linewidth=1, linestyle=(0, (4, 3)))
        if len(curves) <= 4:
            ax.annotate(
                f"{name} ({n} items)",
                (runs[-1], ses[-1]),
                xytext=(6, 5),
                textcoords="offset points",
                color=INK,
                fontsize=8,
            )
    ax.set_xlabel("runs per item", color=MUTED)
    ax.set_ylabel("standard error of the mean score", color=MUTED)
    ax.set_title(
        "Standard error against repeats (dashed: the floor the items set)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    ax.set_xticks(runs)
    ax.set_xlim(0.6, max_runs + 0.22 * max_runs + 0.6)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")
    if len(curves) >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
