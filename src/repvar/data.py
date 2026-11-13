"""Repeated-run scores in long form: one row per (item, run)."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_TRUE = {"true", "t", "yes", "y", "pass", "passed", "resolved"}
_FALSE = {"false", "f", "no", "n", "fail", "failed", "unresolved"}

#: Key used for the single group when the input has no system column.
ALL = "all"


def parse_score(value: object) -> float:
    """Turn a cell into a float score; booleans and pass/fail words become 1 or 0."""
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return 1.0
        if text in _FALSE:
            return 0.0
        try:
            out = float(text)
        except ValueError:
            raise ValueError(f"cannot read score {value!r}") from None
    else:
        raise ValueError(f"cannot read score {value!r}")
    if not math.isfinite(out):
        raise ValueError(f"score is not finite: {value!r}")
    return out


@dataclass(frozen=True)
class Scores:
    """Scores of one system, one entry per observation.

    ``item_index[m]`` is the position in ``item_ids`` of the item that observation ``m``
    belongs to, ``run_labels[m]`` its run label and ``values[m]`` its score.
    ``has_run_labels`` is False when the input had no run column; the labels are then
    counters within each item (or absent) and carry no meaning across items.
    """

    item_ids: tuple[str, ...]
    item_index: np.ndarray
    run_labels: tuple[str, ...]
    values: np.ndarray
    system: str | None = None
    has_run_labels: bool = True
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_records(
        cls,
        records: Iterable[tuple[object, object, object]],
        system: str | None = None,
    ) -> Scores:
        """Build from ``(item, run, score)`` triples; pass ``run=None`` for unlabelled runs."""
        ids: dict[str, int] = {}
        seen: dict[int, int] = {}
        index: list[int] = []
        labels: list[str] = []
        values: list[float] = []
        labelled = True
        for item, run, score in records:
            key = str(item)
            pos = ids.setdefault(key, len(ids))
            seen[pos] = seen.get(pos, 0) + 1
            if run is None:
                labelled = False
                labels.append(str(seen[pos]))
            else:
                labels.append(str(run))
            index.append(pos)
            values.append(parse_score(score))
        if not values:
            raise ValueError("no observations")
        return cls(
            item_ids=tuple(ids),
            item_index=np.asarray(index, dtype=np.int64),
            run_labels=tuple(labels),
            values=np.asarray(values, dtype=float),
            system=system,
            has_run_labels=labelled,
        )

    @classmethod
    def from_lists(
        cls,
        per_item: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
        system: str | None = None,
    ) -> Scores:
        """Build from the scores of each item, runs unlabelled (independent within items)."""
        pairs = per_item.items() if isinstance(per_item, Mapping) else enumerate(per_item)
        return cls.from_records(
            ((item, None, score) for item, scores in pairs for score in scores), system=system
        )

    @classmethod
    def from_matrix(
        cls, matrix: object, system: str | None = None, labelled: bool = True
    ) -> Scores:
        """Build from an items-by-runs array; NaN marks a missing cell.

        Columns are runs, labelled by their position. ``labelled=False`` says the columns
        carry no meaning across items (runs independent within items). Rows and columns
        without any score are dropped.
        """
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            raise ValueError("matrix must be two-dimensional (items by runs)")
        present = ~np.isnan(arr)
        arr = arr[present.any(axis=1)][:, present.any(axis=0)]
        rows, cols = np.nonzero(~np.isnan(arr))
        if rows.size == 0:
            raise ValueError("no observations")
        out = cls(
            item_ids=tuple(str(i) for i in range(arr.shape[0])),
            item_index=rows.astype(np.int64),
            run_labels=tuple(str(c) for c in cols.tolist()) if labelled else (),
            values=arr[rows, cols],
            system=system,
            has_run_labels=labelled,
        )
        out._cache["matrix"] = (arr, tuple(str(c) for c in range(arr.shape[1])))
        return out

    # -- summaries ---------------------------------------------------------------

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_obs(self) -> int:
        return int(self.values.size)

    @property
    def counts(self) -> np.ndarray:
        """Number of runs of each item."""
        if "counts" not in self._cache:
            self._cache["counts"] = np.bincount(self.item_index, minlength=self.n_items)
        return self._cache["counts"]

    @property
    def sums(self) -> np.ndarray:
        """Sum of the scores of each item."""
        if "sums" not in self._cache:
            self._cache["sums"] = np.bincount(
                self.item_index, weights=self.values, minlength=self.n_items
            )
        return self._cache["sums"]

    @property
    def item_means(self) -> np.ndarray:
        return self.sums / self.counts

    @property
    def mean(self) -> float:
        """Mean of the item means: every item counts once, whatever its number of runs."""
        return float(np.mean(self.item_means))

    @property
    def is_binary(self) -> bool:
        return bool(np.all((self.values == 0.0) | (self.values == 1.0)))

    @property
    def balanced(self) -> bool:
        counts = self.counts
        return bool(np.all(counts == counts[0]))

    def first_runs(self, k: int) -> Scores:
        """Balanced subset: the first ``k`` scores, in input order, of each item.

        Items with fewer than ``k`` scores are dropped.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        counts = self.counts
        order = np.argsort(self.item_index, kind="stable")
        rank = np.empty(self.n_obs, dtype=np.int64)
        rank[order] = np.arange(self.n_obs) - np.repeat(np.cumsum(counts) - counts, counts)
        keep = (rank < k) & (counts[self.item_index] >= k)
        kept_items = np.flatnonzero(counts >= k)
        if kept_items.size == 0:
            raise ValueError(f"no item has {k} or more runs")
        remap = np.full(self.n_items, -1, dtype=np.int64)
        remap[kept_items] = np.arange(kept_items.size)
        labels: tuple[str, ...] = ()
        if self.has_run_labels:
            flags = keep.tolist()
            labels = tuple(lab for lab, ok in zip(self.run_labels, flags, strict=True) if ok)
        return Scores(
            item_ids=tuple(self.item_ids[i] for i in kept_items.tolist()),
            item_index=remap[self.item_index[keep]],
            run_labels=labels,
            values=self.values[keep],
            system=self.system,
            has_run_labels=self.has_run_labels,
        )

    def per_item(self) -> list[np.ndarray]:
        """Scores of each item, in ``item_ids`` order."""
        order = np.argsort(self.item_index, kind="stable")
        return np.split(self.values[order], np.cumsum(self.counts)[:-1])

    def matrix(self) -> tuple[np.ndarray, tuple[str, ...]]:
        """Items-by-runs array (NaN for missing cells) and the run label of each column."""
        if "matrix" in self._cache:
            return self._cache["matrix"]
        if not self.has_run_labels:
            raise ValueError("these scores have no run labels to cross with the items")
        runs: dict[str, int] = {}
        cols = np.asarray([runs.setdefault(r, len(runs)) for r in self.run_labels])
        if np.any(np.bincount(self.item_index * len(runs) + cols) > 1):
            raise ValueError("an item has two scores under the same run label")
        out = np.full((self.n_items, len(runs)), np.nan)
        out[self.item_index, cols] = self.values
        return out, tuple(runs)


def _rows(path: Path, fmt: str | None) -> Iterable[Mapping[str, object]]:
    kind = fmt or path.suffix.lower().lstrip(".")
    if kind in {"jsonl", "ndjson", "json"}:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name} line {number}: not valid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path.name} line {number}: expected a JSON object")
                yield row
    elif kind in {"csv", "tsv"}:
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle, delimiter="\t" if kind == "tsv" else ",")
    else:
        raise ValueError(f"unknown format {kind!r}: use a .jsonl, .csv or .tsv file")


def read_long(
    path: str | Path,
    *,
    item: str = "item",
    run: str | None = None,
    score: str = "score",
    system: str | None = None,
    fmt: str | None = None,
) -> dict[str, Scores]:
    """Read a long JSONL or CSV file into one :class:`Scores` per system.

    ``run=None`` uses a column named ``run`` when the file has one and otherwise treats the
    runs of each item as unlabelled. Without ``system`` the result has the single key
    ``"all"``. Rows whose score cell is empty or null are skipped.
    """
    path = Path(path)
    groups: dict[str, list[tuple[object, object, object]]] = {}
    run_col = run if run is not None else "run"
    for number, row in enumerate(_rows(path, fmt), start=1):
        for needed in (item, score) + ((system,) if system else ()):
            if needed not in row:
                raise ValueError(f"{path.name} row {number}: no column {needed!r}")
        if run is not None and run not in row:
            raise ValueError(f"{path.name} row {number}: no column {run!r}")
        cell = row[score]
        if cell is None or (isinstance(cell, str) and not cell.strip()):
            continue
        name = str(row[system]) if system else ALL
        groups.setdefault(name, []).append((row[item], row.get(run_col), cell))
    if not groups:
        raise ValueError(f"{path.name}: no scored rows")
    out: dict[str, Scores] = {}
    for name, records in groups.items():
        if any(r is None for _, r, _ in records):
            records = [(i, None, s) for i, _, s in records]
        out[name] = Scores.from_records(records, system=name if system else None)
    return out
