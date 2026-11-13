"""Fetch three columns of nebius/SWE-agent-trajectories into a long CSV.

The dataset is about 1.1 GB of parquet because every row carries a full trajectory. Only
``instance_id``, ``model_name`` and ``target`` are needed here, so the parquet files are
opened over HTTP and just those column chunks are read.

    uv run --with pyarrow --with huggingface_hub python scripts/fetch_swe_agent_trajectories.py \
        --out swe_agent_runs.csv

The output has the columns ``item,system,score`` (one row per trajectory, in dataset
order) and is not meant to be committed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DATASET = "nebius/SWE-agent-trajectories"
COLUMNS = ["instance_id", "model_name", "target"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="CSV file to write")
    parser.add_argument("--revision", default="main", help="dataset revision (default: main)")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    info = HfApi().dataset_info(DATASET, revision=args.revision)
    files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet"))
    fs = HfFileSystem()
    rows = 0
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "system", "score"])
        for name in files:
            with fs.open(f"datasets/{DATASET}/{name}", revision=info.sha) as remote:
                table = pq.ParquetFile(remote).read(columns=COLUMNS)
            for item, system, target in zip(*(table[c].to_pylist() for c in COLUMNS), strict=True):
                writer.writerow([item, system, int(bool(target))])
                rows += 1
            print(f"{name}: {table.num_rows} rows")
    print(f"revision {info.sha}: wrote {rows} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
