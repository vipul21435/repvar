# repvar

From repeated runs of a stochastic system on the same benchmark items, separate the variance that comes from which
items were chosen from the variance that comes from rerunning.

This is an early version. The list below says what is implemented; everything else is still to come.

## What works so far

- Reading long JSONL, CSV or TSV files with one row per (item, run): an item column, a score column (0/1, true/false
  or a real number), an optional run column and an optional system column. Items may have different numbers of runs.
- Variance components by the analysis of variance:
  - nested design (the runs of one item are unrelated to the runs of another): item and residual variance from the
    one-way random-effects table, with the usual `k0` divisor when items have unequal numbers of runs;
  - crossed design (run `j` is one pass over all items): item, run and residual variance from the two-way table
    without replication; with missing cells the same sums of squares are equated to their expectations for the cells
    that are present.
- The standard error of a mean score over `n` items with `k` runs each, `item / n + run / k + residual / (n k)`,
  split into the part due to items and the part due to reruns.
- `repvar components FILE`, with Markdown or JSON output.

Not implemented yet: the beta-binomial fit, the intraclass correlation and its interval, the decision study,
comparison of two systems, bootstrap intervals, the simulator and the chart.

## Install

```
pip install git+https://github.com/vipul21435/repvar
```

Python 3.10 or later; depends on numpy and scipy.

## Example

`examples/runs.csv` is a small made-up table: 8 items, 4 runs each, 0/1 scores.

```
$ repvar components examples/runs.csv
# Variance components

## all

design: crossed (each run is one pass over the items); method: two-way ANOVA
items: 8; scores: 32; runs per item: 4.00 (balanced); mean score: 0.5625

| component | variance | share |
| --- | --- | --- |
| item | 0.09524 | 35.6 % |
| run | 0.00000 | 0.0 % |
| residual | 0.17262 | 64.4 % |

| design | standard error | item part | rerun part |
| --- | --- | --- | --- |
| 8 items x 1 run(s) | 0.1830 | 0.1091 | 0.1469 |
| 8 items x 4 run(s) | 0.1315 | 0.1091 | 0.0734 |

negative estimate set to zero: run
```

The two "part" columns are standard errors; their squares add up to the square of the total.

## Development

```
uv run --extra dev ruff check
uv run --extra dev pytest -q
```

## Licence

MIT. Copyright (c) 2026 Vipul Raj Jha.

Built by Vipul Raj Jha · https://vipul21435.github.io
