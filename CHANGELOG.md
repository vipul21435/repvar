# Changelog

## 0.1.0

First release.

- Long JSONL, CSV and TSV input with item, score and optional run and system columns; unbalanced designs;
  `--first-runs K` for a balanced subset.
- Variance components by ANOVA for nested designs (item, residual; `k0` for unequal runs) and crossed designs
  (item, run, residual; missing cells allowed).
- Standard error of a mean score for any number of items and runs, split into an item part and a rerun part.
- Intraclass correlation with F intervals, reliability of a k-run item score, and the share of items solved
  always, sometimes and never.
- Beta-binomial fit for 0/1 scores with its likelihood and a likelihood-ratio comparison against a single
  shared pass probability.
- Decision study: standard error of every split of a budget, the best split, and the number of repeats after
  which more repeats stop helping.
- Comparison of two systems on shared items with independent runs: standard error of the difference, its
  item and rerun parts, and the smallest detectable difference.
- Percentile bootstrap over items, items and runs, or runs.
- Simulators with planted components and a coverage check for the intervals.
- Commands `components`, `design`, `compare` and `simulate`, Markdown and JSON output, optional chart of
  standard error against runs per item.
