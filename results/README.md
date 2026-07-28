# Result tables

The repository includes compact numerical tables for the three experiments
reported in the paper. Raw checkpoints and training logs are omitted because
of their size.

## Stage I: backbone selection

`stage1/per_run_metrics.csv` contains the 15 independently trained
populations: five methods and three runs per method. All checkpoints were
reevaluated with the same four 96-step fresh-seed rollouts.
`stage1/aggregate_metrics.csv` reports means and sample standard deviations.

## Stage II: cooperation mechanism

`stage2/per_run_metrics.csv` contains the 12 populations in the 2 x 2
factorial comparison: weighted-sum or MGDA local learning, crossed with
neighborhood or region-archive cooperation. Seeds are 11, 22, and 33.
`stage2/aggregate_metrics.csv` contains the corresponding summaries.

## Stage III: coverage and concentration

`stage3/per_run_metrics.csv` contains 60 independent runs: six semantic
pairs, two methods, and five paired seeds. The statistical replicate is one
trained population, not one member of its 30-solution final population.

`stage3/aggregate_metrics.csv` gives task-level means and sample standard
deviations. `stage3/paired_differences.csv` records MCA-minus-CA differences
for matching seeds, and `stage3/direction_counts.csv` summarizes their
directions. The Cactus--Maple rows use the task-specific prompts
`a tall green cactus` and `a red maple tree`.

All path-bearing fields were removed from the published CSV files.
