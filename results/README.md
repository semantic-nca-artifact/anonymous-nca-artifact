# Result tables

The repository includes compact numerical tables for the three experiments
reported in the paper. These files were derived from completed training
records; they are not synthetic examples. Absolute local paths and checkpoint
fields were removed for anonymous distribution.

## Stage I: backbone selection

`stage1/per_run_metrics.csv` contains the 15 independently trained
populations: five methods and three runs per method. All checkpoints were
reevaluated with the same four 96-step fresh-seed rollouts.
`stage1/aggregate_metrics.csv` reports means and sample standard deviations.

The downloaded records for Weighted Sum, Tchebycheff, EPO, and MOO-SVGD did
not retain their original seed identifiers. Their `seed` cells are deliberately
empty and their replicates must not be interpreted as seed-matched runs. The
three MOEA/D-CA seeds are recorded as 11, 22, and 33.

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

The corresponding final Stage-III model populations are distributed under
`../checkpoints/stage3`. Their compact `summary.json` files contain the 30
canonical objective vectors from which these run-level metrics are computed.
Run directories are mapped in `../checkpoints/stage3/manifest.csv`; individual
population slots resolve through `../checkpoints/stage3/model_index.csv`. Both
metadata and unique model files are covered by the SHA-256 checksum inventory.

## Columns and statistics

The per-run tables use one independently trained population per row. `HV` is
exact two-dimensional maximization hypervolume with reference `(0, 0)`. `ND`
is the number of non-dominated members. `Spacing` is the sample standard
deviation of nearest-neighbor distances on the non-dominated set.

Stage III additionally reports `population_average_worst_similarity`
(`B_pop`), `best_max_min_similarity` (`B_best`), and
`objective_space_extent` (`E_obj`). See the main README for their formulas and
interpretation. Aggregate `*_std` fields are sample standard deviations across
independently trained populations. Paired differences are defined as MCA minus
CA for identical seeds.
