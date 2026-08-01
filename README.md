# Growing Forms Between Concepts

Neural cellular automata (NCAs) are usually trained to grow one prescribed
target. This artifact studies a different question: can a population of local
developmental programs expose a range of viable forms between two concepts
specified in natural language?

We formulate this problem as language-defined Pareto morphospace exploration.
Each NCA develops from the same central seed, while a frozen CLIP ViT-B/32
encoder scores the resulting phenotype against two text prompts. Thirty
preference-indexed programs explore different semantic trade-offs. Persistent
state pools support developmental training, and synchronized fresh-seed
rollouts separate training state from the objectives used for comparison.

The study distinguishes two reproducible population geometries. MOEA/D-CA uses
a fixed-preference weighted-sum gradient and tends to expose a broader semantic
front. MOEA/D-MCA changes only the local gradient construction, combining the
two objective gradients with Frank--Wolfe MGDA; it tends to concentrate more of
the population around forms that remain jointly aligned with both prompts. The
comparison is therefore about coverage and concentration, not a claim that one
geometry is universally preferable.

## Experimental questions

The experiments form a controlled three-stage evidence chain.

| Stage | Question | Comparison | Tasks and replicates |
| --- | --- | --- | --- |
| I | Does a decomposition population provide a strong search substrate? | Weighted Sum, Tchebycheff, EPO, MOO-SVGD, MOEA/D-CA | Butterfly--Caterpillar; three independent runs per method |
| II | Which local-learning and cooperation factors preserve Pareto coverage? | CA, RTA, MCA, and MRTA in a 2 x 2 design | Butterfly--Caterpillar; seeds 11, 22, and 33 |
| III | Does local-gradient construction control population geometry across tasks? | MOEA/D-CA versus MOEA/D-MCA | Six semantic pairs; paired seeds 11, 22, 33, 44, and 55 |

Stage II crosses weighted-sum or MGDA local learning with one-to-one
neighborhood propagation or region-wise archive transfer. Stage III holds the
population, evaluator, persistent-pool protocol, and neighborhood cooperation
fixed; only the local-gradient rule changes. Exact Stage-III prompts are listed
in `configs/semantic_pairs.tsv`.

## Experimental protocol

Every method in Stage I exposes 30 active NCA programs. Each program has a
persistent pool of 1,024 states and receives one Adam update from a batch of
eight states at every outer step. Developmental rollouts are sampled uniformly
from 64 through 95 steps. Formal runs use 2,000 outer updates, learning rate
`5e-4`, and gradient clipping at norm 1.0.

The MOEA/D experiments use 30 uniformly spaced bi-objective preferences and
five-member weight-space neighborhoods. Synchronized evaluation advances all
candidates for 64 steps from the same fresh central seed every ten outer
updates. Cooperation occurs every 50 updates, uses a 3% relative-improvement
gate, and is resolved by global maximum-weight one-to-one matching. Final
objectives average four 96-step fresh-seed rollouts with common random numbers
within each comparison.

Each seed initializes Python, NumPy, PyTorch, and CUDA random-number generators.
One independently trained 30-member population is the statistical replicate;
the individual programs within a population are not treated as independent
observations.

## Main findings represented by the artifact

Stage I selects MOEA/D-CA by mean hypervolume under a common fresh-seed
checkpoint reevaluation. Stage II isolates the effects of local-gradient and
cooperation policies. Across the 30 paired Stage-III comparisons, CA has higher
hypervolume in 28 runs and greater objective-space extent in all 30, whereas
MCA has higher population-wide joint alignment in all 30. The best single
max--min compromise is nearly evenly divided (13 CA, 17 MCA), indicating that
the principal effect is population geometry rather than an isolated champion.

These statements are computed from the corrected task-specific prompts in
`configs/semantic_pairs.tsv`, including `a tall green cactus` and
`a red maple tree` for Cactus--Maple.

## Environment

The recorded execution environment used Ubuntu 22.04.5, Python 3.10, PyTorch 2.7.1 with CUDA
12.8, Transformers 4.47.1, and one NVIDIA RTX 4090 GPU with 24 GB memory per
process. Each job was allocated 14 Intel Xeon Gold 6530 CPU cores and 112 GB of
host memory. Training requires no task-specific image dataset.

The pinned Python dependencies in `requirements.txt` define the artifact
reproduction environment. Create it as follows:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/check_environment.py --require-cuda
```

On Windows, activate the environment with `.venv\Scripts\activate`. Formal
training is intended for CUDA; matrix generation and table inspection do not
require a GPU.

The frozen encoder is `openai/clip-vit-base-patch32` at Hugging Face revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Transformers downloads that
revision when no local model directory is present. Set `CLIP_MODEL_NAME` to a
compatible local snapshot for offline execution. `CLIP_MODEL_REVISION` may be
overridden only when intentionally evaluating another model revision.

### Why EPO fails fast

EPO obtains its update coefficients by solving a constrained optimization
problem with CVXPY and CLARABEL. Replacing a failed solve with normalized
preference weights produces a valid gradient update, but it is no longer the
reported EPO algorithm. Formal EPO runs therefore verify the solver before CLIP
is loaded and stop on any failed or infeasible solve. This check applies on the
machine that executes EPO; it does not depend on the software installed on the
machine used to inspect this repository.

`--allow-epo-fallback` is provided only for diagnostic smoke tests. A run made
with that flag must not be included in the Stage-I comparison. EPO writes
`solver_diagnostics.json`; a formal run must report zero fallbacks.

## Reproduction workflow

### 1. Inspect the experiment matrix

The launcher is non-executing by default:

```bash
python scripts/run_experiments.py
```

It prints 87 independent commands: 15 for Stage I, 12 for Stage II, and 60 for
Stage III. Use `--stage`, `--methods`, `--pairs`, and `--seeds` to select a
subset. The printed commands can be submitted as one-GPU jobs to a cluster
scheduler.

### 2. Verify the released Stage-III models

The final 30-program population from every reported Stage-III run is included
under `checkpoints/stage3`. Verify the 1,800 model files, run metadata, prompt
assignments, and SHA-256 checksums without loading PyTorch:

```bash
python scripts/verify_stage3_checkpoints.py
```

The released `summary.json` files contain the prompt-normalized canonical
objective vectors used by `results/stage3`. They are compact evaluation
records, not training logs. In particular, the Cactus--Maple summaries record
the final `a tall green cactus` evaluation rather than an earlier cached
prompt evaluation.

### 3. Run a short pipeline check

The following command checks model loading, training, evaluation, and final
model serialization for one population. Five updates are not scientifically
meaningful and will not reproduce a reported result.

```bash
python scripts/run_experiments.py \
  --stage 2 \
  --methods moead_mca \
  --seeds 11 \
  --steps 5 \
  --light-output \
  --execute
```

Successful completion produces `run_manifest.json`, an `mca/summary.json`, and
30 final model files under the selected run directory.

### 4. Run the formal experiments

Execute each stage sequentially only if that is appropriate for the available
compute. On a cluster, print the commands and schedule them independently
instead of using `--execute`.

```bash
python scripts/run_experiments.py --stage 1 --execute
python scripts/run_experiments.py --stage 2 --execute
python scripts/run_experiments.py --stage 3 --execute
```

The default output cadence matches the recorded experiments. To retain the
final models and summaries while suppressing pool figures and intermediate
checkpoints, add `--light-output`. This changes storage and diagnostic output,
not the optimization or evaluation protocol.

### 5. Aggregate completed runs

```bash
python scripts/analyze_results.py --stage 1 --metrics-only
python scripts/analyze_results.py --stage 2 --metrics-only
python scripts/analyze_results.py --stage 3 --metrics-only
```

Stage-I checkpoint methods are reevaluated with four shared 96-step rollouts.
Stage-II and Stage-III aggregation validates the final population size,
evaluation horizon, repeat count, seeds, and prompts stored with each run.

For logs produced before the normalized Stage-III directory names were
introduced, `aggregate_stage3_results.py` accepts explicit
`--butterfly-ca-root` and `--butterfly-mca-root` overrides. The script never
searches unrelated directories or selects runs by metric value.

To render the Stage-III overview from completed final checkpoints:

```bash
python scripts/analyze_results.py --stage 3 --build-main-figure
```

## Output structure

New runs use the following top-level layout:

```text
train_log/
|-- stage1/<method>/seed_<seed>/
|-- stage2/<method>/seed_<seed>/
|-- moead_ca_<semantic-pair>/seed_<seed>/
`-- moead_mca_<semantic-pair>/seed_<seed>/
```

Each run root contains `run_manifest.json`, which records the prompts, seed,
method, model revision, training horizon, and output cadence. MOEA/D runs add a
variant directory (`ca`, `rta`, `mca`, or `mrta`) containing `summary.json` and
30 `sub_XX_w[...]` directories. The summary records final canonical objective
vectors and the controlled mechanism settings. Each subproblem directory holds
`final.pt` and, unless disabled, periodic checkpoints and diagnostic figures.

Weighted Sum and Tchebycheff use 30 `w1_..._w2_...` directories; EPO uses 30
`pref_...` directories; MOO-SVGD uses 30 `particle_...` directories. The
analysis commands write stage-specific CSV files beneath
`train_log/analysis/` unless `--output` is supplied.

## Metrics

Let `s_ik` be the canonical CLIP similarity of final program `i` to prompt
`k`. All metrics use the final 30-member active population.

| Metric | Definition and interpretation |
| --- | --- |
| `HV` | Exact two-dimensional maximization hypervolume with reference `(0, 0)`; larger is better. |
| `ND` | Number of non-dominated members in the final population; larger indicates more exposed trade-offs. |
| `Spacing` | Sample standard deviation of nearest-neighbor distances on the non-dominated set; smaller indicates more even spacing, but does not measure front extent. |
| `B_pop` | Population mean of `min_k s_ik`; larger means more of the population remains jointly aligned with both prompts. |
| `B_best` | `max_i min_k s_ik`; larger identifies a stronger single max--min compromise. |
| `E_obj` | Sum, over objectives, of the population similarity range; larger indicates a broader objective-space extent. |

Aggregate tables report the arithmetic mean and sample standard deviation over
independently trained populations. Stage III additionally reports paired
MCA-minus-CA differences for identical training seeds.

## Included results and provenance

The run-level and aggregate tables in `results/stage1`, `results/stage2`, and
`results/stage3` are compact derivatives of the completed training records.
They retain objective metrics, prompts, run identities, evaluation settings,
and recorded seeds where available. Absolute paths and checkpoint fields were
removed for anonymous distribution. `results/README.md` describes every table.

The 60 final Stage-III populations are also distributed in
`checkpoints/stage3`: 30 final NCA programs for each task, method, and seed.
Each run includes an anonymous training manifest and a compact canonical
evaluation summary. `checkpoints/stage3/manifest.csv` maps runs to directories,
and `checkpoints/stage3/SHA256SUMS` protects the complete release payload.

The original working log corpus is not part of the artifact. It occupies about
37.6 GiB and mixes formal runs with superseded experiments, frequent pool
figures, intermediate checkpoints, and machine-specific paths. Uploading that
directory wholesale would not improve reproducibility and would obscure which
runs support the paper. The artifact instead includes only the final Stage-III
models and their prompt-normalized evaluation records. New final models can be
regenerated with `--light-output`.

The seed identifiers of the historical Stage-I Weighted Sum, Tchebycheff, EPO,
and MOO-SVGD runs were not retained in the downloaded records. Their rows are
therefore described as three independent runs, not as seed-matched replicas.
The launcher uses 11, 22, and 33 as a deterministic protocol for new Stage-I
reproductions. Stages II and III retain their reported seed identifiers.

## Repository map

- `main.py` runs one published training strategy and writes its manifest.
- `training/` contains the Stage-I baselines and the shared CA/RTA/MCA/MRTA implementation.
- `configs/semantic_pairs.tsv` records the six Stage-III prompt pairs.
- `scripts/run_experiments.py` constructs the formal experiment matrix.
- `scripts/analyze_results.py` dispatches stage-specific aggregation.
- `visualize_moead_results.py` implements common checkpoint evaluation and population metrics.
- `scripts/aggregate_stage3_results.py` validates and aggregates paired Stage-III runs.
- `scripts/build_stage3_main_figure.py` renders the Stage-III overview.
- `results/` contains the sanitized numerical record distributed with the artifact.
- `checkpoints/stage3/` contains all 1,800 final Stage-III NCA models.
- `scripts/verify_stage3_checkpoints.py` verifies model and metadata integrity.

## Artifact scope

This repository contains the code and compact numerical evidence needed to
inspect and reproduce the three-stage study. Exploratory methods, superseded
runs, manuscript sources, credentials, machine-specific paths, and bulk model
checkpoints are intentionally excluded. Final Stage-III weights are included;
periodic checkpoints, optimizer states, pool snapshots, and training logs are
not. The Git history and repository metadata are anonymous for double-blind
review.
