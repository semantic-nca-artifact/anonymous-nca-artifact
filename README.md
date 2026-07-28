# Three-Stage Neural Cellular Automata Artifact

This repository contains the training and evaluation code for a controlled
study of language-defined morphospace exploration with neural cellular
automata (NCAs). The artifact follows the three experimental stages reported
in the paper and excludes exploratory strategies, paper sources, raw training
logs, and checkpoints.

## Experimental design

| Stage | Purpose | Methods | Tasks and replicates |
| --- | --- | --- | --- |
| I | Select the population-search backbone | Weighted Sum, Tchebycheff, EPO, MOO-SVGD, MOEA/D-CA | Butterfly--Caterpillar; three runs per method |
| II | Factor local learning and cross-subproblem cooperation | MOEA/D-CA, MOEA/D-RTA, MOEA/D-MCA, MOEA/D-MRTA | Butterfly--Caterpillar; seeds 11, 22, and 33 |
| III | Test coverage--concentration control across tasks | MOEA/D-CA and MOEA/D-MCA | Six semantic pairs; paired seeds 11, 22, 33, 44, and 55 |

The 2 x 2 Stage-II design crosses weighted-sum or Frank--Wolfe MGDA local
learning with neighborhood or region-wise archive cooperation. Stage III
holds neighborhood cooperation fixed and changes only the local-gradient
construction.

## Fixed protocol

The formal MOEA/D runs use 30 uniformly spaced bi-objective preferences,
five-member weight-space neighborhoods, and one persistent pool of 1,024 NCA
states per population member. Each local update samples eight states and
advances them for 64--95 NCA steps.

Synchronized fresh-seed evaluation is performed for 64 steps every ten outer
updates. Cooperation occurs every 50 updates with a 3% relative-improvement
gate and global maximum-weight one-to-one matching. Final objectives average
four 96-step fresh-seed rollouts. Training uses 2,000 outer updates, learning
rate 5e-4, and gradient clipping at norm 1.0.

MOEA/D-CA uses a fixed-preference weighted-sum gradient. MOEA/D-MCA computes
one gradient per objective and obtains a minimum-norm convex combination with
Frank--Wolfe. RTA and MRTA replace neighborhood propagation with the
region-wise archive transfer used in Stage II.

## Repository layout

- `main.py`: run one published training strategy;
- `training/`: the four Stage-I baselines and the shared CA/RTA/MCA/MRTA
  implementation;
- `configs/semantic_pairs.tsv`: exact prompts and task identifiers;
- `scripts/run_experiments.py`: prepare or execute the formal Stage-I,
  Stage-II, and Stage-III matrices;
- `scripts/analyze_results.py`: stage-oriented evaluation entry point;
- `visualize_moead_results.py`: shared population evaluation and rendering
  backend;
- `scripts/aggregate_stage3_results.py`: paired Stage-III
  aggregation;
- `scripts/build_stage3_main_figure.py`: build the coverage/concentration
  figure from completed Stage-III runs;
- `results/`: sanitized run-level and aggregate tables for all three stages;

## Environment

The reported environment used Python 3.10, PyTorch 2.7.1 with CUDA 12.8,
Transformers 4.47.1, and CLIP ViT-B/32. A matching setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

On Windows, activate the environment with
`.venv\Scripts\activate`. The default text-image model is
`openai/clip-vit-base-patch32`. Set `CLIP_MODEL_NAME` to a compatible local
directory when running without network access.

## Preparing the experiment matrix

The launcher prints commands by default and performs no training:

```bash
python scripts/run_experiments.py
```

The complete matrix contains 15 Stage-I runs, 12 Stage-II runs, and 60
Stage-III runs. To execute a full stage sequentially:

```bash
python scripts/run_experiments.py --stage 1 --execute
python scripts/run_experiments.py --stage 2 --execute
python scripts/run_experiments.py --stage 3 --execute
```

A short single-run check can be launched with:

```bash
python scripts/run_experiments.py \
  --stage 2 \
  --methods moead_mca \
  --seeds 11 \
  --steps 5 \
  --execute
```

For a cluster or multi-GPU system, omit `--execute`, capture the printed
commands, and schedule them independently. Every command writes to an
isolated method/task/seed directory.

## Evaluation

After training, compute the stage-specific metrics with:

```bash
python scripts/analyze_results.py --stage 1 --metrics-only
python scripts/analyze_results.py --stage 2 --metrics-only
python scripts/analyze_results.py --stage 3 --metrics-only
```

Checkpoint-based plots and the Stage-III paper figure require the completed
checkpoints:

```bash
python scripts/analyze_results.py --stage 3 --build-main-figure
```

Hypervolume is computed in two-dimensional similarity space with reference
`(0, 0)`. The statistical replicate is one independently trained
population. The 30 members within a final population are never treated as
independent replicates.

## Included numerical results

The compact tables under `results/stage1`, `results/stage2`, and
`results/stage3` allow the reported numerical comparisons to be audited
without downloading checkpoints. See `results/README.md` for table
definitions. Absolute paths and checkpoint fields have been removed.

The full experiments are computationally intensive. The reported runs used
one NVIDIA RTX 4090 GPU per process; the command-printing mode is intended to
support external job schedulers.

## Double-blind review

The repository has an anonymous Git history and contains no author metadata,
paper source, credentials, local filesystem paths, raw logs, or model
checkpoints. Please do not attempt to identify the authors during review.
