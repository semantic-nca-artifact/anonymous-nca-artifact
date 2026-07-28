"""Central configuration for training and evaluation."""

import os
from pathlib import Path

import numpy as np


# Neural cellular automaton
CHANNEL_N = 16          # RGBA plus 12 latent channels.
TARGET_PADDING = 16
TARGET_SIZE = 40
CELL_FIRE_RATE = 0.5

# Training
BATCH_SIZE = 8
POOL_SIZE = 1024
TRAIN_STEPS = 2000
RUN_SEED = 0
STEPS_MIN = 64
STEPS_MAX = 96
EVAL_STEPS = 96
FINAL_EVAL_STEPS = 96
FINAL_EVAL_REPEATS = 1
LR = 5e-4
LR_DECAY_STEP = 400
GRAD_CLIP_NORM = 1.0
STATE_CLIP_VALUE = 4.0

# CLIP
# Prefer the conventional local cache when present; otherwise use the Hub ID.
# Override either choice with CLIP_MODEL_NAME=/path/or/model-id.
_LOCAL_CLIP_MODEL = Path(__file__).resolve().parent / "models" / "clip-vit-base-patch32"
CLIP_MODEL_NAME = os.environ.get(
    "CLIP_MODEL_NAME",
    str(_LOCAL_CLIP_MODEL) if _LOCAL_CLIP_MODEL.exists()
    else "openai/clip-vit-base-patch32",
)
CLIP_RENDER_MODE = "soft"
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Objectives and default strategy
PROMPT1 = "a colorful butterfly with symmetric wings"
PROMPT2 = "a green caterpillar"
OBJECTIVE_PROMPTS = [PROMPT1, PROMPT2]
TRAIN_STRATEGY = "weighted_sum"

# Weighted-sum baseline
WEIGHTED_LOSS_WEIGHTS = [
    (round(1.0 - i / 29, 2), round(i / 29, 2))
    for i in range(30)
]

# NCA training regime
EXPERIMENT_TYPE = "Persistent"
EXPERIMENT_MAP = {"Growing": 0, "Persistent": 1, "Regenerating": 2}
EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]
USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
DAMAGE_N = [0, 0, 3][EXPERIMENT_N]

# MOO-SVGD
MOO_SVGD_PARTICLES = 30
MOO_SVGD_BANDWIDTH = -1          # Non-positive values use the median heuristic.
MOO_SVGD_REPULSION_COEF = 0.1

# EPO
EPO_N_PREFS = 30


def _make_pref_vectors(n, k=2):
    eps = 1e-3
    if k == 2:
        weights = np.linspace(eps, 1 - eps, n)
        return [
            [round(float(weight), 6), round(float(1 - weight), 6)]
            for weight in weights
        ]
    rng = np.random.default_rng(42)
    raw = rng.dirichlet(np.ones(k), size=n)
    raw = np.clip(raw, eps, 1 - eps)
    raw /= raw.sum(axis=1, keepdims=True)
    return raw.tolist()


EPO_PREF_VECTORS = _make_pref_vectors(
    EPO_N_PREFS,
    k=len(OBJECTIVE_PROMPTS),
)

# Shared MOEA/D configuration
MOEAD_N_SUBPROBLEMS = 30
MOEAD_DECOMPOSITION = "tchebycheff"
MOEAD_WEIGHT_SEED = 42
MOEAD_EPS = 1e-6
MOEAD_NEIGHBOR_SIZE = 5
MOEAD_MAX_REPLACEMENTS = 1
MOEAD_COOPERATION_INTERVAL = 50

# Factorial ablation policies. Named strategies override these fields with
# immutable policy definitions before training starts.
MOEAD_REPLACEMENT_POLICY = "neighborhood"   # none, neighborhood, global, or stm
MOEAD_POOL_POLICY = "preference_reset"       # preference_reset, uniform, or pareto
MOEAD_GRADIENT_POLICY = "tchebycheff"       # weighted_sum, tchebycheff, mgda, or mgda_svgd
MOEAD_USE_RTA = False
MOEAD_VARIANT_NAME = "ca"
MOEAD_EVAL_INTERVAL = 1                      # Formal synchronized evaluation cadence.

# Pareto-aware pool sampling
MOEAD_PARETO_CANDIDATE_MULTIPLIER = 4
MOEAD_PARETO_EXPLORATION = 0.25

# Parameter-space repulsion
MOEAD_SVGD_BANDWIDTH = MOO_SVGD_BANDWIDTH
MOEAD_SVGD_REPULSION_COEF = MOO_SVGD_REPULSION_COEF

# Output cadence
MOEAD_POOL_FIGURE_INTERVAL = 100
MOEAD_CHECKPOINT_INTERVAL = 100
MOEAD_SAVE_FINAL_MODELS = True

# Region-wise training-state archive
RTA_WARMUP_STEPS = 0
RTA_TRANSFER_INTERVAL = 50
# Shared 3% gate for CA/MCA replacement edges and RTA/MRTA archive edges.
COOPERATION_THRESHOLD = 0.03
RTA_ARCHIVE_SIZE = 3
RTA_ALLOW_SELF_TRANSFER = False
RTA_TRANSFER_POOL = False

# Output root
TRAIN_LOG_ROOT = "train_log"
