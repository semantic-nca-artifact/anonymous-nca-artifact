"""Train one method from the three-stage experimental protocol."""

import argparse
import importlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

import config
from clip_loss import CLIPLoss
from model import make_seed


TRAINERS = {
    "weighted_sum": ("training.weighted_sum", "train_weighted_baseline"),
    "tchebycheff_sum": (
        "training.tchebycheff_sum",
        "train_tchebycheff_baseline",
    ),
    "epo": ("training.epo", "train_epo"),
    "moo_svgd": ("training.moo_svgd", "train_true_moo_svgd"),
    "moead_ca": ("training.moead_ca", "train_moead_ca"),
    "moead_rta": ("training.moead_rta", "train_moead_rta"),
    "moead_mca": ("training.moead_mca", "train_moead_mca"),
    "moead_mrta": ("training.moead_mrta", "train_moead_mrta"),
}
MOEAD_STRATEGIES = {
    "moead_ca",
    "moead_rta",
    "moead_mca",
    "moead_mrta",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-strategy",
        required=True,
        choices=tuple(TRAINERS),
        help="Training method defined by the three-stage protocol.",
    )
    parser.add_argument("--prompt-1", required=True)
    parser.add_argument("--prompt-2", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--log-root",
        type=Path,
        required=True,
        help="Run-specific output directory.",
    )
    return parser.parse_args()


def configure_run(args):
    config.TRAIN_STRATEGY = args.train_strategy
    config.PROMPT1 = args.prompt_1
    config.PROMPT2 = args.prompt_2
    config.OBJECTIVE_PROMPTS = [args.prompt_1, args.prompt_2]
    config.RUN_SEED = int(args.seed)
    config.TRAIN_STEPS = int(args.steps)
    config.TRAIN_LOG_ROOT = str(args.log_root)
    config.EPO_PREF_VECTORS = config._make_pref_vectors(
        config.EPO_N_PREFS,
        k=len(config.OBJECTIVE_PROMPTS),
    )

    if args.train_strategy in MOEAD_STRATEGIES:
        module_name, _ = TRAINERS[args.train_strategy]
        importlib.import_module(module_name).configure_strategy()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def protocol_manifest(args):
    manifest = {
        "seed": int(args.seed),
        "train_strategy": args.train_strategy,
        "objective_prompts": list(config.OBJECTIVE_PROMPTS),
        "experiment": config.EXPERIMENT_TYPE,
        "train_steps": int(config.TRAIN_STEPS),
        "batch_size": int(config.BATCH_SIZE),
        "pool_size": int(config.POOL_SIZE),
        "rollout_steps": [int(config.STEPS_MIN), int(config.STEPS_MAX - 1)],
    }
    if args.train_strategy in MOEAD_STRATEGIES:
        manifest["moead"] = {
            "population_size": int(config.MOEAD_N_SUBPROBLEMS),
            "neighbor_size": int(config.MOEAD_NEIGHBOR_SIZE),
            "gradient_policy": config.MOEAD_GRADIENT_POLICY,
            "cooperation": (
                "region_archive"
                if config.MOEAD_USE_RTA
                else "one_to_one_neighborhood"
            ),
            "cooperation_interval": int(config.MOEAD_COOPERATION_INTERVAL),
            "cooperation_threshold": float(
                config.COOPERATION_THRESHOLD
            ),
            "synchronized_evaluation_interval": int(
                config.MOEAD_EVAL_INTERVAL
            ),
            "synchronized_evaluation_steps": int(config.EVAL_STEPS),
            "reporting_evaluation_steps": int(config.FINAL_EVAL_STEPS),
            "reporting_evaluation_repeats": int(
                config.FINAL_EVAL_REPEATS
            ),
        }
    return manifest


def write_manifest(args):
    args.log_root.mkdir(parents=True, exist_ok=True)
    path = args.log_root / "run_manifest.json"
    path.write_text(
        json.dumps(protocol_manifest(args), indent=2) + "\n",
        encoding="utf-8",
    )


def load_trainer(strategy):
    module_name, function_name = TRAINERS[strategy]
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def main():
    args = parse_args()
    configure_run(args)
    set_random_seed(args.seed)
    write_manifest(args)

    side = config.TARGET_SIZE + 2 * config.TARGET_PADDING
    seed = make_seed(side, n=1)

    print(f"Method: {args.train_strategy}")
    print(f"Prompts: {args.prompt_1!r}; {args.prompt_2!r}")
    print(f"Seed: {args.seed}")
    print(f"Training steps: {args.steps}")

    clip_loss = CLIPLoss()
    trainer = load_trainer(args.train_strategy)
    trainer(clip_loss, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
