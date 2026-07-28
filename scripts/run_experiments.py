"""Prepare or execute the formal three-stage experiment matrix."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS_FILE = ROOT / "configs" / "semantic_pairs.tsv"

STAGE_METHODS = {
    1: (
        "weighted_sum",
        "tchebycheff_sum",
        "epo",
        "moo_svgd",
        "moead_ca",
    ),
    2: (
        "moead_ca",
        "moead_rta",
        "moead_mca",
        "moead_mrta",
    ),
    3: ("moead_ca", "moead_mca"),
}
STAGE_SEEDS = {
    1: (11, 22, 33),
    2: (11, 22, 33),
    3: (11, 22, 33, 44, 55),
}


def comma_set(value):
    if value is None:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_seeds(value):
    if value is None:
        return None
    seeds = tuple(sorted({int(item) for item in comma_set(value)}))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def load_tasks(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "semantic_pair_slug",
        "prompt1",
        "prompt2",
        "ca_root",
        "mca_root",
    }
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Invalid semantic-pair table: {path}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("1", "2", "3", "all"),
        default="all",
    )
    parser.add_argument(
        "--methods",
        help="Optional comma-separated subset of methods.",
    )
    parser.add_argument(
        "--pairs",
        help="Optional comma-separated Stage-III task slugs.",
    )
    parser.add_argument(
        "--seeds",
        help="Optional comma-separated seed override for a partial run.",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--train-log", type=Path, default=ROOT / "train_log")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute commands; otherwise only print the matrix.",
    )
    return parser.parse_args()


def selected_stages(value):
    return (1, 2, 3) if value == "all" else (int(value),)


def run_root(train_log, stage, method, seed, task):
    if stage in (1, 2):
        return train_log / f"stage{stage}" / method / f"seed_{seed}"
    field = "ca_root" if method == "moead_ca" else "mca_root"
    return train_log / task[field] / f"seed_{seed}"


def build_commands(args, tasks):
    stages = selected_stages(args.stage)
    requested_methods = comma_set(args.methods)
    requested_pairs = comma_set(args.pairs)
    seed_override = parse_seeds(args.seeds)

    known_methods = {
        method for stage in stages for method in STAGE_METHODS[stage]
    }
    if requested_methods and not requested_methods <= known_methods:
        unknown = sorted(requested_methods - known_methods)
        raise ValueError(f"Methods are not defined for the selected stage: {unknown}")

    if requested_pairs:
        known_pairs = {task["semantic_pair_slug"] for task in tasks}
        unknown = sorted(requested_pairs - known_pairs)
        if unknown:
            raise ValueError(f"Unknown semantic pairs: {unknown}")

    primary_task = tasks[0]
    commands = []
    for stage in stages:
        stage_tasks = (
            tasks
            if stage == 3
            else (primary_task,)
        )
        if stage == 3 and requested_pairs:
            stage_tasks = tuple(
                task for task in stage_tasks
                if task["semantic_pair_slug"] in requested_pairs
            )

        methods = STAGE_METHODS[stage]
        if requested_methods:
            methods = tuple(
                method for method in methods if method in requested_methods
            )
        seeds = seed_override or STAGE_SEEDS[stage]

        for task in stage_tasks:
            for method in methods:
                for seed in seeds:
                    log_root = run_root(
                        args.train_log,
                        stage,
                        method,
                        seed,
                        task,
                    )
                    commands.append((
                        stage,
                        task["semantic_pair_slug"],
                        [
                            args.python,
                            str(ROOT / "main.py"),
                            "--train-strategy",
                            method,
                            "--prompt-1",
                            task["prompt1"],
                            "--prompt-2",
                            task["prompt2"],
                            "--seed",
                            str(seed),
                            "--steps",
                            str(args.steps),
                            "--log-root",
                            str(log_root),
                        ],
                    ))
    return commands


def main():
    args = parse_args()
    tasks = load_tasks(TASKS_FILE)
    commands = build_commands(args, tasks)

    print(f"Prepared {len(commands)} run(s).")
    environment = os.environ.copy()
    if args.cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    for stage, task, command in commands:
        print(f"[Stage {stage} | {task}] {shlex.join(command)}", flush=True)
        if args.execute:
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
