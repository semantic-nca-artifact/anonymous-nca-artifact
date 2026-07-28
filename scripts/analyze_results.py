"""Evaluate one stage of the formal experimental protocol."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    1: "weighted_sum,tchebycheff_sum,epo,moo_svgd,moead_ca",
    2: "moead_ca,moead_rta,moead_mca,moead_mrta",
}
DEFAULT_SEEDS = {
    1: "11,22,33",
    2: "11,22,33",
    3: "11,22,33,44,55",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--train-log", type=Path, default=ROOT / "train_log")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute tables without rendering checkpoint phenotypes.",
    )
    parser.add_argument(
        "--build-main-figure",
        action="store_true",
        help="Build the Stage-III coverage/concentration figure.",
    )
    return parser.parse_args()


def population_command(args):
    stage = args.stage
    output = args.output or args.train_log / "analysis" / f"stage{stage}"
    command = [
        sys.executable,
        str(ROOT / "visualize_moead_results.py"),
        "--root",
        str(args.train_log / f"stage{stage}"),
        "--output",
        str(output),
        "--methods",
        METHODS[stage],
        "--seeds",
        args.seeds or DEFAULT_SEEDS[stage],
        "--baseline-eval-steps",
        "96",
        "--baseline-eval-repeats",
        "4",
        "--rollout-steps",
        "96",
    ]
    if stage == 2:
        command.append("--summary-only")
    if args.metrics_only:
        command.append("--no-render")
    return command


def stage3_command(args):
    output = args.output or args.train_log / "analysis" / "stage3"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "aggregate_stage3_results.py"),
        "--train-log",
        str(args.train_log),
        "--output",
        str(output),
        "--seeds",
        args.seeds or DEFAULT_SEEDS[3],
    ]
    return command


def main():
    args = parse_args()
    if args.build_main_figure and args.stage != 3:
        raise ValueError("--build-main-figure is available only for Stage III")

    command = (
        population_command(args)
        if args.stage in (1, 2)
        else stage3_command(args)
    )
    subprocess.run(command, cwd=ROOT, check=True)

    if args.build_main_figure:
        figure_output = args.output or ROOT / "figures"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_stage3_main_figure.py"),
                "--train-log",
                str(args.train_log),
                "--output",
                str(figure_output),
                "--device",
                args.device,
                "--rollout-steps",
                "96",
            ],
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
