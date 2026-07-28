"""Aggregate the formal Stage-III CA/MCA comparison."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualize_moead_results import (  # noqa: E402
    discover_runs,
    prepare_repeated_runs,
    run_rollout_seed,
    run_statistics,
)


CA_METHOD = "moead_ca"
MCA_METHOD = "moead_mca"
METHODS = (CA_METHOD, MCA_METHOD)
METHOD_LABELS = {
    CA_METHOD: "MOEA/D-CA",
    MCA_METHOD: "MOEA/D-MCA",
}
METHOD_COLORS = {
    CA_METHOD: "#0072B2",
    MCA_METHOD: "#D55E00",
}
METHOD_MARKERS = {
    CA_METHOD: "o",
    MCA_METHOD: "s",
}
RUN_METRICS = (
    "hypervolume",
    "n_non_dominated",
    "spacing",
    "population_average_worst_similarity",
    "best_max_min_similarity",
    "objective_space_extent",
    "similarity_extent_objective_1",
    "similarity_extent_objective_2",
)


def load_semantic_pairs(path=None):
    path = Path(path or ROOT / "configs" / "semantic_pairs.tsv")
    categories = {
        "Metamorphosis": "Metamorphic relations",
        "Body topology": "Body-topology contrasts",
        "Growth topology": "Growth-topology contrasts",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    pairs = []
    for row in rows:
        pairs.append({
            "slug": row["semantic_pair_slug"],
            "label": row["semantic_pair_slug"].replace("_", " ").title()
                .replace(" ", "--", 1),
            "plot_label": row["semantic_pair_slug"].replace("_", "--").title(),
            "category": categories.get(row["category"], row["category"]),
            CA_METHOD: row["ca_root"],
            MCA_METHOD: row["mca_root"],
            "prompts": (row["prompt1"], row["prompt2"]),
        })
    return tuple(pairs)


SEMANTIC_PAIRS = load_semantic_pairs()


def parse_seeds(value):
    seeds = tuple(sorted({
        int(item.strip())
        for item in str(value).split(",")
        if item.strip()
    }))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def validate_run(run, expected_solutions, expected_steps, expected_repeats):
    stats = run_statistics(run)
    identity = f"{run['strategy']} seed {run['seed']}"
    if stats["n_solutions"] != expected_solutions:
        raise ValueError(
            f"{identity} contains {stats['n_solutions']} solutions; "
            f"expected {expected_solutions}"
        )
    if run.get("evaluation_steps") != expected_steps:
        raise ValueError(
            f"{identity} uses {run.get('evaluation_steps')} evaluation steps; "
            f"expected {expected_steps}"
        )
    if run.get("evaluation_repeats") != expected_repeats:
        raise ValueError(
            f"{identity} uses {run.get('evaluation_repeats')} repeats; "
            f"expected {expected_repeats}"
        )
    return stats


def collect_run_rows(
    train_log,
    seeds,
    expected_solutions=30,
    expected_steps=96,
    expected_repeats=4,
    semantic_pairs=SEMANTIC_PAIRS,
    return_runs=False,
    root_overrides=None,
):
    """Load one completed run for every task, method, and seed."""

    train_log = Path(train_log)
    requested_seeds = set(seeds)
    rows = []
    prompts_by_pair = {}
    run_index = {}
    root_overrides = root_overrides or {}

    for pair in semantic_pairs:
        observed_prompts = set()
        for method in METHODS:
            result_root = Path(root_overrides.get(
                (pair["slug"], method),
                train_log / pair[method],
            ))
            if not result_root.is_dir():
                raise FileNotFoundError(
                    f"Missing Stage-III root for {pair['label']} / {method}: "
                    f"{result_root}"
                )
            discovered = discover_runs(result_root, methods={method})
            selected = prepare_repeated_runs(
                discovered,
                seeds=requested_seeds,
            )
            by_seed = {int(run["seed"]): run for run in selected}
            missing = sorted(requested_seeds - set(by_seed))
            if missing:
                raise ValueError(
                    f"Missing {pair['label']} / {method} seeds: {missing}"
                )
            if len(by_seed) != len(selected):
                raise ValueError(
                    f"Duplicate seeds remain for {pair['label']} / {method}"
                )

            for seed in seeds:
                run = by_seed[int(seed)]
                stats = validate_run(
                    run,
                    expected_solutions,
                    expected_steps,
                    expected_repeats,
                )
                prompts = tuple(run["prompts"])
                observed_prompts.add(prompts)
                run_index[(pair["slug"], method, int(seed))] = run
                rows.append({
                    "semantic_pair": pair["label"],
                    "semantic_pair_slug": pair["slug"],
                    "category": pair["category"],
                    "strategy": method,
                    "method": METHOD_LABELS[method],
                    "seed": int(seed),
                    "run": run["run_name"],
                    **stats,
                    "prompt_1": prompts[0],
                    "prompt_2": prompts[1],
                    "evaluation_steps": run["evaluation_steps"],
                    "evaluation_repeats": run["evaluation_repeats"],
                    "rollout_rng_seed": int(run_rollout_seed(run)),
                })

        if observed_prompts != {tuple(pair["prompts"])}:
            raise ValueError(
                f"Prompt mismatch for {pair['label']}: "
                f"expected {pair['prompts']}, observed {sorted(observed_prompts)}"
            )
        prompts_by_pair[pair["slug"]] = list(pair["prompts"])

    if return_runs:
        return rows, prompts_by_pair, run_index
    return rows, prompts_by_pair


def aggregate_run_rows(run_rows):
    """Compute task- and method-specific summaries across seeds."""

    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["semantic_pair_slug"], row["strategy"])].append(row)

    pair_order = {
        pair["slug"]: index for index, pair in enumerate(SEMANTIC_PAIRS)
    }
    rows = []
    for key in sorted(
        grouped,
        key=lambda item: (pair_order[item[0]], METHODS.index(item[1])),
    ):
        group = grouped[key]
        first = group[0]
        result = {
            "semantic_pair": first["semantic_pair"],
            "semantic_pair_slug": first["semantic_pair_slug"],
            "category": first["category"],
            "strategy": first["strategy"],
            "method": first["method"],
            "run_count": len(group),
            "seeds": ";".join(
                str(row["seed"])
                for row in sorted(group, key=lambda item: item["seed"])
            ),
        }
        for metric in RUN_METRICS:
            values = np.asarray(
                [row[metric] for row in group],
                dtype=np.float64,
            )
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else ""
            )
            result[f"{metric}_min"] = float(values.min())
            result[f"{metric}_max"] = float(values.max())
        rows.append(result)
    return rows


def build_paired_rows(run_rows):
    """Pair CA and MCA by task and seed; differences are MCA minus CA."""

    indexed = {
        (row["semantic_pair_slug"], row["strategy"], row["seed"]): row
        for row in run_rows
    }
    rows = []
    for pair in SEMANTIC_PAIRS:
        seeds = sorted({
            row["seed"]
            for row in run_rows
            if row["semantic_pair_slug"] == pair["slug"]
        })
        for seed in seeds:
            ca = indexed[(pair["slug"], CA_METHOD, seed)]
            mca = indexed[(pair["slug"], MCA_METHOD, seed)]
            result = {
                "semantic_pair": pair["label"],
                "semantic_pair_slug": pair["slug"],
                "category": pair["category"],
                "seed": seed,
                "difference_definition": "moead_mca_minus_moead_ca",
            }
            for metric in RUN_METRICS:
                result[f"{metric}_ca"] = ca[metric]
                result[f"{metric}_mca"] = mca[metric]
                result[f"{metric}_difference"] = (
                    mca[metric] - ca[metric]
                )
            rows.append(result)
    return rows


def build_direction_counts(paired_rows, aggregate_rows):
    """Count paired and task-mean directions without pooling tasks."""

    aggregate_index = {
        (row["semantic_pair_slug"], row["strategy"]): row
        for row in aggregate_rows
    }
    tolerance = 1e-12
    rows = []
    for metric in RUN_METRICS:
        paired = np.asarray(
            [row[f"{metric}_difference"] for row in paired_rows],
            dtype=np.float64,
        )
        task_means = np.asarray([
            aggregate_index[(pair["slug"], MCA_METHOD)][
                f"{metric}_mean"
            ]
            - aggregate_index[(pair["slug"], CA_METHOD)][
                f"{metric}_mean"
            ]
            for pair in SEMANTIC_PAIRS
        ])
        rows.append({
            "metric": metric,
            "difference_definition": "moead_mca_minus_moead_ca",
            "seed_pair_count": len(paired),
            "seed_ca_higher": int(np.sum(paired < -tolerance)),
            "seed_mca_higher": int(np.sum(paired > tolerance)),
            "seed_ties": int(np.sum(np.abs(paired) <= tolerance)),
            "task_count": len(task_means),
            "task_mean_ca_higher": int(
                np.sum(task_means < -tolerance)
            ),
            "task_mean_mca_higher": int(
                np.sum(task_means > tolerance)
            ),
            "task_mean_ties": int(
                np.sum(np.abs(task_means) <= tolerance)
            ),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def aggregate_results(train_log, output, seeds, root_overrides=None):
    run_rows, _ = collect_run_rows(
        train_log,
        seeds,
        root_overrides=root_overrides,
    )
    aggregate_rows = aggregate_run_rows(run_rows)
    paired_rows = build_paired_rows(run_rows)
    direction_rows = build_direction_counts(
        paired_rows,
        aggregate_rows,
    )
    output = Path(output)
    paths = {
        "per_run": write_csv(output / "per_run_metrics.csv", run_rows),
        "aggregate": write_csv(
            output / "aggregate_metrics.csv",
            aggregate_rows,
        ),
        "paired": write_csv(
            output / "paired_differences.csv",
            paired_rows,
        ),
        "directions": write_csv(
            output / "direction_counts.csv",
            direction_rows,
        ),
    }
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", type=Path, default=ROOT / "train_log")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "train_log" / "analysis" / "stage3",
    )
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument(
        "--butterfly-ca-root",
        type=Path,
        help="Explicit CA result root for a historical Butterfly--Caterpillar layout.",
    )
    parser.add_argument(
        "--butterfly-mca-root",
        type=Path,
        help="Explicit MCA result root for a historical Butterfly--Caterpillar layout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_overrides = {}
    if args.butterfly_ca_root:
        root_overrides[("butterfly_caterpillar", CA_METHOD)] = (
            args.butterfly_ca_root
        )
    if args.butterfly_mca_root:
        root_overrides[("butterfly_caterpillar", MCA_METHOD)] = (
            args.butterfly_mca_root
        )
    paths = aggregate_results(
        args.train_log,
        args.output,
        parse_seeds(args.seeds),
        root_overrides=root_overrides,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
