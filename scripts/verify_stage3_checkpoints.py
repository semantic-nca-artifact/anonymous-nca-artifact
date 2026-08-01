"""Verify the released Stage-III model populations and evaluation records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("moead_ca", "moead_mca")
SEEDS = (11, 22, 33, 44, 55)
METRICS = (
    "n_solutions",
    "n_non_dominated",
    "hypervolume",
    "spacing",
    "population_average_worst_similarity",
    "best_max_min_similarity",
    "objective_space_extent",
    "similarity_extent_objective_1",
    "similarity_extent_objective_2",
)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_pairs(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        row["semantic_pair_slug"]: (row["prompt1"], row["prompt2"])
        for row in rows
    }


def pareto_indices(values):
    keep = []
    for index, point in enumerate(values):
        dominated = any(
            other != index
            and all(values[other][axis] <= point[axis] for axis in (0, 1))
            and any(values[other][axis] < point[axis] for axis in (0, 1))
            for other in range(len(values))
        )
        if not dominated:
            keep.append(index)
    return keep


def hypervolume_2d(similarities):
    points = sorted(
        (point for point in similarities if point[0] > 0 and point[1] > 0),
        key=lambda point: point[0],
        reverse=True,
    )
    value = 0.0
    y_max = 0.0
    for index, point in enumerate(points):
        x_next = points[index + 1][0] if index + 1 < len(points) else 0.0
        y_max = max(y_max, point[1])
        value += (point[0] - x_next) * y_max
    return value


def spacing(values):
    if len(values) < 2:
        return 0.0
    nearest = []
    for index, point in enumerate(values):
        nearest.append(min(
            math.dist(point, other)
            for other_index, other in enumerate(values)
            if other_index != index
        ))
    mean = sum(nearest) / len(nearest)
    return math.sqrt(sum((value - mean) ** 2 for value in nearest) / len(nearest))


def run_statistics(objectives):
    front = [objectives[index] for index in pareto_indices(objectives)]
    similarities = [[-value for value in point] for point in objectives]
    worst = [min(point) for point in similarities]
    extents = [
        max(point[axis] for point in similarities)
        - min(point[axis] for point in similarities)
        for axis in (0, 1)
    ]
    return {
        "n_solutions": len(objectives),
        "n_non_dominated": len(front),
        "hypervolume": hypervolume_2d([[-value for value in point] for point in front]),
        "spacing": spacing(front),
        "population_average_worst_similarity": sum(worst) / len(worst),
        "best_max_min_similarity": max(worst),
        "objective_space_extent": sum(extents),
        "similarity_extent_objective_1": extents[0],
        "similarity_extent_objective_2": extents[1],
    }


def safe_child(root: Path, relative: str):
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path escapes checkpoint root: {relative}")
    return path


def verify_checksums(root: Path):
    checksum_path = root / "SHA256SUMS"
    expected = {}
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in expected:
            raise ValueError(f"Duplicate checksum entry: {relative}")
        expected[relative] = digest

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if set(expected) != actual_files:
        missing = sorted(actual_files - set(expected))
        stale = sorted(set(expected) - actual_files)
        raise ValueError(f"Checksum inventory mismatch; missing={missing}, stale={stale}")

    for relative, expected_digest in expected.items():
        path = safe_child(root, relative)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise ValueError(f"Checksum mismatch: {relative}")
    return len(expected)


def verify(root: Path):
    root = root.resolve()
    pairs = load_pairs(ROOT / "configs" / "semantic_pairs.tsv")
    manifest_rows = read_csv(root / "manifest.csv")
    model_rows = read_csv(root / "model_index.csv")
    result_rows = read_csv(ROOT / "results" / "stage3" / "per_run_metrics.csv")
    results = {
        (row["semantic_pair_slug"], row["strategy"], int(row["seed"])): row
        for row in result_rows
    }
    expected_runs = {
        (slug, method, seed)
        for slug in pairs
        for method in METHODS
        for seed in SEEDS
    }
    observed_runs = {
        (row["semantic_pair_slug"], row["strategy"], int(row["seed"]))
        for row in manifest_rows
    }
    if observed_runs != expected_runs or len(manifest_rows) != len(expected_runs):
        raise ValueError("Stage-III manifest does not contain the expected 60 runs")

    indexed_models = {}
    for row in model_rows:
        key = (
            row["semantic_pair_slug"],
            row["strategy"],
            int(row["seed"]),
            int(row["subproblem_index"]),
        )
        if key in indexed_models:
            raise ValueError(f"Duplicate Stage-III model slot: {key}")
        indexed_models[key] = row

    expected_slots = {
        (slug, method, seed, index)
        for slug, method, seed in expected_runs
        for index in range(30)
    }
    if set(indexed_models) != expected_slots or len(model_rows) != 1800:
        raise ValueError(
            "Stage-III model index does not contain the expected 1,800 slots"
        )

    referenced_models = {}
    for row in manifest_rows:
        slug = row["semantic_pair_slug"]
        strategy = row["strategy"]
        seed = int(row["seed"])
        run_dir = safe_child(ROOT, row["relative_run_dir"])
        if run_dir.parent.parent.parent != root:
            raise ValueError(f"Unexpected run directory layout: {run_dir}")

        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        prompts = pairs[slug]
        if int(manifest["seed"]) != seed or manifest["train_strategy"] != strategy:
            raise ValueError(f"Run identity mismatch: {run_dir}")
        if tuple(manifest["objective_prompts"]) != prompts:
            raise ValueError(f"Training prompt mismatch: {run_dir}")
        if tuple(summary["objective_prompts"]) != prompts:
            raise ValueError(f"Evaluation prompt mismatch: {run_dir}")

        objectives = summary["final_objectives"]
        weights = summary["weights"]
        if len(objectives) != 30 or len(weights) != 30:
            raise ValueError(f"Expected 30 objective and weight vectors: {run_dir}")
        if not all(len(point) == 2 and all(math.isfinite(value) for value in point)
                   for point in objectives):
            raise ValueError(f"Invalid objective vectors: {run_dir}")

        expected_run_dir = run_dir.relative_to(root).as_posix()
        if int(row["model_count"]) != 30:
            raise ValueError(f"Expected 30 model slots: {run_dir}")
        for index, weight in enumerate(weights):
            model_row = indexed_models[(slug, strategy, seed, index)]
            if model_row["relative_run_dir"] != expected_run_dir:
                raise ValueError(f"Model-index run mismatch: {run_dir} / {index}")
            indexed_weight = (
                float(model_row["weight_1"]),
                float(model_row["weight_2"]),
            )
            if any(
                not math.isclose(
                    float(observed), float(expected), abs_tol=1e-12
                )
                for observed, expected in zip(indexed_weight, weight)
            ):
                raise ValueError(f"Model-index weight mismatch: {run_dir} / {index}")
            model_path = safe_child(root, model_row["relative_model_path"])
            digest = model_row["model_sha256"]
            if model_path.name != f"{digest}.pt" or not model_path.is_file():
                raise ValueError(f"Invalid indexed model: {model_path}")
            previous = referenced_models.setdefault(model_path, digest)
            if previous != digest:
                raise ValueError(f"Conflicting model digest: {model_path}")

        expected = results[(slug, strategy, seed)]
        computed = run_statistics(objectives)
        for metric in METRICS:
            if not math.isclose(
                float(expected[metric]),
                float(computed[metric]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"Metric mismatch for {run_dir}: {metric}")

    stored_models = set((root / "models" / "sha256").rglob("*.pt"))
    if set(referenced_models) != stored_models:
        raise ValueError(
            "Content-addressed model inventory does not match the model index"
        )
    for model_path, expected_digest in referenced_models.items():
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise ValueError(f"Model digest mismatch: {model_path}")

    checksum_count = verify_checksums(root)
    print(
        f"Verified {len(manifest_rows)} Stage-III runs, "
        f"{len(model_rows)} model slots, {len(stored_models)} unique model files, "
        f"and {checksum_count} checksums."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "checkpoints" / "stage3",
    )
    args = parser.parse_args()
    verify(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
