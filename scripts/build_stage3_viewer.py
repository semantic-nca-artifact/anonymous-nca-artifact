"""Build the offline Stage-III developmental-rollout viewer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import STATE_CLIP_VALUE, TARGET_PADDING, TARGET_SIZE
from model import CAModel, make_seed
from training.common import finite_clip_state


CHECKPOINT_ROOT = ROOT / "checkpoints" / "stage3"
DEFAULT_INDICES = (0, 14, 29)
METHODS = ("moead_ca", "moead_mca")
METHOD_LABELS = {
    "moead_ca": "MOEA/D-CA",
    "moead_mca": "MOEA/D-MCA",
}
SELECTION_LABELS = ("Prompt 2 emphasis", "Balanced", "Prompt 1 emphasis")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "viewer")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--indices", default=",".join(map(str, DEFAULT_INDICES)))
    parser.add_argument("--rollout-steps", type=int, default=96)
    parser.add_argument("--frame-interval", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_pairs():
    path = ROOT / "configs" / "semantic_pairs.tsv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_model_index():
    path = CHECKPOINT_ROOT / "model_index.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (
            row["semantic_pair_slug"],
            row["strategy"],
            int(row["seed"]),
            int(row["subproblem_index"]),
        ): row
        for row in rows
    }


def load_metrics():
    path = ROOT / "results" / "stage3" / "per_run_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["semantic_pair_slug"], row["strategy"], int(row["seed"])): row
        for row in rows
    }


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def display_image(state):
    values = state.detach().cpu().numpy()
    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=STATE_CLIP_VALUE,
        neginf=-STATE_CLIP_VALUE,
    )
    values = np.clip(values, -STATE_CLIP_VALUE, STATE_CLIP_VALUE)
    rgb = 1.0 / (1.0 + np.exp(-values[:, :3]))
    alpha = (values[:, 3:4] > 0.1).astype(np.float32)
    image = 1.0 - alpha + alpha * rgb
    image = np.clip(image[0].transpose(1, 2, 0), 0.0, 1.0)
    return Image.fromarray(np.rint(image * 255).astype(np.uint8))


@torch.no_grad()
def render_trajectory(checkpoint, seed, rollout_steps, frame_interval, device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = CAModel().to(device)
    model.load_state_dict(load_state_dict(checkpoint, device))
    model.eval()

    size = TARGET_SIZE + 2 * TARGET_PADDING
    state = torch.tensor(make_seed(size, n=1), device=device)
    frames = [display_image(state)]
    for step in range(1, rollout_steps + 1):
        state = finite_clip_state(model(state))
        if step % frame_interval == 0 or step == rollout_steps:
            frames.append(display_image(state))
    return frames


def save_sheet(frame_columns, path):
    tile_width, tile_height = frame_columns[0][0].size
    frame_count = len(frame_columns[0])
    sheet = Image.new(
        "RGB",
        (tile_width * len(frame_columns), tile_height * frame_count),
        "white",
    )
    for column, frames in enumerate(frame_columns):
        if len(frames) != frame_count:
            raise ValueError("All rollout trajectories must have equal length")
        for row, frame in enumerate(frames):
            sheet.paste(frame, (column * tile_width, row * tile_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "WEBP", lossless=True, quality=100, method=6, exact=True)
    return tile_width, tile_height, frame_count


def rollout_seed(training_seed):
    return int(training_seed) * 1_000_003 + 10_000_000


def compact_metrics(row):
    keys = (
        "hypervolume",
        "population_average_worst_similarity",
        "best_max_min_similarity",
        "objective_space_extent",
    )
    return {key: float(row[key]) for key in keys}


def build_viewer(output, training_seed, indices, rollout_steps,
                 frame_interval, device):
    if rollout_steps <= 0 or frame_interval <= 0:
        raise ValueError("Rollout and frame interval must be positive")
    if rollout_steps % frame_interval:
        raise ValueError("Rollout steps must be divisible by the frame interval")
    if len(indices) != len(SELECTION_LABELS) or len(set(indices)) != len(indices):
        raise ValueError("Exactly three distinct subproblem indices are required")

    pairs = load_pairs()
    model_index = load_model_index()
    metrics = load_metrics()
    output = output.resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": 1,
        "training_seed": int(training_seed),
        "rollout_seed": rollout_seed(training_seed),
        "rollout_steps": int(rollout_steps),
        "frame_interval": int(frame_interval),
        "selection_rule": {
            "subproblem_indices": list(indices),
            "labels": list(SELECTION_LABELS),
            "description": (
                "One fixed training seed and three fixed preference indices are "
                "used for every task and both methods."
            ),
        },
        "pairs": [],
    }

    tile_size = None
    frame_count = None
    for pair in pairs:
        slug = pair["semantic_pair_slug"]
        pair_record = {
            "slug": slug,
            "category": pair["category"],
            "prompt_1": pair["prompt1"],
            "prompt_2": pair["prompt2"],
            "methods": {},
        }
        for method in METHODS:
            run_dir = CHECKPOINT_ROOT / slug / method / f"seed_{training_seed}"
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            weights = np.asarray(summary["weights"], dtype=np.float64)
            objectives = np.asarray(summary["final_objectives"], dtype=np.float64)
            if weights.shape != (30, 2) or objectives.shape != (30, 2):
                raise ValueError(f"Unexpected Stage-III population shape: {run_dir}")

            selected = []
            trajectories = []
            for label, index in zip(SELECTION_LABELS, indices):
                row = model_index[(slug, method, training_seed, index)]
                checkpoint = CHECKPOINT_ROOT / row["relative_model_path"]
                trajectories.append(
                    render_trajectory(
                        checkpoint,
                        rollout_seed(training_seed),
                        rollout_steps,
                        frame_interval,
                        device,
                    )
                )
                selected.append({
                    "index": int(index),
                    "label": label,
                    "weight": weights[index].tolist(),
                    "similarity": (-objectives[index]).tolist(),
                    "model_sha256": row["model_sha256"],
                })

            asset_name = f"{slug}-{method}.webp"
            width, height, count = save_sheet(trajectories, assets / asset_name)
            tile_size = tile_size or (width, height)
            frame_count = frame_count or count
            if tile_size != (width, height) or frame_count != count:
                raise ValueError("Inconsistent viewer sprite geometry")

            pair_record["methods"][method] = {
                "label": METHOD_LABELS[method],
                "asset": f"assets/{asset_name}",
                "metrics": compact_metrics(metrics[(slug, method, training_seed)]),
                "points": (-objectives).tolist(),
                "selected": selected,
            }
        payload["pairs"].append(pair_record)

    payload["tile_width"], payload["tile_height"] = tile_size
    payload["frame_count"] = frame_count
    data_text = "window.STAGE3_VIEWER_DATA = " + json.dumps(
        payload, ensure_ascii=True, separators=(",", ":")
    ) + ";\n"
    (output / "viewer-data.js").write_text(data_text, encoding="utf-8", newline="\n")

    checksums = []
    for path in sorted(assets.glob("*.webp")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {path.name}")
    (assets / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="ascii", newline="\n"
    )
    return payload


def main():
    args = parse_args()
    indices = tuple(int(value) for value in args.indices.split(","))
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = build_viewer(
        args.output,
        args.seed,
        indices,
        args.rollout_steps,
        args.frame_interval,
        device,
    )
    asset_bytes = sum(path.stat().st_size for path in (args.output / "assets").glob("*.webp"))
    print(
        f"Built {len(payload['pairs'])} tasks, {payload['frame_count']} frames per "
        f"trajectory, {asset_bytes / (1024 * 1024):.2f} MiB of WebP assets."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
