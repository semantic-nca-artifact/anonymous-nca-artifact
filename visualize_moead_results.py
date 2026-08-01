"""Evaluate completed populations under a common fresh-seed protocol.

The Stage-I checkpoint-only methods are reevaluated with shared rollout seeds.
Summary-backed MOEA/D runs retain their recorded canonical objectives. Tables
always treat one independently trained population as one replicate.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import textwrap
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from config import (
    CLIP_MODEL_NAME,
    FINAL_EVAL_REPEATS,
    FINAL_EVAL_STEPS,
    OBJECTIVE_PROMPTS,
    STATE_CLIP_VALUE,
    TARGET_PADDING,
    TARGET_SIZE,
)
from model import CAModel, make_seed
from training.common import finite_clip_state
from utils import to_rgb


# High-contrast, colorblind-safe palette derived from Okabe-Ito. Marker and
# line styles provide redundant channels for grayscale and reduced-size print.
PUBLICATION_COLORS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#7A7A7A",  # neutral gray
    "#8C6D31",  # brown
    "#6A51A3",  # violet
    "#1B9E77",  # teal
    "#E7298A",  # magenta
)
METHOD_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p")
METHOD_LINESTYLES = (
    "-",
    "--",
    "-.",
    (0, (1.2, 1.2)),
    (0, (4.0, 1.4, 1.0, 1.4)),
    (0, (5.0, 1.6)),
)
INK = "#343434"
GRID = "#D8D4CD"


# Paper-facing names for the semantic-path experiment family. Training logs
# retain their immutable internal variant ids (for example ``semantic_b``),
# while visualization, CSV output, and --methods use these canonical names.
SEMANTIC_VARIANT_METHODS = {
    "a": "moead_ca",
    "b": "moead_vreg",
    "c": "moead_brdg",
    "d": "moead_path",
    "full_rta": "moead_frta",
    "pool_rta": "moead_trta",
    "qd_pool_rta": "moead_qrta",
    "pare": "moead_pare",
    "baln": "moead_baln",
    "semantic_a": "moead_ca",
    "semantic_b": "moead_vreg",
    "semantic_c": "moead_brdg",
    "semantic_d": "moead_path",
    "semantic_full_rta": "moead_frta",
    "semantic_pool_rta": "moead_trta",
    "semantic_qd_pool_rta": "moead_qrta",
    "semantic_pare": "moead_pare",
    "semantic_baln": "moead_baln",
    "moead_semantic_a": "moead_ca",
    "moead_semantic_b": "moead_vreg",
    "moead_semantic_c": "moead_brdg",
    "moead_semantic_d": "moead_path",
    "moead_semantic_full_rta": "moead_frta",
    "moead_semantic_pool_rta": "moead_trta",
    "moead_semantic_qd_pool_rta": "moead_qrta",
    "moead_semantic_pare": "moead_pare",
    "moead_semantic_baln": "moead_baln",
}

METHOD_DISPLAY_NAMES = {
    "moead_vreg": "MOEA/D-VREG",
    "moead_brdg": "MOEA/D-BRDG",
    "moead_path": "MOEA/D-PATH",
    "moead_frta": "MOEA/D-FRTA",
    "moead_trta": "MOEA/D-TRTA",
    "moead_qrta": "MOEA/D-QRTA",
    "moead_pare": "MOEA/D-PARE",
    "moead_baln": "MOEA/D-BALN",
    "nsga_nca": "GA-NSGA-II",
    "sms_nca": "Robust SMS-EMOA",
    "mome_nca": "Developmental MOME",
    "mome_base": "MOME-Base",
    "mome_vgat": "MOME-Validity",
    "mome_cvtd": "MOME-CVT",
    "mome_pram": "MOME-ParameterReplay",
    "mome_repl": "MOME-Replay",
}

EXTERNAL_POPULATION_METHODS = {
    "nsga_nca", "sms_nca", "mome_nca",
    "mome_base", "mome_vgat", "mome_cvtd", "mome_pram", "mome_repl",
}

def configure_publication_style():
    """Apply a compact serif style suitable for two-column paper figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


configure_publication_style()


def method_styles(methods):
    methods = sorted(set(methods))
    return {
        method: {
            "color": PUBLICATION_COLORS[index % len(PUBLICATION_COLORS)],
            "marker": METHOD_MARKERS[index % len(METHOD_MARKERS)],
            "linestyle": METHOD_LINESTYLES[index % len(METHOD_LINESTYLES)],
        }
        for index, method in enumerate(methods)
    }


def method_display_name(strategy):
    """Convert a strategy slug to a compact paper-facing label."""
    strategy = canonical_method_name(strategy)
    if strategy in METHOD_DISPLAY_NAMES:
        return METHOD_DISPLAY_NAMES[strategy]
    if strategy.startswith("moead_"):
        return "MOEA/D-" + strategy[len("moead_"):].upper()
    return strategy.replace("_", "-").upper()


METHOD_ALIASES = {
    "epo_sum": "epo",
    "moo_svgd_sum": "moo_svgd",
    "ca": "moead_ca",
}


def canonical_method_name(strategy):
    """Normalize folder-facing aliases to the training strategy name."""
    strategy = str(strategy).strip().lower()
    if strategy in SEMANTIC_VARIANT_METHODS:
        return SEMANTIC_VARIANT_METHODS[strategy]
    return METHOD_ALIASES.get(strategy, strategy)


def strategy_from_summary_variant(variant):
    """Map a summary variant without mislabeling external methods as MOEA/D."""
    variant = str(variant).strip().lower()
    if variant.startswith("semantic_"):
        return canonical_method_name(variant)
    if variant in EXTERNAL_POPULATION_METHODS:
        return variant
    return canonical_method_name(
        variant if variant.startswith("moead_") else f"moead_{variant}"
    )


def parse_method_filter(value):
    """Parse --methods while enforcing canonical semantic method names."""
    if not value:
        return None
    methods = set()
    for item in value.split(","):
        requested = item.strip().lower()
        if not requested:
            continue
        canonical = canonical_method_name(requested)
        if requested in SEMANTIC_VARIANT_METHODS and requested != canonical:
            raise ValueError(
                f"Use canonical method name {canonical!r} instead of "
                f"internal semantic variant {requested!r}."
            )
        methods.add(canonical)
    return methods or None


def run_identity(run):
    """Return an honest run label when legacy logs do not record the seed."""
    seed = run.get("seed")
    if seed is None:
        return f'run={run["run_name"]}'
    return f"seed={seed}"


def run_rollout_seed(run):
    """Use the first canonical repeat when formal summary metadata is available."""
    if (
        run.get("evaluation_source") == "summary"
        and run.get("seed") is not None
    ):
        return int(run["seed"]) * 1_000_003 + 10_000_000
    if run.get("evaluation_seed") is not None:
        return int(run["evaluation_seed"])
    if run.get("rollout_seed") is not None:
        return int(run["rollout_seed"])
    if run.get("seed") is not None:
        return int(run["seed"])
    return 0


def _checkpoint_only_specs(summary_runs, baseline_specs,
                           force_uniform_evaluation=False):
    """Return baseline checkpoint specs that require canonical evaluation.

    When a root mixes legacy checkpoint-only baselines with summary-backed
    methods, ``force_uniform_evaluation`` reevaluates every discovered
    checkpoint-backed run.  This prevents historical summary objectives from
    being compared with newly evaluated checkpoint objectives.
    """
    summarized = {
        Path(run["run_dir"]).resolve()
        for run in summary_runs
    }
    unique = {}
    for spec in baseline_specs:
        run_dir = Path(spec["run_dir"]).resolve()
        if not force_uniform_evaluation and run_dir in summarized:
            continue
        unique.setdefault(run_dir, spec)
    return sorted(
        unique.values(),
        key=lambda spec: (spec["strategy"], str(spec["run_dir"]).lower()),
    )


def save_publication_figure(figure, output_dir, stem, dpi=300):
    """Save a high-resolution raster and a vector PDF from one figure."""
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    return png_path, pdf_path


def pareto_indices(values):
    """Return non-dominated indices for a minimization problem."""
    values = np.asarray(values, dtype=np.float64)
    keep = []
    for index in range(len(values)):
        dominated = any(
            other != index
            and np.all(values[other] <= values[index])
            and np.any(values[other] < values[index])
            for other in range(len(values))
        )
        if not dominated:
            keep.append(index)
    return keep


def hypervolume_2d(similarities, reference=(0.0, 0.0)):
    """Compute two-dimensional maximization hypervolume."""
    points = np.asarray(similarities, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    points = points[
        (points[:, 0] > reference[0])
        & (points[:, 1] > reference[1])
    ]
    if not len(points):
        return 0.0
    points = points[np.argsort(points[:, 0])[::-1]]
    hv = 0.0
    y_max = reference[1]
    for index, point in enumerate(points):
        x_next = (
            points[index + 1, 0]
            if index + 1 < len(points)
            else reference[0]
        )
        y_max = max(y_max, point[1])
        hv += (point[0] - x_next) * (y_max - reference[1])
    return float(hv)


def spacing(values):
    """Compute the standard nearest-neighbor spacing metric."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    distances = np.linalg.norm(
        values[:, None, :] - values[None, :, :],
        axis=2,
    )
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return float(np.sqrt(np.mean((nearest - nearest.mean()) ** 2)))


def final_checkpoint(directory):
    """Return final.pt when present, otherwise the latest numeric checkpoint."""
    directory = Path(directory)
    final_path = directory / "final.pt"
    if final_path.exists():
        return final_path
    numeric = [
        path for path in directory.glob("*.pt")
        if path.stem.isdigit()
    ]
    if not numeric:
        return None
    return max(numeric, key=lambda path: int(path.stem))


_SUBPROBLEM_PATTERN = re.compile(r"^sub_(\d+)(?:_|$)")

_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_BASELINE_SOLUTION_PATTERNS = (
    (
        "tchebycheff_sum",
        re.compile(
            rf"^tch_w1_({_FLOAT_PATTERN})_w2_({_FLOAT_PATTERN})$",
            re.IGNORECASE,
        ),
    ),
    (
        "weighted_sum",
        re.compile(
            rf"^w1_({_FLOAT_PATTERN})_w2_({_FLOAT_PATTERN})$",
            re.IGNORECASE,
        ),
    ),
    (
        "epo",
        re.compile(
            rf"^pref_(\d+)_r\[({_FLOAT_PATTERN})_({_FLOAT_PATTERN})\]$",
            re.IGNORECASE,
        ),
    ),
    (
        "moo_svgd",
        re.compile(r"^particle_(\d+)$", re.IGNORECASE),
    ),
    (
        "moead",
        re.compile(
            rf"^sub_(\d+)_w\[({_FLOAT_PATTERN})_({_FLOAT_PATTERN})\]$",
            re.IGNORECASE,
        ),
    ),
)
_BASELINE_CACHE_VERSION = 2


def _parse_baseline_solution_name(name):
    """Return strategy, stable solution index, and optional weights."""
    for strategy, pattern in _BASELINE_SOLUTION_PATTERNS:
        match = pattern.match(name)
        if match is None:
            continue
        groups = match.groups()
        if strategy in {"weighted_sum", "tchebycheff_sum"}:
            return strategy, None, (float(groups[0]), float(groups[1]))
        if strategy in {"epo", "moead"}:
            return strategy, int(groups[0]), (
                float(groups[1]),
                float(groups[2]),
            )
        return strategy, int(groups[0]), (np.nan, np.nan)
    return None


def _read_json_if_possible(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _moead_strategy_for_run(run_dir):
    summary = _read_json_if_possible(Path(run_dir) / "summary.json")
    if summary:
        variant = str(summary.get("variant", "ca")).strip().lower()
        return strategy_from_summary_variant(variant)
    for part in reversed(Path(run_dir).parts):
        lowered = part.lower()
        match = re.match(r"^(moead_[a-z0-9_]+?)(?:\d+)?$", lowered)
        if match:
            return match.group(1)
    return "moead_ca"


def _metadata_for_baseline_run(run_dir, root, strategy):
    """Recover a seed only when a manifest or summary actually records it."""
    run_dir = Path(run_dir)
    summary = _read_json_if_possible(run_dir / "summary.json")
    if summary and summary.get("run_seed") is not None:
        return int(summary["run_seed"]), "summary"

    candidates = [run_dir / "run_manifest.json"]
    parent = run_dir.parent
    root = Path(root).resolve()
    for _ in range(2):
        candidates.append(parent / "run_manifest.json")
        if parent.resolve() == root or parent.parent == parent:
            break
        parent = parent.parent
    for path in candidates:
        manifest = _read_json_if_possible(path)
        if not manifest or manifest.get("seed") is None:
            continue
        recorded = canonical_method_name(manifest.get("train_strategy", ""))
        if recorded and recorded != strategy:
            continue
        return int(manifest["seed"]), "manifest"
    return None, "unrecorded"


def discover_baseline_runs(root, methods=None, prompt_id=None):
    """Discover checkpoint-backed runs from all supported baseline layouts.

    A run is identified by a directory whose immediate child directories use
    one of the known solution naming conventions.  This deliberately avoids
    relying on the inconsistent extra nesting layers in historical logs.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    methods = (
        {canonical_method_name(method) for method in methods}
        if methods else None
    )
    specs = []
    for current, directory_names, _ in os.walk(root):
        directory_names[:] = [
            name for name in directory_names
            if not name.startswith(".") and name.lower() != "old"
        ]
        parsed_children = []
        recognized_names = set()
        for name in directory_names:
            parsed = _parse_baseline_solution_name(name)
            if parsed is None:
                continue
            child = Path(current) / name
            checkpoint = final_checkpoint(child)
            if checkpoint is None:
                continue
            parsed_children.append((child, checkpoint, *parsed))
            recognized_names.add(name)
        if not parsed_children:
            continue

        # Solution directories contain thousands of images but no additional
        # runs, so prune them from the recursive walk once the parent is known.
        directory_names[:] = [
            name for name in directory_names if name not in recognized_names
        ]
        raw_strategies = {item[2] for item in parsed_children}
        if len(raw_strategies) != 1:
            continue
        raw_strategy = next(iter(raw_strategies))
        run_dir = Path(current)
        strategy = (
            _moead_strategy_for_run(run_dir)
            if raw_strategy == "moead" else raw_strategy
        )
        strategy = canonical_method_name(strategy)
        if methods and strategy not in methods:
            continue
        if prompt_id and prompt_id not in run_dir.parts:
            continue

        solutions = []
        used_indices = set()
        implicit_index = 0
        for child, checkpoint, _, parsed_index, weight in sorted(
            parsed_children,
            key=lambda item: item[0].name,
        ):
            index = parsed_index
            if index is None:
                while implicit_index in used_indices:
                    implicit_index += 1
                index = implicit_index
            if index in used_indices:
                raise ValueError(
                    f"Duplicate solution index {index} under {run_dir}."
                )
            used_indices.add(index)
            solutions.append({
                "index": int(index),
                "weight": tuple(weight),
                "directory": child,
                "checkpoint": checkpoint,
            })
        solutions.sort(key=lambda item: item["index"])
        seed, seed_source = _metadata_for_baseline_run(
            run_dir,
            root,
            strategy,
        )
        specs.append({
            "strategy": strategy,
            "variant": strategy,
            "seed": seed,
            "seed_source": seed_source,
            "run_dir": run_dir,
            "run_name": run_dir.name,
            "solutions": solutions,
        })

    specs.sort(key=lambda item: (
        item["strategy"],
        str(item["run_dir"]).lower(),
    ))
    return specs


def _checkpoint_fingerprints(spec):
    fingerprints = []
    for solution in spec["solutions"]:
        checkpoint = Path(solution["checkpoint"]).resolve()
        stat = checkpoint.stat()
        fingerprints.append({
            "index": int(solution["index"]),
            "path": str(checkpoint),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    return fingerprints


def _baseline_cache_path(cache_dir, spec):
    identity = str(Path(spec["run_dir"]).resolve()).encode("utf-8")
    digest = hashlib.sha1(identity).hexdigest()[:12]
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", spec["run_name"])
    return Path(cache_dir) / spec["strategy"] / f"{safe_name}_{digest}.json"


def _baseline_evaluation_config(prompts, steps, repeats, rng_seed):
    return {
        "prompts": list(prompts),
        "clip_model": str(CLIP_MODEL_NAME),
        "rollout_steps": int(steps),
        "repeats": int(repeats),
        "rng_seed": int(rng_seed),
    }


def _run_from_baseline_objectives(spec, objectives, prompts, cache_path,
                                  evaluation):
    objectives = np.asarray(objectives, dtype=np.float64)
    expected_shape = (len(spec["solutions"]), 2)
    if objectives.shape != expected_shape:
        raise ValueError(
            f"Expected baseline objectives shaped {expected_shape}, "
            f"got {objectives.shape} for {spec['run_dir']}."
        )
    weights = np.asarray([
        solution["weight"] for solution in spec["solutions"]
    ], dtype=np.float64)
    front = set(pareto_indices(objectives))
    records = []
    for position, (solution, objective) in enumerate(zip(
        spec["solutions"], objectives
    )):
        records.append({
            "index": int(solution["index"]),
            "objective": objective,
            "similarity": -objective,
            "weight": weights[position],
            "checkpoint": Path(solution["checkpoint"]),
            "is_non_dominated": position in front,
        })
    return {
        "strategy": spec["strategy"],
        "variant": spec["variant"],
        "seed": spec["seed"],
        "seed_source": spec["seed_source"],
        "rollout_seed": int(evaluation["rng_seed"]),
        "prompts": tuple(prompts),
        "run_dir": Path(spec["run_dir"]),
        "run_name": spec["run_name"],
        "summary_path": Path(cache_path),
        "ideal_point_rule": "common_checkpoint_reevaluation",
        "evaluation_source": "checkpoint_reevaluation",
        "evaluation_steps": int(evaluation["rollout_steps"]),
        "evaluation_repeats": int(evaluation["repeats"]),
        "evaluation_seed": int(evaluation["rng_seed"]),
        "records": records,
    }


def _load_baseline_cache(spec, cache_path, evaluation, refresh=False):
    if refresh or not Path(cache_path).is_file():
        return None
    payload = _read_json_if_possible(cache_path)
    if not payload:
        return None
    if payload.get("version") != _BASELINE_CACHE_VERSION:
        return None
    if payload.get("evaluation") != evaluation:
        return None
    if payload.get("checkpoints") != _checkpoint_fingerprints(spec):
        return None
    try:
        return _run_from_baseline_objectives(
            spec,
            payload["objectives"],
            evaluation["prompts"],
            cache_path,
            evaluation,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _write_baseline_cache(spec, cache_path, evaluation, objectives):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _BASELINE_CACHE_VERSION,
        "strategy": spec["strategy"],
        "run_name": spec["run_name"],
        "run_dir": str(Path(spec["run_dir"]).resolve()),
        "seed": spec["seed"],
        "seed_source": spec["seed_source"],
        "evaluation": evaluation,
        "checkpoints": _checkpoint_fingerprints(spec),
        "objectives": np.asarray(objectives, dtype=np.float64).tolist(),
    }
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, cache_path)


@torch.no_grad()
def _rollout_checkpoint_repeats(path, rollout_steps, repeats, rng_seed,
                                device, seed_preference=None):
    model = CAModel().to(device)
    model.load_state_dict(load_state_dict(path, device))
    model.eval()
    size = TARGET_SIZE + 2 * TARGET_PADDING
    devices = []
    if device.type == "cuda":
        devices = [
            device.index
            if device.index is not None else torch.cuda.current_device()
        ]
    states = []
    for repeat in range(int(repeats)):
        with torch.random.fork_rng(devices=devices):
            repeat_seed = int(rng_seed) + repeat
            torch.manual_seed(repeat_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(repeat_seed)
            initial = make_seed(size, n=1)
            if seed_preference is not None:
                preference = np.asarray(seed_preference, dtype=np.float32).reshape(-1)
                center_row = initial.shape[-2] // 2
                center_column = initial.shape[-1] // 2
                initial[
                    :, 4:4 + len(preference), center_row, center_column
                ] = preference[None]
            state = torch.tensor(initial, device=device)
            for _ in range(int(rollout_steps)):
                state = finite_clip_state(model(state))
            states.append(state.detach())
    del model
    return torch.cat(states, dim=0)


@torch.no_grad()
def _evaluate_baseline_spec(spec, clip_loss, text_embeddings, evaluation,
                            batch_size):
    device = next(clip_loss.model.parameters()).device
    objectives = {}
    pending = []
    pending_count = 0

    def flush():
        nonlocal pending, pending_count
        if not pending:
            return
        states = torch.cat([item[1] for item in pending], dim=0)
        losses = clip_loss.compute_objective_losses(states, text_embeddings)
        offset = 0
        for solution, state in pending:
            count = len(state)
            vector = [
                float(loss[offset:offset + count].mean().item())
                for loss in losses
            ]
            if not np.all(np.isfinite(vector)):
                raise ValueError(
                    f"Non-finite CLIP objectives for {solution['checkpoint']}."
                )
            objectives[int(solution["index"])] = vector
            offset += count
        del states
        pending = []
        pending_count = 0

    for solution in spec["solutions"]:
        try:
            state = _rollout_checkpoint_repeats(
                solution["checkpoint"],
                evaluation["rollout_steps"],
                evaluation["repeats"],
                evaluation["rng_seed"],
                device,
                seed_preference=None,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to evaluate checkpoint {solution['checkpoint']}: {exc}"
            ) from exc
        if pending and pending_count + len(state) > int(batch_size):
            flush()
        pending.append((solution, state))
        pending_count += len(state)
    flush()
    return [objectives[int(solution["index"])] for solution in spec["solutions"]]


def evaluate_baseline_runs(specs, prompts, cache_dir, rollout_steps,
                           repeats, rng_seed, batch_size=8, refresh=False):
    """Load cached objectives or uniformly reevaluate baseline checkpoints."""
    if len(prompts) != 2:
        raise ValueError(
            "Baseline comparison currently requires exactly two prompts, "
            f"got {len(prompts)}."
        )
    rollout_steps = max(1, int(rollout_steps))
    repeats = max(1, int(repeats))
    batch_size = max(1, int(batch_size))
    evaluation = _baseline_evaluation_config(
        prompts,
        rollout_steps,
        repeats,
        rng_seed,
    )
    runs_by_path = {}
    pending = []
    for spec in specs:
        cache_path = _baseline_cache_path(cache_dir, spec)
        run = _load_baseline_cache(
            spec,
            cache_path,
            evaluation,
            refresh=refresh,
        )
        if run is None:
            pending.append((spec, cache_path))
        else:
            runs_by_path[Path(spec["run_dir"]).resolve()] = run
            print(
                f'Loaded baseline cache: {spec["strategy"]} / '
                f'{spec["run_name"]}'
            )

    if pending:
        from clip_loss import CLIPLoss

        print(
            f"Loading CLIP to evaluate {len(pending)} uncached baseline "
            f"run(s) with {rollout_steps} rollout steps x {repeats} repeat(s)."
        )
        clip_loss = CLIPLoss()
        text_embeddings = clip_loss.embed_objective_prompts(prompts)
        for spec, cache_path in pending:
            print(
                f'Evaluating {spec["strategy"]} / {spec["run_name"]}: '
                f'{len(spec["solutions"])} checkpoint(s)...',
                flush=True,
            )
            objectives = _evaluate_baseline_spec(
                spec,
                clip_loss,
                text_embeddings,
                evaluation,
                batch_size,
            )
            _write_baseline_cache(
                spec,
                cache_path,
                evaluation,
                objectives,
            )
            runs_by_path[Path(spec["run_dir"]).resolve()] = (
                _run_from_baseline_objectives(
                    spec,
                    objectives,
                    prompts,
                    cache_path,
                    evaluation,
                )
            )
            print(f"Saved baseline cache: {cache_path}")
        del clip_loss

    return [
        runs_by_path[Path(spec["run_dir"]).resolve()]
        for spec in specs
    ]


@lru_cache(maxsize=None)
def _read_release_model_index(path):
    """Read a content-addressed release index keyed by run and slot."""

    path = Path(path)
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["relative_run_dir"], int(row["subproblem_index"]))
            if key in rows:
                raise ValueError(f"Duplicate model-index slot: {key}")
            rows[key] = row["relative_model_path"]
    return rows


def _indexed_subproblem_checkpoints(run_dir, solution_count):
    """Resolve release checkpoints from the nearest model index, if present."""

    run_dir = Path(run_dir).resolve()
    for release_root in (run_dir, *run_dir.parents):
        index_path = release_root / "model_index.csv"
        if not index_path.is_file():
            continue
        try:
            relative_run = run_dir.relative_to(release_root).as_posix()
        except ValueError:
            continue
        index = _read_release_model_index(index_path.resolve())
        checkpoints = []
        for solution_index in range(int(solution_count)):
            relative_model = index.get((relative_run, solution_index))
            if relative_model is None:
                raise ValueError(
                    f"Missing model-index slot {relative_run} / {solution_index}"
                )
            checkpoint = (release_root / relative_model).resolve()
            if release_root not in checkpoint.parents or not checkpoint.is_file():
                raise ValueError(f"Invalid indexed checkpoint: {relative_model}")
            checkpoints.append(checkpoint)
        return checkpoints
    return None


def _load_subproblem_checkpoints(summary_path, run_dir, solution_count):
    """Find checkpoints by explicit subproblem index across legacy run layers."""
    roots = [Path(summary_path).parent, Path(run_dir)]
    summary_parent = Path(summary_path).parent.parent
    roots.append(summary_parent)
    unique_roots = []
    seen_roots = set()
    for root in roots:
        root = root.resolve()
        if root in seen_roots or not root.is_dir():
            continue
        seen_roots.add(root)
        unique_roots.append(root)

    checkpoints = {}
    for root in unique_roots:
        for directory in sorted(root.glob("sub_*")):
            if not directory.is_dir():
                continue
            match = _SUBPROBLEM_PATTERN.match(directory.name)
            if match is None:
                continue
            index = int(match.group(1))
            if index >= int(solution_count) or index in checkpoints:
                continue
            checkpoint = final_checkpoint(directory)
            if checkpoint is not None:
                checkpoints[index] = checkpoint
    indexed = _indexed_subproblem_checkpoints(run_dir, solution_count)
    if indexed is not None:
        for index, checkpoint in enumerate(indexed):
            checkpoints.setdefault(index, checkpoint)
    return [checkpoints.get(index) for index in range(int(solution_count))]


def _load_completed_run(summary_path, strategy, variant, seed, prompts,
                        run_dir):
    """Load one manifest-backed or legacy summary-backed MOEA/D run."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    objectives = np.asarray(summary["final_objectives"], dtype=np.float64)
    if objectives.ndim != 2 or objectives.shape[1] != 2:
        raise ValueError(
            f"Expected two objectives in {summary_path}, got {objectives.shape}."
        )

    checkpoints = _load_subproblem_checkpoints(
        summary_path,
        run_dir,
        len(objectives),
    )
    weights = np.asarray(summary.get("weights", []), dtype=np.float64)
    if weights.shape != objectives.shape:
        weights = np.full_like(objectives, np.nan)

    front = set(pareto_indices(objectives))
    records = []
    for index, objective in enumerate(objectives):
        records.append({
            "index": index,
            "objective": objective,
            "similarity": -objective,
            "weight": weights[index],
            "checkpoint": checkpoints[index] if index < len(checkpoints) else None,
            "is_non_dominated": index in front,
        })

    return {
        "strategy": strategy,
        "variant": variant,
        "seed": int(seed),
        "seed_source": "summary_or_manifest",
        "rollout_seed": int(seed),
        "prompts": tuple(prompts),
        "run_dir": Path(run_dir),
        "run_name": Path(run_dir).name,
        "summary_path": summary_path,
        "ideal_point_rule": summary.get("ideal_point_rule", "unspecified"),
        "evaluation_source": "summary",
        "evaluation_steps": summary.get("final_eval_steps"),
        "evaluation_repeats": summary.get("final_eval_repeats"),
        "evaluation_seed": None,
        "records": records,
    }


def discover_runs(root, methods=None, prompt_id=None):
    """Discover manifest-backed runs and legacy MOEA/D summary directories."""
    root = Path(root)
    methods = (
        {canonical_method_name(method) for method in methods}
        if methods else None
    )
    runs = []
    loaded_summaries = set()

    # Semantic-path runs use a separate manifest so the original main.py and
    # its historical run_manifest.json contract remain untouched.
    for manifest_path in sorted(root.rglob("semantic_run_manifest.json")):
        run_dir = manifest_path.parent
        if "old" in {part.lower() for part in run_dir.parts}:
            continue
        if prompt_id and prompt_id not in run_dir.parts:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        internal_variant = str(manifest.get("variant", "")).strip().lower()
        strategy = canonical_method_name(internal_variant)
        if not strategy.startswith("moead_"):
            continue
        if methods and strategy not in methods:
            continue
        summary_variant = f"semantic_{internal_variant}"
        summary_path = run_dir / summary_variant / "summary.json"
        if not summary_path.exists():
            continue
        prompts = (
            str(manifest.get("prompt1", OBJECTIVE_PROMPTS[0])),
            str(manifest.get("prompt2", OBJECTIVE_PROMPTS[1])),
        )
        runs.append(_load_completed_run(
            summary_path=summary_path,
            strategy=strategy,
            variant=summary_variant,
            seed=int(manifest.get("seed", 0)),
            prompts=prompts,
            run_dir=run_dir,
        ))
        loaded_summaries.add(summary_path.resolve())

    for manifest_path in sorted(root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        if "old" in {part.lower() for part in run_dir.parts}:
            continue
        if prompt_id and prompt_id not in run_dir.parts:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "train_strategy" in manifest:
            strategy = canonical_method_name(manifest["train_strategy"])
            variant = str(manifest.get("variant_name", strategy)).strip().lower()
            summary_path = run_dir / variant / "summary.json"
            if not summary_path.exists():
                # Current main.py writes run_manifest.json beside summary.json;
                # retain the nested path above for older experiment layouts.
                summary_path = run_dir / "summary.json"
        else:
            # Standalone state-aware baselines keep their summary beside the
            # manifest and record a non-MOEA/D variant explicitly.
            variant = str(manifest.get("variant", "")).strip().lower()
            if variant not in EXTERNAL_POPULATION_METHODS:
                continue
            strategy = variant
            summary_path = run_dir / "summary.json"
        if methods and strategy not in methods:
            continue
        if not summary_path.exists():
            continue
        runs.append(_load_completed_run(
            summary_path=summary_path,
            strategy=strategy,
            variant=variant,
            seed=int(manifest.get("seed", 0)),
            prompts=manifest.get(
                "objective_prompts",
                (
                    manifest.get("prompt1", OBJECTIVE_PROMPTS[0]),
                    manifest.get("prompt2", OBJECTIVE_PROMPTS[1]),
                ),
            ),
            run_dir=run_dir,
        ))
        loaded_summaries.add(summary_path.resolve())

    # Compatibility with completed runs created before run_manifest.json was
    # introduced. Archive-only summaries are excluded because they do not
    # represent a final active population.
    for summary_path in sorted(root.rglob("summary.json")):
        if summary_path.resolve() in loaded_summaries:
            continue
        if "old" in {part.lower() for part in summary_path.parts}:
            continue
        if summary_path.parent.name.lower() == "archive":
            continue
        if prompt_id and prompt_id not in summary_path.parts:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "final_objectives" not in summary:
            continue
        variant = str(summary.get("variant", summary_path.parent.name)).strip().lower()
        strategy = strategy_from_summary_variant(variant)
        if methods and strategy not in methods:
            continue
        seed = summary.get("run_seed", 0)
        if seed is None:
            seed = 0
        runs.append(_load_completed_run(
            summary_path=summary_path,
            strategy=strategy,
            variant=variant,
            seed=seed,
            prompts=summary.get("objective_prompts", OBJECTIVE_PROMPTS),
            run_dir=summary_path.parent,
        ))
        loaded_summaries.add(summary_path.resolve())

    return runs


def run_statistics(run):
    """Compute final active-population metrics for one completed run."""
    objectives = np.asarray([
        record["objective"] for record in run["records"]
    ], dtype=np.float64)
    if objectives.ndim != 2 or objectives.shape[1] != 2 or not len(objectives):
        raise ValueError(
            "Final active-population objectives must have shape [N, 2]; "
            f"received {objectives.shape!r} for {run.get('run_name', 'run')}."
        )
    if not np.isfinite(objectives).all():
        raise ValueError(
            "Final active-population objectives contain non-finite values for "
            f"{run.get('run_name', 'run')}."
        )
    front = objectives[pareto_indices(objectives)]
    similarities = -objectives
    worst_objective_similarities = similarities.min(axis=1)
    similarity_extents = np.ptp(similarities, axis=0)
    return {
        "n_solutions": len(objectives),
        "n_non_dominated": len(front),
        "hypervolume": hypervolume_2d(-front),
        "spacing": spacing(front),
        # Population-average balance and best attainable compromise are kept
        # separate: a concentrated population can improve the former without
        # improving the latter.
        "population_average_worst_similarity": float(
            worst_objective_similarities.mean()
        ),
        "best_max_min_similarity": float(
            worst_objective_similarities.max()
        ),
        # The scalar extent is the L1 sum of the two per-objective similarity
        # ranges.  Per-axis ranges are retained so the diagnostic is auditable.
        "objective_space_extent": float(similarity_extents.sum()),
        "similarity_extent_objective_1": float(similarity_extents[0]),
        "similarity_extent_objective_2": float(similarity_extents[1]),
    }


def parse_seed_filter(value):
    """Parse an optional comma-separated set of formal experiment seeds."""

    if value is None:
        return None
    seeds = {int(item.strip()) for item in str(value).split(",") if item.strip()}
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    return seeds


def _run_result_fingerprint(run):
    objectives = np.asarray([
        record["objective"] for record in run["records"]
    ], dtype=np.float64)
    weights = np.asarray([
        record["weight"] for record in run["records"]
    ], dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(objectives.tobytes())
    digest.update(weights.tobytes())
    digest.update(str(run.get("evaluation_steps")).encode("ascii"))
    digest.update(str(run.get("evaluation_repeats")).encode("ascii"))
    return digest.hexdigest()


def prepare_repeated_runs(runs, seeds=None):
    """Filter formal seeds, remove exact copies, and reject seed conflicts."""

    filtered = [
        run for run in runs
        if seeds is None or (
            run.get("seed") is not None and int(run["seed"]) in seeds
        )
    ]
    exact = {}
    for run in filtered:
        key = (
            run["strategy"],
            int(run["seed"]) if run.get("seed") is not None else None,
            _run_result_fingerprint(run),
        )
        current = exact.get(key)
        if current is None or str(run["summary_path"]) < str(current["summary_path"]):
            exact[key] = run
    unique = list(exact.values())

    if seeds is not None:
        by_method_seed = defaultdict(list)
        for run in unique:
            by_method_seed[(run["strategy"], int(run["seed"]))].append(run)
        conflicts = {
            key: candidates
            for key, candidates in by_method_seed.items()
            if len(candidates) > 1
        }
        if conflicts:
            details = []
            for (strategy, seed), candidates in sorted(conflicts.items()):
                paths = ", ".join(str(run["summary_path"]) for run in candidates)
                details.append(f"{strategy} seed {seed}: {paths}")
            raise ValueError(
                "Conflicting results exist for the same formal method/seed. "
                "Restrict --root before aggregating: " + " | ".join(details)
            )
    return unique


def split_statistical_and_selection_runs(runs, statistical_seeds=None,
                                         selected_run_seeds=None):
    """Separate aggregation seeds from best-run phenotype-selection seeds."""
    # Do not silently discard legacy runs whose directories predate seed
    # manifests merely because another method in the same comparison records
    # seeds.  They remain valid for exploratory visualization, while an
    # explicit --seeds filter stays strict and therefore requires metadata.
    all_formal_runs = prepare_repeated_runs(runs, seeds=None)
    statistical_runs = (
        prepare_repeated_runs(runs, seeds=statistical_seeds)
        if statistical_seeds is not None else all_formal_runs
    )
    selection_runs = (
        prepare_repeated_runs(runs, seeds=selected_run_seeds)
        if selected_run_seeds is not None else all_formal_runs
    )
    return statistical_runs, selection_runs


def select_best_runs(runs):
    """Select one run per strategy, using maximum hypervolume across repeats."""
    grouped = defaultdict(list)
    for run in runs:
        run["statistics"] = run_statistics(run)
        grouped[run["strategy"]].append(run)

    selected = []
    for strategy in sorted(grouped):
        candidates = grouped[strategy]
        best = max(
            candidates,
            key=lambda run: (
                run["statistics"]["hypervolume"],
                run["statistics"]["n_non_dominated"],
                -run["statistics"]["spacing"],
                -int(run["seed"]) if run.get("seed") is not None else 0,
                run["run_name"],
            ),
        )
        best["candidate_run_count"] = len(candidates)
        selected.append(best)
    return selected


def write_metrics(runs, output_dir):
    rows = []
    for run in runs:
        stats = run.get("statistics") or run_statistics(run)
        rows.append({
            "strategy": run["strategy"],
            "selected_run": run["run_name"],
            "selected_seed": (
                run["seed"] if run.get("seed") is not None else ""
            ),
            "candidate_runs": int(run.get("candidate_run_count", 1)),
            **stats,
            "ideal_point_rule": run["ideal_point_rule"],
            "evaluation_source": run.get("evaluation_source", "unspecified"),
            "evaluation_steps": run.get("evaluation_steps") or "",
            "evaluation_repeats": run.get("evaluation_repeats") or "",
            "evaluation_seed": (
                run.get("evaluation_seed")
                if run.get("evaluation_seed") is not None else ""
            ),
            "summary_path": str(run["summary_path"]),
        })
    path = output_dir / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows, path


def write_repeated_run_metrics(runs, output_dir):
    """Write unbiased per-run and aggregate metrics for repeated experiments."""

    per_run_rows = []
    grouped = defaultdict(list)
    for run in sorted(
        runs,
        key=lambda item: (
            item["strategy"],
            int(item["seed"]) if item.get("seed") is not None else -1,
            item["run_name"],
        ),
    ):
        stats = run.get("statistics") or run_statistics(run)
        run["statistics"] = stats
        grouped[run["strategy"]].append(run)
        per_run_rows.append({
            "strategy": run["strategy"],
            "run": run["run_name"],
            "seed": run["seed"] if run.get("seed") is not None else "",
            **stats,
            "evaluation_source": run.get("evaluation_source", "unspecified"),
            "evaluation_steps": run.get("evaluation_steps") or "",
            "evaluation_repeats": run.get("evaluation_repeats") or "",
            "summary_path": str(run["summary_path"]),
        })

    per_run_path = output_dir / "per_run_metrics.csv"
    with per_run_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_run_rows[0]))
        writer.writeheader()
        writer.writerows(per_run_rows)

    aggregate_rows = []
    for strategy in sorted(grouped):
        strategy_runs = grouped[strategy]
        hvs = np.asarray([
            run["statistics"]["hypervolume"] for run in strategy_runs
        ], dtype=np.float64)
        non_dominated = np.asarray([
            run["statistics"]["n_non_dominated"] for run in strategy_runs
        ], dtype=np.float64)
        spacings = np.asarray([
            run["statistics"]["spacing"] for run in strategy_runs
        ], dtype=np.float64)
        population_balance = np.asarray([
            run["statistics"]["population_average_worst_similarity"]
            for run in strategy_runs
        ], dtype=np.float64)
        best_compromise = np.asarray([
            run["statistics"]["best_max_min_similarity"]
            for run in strategy_runs
        ], dtype=np.float64)
        objective_extents = np.asarray([
            run["statistics"]["objective_space_extent"]
            for run in strategy_runs
        ], dtype=np.float64)
        objective_1_extents = np.asarray([
            run["statistics"]["similarity_extent_objective_1"]
            for run in strategy_runs
        ], dtype=np.float64)
        objective_2_extents = np.asarray([
            run["statistics"]["similarity_extent_objective_2"]
            for run in strategy_runs
        ], dtype=np.float64)
        seeds = sorted({
            int(run["seed"])
            for run in strategy_runs
            if run.get("seed") is not None
        })
        aggregate_rows.append({
            "strategy": strategy,
            "run_count": len(strategy_runs),
            "seeds": ";".join(str(seed) for seed in seeds),
            "hypervolume_mean": float(hvs.mean()),
            "hypervolume_std": (
                float(hvs.std(ddof=1)) if len(hvs) > 1 else ""
            ),
            "hypervolume_min": float(hvs.min()),
            "hypervolume_max": float(hvs.max()),
            "n_non_dominated_mean": float(non_dominated.mean()),
            "n_non_dominated_std": (
                float(non_dominated.std(ddof=1))
                if len(non_dominated) > 1 else ""
            ),
            "spacing_mean": float(spacings.mean()),
            "spacing_std": (
                float(spacings.std(ddof=1)) if len(spacings) > 1 else ""
            ),
            "population_average_worst_similarity_mean": float(
                population_balance.mean()
            ),
            "population_average_worst_similarity_std": (
                float(population_balance.std(ddof=1))
                if len(population_balance) > 1 else ""
            ),
            "population_average_worst_similarity_min": float(
                population_balance.min()
            ),
            "population_average_worst_similarity_max": float(
                population_balance.max()
            ),
            "best_max_min_similarity_mean": float(best_compromise.mean()),
            "best_max_min_similarity_std": (
                float(best_compromise.std(ddof=1))
                if len(best_compromise) > 1 else ""
            ),
            "best_max_min_similarity_min": float(best_compromise.min()),
            "best_max_min_similarity_max": float(best_compromise.max()),
            "objective_space_extent_mean": float(objective_extents.mean()),
            "objective_space_extent_std": (
                float(objective_extents.std(ddof=1))
                if len(objective_extents) > 1 else ""
            ),
            "objective_space_extent_min": float(objective_extents.min()),
            "objective_space_extent_max": float(objective_extents.max()),
            "similarity_extent_objective_1_mean": float(
                objective_1_extents.mean()
            ),
            "similarity_extent_objective_1_std": (
                float(objective_1_extents.std(ddof=1))
                if len(objective_1_extents) > 1 else ""
            ),
            "similarity_extent_objective_2_mean": float(
                objective_2_extents.mean()
            ),
            "similarity_extent_objective_2_std": (
                float(objective_2_extents.std(ddof=1))
                if len(objective_2_extents) > 1 else ""
            ),
        })

    aggregate_path = output_dir / "aggregate_metrics.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return per_run_rows, per_run_path, aggregate_rows, aggregate_path


PAIRED_METRICS = (
    "hypervolume",
    "n_non_dominated",
    "spacing",
    "population_average_worst_similarity",
    "best_max_min_similarity",
    "objective_space_extent",
    "similarity_extent_objective_1",
    "similarity_extent_objective_2",
)


def write_paired_run_differences(runs, output_dir):
    """Write same-seed pairwise method differences using runs as replicates.

    Each ``*_difference`` column is explicitly strategy B minus strategy A.
    Runs without recorded seed metadata cannot be paired and are omitted.
    """

    by_method_seed = {}
    for run in runs:
        if run.get("seed") is None:
            continue
        key = (run["strategy"], int(run["seed"]))
        if key in by_method_seed:
            raise ValueError(
                "Paired statistics require exactly one run per method/seed; "
                f"found multiple runs for {key[0]} seed {key[1]}."
            )
        run["statistics"] = run.get("statistics") or run_statistics(run)
        by_method_seed[key] = run

    strategies = sorted({strategy for strategy, _ in by_method_seed})
    rows = []
    for strategy_a, strategy_b in itertools.combinations(strategies, 2):
        seeds_a = {
            seed for strategy, seed in by_method_seed if strategy == strategy_a
        }
        seeds_b = {
            seed for strategy, seed in by_method_seed if strategy == strategy_b
        }
        for seed in sorted(seeds_a & seeds_b):
            run_a = by_method_seed[(strategy_a, seed)]
            run_b = by_method_seed[(strategy_b, seed)]
            row = {
                "strategy_a": strategy_a,
                "strategy_b": strategy_b,
                "seed": seed,
                "difference_definition": f"{strategy_b}_minus_{strategy_a}",
                "run_a": run_a["run_name"],
                "run_b": run_b["run_name"],
            }
            for metric in PAIRED_METRICS:
                value_a = float(run_a["statistics"][metric])
                value_b = float(run_b["statistics"][metric])
                row[f"{metric}_a"] = value_a
                row[f"{metric}_b"] = value_b
                row[f"{metric}_difference"] = value_b - value_a
            row["summary_path_a"] = str(run_a["summary_path"])
            row["summary_path_b"] = str(run_b["summary_path"])
            rows.append(row)

    fieldnames = [
        "strategy_a",
        "strategy_b",
        "seed",
        "difference_definition",
        "run_a",
        "run_b",
    ]
    for metric in PAIRED_METRICS:
        fieldnames.extend((
            f"{metric}_a",
            f"{metric}_b",
            f"{metric}_difference",
        ))
    fieldnames.extend(("summary_path_a", "summary_path_b"))
    path = output_dir / "paired_run_differences.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows, path


def draw_objective_space(axis, runs, prompts, compact=False):
    """Draw one connected empirical Pareto front for each selected strategy."""
    methods = sorted(run["strategy"] for run in runs)
    styles = method_styles(methods)
    method_handles = []

    for run in sorted(runs, key=lambda item: item["strategy"]):
        method = run["strategy"]
        style = styles[method]
        records = [record for record in run["records"] if record["is_non_dominated"]]
        if records:
            similarities = np.asarray([
                record["similarity"] for record in records
            ])
            order = np.argsort(similarities[:, 0], kind="stable")
            similarities = similarities[order]
            axis.plot(
                similarities[:, 0],
                similarities[:, 1],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.7 if compact else 1.8,
                marker=style["marker"],
                markersize=4.8 if compact else 5.2,
                markerfacecolor=style["color"],
                markeredgecolor=INK,
                markeredgewidth=0.45,
                alpha=0.97,
                solid_capstyle="round",
                dash_capstyle="round",
                zorder=3,
            )
        stats = run.get("statistics") or run_statistics(run)
        method_handles.append(Line2D(
            [0], [0],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5.0,
            linewidth=1.7,
            markerfacecolor=style["color"],
            markeredgecolor=INK,
            markeredgewidth=0.45,
            label=(
                method_display_name(method)
                if compact else
                f'{method_display_name(method)}  ({run_identity(run)}, '
                f'HV={stats["hypervolume"]:.3f})'
            ),
        ))

    prompt1 = prompts[0] if len(prompts) > 0 else "objective 1"
    prompt2 = prompts[1] if len(prompts) > 1 else "objective 2"
    if compact:
        x_prompt = "\n".join(textwrap.wrap(
            prompt1, width=34, break_long_words=False, break_on_hyphens=False,
        ))
        y_prompt = "\n".join(textwrap.wrap(
            prompt2, width=34, break_long_words=False, break_on_hyphens=False,
        ))
        axis.set_xlabel(f'Similarity to "{x_prompt}"', labelpad=5)
        axis.set_ylabel(f'Similarity to "{y_prompt}"', labelpad=5)
    else:
        axis.set_xlabel(f'CLIP similarity: "{prompt1}"')
        axis.set_ylabel(f'CLIP similarity: "{prompt2}"')
        axis.set_title("Connected empirical Pareto fronts", pad=8,
                       fontweight="semibold")
    axis.margins(x=0.055, y=0.065)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5 if compact else 6))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.minorticks_off()
    axis.grid(
        True,
        which="major",
        axis="both",
        color=GRID,
        linewidth=0.5,
        alpha=0.58,
    )
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", which="major", direction="out", length=3.2,
                     width=0.7, pad=3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_aspect("auto" if compact else "equal", adjustable="box")
    if compact:
        axis.legend(
            handles=method_handles,
            loc="best",
            fontsize=7,
            handlelength=2.25,
            handletextpad=0.55,
            labelspacing=0.3,
            borderpad=0.35,
            frameon=True,
            framealpha=0.9,
            facecolor="white",
            edgecolor="#C7C7C7",
        )
    else:
        axis.legend(
            handles=method_handles,
            title="Selected run",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            ncol=2 if len(methods) > 10 else 1,
        )
    return styles


def plot_objective_space(runs, prompts, output_dir):
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    draw_objective_space(axis, runs, prompts, compact=False)
    figure.tight_layout()
    path, _ = save_publication_figure(figure, output_dir, "objective_space")
    plt.close(figure)
    return path


def plot_hypervolume(metrics, output_dir):
    """Compare the best available-HV run of every strategy."""
    rows = sorted(metrics, key=lambda row: (-float(row["hypervolume"]), row["strategy"]))
    methods = [row["strategy"] for row in rows]
    values = [float(row["hypervolume"]) for row in rows]
    styles = method_styles(methods)
    colors = [styles[method]["color"] for method in methods]
    upper = max(values, default=1.0) * 1.16

    if len(methods) <= 8:
        figure, axis = plt.subplots(figsize=(max(5.6, len(methods) * 0.82), 4.2))
        positions = np.arange(len(methods))
        bars = axis.bar(
            positions,
            values,
            width=0.66,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [
                f'{method_display_name(row["strategy"])}\n'
                f'seed={row["selected_seed"]}'
                for row in rows
            ],
            rotation=28,
            ha="right",
        )
        axis.set_ylabel("Hypervolume")
        axis.set_ylim(0.0, upper)
        axis.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.7)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + upper * 0.015,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK,
            )
    else:
        figure, axis = plt.subplots(figsize=(7.2, max(5.0, len(methods) * 0.25)))
        positions = np.arange(len(methods))
        bars = axis.barh(
            positions,
            values,
            height=0.72,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
        )
        axis.set_yticks(positions)
        axis.set_yticklabels([
            f'{method_display_name(row["strategy"])} '
            f'(seed={row["selected_seed"]})'
            for row in rows
        ])
        axis.invert_yaxis()
        axis.set_xlabel("Hypervolume")
        axis.set_xlim(0.0, upper)
        axis.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.7)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_width() + upper * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}",
                ha="left",
                va="center",
                fontsize=7,
                color=INK,
            )

    axis.set_title(
        "Best available-run hypervolume",
        pad=8,
        fontweight="semibold",
    )
    axis.text(
        1.0,
        1.01,
        "Best seed selected independently per method; descriptive only",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#66615C",
    )
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    path, _ = save_publication_figure(figure, output_dir, "hypervolume")
    plt.close(figure)
    return path


def plot_aggregate_hypervolume(metrics, output_dir):
    """Plot mean hypervolume with sample-standard-deviation error bars."""

    rows = sorted(
        metrics,
        key=lambda row: (-float(row["hypervolume_mean"]), row["strategy"]),
    )
    methods = [row["strategy"] for row in rows]
    values = np.asarray([
        float(row["hypervolume_mean"]) for row in rows
    ], dtype=np.float64)
    errors = np.asarray([
        float(row["hypervolume_std"])
        if row["hypervolume_std"] != "" else 0.0
        for row in rows
    ], dtype=np.float64)
    styles = method_styles(methods)
    colors = [styles[method]["color"] for method in methods]
    upper = max((values + errors).max(initial=0.0) * 1.18, 1e-6)

    figure, axis = plt.subplots(
        figsize=(max(6.2, len(methods) * 0.9), 4.4)
    )
    positions = np.arange(len(methods))
    bars = axis.bar(
        positions,
        values,
        yerr=errors,
        capsize=3,
        width=0.66,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        error_kw={"elinewidth": 0.9, "ecolor": INK},
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [method_display_name(method) for method in methods],
        rotation=28,
        ha="right",
    )
    axis.set_ylabel("Hypervolume")
    axis.set_ylim(0.0, upper)
    axis.set_title("Mean hypervolume across formal seeds", fontweight="semibold")
    axis.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for bar, row, value in zip(bars, rows, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + upper * 0.025,
            f'{value:.4f}\nn={row["run_count"]}',
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )
    figure.tight_layout()
    path, _ = save_publication_figure(
        figure, output_dir, "hypervolume_mean_std"
    )
    plt.close(figure)
    return path


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def render_checkpoint(path, rollout_steps, seed_value, device,
                      seed_preference=None, return_state=False):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    model = CAModel().to(device)
    model.load_state_dict(load_state_dict(path, device))
    model.eval()
    size = TARGET_SIZE + 2 * TARGET_PADDING
    initial = make_seed(size, n=1)
    if seed_preference is not None:
        preference = np.asarray(seed_preference, dtype=np.float32).reshape(-1)
        center_row = initial.shape[-2] // 2
        center_column = initial.shape[-1] // 2
        initial[:, 4:4 + len(preference), center_row, center_column] = (
            preference[None]
        )
    state = torch.tensor(initial, device=device)
    for _ in range(rollout_steps):
        state = finite_clip_state(model(state))
    image = to_rgb(state)[0]
    if return_state:
        return image, state.detach().cpu()
    return image


def _alpha_foreground_bbox(state, threshold=0.1, padding=3):
    """Return a padded alpha-mask bounding box and occupied canvas fraction."""
    alpha = state[0, 3].detach().cpu().numpy()
    mask = np.isfinite(alpha) & (alpha > float(threshold))
    occupied_fraction = float(mask.mean())
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return None, occupied_fraction
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0) + 1
    top = max(0, int(top) - int(padding))
    left = max(0, int(left) - int(padding))
    bottom = min(mask.shape[0], int(bottom) + int(padding))
    right = min(mask.shape[1], int(right) + int(padding))
    return (top, bottom, left, right), occupied_fraction


def _crop_to_bbox(image, bbox):
    if bbox is None:
        return image
    top, bottom, left, right = bbox
    cropped = np.asarray(image)[top:bottom, left:right]
    height, width = cropped.shape[:2]
    side = max(height, width)
    if side == 0 or height == width:
        return cropped
    pad_top = (side - height) // 2
    pad_bottom = side - height - pad_top
    pad_left = (side - width) // 2
    pad_right = side - width - pad_left
    padding = ((pad_top, pad_bottom), (pad_left, pad_right))
    if cropped.ndim == 3:
        padding += ((0, 0),)
    return np.pad(cropped, padding, mode="constant", constant_values=1.0)


def _human_readable_image(state, alpha_threshold=0.1):
    """Render one CA state with sigmoid color and a binary alive mask."""
    values = state.detach().cpu().numpy()
    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=STATE_CLIP_VALUE,
        neginf=-STATE_CLIP_VALUE,
    )
    values = np.clip(values, -STATE_CLIP_VALUE, STATE_CLIP_VALUE)
    rgb = 1.0 / (1.0 + np.exp(-values[:, :3]))
    alpha = (values[:, 3:4] > float(alpha_threshold)).astype(np.float32)
    image = 1.0 - alpha + alpha * rgb
    return np.clip(image[0].transpose(1, 2, 0), 0.0, 1.0)


@torch.no_grad()
def _attach_displayed_similarities(rendered, clip_loss, text_embeddings,
                                   batch_size=32, human_render=True):
    """Score rollout states while independently choosing their display view."""
    if not rendered:
        return []
    device = next(clip_loss.model.parameters()).device
    result = []
    for start in range(0, len(rendered), int(batch_size)):
        chunk = rendered[start:start + int(batch_size)]
        states = torch.cat([state for _, _, state in chunk], dim=0).to(device)
        losses = clip_loss.compute_objective_losses(states, text_embeddings)
        similarities = torch.stack([-loss for loss in losses], dim=1)
        similarities = similarities.detach().cpu().numpy()
        for (record, canonical_image, state), similarity in zip(
            chunk, similarities
        ):
            scored_record = dict(record)
            scored_record["displayed_similarity"] = np.asarray(
                similarity, dtype=np.float64
            )
            bbox, occupied_fraction = _alpha_foreground_bbox(state)
            scored_record["display_foreground_bbox"] = bbox
            scored_record["display_occupied_fraction"] = occupied_fraction
            scored_record["display_render"] = (
                "human_sigmoid_rgb_binary_alpha"
                if human_render else "canonical_soft"
            )
            image = (
                _human_readable_image(state)
                if human_render else canonical_image
            )
            result.append((scored_record, image))
    return result


def _render_and_score_records(run, records, rollout_steps, device,
                              clip_loss, text_embeddings, human_render=True):
    rendered = []
    rollout_seed = run_rollout_seed(run)
    for record in records:
        image, state = render_checkpoint(
            record["checkpoint"],
            rollout_steps,
            rollout_seed,
            device,
            seed_preference=None,
            return_state=True,
        )
        rendered.append((record, image, state))
    return _attach_displayed_similarities(
        rendered,
        clip_loss,
        text_embeddings,
        human_render=human_render,
    )


def _image_descriptor(image, grid_size=12):
    """Return a compact foreground-aware descriptor for one rendered CA."""
    image = np.asarray(image, dtype=np.float32)
    height, width, _ = image.shape
    row_edges = np.linspace(0, height, grid_size + 1, dtype=int)
    col_edges = np.linspace(0, width, grid_size + 1, dtype=int)
    # Distance from the white canvas emphasizes organism color and silhouette.
    foreground = 1.0 - image
    pooled = np.empty((grid_size, grid_size, 3), dtype=np.float32)
    for row in range(grid_size):
        for column in range(grid_size):
            patch = foreground[
                row_edges[row]:row_edges[row + 1],
                col_edges[column]:col_edges[column + 1],
            ]
            pooled[row, column] = patch.mean(axis=(0, 1))
    return pooled.reshape(-1)


def _normalized_pairwise_distances(features, standardize=True):
    features = np.asarray(features, dtype=np.float64)
    if len(features) <= 1:
        return np.zeros((len(features), len(features)), dtype=np.float64)
    if standardize:
        scale = features.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        normalized = (features - features.mean(axis=0)) / scale
    else:
        normalized = features
    distances = np.linalg.norm(
        normalized[:, None, :] - normalized[None, :, :],
        axis=2,
    )
    maximum = float(distances.max())
    return distances / maximum if maximum > 0 else distances


def _deterministic_k_medoids(distances, records, count, max_iterations=30):
    """Cluster a precomputed distance matrix and return real center samples."""
    distances = np.asarray(distances, dtype=np.float64)
    sample_count = len(distances)
    selected_n = min(int(count), sample_count)
    if selected_n <= 0:
        return [], np.empty(0, dtype=int)
    if selected_n == sample_count:
        return list(range(sample_count)), np.arange(sample_count, dtype=int)

    # Begin at the global medoid, then seed uncovered visual modes by their
    # distance to the closest existing center.
    medoids = [min(
        range(sample_count),
        key=lambda index: (
            float(distances[index].mean()),
            -float(records[index]["similarity"].sum()),
            int(records[index]["index"]),
        ),
    )]
    while len(medoids) < selected_n:
        nearest = distances[:, medoids].min(axis=1)
        candidates = [index for index in range(sample_count) if index not in medoids]
        medoids.append(max(
            candidates,
            key=lambda index: (
                float(nearest[index]),
                float(records[index]["similarity"].sum()),
                -int(records[index]["index"]),
            ),
        ))

    for _ in range(max_iterations):
        assignments = np.argmin(distances[:, medoids], axis=1)
        updated = []
        for cluster_index, current_medoid in enumerate(medoids):
            members = np.flatnonzero(assignments == cluster_index)
            if not len(members):
                updated.append(current_medoid)
                continue
            cluster_distances = distances[np.ix_(members, members)]
            costs = cluster_distances.mean(axis=1)
            best_member = min(
                range(len(members)),
                key=lambda local_index: (
                    float(costs[local_index]),
                    -float(records[members[local_index]]["similarity"].sum()),
                    int(records[members[local_index]]["index"]),
                ),
            )
            updated.append(int(members[best_member]))
        if updated == medoids:
            break
        medoids = updated

    assignments = np.argmin(distances[:, medoids], axis=1)
    return medoids, assignments


def select_representative_records(run, rollout_steps, device, clip_loss,
                                  text_embeddings, count=6,
                                  human_render=True):
    """Select visual cluster medoids from all active-population members.

    Every candidate is rendered with a common random seed. Deterministic
    k-medoids uses 85% phenotype distance and 15% objective-space distance, so
    selected checkpoints represent distinct visual modes rather than index
    spacing or isolated visual outliers.
    """
    available = [
        record for record in run["records"]
        if record["checkpoint"] is not None
    ]
    if not available:
        return []

    rendered = _render_and_score_records(
        run,
        available,
        rollout_steps,
        device,
        clip_loss,
        text_embeddings,
        human_render=human_render,
    )

    similarities = np.asarray([
        record["similarity"] for record, _ in rendered
    ])
    image_features = np.asarray([
        _image_descriptor(image) for _, image in rendered
    ])
    objective_distance = _normalized_pairwise_distances(similarities)
    # Do not standardize individual pixels: doing so would amplify tiny rollout
    # noise in otherwise identical phenotypes.
    phenotype_distance = _normalized_pairwise_distances(
        image_features,
        standardize=False,
    )
    combined_distance = 0.15 * objective_distance + 0.85 * phenotype_distance
    selected, assignments = _deterministic_k_medoids(
        combined_distance,
        [record for record, _ in rendered],
        count,
    )

    result = []
    for cluster_index, position in enumerate(selected):
        record, image = rendered[position]
        members = np.flatnonzero(assignments == cluster_index)
        representative = dict(record)
        representative["representative_cluster_size"] = int(len(members))
        representative["mean_cluster_visual_distance"] = float(
            phenotype_distance[position, members].mean()
        )
        result.append((representative, image))
    result.sort(key=lambda item: (item[0]["similarity"][0], item[0]["index"]))
    return result


def render_pareto_rollout_composite(runs, prompts, output_dir, rollout_steps,
                                    solution_count=6, foreground_crop=False,
                                    pixelated_crop=False,
                                    human_render=True):
    """Compose HV and dominated-solution objective/rollout panels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    styles = method_styles(run["strategy"] for run in runs)
    from clip_loss import CLIPLoss
    clip_loss = CLIPLoss()
    text_embeddings = clip_loss.embed_objective_prompts(prompts)
    selected_by_run = [
        select_representative_records(
            run,
            rollout_steps=rollout_steps,
            device=device,
            clip_loss=clip_loss,
            text_embeddings=text_embeddings,
            count=solution_count,
            human_render=human_render,
        )
        for run in runs
    ]
    columns = max((len(records) for records in selected_by_run), default=0)
    if columns == 0:
        return None

    row_count = len(runs)
    row_height = 2.08 if foreground_crop else 1.92
    figure = plt.figure(
        figsize=(4.1 + columns * 2.05, 1.75 + row_count * row_height)
    )
    grid = figure.add_gridspec(
        row_count + 2,
        columns + 2,
        width_ratios=[2.15, 0.16] + [1.0] * columns,
        height_ratios=[0.72, 0.16] + [1.0] * row_count,
        left=0.045,
        right=0.99,
        bottom=0.065,
        top=0.975,
        wspace=0.08,
        hspace=0.30 if foreground_crop else 0.22,
    )

    # (a) Hypervolume comparison across the selected run of every method.
    hv_axis = figure.add_subplot(grid[0, :])
    hv_runs = sorted(
        runs,
        key=lambda run: (
            -(run.get("statistics") or run_statistics(run))["hypervolume"],
            run["strategy"],
        ),
    )
    hv_positions = np.arange(len(hv_runs))
    hv_values = [
        (run.get("statistics") or run_statistics(run))["hypervolume"]
        for run in hv_runs
    ]
    hv_colors = [styles[run["strategy"]]["color"] for run in hv_runs]
    bars = hv_axis.bar(
        hv_positions,
        hv_values,
        width=0.62,
        color=hv_colors,
        edgecolor="white",
        linewidth=0.7,
    )
    hv_axis.set_xticks(hv_positions)
    hv_axis.set_xticklabels(
        [method_display_name(run["strategy"]) for run in hv_runs],
        rotation=22 if len(hv_runs) > 5 else 0,
        ha="right" if len(hv_runs) > 5 else "center",
    )
    hv_upper = max(hv_values, default=1.0) * 1.23
    hv_axis.set_ylim(0.0, hv_upper)
    hv_axis.set_ylabel("HV")
    hv_axis.set_title(
        "(a) Best available-run hypervolume  ↑",
        loc="left",
        pad=4,
        fontweight="semibold",
    )
    hv_axis.text(
        1.0, 1.02, "reference point: (0, 0)",
        transform=hv_axis.transAxes,
        ha="right", va="bottom", fontsize=7, color="#66615C",
    )
    hv_axis.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.65)
    hv_axis.set_axisbelow(True)
    hv_axis.spines["top"].set_visible(False)
    hv_axis.spines["right"].set_visible(False)
    for bar, value in zip(bars, hv_values):
        hv_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + hv_upper * 0.025,
            f"{value:.4f}",
            ha="center", va="bottom", fontsize=7, color=INK,
        )

    # (b) One shared objective-space panel spanning all method rows.
    objective_axis = figure.add_subplot(grid[1:, 0])
    draw_objective_space(objective_axis, runs, prompts, compact=True)
    objective_axis.set_title("(b) Objective space", loc="left", pad=7,
                             fontweight="semibold")

    prompt1 = prompts[0] if len(prompts) > 0 else "objective 1"
    prompt2 = prompts[1] if len(prompts) > 1 else "objective 2"
    header_axis = figure.add_subplot(grid[1, 2:])
    header_axis.axis("off")
    header_axis.text(
        0.0, 0.45, f'\u2190  "{prompt2}"',
        transform=header_axis.transAxes,
        ha="left", va="center", fontsize=8.5, style="italic", color="#5F5A56",
    )
    header_axis.text(
        0.5, 0.45,
        (
            "(c) Human-readable hard-alpha representatives "
            "(ordered by $s_1$ \u2191)"
            if human_render
            else "(c) Foreground-normalized representatives "
            "(ordered by $s_1$ \u2191)"
            if foreground_crop
            else "(c) Representative solutions (ordered by $s_1$ \u2191)"
        ),
        transform=header_axis.transAxes,
        ha="center", va="center", fontsize=8, color="#66615C",
    )
    header_axis.text(
        1.0, 0.45, f'"{prompt1}"  \u2192',
        transform=header_axis.transAxes,
        ha="right", va="center", fontsize=8.5, style="italic", color="#5F5A56",
    )

    for row, (run, records) in enumerate(zip(runs, selected_by_run), start=2):
        style = styles[run["strategy"]]
        stats = run.get("statistics") or run_statistics(run)
        label_axis = figure.add_subplot(grid[row, 1])
        label_axis.axis("off")
        label_axis.axvline(0.82, color=style["color"], linewidth=4.0)
        label_axis.text(
            0.33,
            0.5,
            f'{method_display_name(run["strategy"])}\n'
            f'{run_identity(run)}  HV={stats["hypervolume"]:.3f}',
            rotation=90,
            transform=label_axis.transAxes,
            ha="center",
            va="center",
            fontsize=7.5,
            color=style["color"],
            fontweight="semibold",
        )

        for column in range(columns):
            axis = figure.add_subplot(grid[row, column + 2])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if column >= len(records):
                axis.axis("off")
                continue

            record, image = records[column]
            display_image = (
                _crop_to_bbox(image, record.get("display_foreground_bbox"))
                if foreground_crop else image
            )
            axis.imshow(
                display_image,
                interpolation=(
                    "nearest" if foreground_crop and pixelated_crop
                    else "lanczos" if foreground_crop or human_render
                    else None
                ),
            )
            axis.set_facecolor("#F6F4F0")
            canonical_similarity = record["similarity"]
            displayed_similarity = record["displayed_similarity"]
            weight = record["weight"]
            cluster_size = record.get("representative_cluster_size", 1)
            if np.all(np.isfinite(weight)):
                top_text = (
                    f'i={record["index"]:02d}  n={cluster_size}  '
                    f'w=({weight[0]:.2f},{weight[1]:.2f})'
                )
            else:
                top_text = f'i={record["index"]:02d}  n={cluster_size}'
            axis.set_title(top_text, fontsize=6.8, pad=3, color=INK)
            axis.set_xlabel(
                f'd={displayed_similarity[0]:.3f}/{displayed_similarity[1]:.3f}\n'
                f'c={canonical_similarity[0]:.3f}/{canonical_similarity[1]:.3f}',
                fontsize=6.2,
                labelpad=2,
                color=INK,
            )

    representative_rows = []
    for run, records in zip(runs, selected_by_run):
        for display_order, (record, _) in enumerate(records, start=1):
            weight = record["weight"]
            representative_rows.append({
                "strategy": run["strategy"],
                "selected_run": run["run_name"],
                "seed": run["seed"] if run.get("seed") is not None else "",
                "display_order": display_order,
                "subproblem_index": record["index"],
                "weight_1": weight[0] if np.isfinite(weight[0]) else "",
                "weight_2": weight[1] if np.isfinite(weight[1]) else "",
                "similarity_1": record["similarity"][0],
                "similarity_2": record["similarity"][1],
                "canonical_similarity_1": record["similarity"][0],
                "canonical_similarity_2": record["similarity"][1],
                "displayed_similarity_1": record["displayed_similarity"][0],
                "displayed_similarity_2": record["displayed_similarity"][1],
                "display_similarity_delta_linf": float(np.max(np.abs(
                    record["displayed_similarity"] - record["similarity"]
                ))),
                "display_rollout_steps": int(rollout_steps),
                "display_rng_seed": int(run_rollout_seed(run)),
                "display_view": (
                    "foreground_pixel_grid"
                    if foreground_crop and pixelated_crop
                    else "human_hard_alpha_foreground"
                    if foreground_crop and human_render
                    else "foreground_normalized"
                    if foreground_crop else "full_canvas"
                ),
                "display_render": record.get("display_render", ""),
                "display_occupied_fraction": record.get(
                    "display_occupied_fraction", ""
                ),
                "canonical_evaluation_steps": run.get("evaluation_steps") or "",
                "canonical_evaluation_repeats": run.get("evaluation_repeats") or "",
                "canonical_evaluation_source": run.get("evaluation_source", ""),
                "is_non_dominated": record["is_non_dominated"],
                "visual_cluster_size": record["representative_cluster_size"],
                "mean_cluster_visual_distance": record[
                    "mean_cluster_visual_distance"
                ],
                "checkpoint": str(record["checkpoint"]),
            })
    representative_path = output_dir / "representative_solutions.csv"
    with representative_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(representative_rows[0]),
        )
        writer.writeheader()
        writer.writerows(representative_rows)

    path, _ = save_publication_figure(
        figure,
        output_dir,
        (
            "comparison_figure_foreground_pixels"
            if foreground_crop and pixelated_crop
            else
            "comparison_figure_foreground"
            if foreground_crop else "comparison_figure"
        ),
        dpi=300,
    )
    plt.close(figure)
    del clip_loss
    return path


def _rollout_run_stem(run):
    """Return a stable filesystem-safe stem for one rendered training run."""
    strategy = re.sub(r"[^A-Za-z0-9._-]+", "_", str(run["strategy"]))
    seed = (
        f'seed_{int(run["seed"])}'
        if run.get("seed") is not None else "seed_unknown"
    )
    run_name = re.sub(
        r"[^A-Za-z0-9._-]+", "_", str(run.get("run_name") or "run")
    ).strip("._-") or "run"
    return f"{strategy}_{seed}_{run_name}_rollouts"


def _save_rollout_contact_sheet(rendered_by_run, output_dir, rollout_steps,
                                columns, human_render, styles, stem):
    """Save one or more already-rendered runs as a contact sheet and CSV."""
    if not rendered_by_run:
        return None, None, []
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = max(1, int(columns))
    columns = min(columns, max(len(items) for _, items in rendered_by_run))
    block_rows = [
        int(np.ceil(len(items) / columns))
        for _, items in rendered_by_run
    ]
    total_rows = sum(block_rows)
    figure = plt.figure(
        figsize=(1.35 + columns * 1.18, 1.0 + total_rows * 1.22),
    )
    grid = figure.add_gridspec(
        total_rows,
        columns + 1,
        width_ratios=[0.72] + [1.0] * columns,
        left=0.025,
        right=0.995,
        bottom=0.035,
        top=0.94,
        wspace=0.08,
        hspace=0.42,
    )

    csv_rows = []
    row_cursor = 0
    for (run, rendered), rows_for_run in zip(rendered_by_run, block_rows):
        style = styles[run["strategy"]]
        stats = run.get("statistics") or run_statistics(run)
        label_axis = figure.add_subplot(
            grid[row_cursor:row_cursor + rows_for_run, 0]
        )
        label_axis.axis("off")
        label_axis.axvline(0.82, color=style["color"], linewidth=4.0)
        label_axis.text(
            0.30,
            0.5,
            f'{method_display_name(run["strategy"])}\n'
            f'{run_identity(run)}\n'
            f'HV={stats["hypervolume"]:.3f}\n'
            f'n={len(rendered)}',
            transform=label_axis.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=7.2,
            color=style["color"],
            fontweight="semibold",
        )

        for position, (record, image) in enumerate(rendered):
            local_row, column = divmod(position, columns)
            axis = figure.add_subplot(
                grid[row_cursor + local_row, column + 1]
            )
            axis.imshow(
                image,
                interpolation="lanczos" if human_render else None,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor("#F6F4F0")
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color(style["color"])
                spine.set_linewidth(0.8)
            canonical_similarity = record["similarity"]
            displayed_similarity = record["displayed_similarity"]
            axis.set_title(
                f'i={record["index"]:02d}',
                fontsize=6.4,
                pad=2,
                color=INK,
            )
            axis.set_xlabel(
                f'd={displayed_similarity[0]:.3f}/{displayed_similarity[1]:.3f}\n'
                f'c={canonical_similarity[0]:.3f}/{canonical_similarity[1]:.3f}',
                fontsize=5.2,
                labelpad=1,
                color=INK,
            )
            weight = record["weight"]
            csv_rows.append({
                "strategy": run["strategy"],
                "selected_run": run["run_name"],
                "seed": run["seed"] if run.get("seed") is not None else "",
                "display_order": position + 1,
                "subproblem_index": record["index"],
                "weight_1": weight[0] if np.isfinite(weight[0]) else "",
                "weight_2": weight[1] if np.isfinite(weight[1]) else "",
                "similarity_1": canonical_similarity[0],
                "similarity_2": canonical_similarity[1],
                "canonical_similarity_1": canonical_similarity[0],
                "canonical_similarity_2": canonical_similarity[1],
                "displayed_similarity_1": displayed_similarity[0],
                "displayed_similarity_2": displayed_similarity[1],
                "display_similarity_delta_linf": float(np.max(np.abs(
                    displayed_similarity - canonical_similarity
                ))),
                "display_rollout_steps": int(rollout_steps),
                "display_rng_seed": int(run_rollout_seed(run)),
                "display_render": record.get("display_render", ""),
                "canonical_evaluation_steps": run.get("evaluation_steps") or "",
                "canonical_evaluation_repeats": run.get("evaluation_repeats") or "",
                "canonical_evaluation_source": run.get("evaluation_source", ""),
                "is_non_dominated": record["is_non_dominated"],
                "checkpoint": str(record["checkpoint"]),
            })
        row_cursor += rows_for_run

    figure.suptitle(
        f"Rollouts after {rollout_steps} CA steps "
        "(human-readable hard-alpha view; "
        "d=rollout similarity, c=canonical mean)"
        if human_render
        else f"Rollouts after {rollout_steps} CA steps "
        "(d=rollout similarity, c=canonical mean)",
        y=0.985,
        fontsize=10.5,
        fontweight="semibold",
    )
    path, _ = save_publication_figure(
        figure,
        output_dir,
        stem,
        dpi=300,
    )
    plt.close(figure)

    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return path, csv_path, csv_rows


def render_all_rollouts(runs, output_dir, rollout_steps, columns=10,
                        human_render=True, split_by_run=False):
    """Render all checkpoints, optionally saving one contact sheet per run."""
    if not runs:
        return [] if split_by_run else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    styles = method_styles(run["strategy"] for run in runs)
    from clip_loss import CLIPLoss
    clip_loss = CLIPLoss()
    prompts = runs[0]["prompts"]
    text_embeddings = clip_loss.embed_objective_prompts(prompts)
    rendered_by_run = []
    ordered_runs = sorted(
        runs,
        key=lambda item: (
            item["strategy"],
            item.get("seed") is None,
            int(item["seed"]) if item.get("seed") is not None else 0,
            item.get("run_name") or "",
        ),
    )
    for run in ordered_runs:
        records = [
            record for record in run["records"]
            if record["checkpoint"] is not None
        ]
        records.sort(key=lambda record: record["index"])
        rendered = _render_and_score_records(
            run,
            records,
            rollout_steps,
            device,
            clip_loss,
            text_embeddings,
            human_render=human_render,
        )
        if rendered:
            rendered_by_run.append((run, rendered))

    if not rendered_by_run:
        del clip_loss
        return [] if split_by_run else None

    if split_by_run:
        per_run_dir = output_dir / "every_run_rollouts"
        paths = []
        combined_rows = []
        for run, rendered in rendered_by_run:
            stem = _rollout_run_stem(run)
            path, _, rows = _save_rollout_contact_sheet(
                [(run, rendered)],
                per_run_dir,
                rollout_steps,
                columns,
                human_render,
                styles,
                stem,
            )
            if path is not None:
                paths.append(path)
                combined_rows.extend(rows)
        if combined_rows:
            manifest_path = per_run_dir / "every_run_rollout_solutions.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(combined_rows[0])
                )
                writer.writeheader()
                writer.writerows(combined_rows)
        del clip_loss
        return paths

    path, _, _ = _save_rollout_contact_sheet(
        rendered_by_run,
        output_dir,
        rollout_steps,
        columns,
        human_render,
        styles,
        "all_rollouts",
    )
    del clip_loss
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("train_log/moead_screen"))
    parser.add_argument(
        "--additional-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional result root to scan. Repeat this option to combine "
            "isolated formal result directories without scanning stale runs."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("train_log/visualizations/moead"))
    parser.add_argument(
        "--methods",
        default=None,
        help=(
            "Comma-separated strategy names from the formal three-stage "
            "protocol."
        ),
    )
    parser.add_argument("--prompt-id", default=None, help="Filter runs by a path component.")
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "Comma-separated formal seeds used for per-run and mean/std "
            "statistics, for example 11,22,33. Best-run phenotype rendering "
            "uses every discovered formal seed unless --selected-run-seeds "
            "is provided."
        ),
    )
    parser.add_argument(
        "--selected-run-seeds",
        default=None,
        help=(
            "Optional comma-separated seeds eligible for best-HV phenotype "
            "rendering. The default is every formal seed found under the "
            "explicit result roots."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Disable checkpoint-only baseline discovery and use summaries only.",
    )
    parser.add_argument(
        "--baseline-cache",
        type=Path,
        default=None,
        help=(
            "Baseline objective-cache directory. Defaults to "
            "<output>/baseline_evaluation_cache."
        ),
    )
    parser.add_argument(
        "--baseline-eval-steps",
        type=int,
        default=FINAL_EVAL_STEPS,
        help="Common CA rollout horizon used to evaluate every baseline checkpoint.",
    )
    parser.add_argument(
        "--baseline-eval-repeats",
        type=int,
        default=FINAL_EVAL_REPEATS,
        help="Number of deterministic stochastic-rollout repeats per checkpoint.",
    )
    parser.add_argument(
        "--baseline-eval-seed",
        type=int,
        default=10_000_000,
        help="Common RNG seed used for paired checkpoint evaluation.",
    )
    parser.add_argument(
        "--baseline-eval-batch-size",
        type=int,
        default=8,
        help="Maximum number of rendered states in one CLIP image batch.",
    )
    parser.add_argument(
        "--refresh-baseline-cache",
        action="store_true",
        help="Ignore compatible cached objectives and reevaluate checkpoints.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip checkpoint rollout rendering and generate metrics/plots only.",
    )
    parser.add_argument(
        "--all-rollouts",
        action="store_true",
        help=(
            "Render every available solution from each selected best-HV run "
            "in one contact-sheet figure."
        ),
    )
    parser.add_argument(
        "--every-run-rollouts",
        action="store_true",
        help=(
            "Render every available solution from every eligible run, writing "
            "one PNG/PDF contact sheet and CSV per run. Best-run plots and "
            "formal statistics remain unchanged."
        ),
    )
    parser.add_argument(
        "--all-rollout-columns",
        type=int,
        default=10,
        help="Number of solution columns in the all-rollouts contact sheet.",
    )
    parser.add_argument(
        "--foreground-crop",
        action="store_true",
        help=(
            "Crop representative tiles to the raw alpha>0.1 foreground with "
            "three-pixel padding and Lanczos display resampling. Absolute size and "
            "position are intentionally removed."
        ),
    )
    parser.add_argument(
        "--pixelated-crop",
        action="store_true",
        help=(
            "With --foreground-crop, use nearest-neighbor display to expose "
            "the native CA cell grid instead of smoothing it."
        ),
    )
    parser.add_argument(
        "--canonical-render",
        action="store_true",
        help=(
            "Display the canonical soft CLIP renderer instead of the default "
            "human-readable sigmoid-RGB, binary-alpha renderer. Numerical "
            "evaluation is unchanged."
        ),
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=FINAL_EVAL_STEPS,
        help=(
            "CA steps used for displayed rollouts. Defaults to the canonical "
            "final-evaluation horizon; use 128 only for long-horizon diagnostics."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_render and (args.all_rollouts or args.every_run_rollouts):
        raise SystemExit(
            "--no-render cannot be combined with rollout rendering modes."
        )
    if args.all_rollouts and args.every_run_rollouts:
        raise SystemExit(
            "Choose either --all-rollouts or --every-run-rollouts, not both."
        )
    try:
        methods = parse_method_filter(args.methods)
        seeds = parse_seed_filter(args.seeds)
        selected_run_seeds = parse_seed_filter(args.selected_run_seeds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output.mkdir(parents=True, exist_ok=True)
    roots = [args.root, *args.additional_root]
    summary_runs = []
    baseline_specs = []
    for root in roots:
        summary_runs.extend(discover_runs(
            root,
            methods=methods,
            prompt_id=args.prompt_id,
        ))
        if not args.summary_only:
            baseline_specs.extend(discover_baseline_runs(
                root,
                methods=methods,
                prompt_id=args.prompt_id,
            ))
    initial_checkpoint_only_specs = _checkpoint_only_specs(
        summary_runs, baseline_specs
    )
    force_uniform_evaluation = bool(initial_checkpoint_only_specs)
    checkpoint_only_specs = _checkpoint_only_specs(
        summary_runs,
        baseline_specs,
        force_uniform_evaluation=force_uniform_evaluation,
    )
    if checkpoint_only_specs:
        prompts = tuple(OBJECTIVE_PROMPTS)
        cache_dir = (
            args.baseline_cache
            if args.baseline_cache is not None
            else args.output / "baseline_evaluation_cache"
        )
        if force_uniform_evaluation:
            print(
                f"Detected {len(initial_checkpoint_only_specs)} legacy "
                "checkpoint-only baseline run(s); reevaluating all "
                f"{len(checkpoint_only_specs)} discovered checkpoint-backed "
                "runs under one canonical protocol."
            )
        else:
            print(
                f"Detected {len(checkpoint_only_specs)} checkpoint-only "
                "baseline run(s)."
            )
        checkpoint_runs = evaluate_baseline_runs(
            checkpoint_only_specs,
            prompts,
            cache_dir,
            rollout_steps=args.baseline_eval_steps,
            repeats=args.baseline_eval_repeats,
            rng_seed=args.baseline_eval_seed,
            batch_size=args.baseline_eval_batch_size,
            refresh=args.refresh_baseline_cache,
        )
        reevaluated_run_dirs = {
            Path(spec["run_dir"]).resolve()
            for spec in checkpoint_only_specs
        }
        runs = [
            run for run in summary_runs
            if Path(run["run_dir"]).resolve() not in reevaluated_run_dirs
        ]
        runs.extend(checkpoint_runs)
    else:
        runs = summary_runs

    unrecorded = [
        run for run in runs if run.get("seed") is None
    ]
    if unrecorded and args.seeds is None:
        print(
            "WARNING: "
            f"{len(unrecorded)} run(s) have no recorded seed metadata. "
            "They are included for exploratory plots and unlabeled "
            "three-run aggregates, but not certified formal seed statistics."
        )
    try:
        statistical_runs, selection_runs = split_statistical_and_selection_runs(
            runs,
            statistical_seeds=seeds,
            selected_run_seeds=selected_run_seeds,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not statistical_runs:
        raise SystemExit(
            "No completed strategy runs match the statistical seed filter under: "
            + ", ".join(str(root) for root in roots)
        )
    if not selection_runs:
        raise SystemExit(
            "No completed strategy runs match the selected-run seed filter under: "
            + ", ".join(str(root) for root in roots)
        )
    if methods:
        statistical_methods = {run["strategy"] for run in statistical_runs}
        selection_methods = {run["strategy"] for run in selection_runs}
        missing_statistics = sorted(methods - statistical_methods)
        missing_selection = sorted(methods - selection_methods)
        if missing_statistics:
            raise SystemExit(
                "No statistical runs were found for: "
                + ", ".join(missing_statistics)
            )
        if missing_selection:
            raise SystemExit(
                "No best-run candidates were found for: "
                + ", ".join(missing_selection)
            )

    prompt_sets = {
        run["prompts"] for run in [*statistical_runs, *selection_runs]
    }
    if len(prompt_sets) != 1:
        raise SystemExit(
            "Multiple prompt sets were found. Restrict --root or use --prompt-id "
            "so that one figure contains only one objective definition."
        )

    prompts = next(iter(prompt_sets))
    selected_runs = select_best_runs(selection_runs)
    metrics, metrics_path = write_metrics(selected_runs, args.output)
    _, per_run_path, aggregate_metrics, aggregate_path = (
        write_repeated_run_metrics(statistical_runs, args.output)
    )
    try:
        paired_rows, paired_path = write_paired_run_differences(
            statistical_runs, args.output
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"Using {len(statistical_runs)} run(s) for formal statistics and "
        f"{len(selection_runs)} run(s) as best-run candidates; selected "
        f"{len(selected_runs)} strategy runs."
    )
    for run in selected_runs:
        stats = run["statistics"]
        print(
            f'  {run["strategy"]}: {run_identity(run)}, '
            f'HV={stats["hypervolume"]:.6f}, '
            f'best of {run["candidate_run_count"]} run(s), '
            f'run={run["run_name"]}'
        )
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved per-run metrics: {per_run_path}")
    print(f"Saved aggregate mean/std metrics: {aggregate_path}")
    print(
        f"Saved {len(paired_rows)} same-seed paired comparison row(s): "
        f"{paired_path}"
    )
    for row in sorted(
        aggregate_metrics,
        key=lambda item: (-float(item["hypervolume_mean"]), item["strategy"]),
    ):
        std = row["hypervolume_std"]
        std_text = f"{float(std):.6f}" if std != "" else "n/a"
        print(
            f'  aggregate {row["strategy"]}: '
            f'HV={float(row["hypervolume_mean"]):.6f} +/- {std_text}, '
            f'B_pop={float(row["population_average_worst_similarity_mean"]):.6f}, '
            f'B_best={float(row["best_max_min_similarity_mean"]):.6f}, '
            f'n={row["run_count"]}, seeds={row["seeds"]}'
        )
    aggregate_hypervolume_path = plot_aggregate_hypervolume(
        aggregate_metrics, args.output
    )
    print(f"Saved aggregate hypervolume plot: {aggregate_hypervolume_path}")

    if args.no_render:
        objective_path = plot_objective_space(selected_runs, prompts, args.output)
        hypervolume_path = plot_hypervolume(metrics, args.output)
        print(f"Saved objective-space plot: {objective_path}")
        print(f"Saved hypervolume plot: {hypervolume_path}")
    elif args.every_run_rollouts:
        objective_path = plot_objective_space(selected_runs, prompts, args.output)
        hypervolume_path = plot_hypervolume(metrics, args.output)
        rollout_paths = render_all_rollouts(
            selection_runs,
            args.output,
            args.rollout_steps,
            columns=args.all_rollout_columns,
            human_render=not args.canonical_render,
            split_by_run=True,
        )
        print(f"Saved connected Pareto-front plot: {objective_path}")
        print(f"Saved hypervolume plot: {hypervolume_path}")
        if rollout_paths:
            print(
                f"Saved {len(rollout_paths)} per-run rollout figures under: "
                f'{args.output / "every_run_rollouts"}'
            )
        else:
            print("No solution checkpoints were available for rollout rendering.")
    elif args.all_rollouts:
        objective_path = plot_objective_space(selected_runs, prompts, args.output)
        hypervolume_path = plot_hypervolume(metrics, args.output)
        rollout_path = render_all_rollouts(
            selected_runs,
            args.output,
            args.rollout_steps,
            columns=args.all_rollout_columns,
            human_render=not args.canonical_render,
        )
        print(f"Saved connected Pareto-front plot: {objective_path}")
        print(f"Saved hypervolume plot: {hypervolume_path}")
        if rollout_path is not None:
            print(f"Saved all-solutions rollout figure: {rollout_path}")
        else:
            print("No solution checkpoints were available for rollout rendering.")
    else:
        path = render_pareto_rollout_composite(
            selected_runs,
            prompts,
            args.output,
            args.rollout_steps,
            solution_count=6,
            foreground_crop=args.foreground_crop,
            pixelated_crop=args.pixelated_crop,
            human_render=not args.canonical_render,
        )
        if path is not None:
            print(f"Saved comparison figure: {path}")
        else:
            print(
                "No solution checkpoints were available; "
                "comparison rendering was skipped."
            )


if __name__ == "__main__":
    main()
