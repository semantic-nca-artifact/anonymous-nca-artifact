"""Build the paper-facing Stage-III coverage/concentration figure.

The quantitative panels use all five same-seed CA/MCA run pairs for each of
the six semantic tasks. The phenotype strip is qualitative: one pre-specified
common seed and fixed subproblem indices are rendered for both methods.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from aggregate_stage3_results import (
    CA_METHOD,
    MCA_METHOD,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    METHODS,
    SEMANTIC_PAIRS,
    collect_run_rows,
)
from visualize_moead_results import (
    INK,
    _human_readable_image,
    render_checkpoint,
    run_rollout_seed,
    save_publication_figure,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (11, 22, 33, 44, 55)
DISPLAY_PAIR_SLUG = "butterfly_caterpillar"
DISPLAY_SEED = 22
DISPLAY_INDICES = (0, 14, 29)


def _metric_axis(axis, run_rows, metric, title, show_task_labels):
    indexed = defaultdict(dict)
    for row in run_rows:
        indexed[(row["semantic_pair_slug"], int(row["seed"]))][
            row["strategy"]
        ] = row

    y_positions = np.arange(len(SEMANTIC_PAIRS), dtype=np.float64)
    for y, pair in zip(y_positions, SEMANTIC_PAIRS):
        seeds = sorted(
            seed for slug, seed in indexed if slug == pair["slug"]
        )
        jitters = np.linspace(-0.15, 0.15, len(seeds))
        values = defaultdict(list)
        for jitter, seed in zip(jitters, seeds):
            method_rows = indexed[(pair["slug"], seed)]
            ca_value = float(method_rows[CA_METHOD][metric])
            mca_value = float(method_rows[MCA_METHOD][metric])
            axis.plot(
                (ca_value, mca_value), (y + jitter, y + jitter),
                color="#C7C7C7", linewidth=0.55, zorder=1,
            )
            for method, value in (
                (CA_METHOD, ca_value), (MCA_METHOD, mca_value)
            ):
                values[method].append(value)
                axis.scatter(
                    value, y + jitter, s=10,
                    marker=METHOD_MARKERS[method],
                    color=METHOD_COLORS[method], alpha=0.5,
                    linewidths=0, zorder=2,
                )

        means = {method: float(np.mean(values[method])) for method in METHODS}
        axis.plot(
            (means[CA_METHOD], means[MCA_METHOD]), (y, y),
            color="#737373", linewidth=1.05, zorder=3,
        )
        for method in METHODS:
            method_values = np.asarray(values[method], dtype=np.float64)
            axis.errorbar(
                means[method], y,
                xerr=float(method_values.std(ddof=1)),
                fmt=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                markeredgecolor=INK, markeredgewidth=0.4,
                markersize=5.1, elinewidth=0.95, capsize=2.0, zorder=4,
            )

    axis.set_title(title, fontsize=8.2, fontweight="semibold", pad=4)
    axis.grid(axis="x", color="#DDDAD4", linewidth=0.5, alpha=0.85)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=6.7)
    axis.tick_params(axis="y", length=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.margins(x=0.08, y=0.09)
    axis.set_yticks(y_positions)
    if show_task_labels:
        axis.set_yticklabels([pair["label"] for pair in SEMANTIC_PAIRS])
    else:
        axis.set_yticklabels([])
    axis.invert_yaxis()


def _display_device(value):
    if str(value).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def build_figure(train_log, output_dir, device="cuda", rollout_steps=96):
    run_rows, _, run_index = collect_run_rows(
        train_log,
        DEFAULT_SEEDS,
        expected_solutions=30,
        expected_steps=96,
        expected_repeats=4,
        return_runs=True,
    )

    figure = plt.figure(figsize=(7.25, 5.3))
    grid = figure.add_gridspec(
        2, 6, height_ratios=(2.65, 1.0),
        left=0.145, right=0.992, top=0.94, bottom=0.075,
        wspace=0.22, hspace=0.72,
    )
    hv_axis = figure.add_subplot(grid[0, :3])
    bpop_axis = figure.add_subplot(grid[0, 3:])
    _metric_axis(
        hv_axis, run_rows, "hypervolume",
        r"(a) Coverage: hypervolume $\uparrow$", True,
    )
    _metric_axis(
        bpop_axis, run_rows, "population_average_worst_similarity",
        r"(b) Concentration: $B_{\mathrm{pop}}$ $\uparrow$", False,
    )

    handles = [
        Line2D(
            [0], [0], marker=METHOD_MARKERS[method], linestyle="none",
            markerfacecolor=METHOD_COLORS[method], markeredgecolor=INK,
            markeredgewidth=0.4, markersize=5.2,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    figure.legend(
        handles=handles, loc="upper center", ncol=2,
        bbox_to_anchor=(0.52, 0.405), frameon=False,
        fontsize=7.1, handletextpad=0.4, columnspacing=1.2,
    )

    display_pair = next(
        pair for pair in SEMANTIC_PAIRS if pair["slug"] == DISPLAY_PAIR_SLUG
    )
    render_device = _display_device(device)
    selection_rows = []
    index_labels = ("prompt 2", "balanced", "prompt 1")
    for method_index, method in enumerate(METHODS):
        run = run_index[(display_pair["slug"], method, DISPLAY_SEED)]
        records = {int(record["index"]): record for record in run["records"]}
        for local_index, (solution_index, index_label) in enumerate(
            zip(DISPLAY_INDICES, index_labels)
        ):
            column = method_index * len(DISPLAY_INDICES) + local_index
            axis = figure.add_subplot(grid[1, column])
            record = records[int(solution_index)]
            _, state = render_checkpoint(
                record["checkpoint"],
                rollout_steps=int(rollout_steps),
                seed_value=run_rollout_seed(run),
                device=render_device,
                return_state=True,
            )
            axis.imshow(_human_readable_image(state), interpolation="lanczos")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(
                f"i={solution_index:02d}  {index_label}", fontsize=5.8, pad=2,
            )
            axis.set_xlabel(
                f"s={record['similarity'][0]:.3f}/{record['similarity'][1]:.3f}",
                fontsize=5.0, labelpad=1,
            )
            for spine in axis.spines.values():
                spine.set_color(METHOD_COLORS[method])
                spine.set_linewidth(0.9)
            selection_rows.append({
                "semantic_pair": display_pair["label"],
                "strategy": method,
                "method": METHOD_LABELS[method],
                "seed": DISPLAY_SEED,
                "subproblem_index": int(solution_index),
                "canonical_similarity": [
                    float(record["similarity"][0]),
                    float(record["similarity"][1]),
                ],
                "checkpoint": str(record["checkpoint"]),
                "display_rollout_steps": int(rollout_steps),
                "display_rng_seed": int(run_rollout_seed(run)),
            })

    figure.text(
        0.355, 0.327, "(c) Common-seed phenotypes: MOEA/D-CA",
        color=METHOD_COLORS[CA_METHOD], ha="center", va="center",
        fontsize=7.5, fontweight="semibold",
    )
    figure.text(
        0.765, 0.327, "MOEA/D-MCA",
        color=METHOD_COLORS[MCA_METHOD], ha="center", va="center",
        fontsize=7.5, fontweight="semibold",
    )
    figure.text(
        0.018, 0.185, "Butterfly--\nCaterpillar\nseed 22",
        ha="left", va="center", fontsize=6.5, fontweight="semibold",
    )

    output_dir = Path(output_dir)
    png_path, pdf_path = save_publication_figure(
        figure, output_dir, "stage3_ca_mca_control", dpi=400
    )
    plt.close(figure)
    manifest_path = output_dir / "stage3_ca_mca_control.json"
    manifest_path.write_text(
        json.dumps({
            "statistical_unit": "independently trained run/seed",
            "paired_seeds": list(DEFAULT_SEEDS),
            "semantic_pairs": [pair["label"] for pair in SEMANTIC_PAIRS],
            "quantitative_panels": [
                "hypervolume",
                "population_average_worst_similarity",
            ],
            "phenotype_selection_rule": (
                "Pre-specified common seed and fixed subproblem indices; "
                "qualitative display only."
            ),
            "phenotypes": selection_rows,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return png_path, pdf_path, manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", type=Path, default=ROOT / "train_log")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollout-steps", type=int, default=96)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = build_figure(
        args.train_log,
        args.output,
        device=args.device,
        rollout_steps=args.rollout_steps,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
