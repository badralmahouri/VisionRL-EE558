#!/usr/bin/env python3
"""Plot synthetic line-drawing RL accuracy and training dynamics.

The main curve shows held-out test accuracy by checkpoint, with direct baselines
as horizontal lines. When a VeRL training log is provided, extra panels show the
training reward, validation reward, KL loss, and policy entropy over steps.

Examples:
    python evaluation/plot_line_drawing_rl_curve.py

    python evaluation/plot_line_drawing_rl_curve.py \
      --training-log logs/line_drawing_rl_2236267.out \
      --rl-result global_step_10 results/line_drawing_rl/checkpoints/global_step_10/accuracy.json \
      --rl-result global_step_20 results/line_drawing_rl/checkpoints/global_step_20/accuracy.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep:(\d+)\b")
METRIC_RE = re.compile(r"([A-Za-z0-9_./@-]+):([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def load_accuracy(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if "overall_accuracy" not in data and "accuracy" not in data:
        raise ValueError(f"{path} does not contain overall_accuracy or accuracy")
    if "overall_accuracy" not in data:
        data["overall_accuracy"] = data["accuracy"]
    if "num_correct" not in data and "num_soft_correct" in data:
        data["num_correct"] = data["num_soft_correct"]
    return data


def infer_step(label: str, path: Path, data: dict[str, Any]) -> int:
    match = re.search(r"global_step_(\d+)", label) or re.search(r"global_step_(\d+)", str(path))
    if match:
        return int(match.group(1))
    by_step = data.get("by_step") or {}
    numeric_steps = sorted(int(step) for step in by_step if str(step).isdigit())
    if numeric_steps:
        return numeric_steps[-1]
    match = re.search(r"(\d+)", label)
    return int(match.group(1)) if match else 0


def load_rl_points(items: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    points = []
    for label, path in items:
        data = load_accuracy(path)
        step = infer_step(label, path, data)
        points.append(
            {
                "label": label,
                "step": step,
                "accuracy": float(data["overall_accuracy"]),
                "num_samples": data.get("num_samples", ""),
                "num_correct": data.get("num_correct", ""),
                "path": str(path),
            }
        )
    return sorted(points, key=lambda row: row["step"])


def parse_training_log(path: Path | None) -> list[dict[str, float]]:
    if path is None or not path.exists():
        return []

    by_step: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = ANSI_RE.sub("", raw_line)
            step_match = STEP_RE.search(line)
            if not step_match:
                continue
            step = int(step_match.group(1))
            metrics = {key: float(value) for key, value in METRIC_RE.findall(line)}
            if not metrics:
                continue
            row = by_step.setdefault(step, {"step": float(step)})
            row.update(metrics)

    return [by_step[step] for step in sorted(by_step)]


def metric_series(rows: list[dict[str, float]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        if key in row and math.isfinite(row[key]):
            xs.append(int(row["step"]))
            ys.append(float(row[key]))
    return xs, ys


def write_csv(
    path: Path,
    baselines: list[dict[str, Any]],
    rl_points: list[dict[str, Any]],
    extra_rl_series: list[tuple[str, list[dict[str, Any]]]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["kind", "name", "step", "accuracy", "num_samples", "num_correct", "path"],
        )
        writer.writeheader()
        for row in baselines:
            writer.writerow({"kind": "baseline", **row})
        for row in rl_points:
            writer.writerow(
                {
                    "kind": "rl_checkpoint",
                    "name": row["label"],
                    "step": row["step"],
                    "accuracy": row["accuracy"],
                    "num_samples": row["num_samples"],
                    "num_correct": row["num_correct"],
                    "path": row["path"],
                }
            )
        for series_name, points in extra_rl_series or []:
            for row in points:
                writer.writerow(
                    {
                        "kind": "extra_rl_checkpoint",
                        "name": f"{series_name}: {row['label']}",
                        "step": row["step"],
                        "accuracy": row["accuracy"],
                        "num_samples": row["num_samples"],
                        "num_correct": row["num_correct"],
                        "path": row["path"],
                    }
                )


def write_metrics_csv(path: Path, metric_rows: list[dict[str, float]], metric_prefix: str = "line_drawing_rl") -> None:
    if not metric_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "step",
        "critic/rewards/mean",
        "critic/rewards/min",
        "critic/rewards/max",
        "critic/score/mean",
        f"val-aux/{metric_prefix}/reward/mean@1",
        f"val-core/{metric_prefix}/acc/mean@1",
        "actor/kl_loss",
        "actor/ppo_kl",
        "actor/entropy",
        "actor/pg_loss",
        "actor/grad_norm",
        "timing_s/agent_loop/tool_calls/mean",
    ]
    extra = sorted({key for row in metric_rows for key in row} - set(keys))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys + extra)
        writer.writeheader()
        writer.writerows(metric_rows)


def plot_accuracy_panel(
    ax,
    baselines: list[dict[str, Any]],
    rl_points: list[dict[str, Any]],
    metric_rows: list[dict[str, float]] | None = None,
    metric_prefix: str = "line_drawing_rl",
    extra_val_curves: list[tuple[str, list[dict[str, float]], str]] | None = None,
    extra_rl_series: list[tuple[str, list[dict[str, Any]]]] | None = None,
) -> None:
    if rl_points:
        xs = [row["step"] for row in rl_points]
        ys = [row["accuracy"] for row in rl_points]
        ax.plot(xs, ys, marker="o", linewidth=2.4, color="#2563eb", label="Test accuracy")
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.025, f"{y:.2f}", ha="center", va="bottom", color="#2563eb", fontweight="bold")
    else:
        ax.plot([], [], marker="o", linewidth=2.4, color="#2563eb", label="Test accuracy")

    extra_rl_steps: list[int] = []
    extra_rl_values: list[float] = []
    extra_rl_styles = [("#7c3aed", "-."), ("#0891b2", ":"), ("#ea580c", "-.")]
    for idx, (series_name, points) in enumerate(extra_rl_series or []):
        if not points:
            continue
        xs = [row["step"] for row in points]
        ys = [row["accuracy"] for row in points]
        color, linestyle = extra_rl_styles[idx % len(extra_rl_styles)]
        ax.plot(xs, ys, marker="^", linewidth=2.2, linestyle=linestyle, color=color, label=f"{series_name} test accuracy")
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.025, f"{y:.2f}", ha="center", va="bottom", color=color, fontweight="bold")
        extra_rl_steps.extend(xs)
        extra_rl_values.extend(ys)

    val_acc_x: list[int] = []
    val_acc: list[float] = []
    if metric_rows:
        val_acc_x, val_acc = metric_series(metric_rows, f"val-core/{metric_prefix}/acc/mean@1")
        if val_acc_x and val_acc:
            ax.plot(
                val_acc_x,
                val_acc,
                marker="s",
                markersize=4,
                linewidth=2.0,
                linestyle="--",
                color="#dc2626",
                label="Validation accuracy",
            )

    extra_steps: list[int] = []
    extra_values: list[float] = []
    extra_styles = [("#9333ea", "-."), ("#0891b2", ":"), ("#ea580c", "-.")]
    for idx, (label, rows, prefix) in enumerate(extra_val_curves or []):
        extra_x, extra_y = metric_series(rows, f"val-core/{prefix}/acc/mean@1")
        if not extra_x or not extra_y:
            continue
        color, linestyle = extra_styles[idx % len(extra_styles)]
        ax.plot(
            extra_x,
            extra_y,
            marker="D",
            markersize=3.5,
            linewidth=1.8,
            linestyle=linestyle,
            color=color,
            label=f"{label} validation accuracy",
        )
        extra_steps.extend(extra_x)
        extra_values.extend(extra_y)

    baseline_styles = [("#6b7280", "--"), ("#ef4444", "--"), ("#16a34a", "--")]
    for idx, row in enumerate(baselines):
        color, linestyle = baseline_styles[idx % len(baseline_styles)]
        value = row["accuracy"]
        ax.axhline(value, color=color, linestyle=linestyle, linewidth=2, label=f"{row['name']} ({value:.2f})")

    plotted_steps = [row["step"] for row in rl_points] + extra_rl_steps + val_acc_x + extra_steps
    if plotted_steps:
        min_step = min(plotted_steps)
        max_step = max(plotted_steps)
        pad = max(1, int((max_step - min_step) * 0.12))
        ax.set_xlim(min_step - pad, max_step + pad)
        ax.set_xticks(sorted(set([row["step"] for row in rl_points] + extra_rl_steps + val_acc_x + extra_steps)))
    else:
        ax.set_xlim(0, 1)

    max_acc = max([row["accuracy"] for row in rl_points] + extra_rl_values + [row["accuracy"] for row in baselines] + val_acc + extra_values + [0.05])
    ax.set_ylim(0, min(1.0, max(0.1, max_acc + 0.12)))
    ax.set_xlabel("Checkpoint global step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Held-Out and Validation Accuracy")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")


def plot_reward_panel(ax, metric_rows: list[dict[str, float]], metric_prefix: str = "line_drawing_rl") -> None:
    xs, train = metric_series(metric_rows, "critic/rewards/mean")
    _, reward_min = metric_series(metric_rows, "critic/rewards/min")
    _, reward_max = metric_series(metric_rows, "critic/rewards/max")
    val_x, val = metric_series(metric_rows, f"val-aux/{metric_prefix}/reward/mean@1")

    if xs and train:
        ax.plot(xs, train, marker="o", markersize=3, linewidth=1.5, color="#2563eb", label="Train reward mean")
        if len(reward_min) == len(train) and len(reward_max) == len(train):
            ax.fill_between(xs, reward_min, reward_max, color="#2563eb", alpha=0.15, label="Train reward min/max")
    if val_x and val:
        ax.plot(val_x, val, marker="s", markersize=4, linewidth=2.0, color="#ef4444", label="Val reward mean")

    ax.set_title("Training and Validation Rewards")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward / score")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")


def plot_kl_panel(ax, metric_rows: list[dict[str, float]]) -> None:
    xs, kl = metric_series(metric_rows, "actor/kl_loss")
    ppo_xs, ppo_kl = metric_series(metric_rows, "actor/ppo_kl")
    if xs and kl:
        positive = [y for y in kl if y > 0]
        ax.plot(xs, kl, marker="o", markersize=3, linewidth=1.4, color="#1f77b4", label="actor/kl_loss")
        if positive:
            ax.set_yscale("log")
    if ppo_xs and ppo_kl and any(y != 0 for y in ppo_kl):
        ax.plot(ppo_xs, ppo_kl, marker=".", linewidth=1.2, color="#7c3aed", label="actor/ppo_kl")
    ax.axhline(0.01, color="#ef4444", linestyle="--", linewidth=1, label="Target min (0.01)")
    ax.axhline(0.1, color="#f59e0b", linestyle="--", linewidth=1, label="Target max (0.1)")
    ax.set_title("KL Divergence")
    ax.set_xlabel("Training step")
    ax.set_ylabel("KL loss")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best")


def plot_entropy_panel(ax, metric_rows: list[dict[str, float]]) -> None:
    xs, entropy = metric_series(metric_rows, "actor/entropy")
    if xs and entropy:
        ax.plot(xs, entropy, marker="o", markersize=3, linewidth=1.4, color="#15803d", label="actor/entropy")
    ax.set_title("Policy Entropy")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Entropy")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")


def write_plot(
    path: Path,
    baselines: list[dict[str, Any]],
    rl_points: list[dict[str, Any]],
    metric_rows: list[dict[str, float]],
    metric_prefix: str = "line_drawing_rl",
    extra_val_curves: list[tuple[str, list[dict[str, float]], str]] | None = None,
    extra_rl_series: list[tuple[str, list[dict[str, Any]]]] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is not installed; CSV was still written") from exc

    if metric_rows:
        fig, axes = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True)
        plot_accuracy_panel(
            axes[0],
            baselines,
            rl_points,
            metric_rows,
            metric_prefix=metric_prefix,
            extra_val_curves=extra_val_curves,
            extra_rl_series=extra_rl_series,
        )
        plot_reward_panel(axes[1], metric_rows, metric_prefix=metric_prefix)
        plot_kl_panel(axes[2], metric_rows)
        plot_entropy_panel(axes[3], metric_rows)
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_accuracy_panel(ax, baselines, rl_points, extra_val_curves=extra_val_curves, extra_rl_series=extra_rl_series)
        fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot line-drawing RL checkpoint curve and training dynamics")
    parser.add_argument(
        "--apertus-baseline",
        default="results/line_drawing_rl/apertus_baseline/accuracy.json",
    )
    parser.add_argument(
        "--qwen-baseline",
        default="results/line_drawing_rl/qwen_baseline/accuracy.json",
    )
    parser.add_argument(
        "--rl-result",
        nargs=2,
        action="append",
        metavar=("LABEL", "JSON"),
        default=None,
        help="RL checkpoint label and accuracy JSON. Can be repeated.",
    )
    parser.add_argument(
        "--training-log",
        default=None,
        help="Optional VeRL training .out log to parse for reward, KL, and entropy curves.",
    )
    parser.add_argument(
        "--output",
        default="results/plots/line_drawing_rl_curve.png",
    )
    parser.add_argument("--csv", default=None)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--metric-prefix", default="line_drawing_rl", help="Metric namespace in VeRL logs, e.g. line_drawing_rl or toolvqa_rl.")
    parser.add_argument(
        "--extra-val-log",
        nargs=3,
        action="append",
        metavar=("LABEL", "LOG", "PREFIX"),
        default=None,
        help="Overlay an additional validation accuracy curve on the first panel. Can be repeated.",
    )
    parser.add_argument(
        "--extra-rl-result",
        nargs=3,
        action="append",
        metavar=("SERIES", "LABEL", "JSON"),
        default=None,
        help="Overlay an additional test-accuracy checkpoint series on the first panel. Can be repeated.",
    )
    args = parser.parse_args()

    baselines = []
    for name, path_str in [
        ("Apertus baseline", args.apertus_baseline),
        ("Qwen baseline", args.qwen_baseline),
    ]:
        path = Path(path_str)
        data = load_accuracy(path)
        baselines.append(
            {
                "name": name,
                "step": "",
                "accuracy": float(data["overall_accuracy"]),
                "num_samples": data.get("num_samples", ""),
                "num_correct": data.get("num_correct", ""),
                "path": str(path),
            }
        )

    rl_items = args.rl_result or [("global_step_1", "results/line_drawing_rl/trained_tool/accuracy.json")]
    rl_points = load_rl_points([(label, Path(path)) for label, path in rl_items if Path(path).exists()])
    metric_rows = parse_training_log(Path(args.training_log)) if args.training_log else []
    grouped_extra_rl: dict[str, list[tuple[str, Path]]] = {}
    for series, label, json_path in args.extra_rl_result or []:
        path = Path(json_path)
        if path.exists():
            grouped_extra_rl.setdefault(series, []).append((label, path))
    extra_rl_series = [(series, load_rl_points(items)) for series, items in grouped_extra_rl.items()]
    extra_val_curves = []
    for label, log_path, prefix in args.extra_val_log or []:
        rows = parse_training_log(Path(log_path))
        if rows:
            extra_val_curves.append((label, rows, prefix))

    output_path = Path(args.output)
    csv_path = Path(args.csv) if args.csv else output_path.with_suffix(".csv")
    metrics_csv_path = Path(args.metrics_csv) if args.metrics_csv else output_path.with_name(output_path.stem + "_metrics.csv")
    write_csv(csv_path, baselines, rl_points, extra_rl_series=extra_rl_series)
    write_metrics_csv(metrics_csv_path, metric_rows, metric_prefix=args.metric_prefix)
    try:
        write_plot(
            output_path,
            baselines,
            rl_points,
            metric_rows,
            metric_prefix=args.metric_prefix,
            extra_val_curves=extra_val_curves,
            extra_rl_series=extra_rl_series,
        )
        print(f"Wrote plot: {output_path}")
    except RuntimeError as exc:
        print(f"Skipping plot: {exc}")
    print(f"Wrote CSV:  {csv_path}")
    if metric_rows:
        print(f"Wrote metrics CSV: {metrics_csv_path}")
        print(f"Parsed training metrics from {args.training_log}: {len(metric_rows)} step rows")
    for label, rows, prefix in extra_val_curves:
        print(f"Parsed extra validation curve {label} ({prefix}): {len(rows)} step rows")
    for series, points in extra_rl_series:
        print(f"Parsed extra RL series {series}: {len(points)} checkpoint rows")

    for row in baselines:
        print(f"{row['name']}: {row['accuracy'] * 100:.2f}% ({row['num_correct']}/{row['num_samples']})")
    for row in rl_points:
        print(f"{row['label']}: {row['accuracy'] * 100:.2f}% ({row['num_correct']}/{row['num_samples']})")
    for series, points in extra_rl_series:
        for row in points:
            print(f"{series} {row['label']}: {row['accuracy'] * 100:.2f}% ({row['num_correct']}/{row['num_samples']})")


if __name__ == "__main__":
    main()
