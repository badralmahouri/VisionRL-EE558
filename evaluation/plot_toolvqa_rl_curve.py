#!/usr/bin/env python3
"""Plot ToolVQA RL accuracy and training dynamics.

This is the ToolVQA wrapper around plot_line_drawing_rl_curve.py. By default it
plots the current ToolVQA baselines plus the step-100 tool-calling RL result.

Example:
    python evaluation/plot_toolvqa_rl_curve.py

    python evaluation/plot_toolvqa_rl_curve.py \
      --rl-result global_step_90 results/toolvqa_rl/checkpoints/global_step_90/accuracy.json \
      --rl-result global_step_100 results/toolvqa_rl/checkpoints/global_step_100/accuracy.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_line_drawing_rl_curve import (
    load_accuracy,
    load_rl_points,
    parse_training_log,
    write_csv,
    write_metrics_csv,
    write_plot,
)


def existing_default_results() -> list[tuple[str, str]]:
    candidates = [
        ("global_step_100", "results/toolvqa_rl/checkpoints/global_step_100/accuracy.json"),
    ]
    return [(label, path) for label, path in candidates if Path(path).exists()]


def existing_default_no_tool_results() -> list[tuple[str, str]]:
    candidates = [
        ("global_step_20", "results/toolvqa_rl/no_tool_checkpoints/global_step_20/accuracy.json"),
        ("global_step_100", "results/toolvqa_rl/no_tool_checkpoints/global_step_100/accuracy.json"),
    ]
    return [(label, path) for label, path in candidates if Path(path).exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ToolVQA RL checkpoint curve and training dynamics")
    parser.add_argument("--apertus-baseline", default="results/toolvqa_rl/apertus_baseline/accuracy.json")
    parser.add_argument("--qwen-baseline", default="results/toolvqa_rl/qwen_baseline/accuracy.json")
    parser.add_argument(
        "--rl-result",
        nargs=2,
        action="append",
        metavar=("LABEL", "JSON"),
        default=None,
        help="With-tool RL checkpoint label and accuracy JSON. Can be repeated.",
    )
    parser.add_argument(
        "--no-tool-rl-result",
        nargs=2,
        action="append",
        metavar=("LABEL", "JSON"),
        default=None,
        help="No-tool RL checkpoint label and accuracy JSON. Can be repeated.",
    )
    parser.add_argument("--training-log", default="logs/toolvqa_rl_2293013.out")
    parser.add_argument("--no-tool-training-log", default="logs/toolvqa_no_tool_2319153.out")
    parser.add_argument("--output", default="results/plots/toolvqa_rl_curve_with_metrics.png")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--metrics-csv", default=None)
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

    rl_items = args.rl_result or existing_default_results()
    rl_points = load_rl_points([(label, Path(path)) for label, path in rl_items if Path(path).exists()])
    no_tool_items = args.no_tool_rl_result or existing_default_no_tool_results()
    no_tool_points = load_rl_points([(label, Path(path)) for label, path in no_tool_items if Path(path).exists()])
    extra_rl_series = [("No-tool", no_tool_points)] if no_tool_points else []
    metric_rows = parse_training_log(Path(args.training_log)) if args.training_log else []
    extra_val_curves = []
    no_tool_log = Path(args.no_tool_training_log) if args.no_tool_training_log else None
    if no_tool_log and no_tool_log.exists():
        no_tool_rows = parse_training_log(no_tool_log)
        if no_tool_rows:
            extra_val_curves.append(("No-tool", no_tool_rows, "toolvqa_rl"))

    output_path = Path(args.output)
    csv_path = Path(args.csv) if args.csv else output_path.with_suffix(".csv")
    metrics_csv_path = (
        Path(args.metrics_csv) if args.metrics_csv else output_path.with_name(output_path.stem + "_metrics.csv")
    )

    write_csv(csv_path, baselines, rl_points, extra_rl_series=extra_rl_series)
    write_metrics_csv(metrics_csv_path, metric_rows, metric_prefix="toolvqa_rl")
    try:
        write_plot(
            output_path,
            baselines,
            rl_points,
            metric_rows,
            metric_prefix="toolvqa_rl",
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
