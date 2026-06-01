"""End-of-D.7.10 deliverable plot for the lens-search RL track.

Parses metrics from the SLURM stdout log of a successful Arm B run (e.g.
logs/lens_rl_armB_2409997.out) and produces a single PNG with 2 side-by-side
panels:

  Panel A — Validation accuracy vs training step (val-core/lens_search_rl/acc/mean@1),
            with the cached-mode oracle ceiling and the SFT baseline overlaid.
  Panel B — Train critic/rewards/mean per step + 10-step bucket means.

Output filename intentionally parallels but is distinct from the visual-math
plot at evaluation/visual_math_rl_results_v1.png. They live side-by-side.

Usage (verl conda env has matplotlib):
    /users/badralmahouri/miniconda3/envs/verl/bin/python3 inference/lens_search_rl_plots.py \\
        --log    logs/lens_rl_armB_2409997.out \\
        --output evaluation/lens_search_rl_results_v1.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


VAL_ACC_PAT = re.compile(r"val-core/lens_search_rl/acc/mean@1:([0-9.eE-]+)")
STEP_PAT = re.compile(r"training/global_step:(\d+)")
TRAIN_R_PAT = re.compile(r"critic/rewards/mean:([0-9.eE-]+)")


def parse_log(log_path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return (val_curve, train_curve) as lists of (step, value)."""
    text = log_path.read_text()
    # Per-line scan: each training step prints one line containing the train metric +
    # any val metric that was computed at that step. We also pick up the standalone
    # "step:0 - val-aux ..." line that contains val before any training step.
    val_curve: list[tuple[int, float]] = []
    train_curve: list[tuple[int, float]] = []
    seen_val_steps: set[int] = set()
    seen_train_steps: set[int] = set()

    for line in text.splitlines():
        # Per-step training line
        m_step = STEP_PAT.search(line)
        if m_step:
            step = int(m_step.group(1))
            m_train = TRAIN_R_PAT.search(line)
            if m_train and step not in seen_train_steps:
                train_curve.append((step, float(m_train.group(1))))
                seen_train_steps.add(step)
            m_val = VAL_ACC_PAT.search(line)
            if m_val and step not in seen_val_steps:
                val_curve.append((step, float(m_val.group(1))))
                seen_val_steps.add(step)
            continue

        # Standalone "step:0 - val-aux ..." line (no training/global_step yet)
        if line.lstrip().startswith("step:0 ") or " step:0 " in line:
            m_val = VAL_ACC_PAT.search(line)
            if m_val and 0 not in seen_val_steps:
                val_curve.append((0, float(m_val.group(1))))
                seen_val_steps.add(0)

    val_curve.sort()
    train_curve.sort()
    return val_curve, train_curve


def bucket_means(train_curve: list[tuple[int, float]], width: int = 10) -> list[tuple[int, float]]:
    """Return [(bucket_center_step, mean_reward), ...] over consecutive windows of size `width`."""
    out: list[tuple[int, float]] = []
    if not train_curve:
        return out
    steps = [s for s, _ in train_curve]
    vals = [v for _, v in train_curve]
    max_step = max(steps)
    for start in range(1, max_step + 1, width):
        end = start + width - 1
        chunk = [v for s, v in train_curve if start <= s <= end]
        if not chunk:
            continue
        center = (start + min(end, max_step)) / 2.0
        out.append((center, sum(chunk) / len(chunk)))
    return out


def plot(
    val_curve: list[tuple[int, float]],
    train_curve: list[tuple[int, float]],
    output: Path,
    *,
    sft_baseline: float | None = None,
    cached_ceiling: float | None = None,
) -> None:
    """Two-panel figure: Panel A val acc, Panel B train reward."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6))

    # --- Panel A: validation accuracy ---
    if val_curve:
        xs = [s for s, _ in val_curve]
        ys = [v for _, v in val_curve]
        axA.plot(xs, ys, marker="o", linewidth=2.2, markersize=8, color="#1f77b4", label="val acc@1 (n=200, greedy)")
        # Peak marker
        peak_idx = max(range(len(ys)), key=ys.__getitem__)
        axA.plot(xs[peak_idx], ys[peak_idx], marker="*", markersize=18, color="#d62728",
                 zorder=5, label=f"peak @ step {xs[peak_idx]}: {ys[peak_idx]:.3f}")
        # Annotate every point
        for s, v in val_curve:
            axA.annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(6, 8), fontsize=9)

    if sft_baseline is not None:
        axA.axhline(sft_baseline, linestyle=":", color="gray", linewidth=1.2,
                    label=f"SFT baseline = {sft_baseline:.3f}")
    if cached_ceiling is not None:
        axA.axhline(cached_ceiling, linestyle="--", color="#2ca02c", linewidth=1.4,
                    label=f"cached oracle ceiling = {cached_ceiling:.3f}")

    axA.set_xlabel("training step")
    axA.set_ylabel("val-core/lens_search_rl/acc/mean@1")
    axA.set_title("A. Validation accuracy (held-out FVQA, n=200, greedy)")
    axA.set_ylim(0.34, 0.43)
    axA.grid(True, alpha=0.3)
    axA.legend(loc="lower right", fontsize=9)

    # --- Panel B: train reward ---
    if train_curve:
        xs = [s for s, _ in train_curve]
        ys = [v for _, v in train_curve]
        axB.plot(xs, ys, marker=".", linewidth=1.0, alpha=0.45, color="#1f77b4",
                 label="per-step (n=8 sampling, batch 64)")
        # 10-step bucket means
        bm = bucket_means(train_curve, width=10)
        if bm:
            bxs = [s for s, _ in bm]
            bys = [v for _, v in bm]
            axB.plot(bxs, bys, marker="o", linewidth=2.2, markersize=9, color="#ff7f0e",
                     label="10-step bucket mean")
            for s, v in bm:
                axB.annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(0, 10),
                             fontsize=9, color="#ff7f0e", ha="center")

    axB.set_xlabel("training step")
    axB.set_ylabel("critic/rewards/mean")
    axB.set_title("B. Train reward (per-step + 10-step bucket means)")
    axB.grid(True, alpha=0.3)
    axB.legend(loc="lower right", fontsize=9)

    fig.suptitle("Apertus-8B SFT→RL on FVQA (cached lens_search) — D.7.10", fontsize=13, y=1.02)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="logs/lens_rl_armB_2409997.out",
                   help="Path to the Arm B RL stdout log to parse.")
    p.add_argument("--output", default="evaluation/lens_search_rl_results_v1.png",
                   help="PNG output path. v1 is the D.7.10 50-step run; "
                        "do not collide with evaluation/visual_math_rl_results_v1.png.")
    p.add_argument("--sft-baseline", type=float, default=0.370,
                   help="Val acc at step 0 (SFT only, before any RL update).")
    p.add_argument("--cached-ceiling", type=float, default=0.389,
                   help="Oracle ceiling: fraction of FVQA rows whose GT answer appears "
                        "in the cached [Search results:...] block (measured on step-50 "
                        "rollouts, 199/512 = 38.9%).")
    args = p.parse_args()

    log_path = Path(args.log)
    if not log_path.is_absolute():
        repo = Path(__file__).resolve().parent.parent
        log_path = repo / log_path
    output = Path(args.output)
    if not output.is_absolute():
        repo = Path(__file__).resolve().parent.parent
        output = repo / output

    val_curve, train_curve = parse_log(log_path)

    print(f"Parsed {len(val_curve)} val points and {len(train_curve)} train points from {log_path}")
    print(f"  val:   {val_curve}")
    bm = bucket_means(train_curve, width=10)
    print(f"  bucket means: {[(int(s), round(v, 4)) for s, v in bm]}")

    plot(val_curve, train_curve, output,
         sft_baseline=args.sft_baseline,
         cached_ceiling=args.cached_ceiling)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
