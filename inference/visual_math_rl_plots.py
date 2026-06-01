"""4-stacked-plot end-of-Phase-2 deliverable.

Reads metrics from the SLURM stdout log of a successful real-RL run (e.g.
logs/vmath_rl_2355716.out) and produces a single PNG with 4 vertically
stacked subplots:

  1. Validation accuracy vs checkpoint global step (with Apertus + Qwen baselines)
  2. Training reward + validation reward vs training step
  3. KL loss vs training step
  4. Policy entropy vs training step

Usage (on a node with matplotlib in the conda env):
    python inference/visual_math_rl_plots.py \\
        --log    logs/vmath_rl_2355716.out \\
        --output evaluation/visual_math_rl_results.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Baseline constants — UPDATE THESE WITH YOUR PREFERRED NUMBERS
# ---------------------------------------------------------------------------
# Apertus baseline = the actual step-0 (val_before_train) numbers from THIS run.
# These are auto-extracted from the log if available, but the constants here
# act as a fallback / sanity check. Update if you want to override.
APERTUS_BASELINE = {
    "dynamath": 0.245,                # 24.5% — extracted from job 2355716 step 0
    "mathvision": 0.080,              # 8.0%
    "mathverse_visionOnly": 0.065,    # 6.5%
}

# Qwen baseline = published Qwen2-VL-7B numbers on the FULL benchmarks.
# CAVEAT: these are on the FULL eval sets, not our 200-row subsets — within
# ~3pp of subset numbers typically, but not exact.
# Sources:
#   - MathVision: Qwen2-VL-7B paper / MathVision leaderboard (approx)
#   - MathVerse:  MathVerse leaderboard, Vision-Only subset (approx)
#   - DynaMath:   DynaMath paper Table 1 (approx)
# UPDATE these with the specific Qwen variant + paper number you cite.
QWEN_BASELINE = {
    "dynamath": 0.36,                 # PLACEHOLDER — Qwen2-VL-7B, DynaMath paper
    "mathvision": 0.17,               # PLACEHOLDER — Qwen2-VL-7B, MathVision leaderboard
    "mathverse_visionOnly": 0.30,     # PLACEHOLDER — Qwen2-VL-7B, MathVerse Vision-Only
}

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# Strip Ray actor prefix `[36m(TaskRunner pid=XXXX)[0m ` etc.
ANSI_PREFIX_RE = re.compile(r"\x1b\[\d+m|\(\w+ pid=\d+\)")

# Match a metric line: contains "step:N" and at least one "key:value" pair.
STEP_LINE_RE = re.compile(r"step:(\d+)\s*-(.+)")
# Key chars: word chars + - / @ . (hyphen MUST be in here because keys like
# `val-core/dynamath/acc/mean@1` use hyphens). Inside [], hyphen at the start
# or end is literal; we put it after \w to be safe.
KV_RE = re.compile(r"([\w\-/@.]+):(-?\d+\.?\d*(?:[eE][+-]?\d+)?)")
# v3 logs wrap metric values as np.float64(0.22); unwrap to a bare number so KV_RE matches.
NPFLOAT_RE = re.compile(r"np\.float64\((-?\d+\.?\d*(?:[eE][+-]?\d+)?)\)")


def parse_log(log_path: Path) -> pd.DataFrame:
    """Parse the SLURM stdout log into a DataFrame indexed by step.

    Each row = one step. Columns = every metric key seen for that step.
    """
    rows: dict[int, dict] = {}
    with open(log_path) as f:
        for line in f:
            line = ANSI_PREFIX_RE.sub("", line).strip()
            m = STEP_LINE_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            tail = NPFLOAT_RE.sub(r"\1", m.group(2))
            kvs = dict(KV_RE.findall(tail))
            # Convert to float
            kvs_f = {}
            for k, v in kvs.items():
                try:
                    kvs_f[k] = float(v)
                except ValueError:
                    continue
            # Skip lines that didn't yield any numeric values
            if not kvs_f:
                continue
            # Merge into existing step row (a step may print multiple lines)
            if step not in rows:
                rows[step] = {"step": step}
            rows[step].update(kvs_f)

    df = pd.DataFrame(sorted(rows.values(), key=lambda r: r["step"]))
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_stack(df: pd.DataFrame, output: Path) -> None:
    """Build the 4-stacked-plot figure and save as PNG."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 16), sharex=False)

    # --------------------------------------------------------------------
    # Plot 1: validation accuracy vs checkpoint global step
    # --------------------------------------------------------------------
    ax = axes[0]
    # v1/v2 log val-core/<ds>/acc/mean@1; v3 (pure judge) logs reward/mean@1.
    _esuf = "acc/mean@1" if "val-core/dynamath/acc/mean@1" in df.columns else "reward/mean@1"
    eval_keys = {
        "DynaMath in-dist": f"val-core/dynamath/{_esuf}",
        "MathVision OOD": f"val-core/mathvision/{_esuf}",
        "MathVerse Vision-Only OOD": f"val-core/mathverse_visionOnly/{_esuf}",
    }
    colors = {"DynaMath in-dist": "tab:blue", "MathVision OOD": "tab:green",
              "MathVerse Vision-Only OOD": "tab:orange"}
    for label, key in eval_keys.items():
        if key in df.columns:
            sub = df.dropna(subset=[key])
            ax.plot(sub["step"], sub[key], "o-", label=label, color=colors[label], linewidth=2,
                    markersize=8)

    # Apertus base is NOT plotted as a separate dotted line — it's already the
    # step-0 point on each solid line (redundant).
    # Qwen2-VL-7B baseline only (dashed horizontal per dataset).
    for label, baseline_key in [
        ("DynaMath in-dist", "dynamath"),
        ("MathVision OOD", "mathvision"),
        ("MathVerse Vision-Only OOD", "mathverse_visionOnly"),
    ]:
        ax.axhline(QWEN_BASELINE[baseline_key], linestyle="--", alpha=0.4,
                   color=colors[label], label=f"Qwen2-VL-7B {label.split()[0]}")

    ax.set_xlabel("Training step (global)")
    ax.set_ylabel("Accuracy")
    ax.set_title("1. Held-out + Validation Accuracy vs Checkpoint\n"
                 "(solid = Apertus-RL trajectory; dashed = Qwen2-VL-7B published baseline)")
    ax.legend(loc="upper right", fontsize=9)
    _present = [k for k in eval_keys.values() if k in df.columns]
    ax.set_ylim(0, max(0.45, df[_present].max().max() * 1.2) if _present else 0.45)
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------------------------
    # Plot 2: training + validation reward vs training step
    # --------------------------------------------------------------------
    ax = axes[1]
    train_reward_key = "critic/rewards/mean"
    if train_reward_key in df.columns:
        sub = df.dropna(subset=[train_reward_key])
        ax.plot(sub["step"], sub[train_reward_key], "-", alpha=0.6,
                label="Training reward (per-iter)", color="tab:blue", linewidth=1)
        # 10-iter moving average for clarity
        if len(sub) >= 10:
            ax.plot(sub["step"], sub[train_reward_key].rolling(window=10, min_periods=1).mean(),
                    "-", color="tab:blue", linewidth=2.5, label="Training reward (10-iter MA)")

    val_reward_keys = {
        "Val reward — DynaMath": "val-aux/dynamath/reward/mean@1",
        "Val reward — MathVision": "val-aux/mathvision/reward/mean@1",
        "Val reward — MathVerse VO": "val-aux/mathverse_visionOnly/reward/mean@1",
    }
    val_colors = {"Val reward — DynaMath": "tab:purple",
                  "Val reward — MathVision": "tab:green",
                  "Val reward — MathVerse VO": "tab:orange"}
    for label, key in val_reward_keys.items():
        if key in df.columns:
            sub = df.dropna(subset=[key])
            ax.plot(sub["step"], sub[key], "s--", label=label, color=val_colors[label],
                    linewidth=1.5, markersize=7)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward score")
    ax.set_title("2. Training + Validation Reward vs Training Step")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------------------------
    # Plot 3: KL loss vs training step
    # --------------------------------------------------------------------
    ax = axes[2]
    kl_key = "actor/kl_loss"
    if kl_key in df.columns:
        sub = df.dropna(subset=[kl_key])
        ax.plot(sub["step"], sub[kl_key], "-", color="tab:red", linewidth=1.5, alpha=0.7,
                label="KL loss (per iter)")
        if len(sub) >= 10:
            ax.plot(sub["step"], sub[kl_key].rolling(window=10, min_periods=1).mean(),
                    "-", color="darkred", linewidth=2.5, label="10-iter MA")
    ax.set_xlabel("Training step")
    ax.set_ylabel("KL divergence (actor vs ref)")
    ax.set_title("3. KL Loss vs Training Step")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------------------------
    # Plot 4: policy entropy vs training step
    # --------------------------------------------------------------------
    ax = axes[3]
    ent_key = "actor/entropy"
    if ent_key in df.columns:
        sub = df.dropna(subset=[ent_key])
        ax.plot(sub["step"], sub[ent_key], "-", color="tab:olive", linewidth=1.5, alpha=0.7,
                label="Entropy (per iter)")
        if len(sub) >= 10:
            ax.plot(sub["step"], sub[ent_key].rolling(window=10, min_periods=1).mean(),
                    "-", color="darkgreen", linewidth=2.5, label="10-iter MA")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Policy entropy")
    ax.set_title("4. Policy Entropy vs Training Step")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    print(f"Wrote {output} ({output.stat().st_size // 1024} KB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=None,
                   help="Path to SLURM stdout log. Default: auto-pick latest "
                        "logs/vmath_rl_*.out (sorted by mtime)")
    p.add_argument("--output", default="evaluation/visual_math_rl_results_v2.png",
                   help="PNG output path. Default: _v2.png (v1 lives at _v1.png)")
    args = p.parse_args()

    if args.log is None:
        # Auto-pick latest vmath_rl_*.out by mtime
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        candidates = sorted(log_dir.glob("vmath_rl_*.out"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"No vmath_rl_*.out logs found in {log_dir}")
            raise SystemExit(1)
        log_path = candidates[0]
        print(f"Auto-selected latest log: {log_path.name}")
    else:
        log_path = Path(args.log)
        if not log_path.is_absolute():
            log_path = Path(__file__).resolve().parent.parent / log_path
    output = Path(args.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parent.parent / output

    print(f"Parsing {log_path}...")
    df = parse_log(log_path)
    print(f"  Found {len(df)} step rows, {len(df.columns)} unique metric columns")

    # Quick sanity: print the step + eval columns we found
    eval_cols = [c for c in df.columns if c.startswith("val-core/")]
    print(f"  Eval cols: {eval_cols}")
    for c in eval_cols:
        sub = df.dropna(subset=[c])
        print(f"    {c}: {len(sub)} eval points, steps={sub['step'].tolist()}")

    plot_stack(df, output)


if __name__ == "__main__":
    main()
