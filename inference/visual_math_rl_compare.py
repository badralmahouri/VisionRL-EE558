"""v1 vs v2 vs v3 comparison figure for Apertus visual-math RL.

Parses the three SLURM stdout logs and overlays, per training step:
  panel 1: DynaMath in-dist accuracy
  panel 2: MathVision OOD accuracy
  panel 3: MathVerse-VisionOnly OOD accuracy
  panel 4: response_length/mean  (the cleanest cross-run signal)

METRIC CAVEAT (annotated on the figure): v1/v2 in-run eval uses
`val-core/<ds>/acc/mean@1` (regex-extracted correctness); v3 uses
`val-core/<ds>/reward/mean@1` (Qwen3-32B judge YES/NO). Both are "fraction
correct" but measured differently, so absolute levels are not strictly
comparable across reward versions — the SHAPE (regression / collapse /
stability) is the point. response_length IS directly comparable.

Run in-container (matplotlib lives in verl_env):
  srun ... --environment=verl_env python3 inference/visual_math_rl_compare.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/capstor/scratch/cscs/badralmahouri/verl-apertus")
RUNS = {
    "v1 (binary regex)":      BASE / "logs/vmath_rl_2355716.out",
    "v2 (continuous +pen)":   BASE / "logs/vmath_rl_2378427.out",
    "v3 (pure judge)":        BASE / "logs/vmath_rl_v3_2436122.out",
}
COLORS = {"v1 (binary regex)": "tab:orange", "v2 (continuous +pen)": "tab:red", "v3 (pure judge)": "tab:green"}
DATASETS = ["dynamath", "mathvision", "mathverse_visionOnly"]
TITLES = {"dynamath": "DynaMath (in-dist) accuracy",
          "mathvision": "MathVision (OOD) accuracy",
          "mathverse_visionOnly": "MathVerse-VisionOnly (OOD) accuracy"}

# val-core/<ds>/(acc|reward)/mean@1:[np.float64(]0.22[)]
def _metric(ds: str):
    return re.compile(rf"val-core/{re.escape(ds)}/(?:acc|reward)/mean@1:(?:np\.float64\()?([0-9.]+)")
_STEP = re.compile(r"step:(\d+)")
_RLEN = re.compile(r"\bresponse_length/mean:([0-9.]+)")

def parse(log: Path):
    """Return {ds: [(step,val)...]} and [(step, resp_len)...]."""
    acc = {ds: [] for ds in DATASETS}
    rlen = []
    if not log.exists():
        return acc, rlen
    for line in log.read_text(errors="ignore").splitlines():
        if "step:" not in line:
            continue
        ms = _STEP.search(line)
        if not ms:
            continue
        step = int(ms.group(1))
        for ds in DATASETS:
            m = _metric(ds).search(line)
            if m:
                acc[ds].append((step, float(m.group(1))))
        mr = _RLEN.search(line)
        if mr:
            rlen.append((step, float(mr.group(1))))
    # de-dup by step (keep last), sort
    for ds in DATASETS:
        d = dict(acc[ds]); acc[ds] = sorted(d.items())
    rlen = sorted(dict(rlen).items())
    return acc, rlen

data = {name: parse(log) for name, log in RUNS.items()}

fig, axes = plt.subplots(4, 1, figsize=(9, 14))
for ax, ds in zip(axes[:3], DATASETS):
    for name in RUNS:
        pts = data[name][0][ds]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=name, color=COLORS[name])
    ax.set_title(TITLES[ds]); ax.set_xlabel("training step"); ax.set_ylabel("accuracy")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[3]
for name in RUNS:
    pts = data[name][1]
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=name, color=COLORS[name])
ax.axhline(50, ls="--", color="gray", lw=1, label="50-tok floor (anti-collapse)")
ax.set_title("response_length/mean  (directly comparable; v2 collapse vs v1/v3 stability)")
ax.set_xlabel("training step"); ax.set_ylabel("mean response length (tokens)")
ax.grid(alpha=0.3); ax.legend(fontsize=8)

fig.suptitle("Apertus visual-math RL: v1 vs v2 vs v3  (DynaMath arm)\n"
             "acc panels: v1/v2=regex-acc, v3=judge-reward (levels not strictly comparable; shape is)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = BASE / "evaluation/visual_math_rl_v1_v2_v3.png"
fig.savefig(out, dpi=130)
print(f"WROTE {out}")

# Print the parsed tables for the record
for name in RUNS:
    acc, rlen = data[name]
    print(f"\n## {name}")
    steps = sorted({s for ds in DATASETS for s, _ in acc[ds]})
    for s in steps:
        row = {ds: dict(acc[ds]).get(s) for ds in DATASETS}
        rl = dict(rlen).get(s)
        print(f"  step {s:>3}: dyna={row['dynamath']} mv={row['mathvision']} "
              f"ms={row['mathverse_visionOnly']} rlen={rl}")
