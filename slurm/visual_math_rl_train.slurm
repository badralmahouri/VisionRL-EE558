#!/bin/bash
#SBATCH --job-name=vmath_rl
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=6:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Visual-math RL real training run (Step 4 of plan).
# Target 100 iters via trainer.total_training_steps (overrides epoch-based count).
# Eval cadence: every 25 iters → evals at 25, 50, 75, 100.
# Tune via Hydra override at submit time:
#   sbatch ... slurm/visual_math_rl_train.slurm trainer.total_training_steps=200
#
# Submit per arm via env var:
#   sbatch --export=ARM=mathvr   slurm/visual_math_rl_train.slurm
#   sbatch --export=ARM=dynamath slurm/visual_math_rl_train.slurm
#
# Both arms can run in parallel on separate 4-GPU nodes (same OOD eval, different
# in-dist eval, same reward function, same starting checkpoint).

set -euo pipefail
ulimit -n 65535
echo "=== Job started at $(date '+%Y-%m-%d %H:%M:%S') ==="

source /users/$USER/miniconda3/etc/profile.d/conda.sh
conda activate verl

BASE="/capstor/scratch/cscs/$USER/verl-apertus"
VERL="/users/$USER/verl-vision"
EMU3_SRC="/users/$USER/Emu3.5/src"

export PYTHONPATH="${BASE}:${VERL}:${EMU3_SRC}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH:-/iopsstor/scratch/cscs/$USER}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline    # local files only; no W&B account / network needed

ARM="${ARM:-mathvr}"
echo "=== ARM = $ARM (real RL) ==="

case "$ARM" in
  mathvr)
    TRAIN_PARQUET="$BASE/data_prep/visual_math_rl/mathvr_train.parquet"
    IDIST_VAL="$BASE/data_prep/visual_math_rl/mathvr_indist_eval.parquet"
    # EXPERIMENT_TAG default = v3 (pure judge reward). Distinct ckpt dir from
    # v1/v2 so verl resume_mode='auto' doesn't resume from a stale checkpoint
    # (the bug that wasted v2 run 1, job 2372407).
    EXPERIMENT="apertus8b_grpo_visual_math_mathvr_${EXPERIMENT_TAG:-v3}"
    ;;
  dynamath)
    TRAIN_PARQUET="$BASE/data_prep/visual_math_rl/dynamath_train.parquet"
    IDIST_VAL="$BASE/data_prep/visual_math_rl/dynamath_indist_eval.parquet"
    EXPERIMENT="apertus8b_grpo_visual_math_dynamath_${EXPERIMENT_TAG:-v3}"
    ;;
  *)
    echo "ERROR: unknown ARM=$ARM (use mathvr|dynamath)"
    exit 1
    ;;
esac

MV_VAL="$BASE/data_prep/visual_math_rl/mathvision_eval.parquet"
MS_VAL="$BASE/data_prep/visual_math_rl/mathverse_visionOnly_eval.parquet"

# --- pre-flight checks ---
echo "=== Pre-flight Checks ==="
ERR=0
check() { if [ -e "$2" ]; then echo "  OK: $1"; else echo "  FAIL: $1 ($2)"; ERR=$((ERR+1)); fi; }
check "verl repo"           "$VERL/verl/__init__.py"
check "trainer yaml"        "$BASE/configs/visual_math_rl_grpo.yaml"
check "reward fn"           "$BASE/rewards/visual_math_rl_reward.py"
check "train parquet"       "$TRAIN_PARQUET"
check "in-dist val"         "$IDIST_VAL"
check "mathvision val"      "$MV_VAL"
check "mathverse val"       "$MS_VAL"
check "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"
[ $ERR -eq 0 ] || { echo "ERROR: $ERR pre-flight failed."; exit 1; }

nvidia-smi --list-gpus

# ---------------------------------------------------------------------------
# v3 reward: read Qwen3-32B judge URL written by slurm/qwen_judge_server.slurm
# and block until /health responds. v2 used regex-only and didn't need this.
# ---------------------------------------------------------------------------
URL_FILE="$BASE/slurm/judge_url.txt"
if [ ! -f "$URL_FILE" ]; then
  echo "ERROR: judge URL file missing at $URL_FILE — has the judge SLURM job started?"
  exit 1
fi
# First line is the URL; subsequent lines are comments (# job_id, host, ts)
VISUAL_MATH_JUDGE_URL="$(grep -v '^#' "$URL_FILE" | head -1 | tr -d '[:space:]')"
if [ -z "$VISUAL_MATH_JUDGE_URL" ]; then
  echo "ERROR: judge URL file at $URL_FILE is empty or malformed"
  cat "$URL_FILE"
  exit 1
fi
export VISUAL_MATH_JUDGE_URL
echo "=== Judge URL: $VISUAL_MATH_JUDGE_URL"

echo "=== Waiting for judge /health (timeout 1800s)..."
python3 -c "
import sys
sys.path.insert(0, '$BASE')
from inference.llm_judge_math import wait_for_server
wait_for_server('$VISUAL_MATH_JUDGE_URL', timeout_s=1800)
"
echo "=== Judge is READY"

TOTAL_STEPS="${TOTAL_STEPS:-50}"     # v3: same as v2 (50 steps); v1 peaked at step 40-50 then overshoot
SAVE_FREQ="${SAVE_FREQ:-10}"          # v3: save every 10 steps -> ckpts at 10/20/30/40/50
TEST_FREQ="${TEST_FREQ:-10}"          # v3: eval every 10 steps -> 6 eval points (0 + 10/20/30/40/50)
echo "== Job started at $(date)"
echo "=== Starting GRPO training (arm=$ARM, total_steps=$TOTAL_STEPS, save@$SAVE_FREQ, test@$TEST_FREQ) ==="
cd "$BASE"
python3 -m verl.trainer.main_ppo \
    --config-path="$BASE/configs" \
    --config-name=visual_math_rl_grpo \
    data.train_files="$TRAIN_PARQUET" \
    "data.val_files=[$IDIST_VAL,$MV_VAL,$MS_VAL]" \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.total_epochs=10 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    "$@"

echo "=== Job ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "=== Real RL ($ARM) complete ==="
