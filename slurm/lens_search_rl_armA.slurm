#!/bin/bash
#SBATCH --job-name=lens_rl_armA
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=01:25:00
#SBATCH --environment=verl_env
#SBATCH --partition=debug
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Arm A — pure RL on base IBQ ckpt (no SFT cold-start).
#
# Env-var overrides:
#   TOTAL_STEPS    — total GRPO training steps (default 50 for smoke; bump to 150 for real)
#   TOOL_MODE      — 'cached' (FVQA pkl, zero GCP cost) or 'live' (GCP Vision API). default 'cached'.
#   LENS_REWARD_TIER — '1' (correctness only, default) or '2' (gated tool bonus, DeepEyes)

set -euo pipefail
ulimit -n 65535
echo "=== Job started at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "  TOTAL_STEPS=${TOTAL_STEPS:-50}  TOOL_MODE=${TOOL_MODE:-cached}  LENS_REWARD_TIER=${LENS_REWARD_TIER:-1}"

source /users/$USER/miniconda3/etc/profile.d/conda.sh
conda activate verl

BASE="/capstor/scratch/cscs/$USER/verl-apertus"
VERL="/users/$USER/verl-vision"
EMU3_SRC="/users/$USER/Emu3.5/src"

cd "$BASE"
export PYTHONPATH="${BASE}:${VERL}:${EMU3_SRC}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH:-/iopsstor/scratch/cscs/$USER}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
unset SSL_CERT_FILE

# Toggle for the tool wrapper + reward function
export TOOL_MODE="${TOOL_MODE:-cached}"
export LENS_TOOL_MODE="$TOOL_MODE"
export TOTAL_STEPS="${TOTAL_STEPS:-50}"
export LENS_REWARD_TIER="${LENS_REWARD_TIER:-1}"

rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true

mkdir -p logs slurm/rollout_logs/armA

# --- Pre-flight checks ---
echo ""
echo "=== Pre-flight ==="
ERR=0
check() { if [ -e "$2" ]; then echo "  OK: $1"; else echo "  FAIL: $1 ($2)"; ERR=$((ERR+1)); fi; }
check "verl repo"             "$VERL/verl/__init__.py"
check "lens_search_tool.py"   "$BASE/tools/lens_search_tool.py"
check "tool config"           "$BASE/configs/lens_search_tool_config.yaml"
check "agent loop config"     "$BASE/configs/apertus_agent_loop.yaml"
check "GRPO config (Arm A)"   "$BASE/configs/lens_search_rl_grpo_armA.yaml"
check "reward fn"             "$BASE/rewards/lens_search_rl_reward.py"
check "RL train parquet"      "$BASE/data_prep/lens_search/rl_fvqa_train.parquet"
check "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"

# Cached-mode also needs the pkl
if [ "$TOOL_MODE" = "cached" ]; then
    check "FVQA cache pkl" "/iopsstor/scratch/cscs/$USER/hf_home/hub/datasets--lmms-lab--FVQA/snapshots/bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5/fvqa_train_image_search_results_cache.pkl"
fi
# Live-mode needs GCP creds
if [ "$TOOL_MODE" = "live" ]; then
    if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] || [ ! -f "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
        echo "  FAIL: GOOGLE_APPLICATION_CREDENTIALS unset or file missing (live mode)"
        ERR=$((ERR+1))
    else
        echo "  OK: GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS"
    fi
fi

if [ $ERR -gt 0 ]; then echo "ERROR: $ERR pre-flight failed."; exit 1; fi

nvidia-smi --list-gpus

# --- Launch GRPO ---
echo ""
echo "=== Starting Arm A GRPO ==="
T0=$(date +%s)
python3 -m verl.trainer.main_ppo \
    --config-path="$BASE/configs" \
    --config-name=lens_search_rl_grpo_armA "$@"
T1=$(date +%s)
echo ""
echo "=== GRPO elapsed: $((T1 - T0))s ==="

echo "=== Job ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
