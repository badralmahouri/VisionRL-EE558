#!/bin/bash
#SBATCH --job-name=lens_rl_armB
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

# Arm B — SFT-trained ckpt → RL.
# REQUIRES Arm B SFT (slurm/lens_search_sft.slurm) to have completed first.
#
# Env-var overrides:
#   TOTAL_STEPS       — total GRPO training steps (default 50 for smoke; 150 for real)
#   TOOL_MODE         — 'cached' or 'live' (default 'cached')
#   LENS_REWARD_TIER  — '1' or '2' (default '1')
#   SFT_CKPT_PATH     — path to the SFT-trained HF ckpt. Default:
#                       checkpoints/apertus8b-lens-search-sft/global_step_<last>/actor/huggingface
#                       Set explicitly via --export=SFT_CKPT_PATH=... at sbatch time.

set -euo pipefail
ulimit -n 65535
echo "=== Job started at $(date '+%Y-%m-%d %H:%M:%S') ==="

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

export TOOL_MODE="${TOOL_MODE:-cached}"
export LENS_TOOL_MODE="$TOOL_MODE"
export TOTAL_STEPS="${TOTAL_STEPS:-50}"
export LENS_REWARD_TIER="${LENS_REWARD_TIER:-1}"

# Auto-discover the latest SFT HF ckpt if SFT_CKPT_PATH not set.
# D.7.3 SFT writes to apertus8b-lens-search-sft-toolvqa/ and the merge script
# produces global_step_<N>_hf/ dirs with config.json directly inside.
if [ -z "${SFT_CKPT_PATH:-}" ]; then
    SFT_DIR="$BASE/checkpoints/apertus8b-lens-search-sft-toolvqa"
    if [ -d "$SFT_DIR" ]; then
        # Pick highest-numbered global_step_*_hf dir that contains config.json
        LATEST=$(ls -1d "$SFT_DIR"/global_step_*_hf 2>/dev/null | sort -t_ -k3 -n | tail -1)
        if [ -n "$LATEST" ] && [ -f "$LATEST/config.json" ]; then
            export SFT_CKPT_PATH="$LATEST"
        fi
    fi
fi
echo "  TOTAL_STEPS=$TOTAL_STEPS  TOOL_MODE=$TOOL_MODE  LENS_REWARD_TIER=$LENS_REWARD_TIER"
echo "  SFT_CKPT_PATH=${SFT_CKPT_PATH:-<UNSET — config default will be used>}"

rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true

mkdir -p logs slurm/rollout_logs/armB

# --- Pre-flight checks ---
echo ""
echo "=== Pre-flight ==="
ERR=0
check() { if [ -e "$2" ]; then echo "  OK: $1"; else echo "  FAIL: $1 ($2)"; ERR=$((ERR+1)); fi; }
check "verl repo"             "$VERL/verl/__init__.py"
check "lens_search_tool.py"   "$BASE/tools/lens_search_tool.py"
check "tool config"           "$BASE/configs/lens_search_tool_config.yaml"
check "agent loop config"     "$BASE/configs/apertus_agent_loop.yaml"
check "GRPO config (Arm B)"   "$BASE/configs/lens_search_rl_grpo_armB.yaml"
check "reward fn"             "$BASE/rewards/lens_search_rl_reward.py"
check "RL train parquet"      "$BASE/data_prep/lens_search/rl_fvqa_train.parquet"
check "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"

# Must have a real SFT ckpt to run Arm B
if [ -z "${SFT_CKPT_PATH:-}" ] || [ ! -d "${SFT_CKPT_PATH:-}" ]; then
    echo "  FAIL: SFT_CKPT_PATH unset or missing — Arm B requires the SFT job to have completed."
    ERR=$((ERR+1))
else
    check "SFT ckpt config.json" "$SFT_CKPT_PATH/config.json"
fi

if [ "$TOOL_MODE" = "cached" ]; then
    check "FVQA cache pkl" "/iopsstor/scratch/cscs/$USER/hf_home/hub/datasets--lmms-lab--FVQA/snapshots/bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5/fvqa_train_image_search_results_cache.pkl"
fi
if [ "$TOOL_MODE" = "live" ]; then
    if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] || [ ! -f "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
        echo "  FAIL: GOOGLE_APPLICATION_CREDENTIALS unset (live mode)"
        ERR=$((ERR+1))
    else
        echo "  OK: GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS"
    fi
fi

if [ $ERR -gt 0 ]; then echo "ERROR: $ERR pre-flight failed."; exit 1; fi

# --- Verify merged HF ckpt loads cleanly (catches merge bugs before GRPO init) ---
echo ""
echo "=== Pre-flight: verify SFT_CKPT_PATH loads as HF model ==="
python3 -c "
import sys
from transformers import AutoConfig, AutoTokenizer
ckpt = '$SFT_CKPT_PATH'
try:
    cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    print(f'  OK: config loaded (model_type={cfg.model_type}, vocab_size={cfg.vocab_size})')
    print(f'  OK: tokenizer loaded ({type(tok).__name__})')
except Exception as e:
    print(f'  FAIL: {type(e).__name__}: {e}')
    sys.exit(1)
" || { echo "ERROR: merged HF ckpt failed to load — aborting before GRPO init."; exit 1; }

nvidia-smi --list-gpus

# --- Launch GRPO ---
echo ""
echo "=== Starting Arm B GRPO ==="
T0=$(date +%s)
python3 -m verl.trainer.main_ppo \
    --config-path="$BASE/configs" \
    --config-name=lens_search_rl_grpo_armB "$@"
T1=$(date +%s)
echo ""
echo "=== GRPO elapsed: $((T1 - T0))s ==="

echo "=== Job ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
