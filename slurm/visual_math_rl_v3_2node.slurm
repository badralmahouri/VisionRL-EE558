#!/bin/bash
#SBATCH --job-name=vmath_rl_v3
#SBATCH --account=infra01
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=6:00:00
#SBATCH --partition=normal
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Visual-math v3 RL: judge + training co-scheduled on 2 nodes in one allocation.
#
# Why 2-node single sbatch instead of two separate jobs:
#   Two separate 1-node jobs cannot guarantee temporal co-scheduling on a busy
#   partition. The judge can start at an unexpected time, eat its 12h window,
#   and expire before the RL job is ever scheduled (observed: jobs 2386229 +
#   2395263 on 2026-05-27). One 2-node allocation eliminates the race: both
#   nodes land simultaneously, judge and RL share the same 6h window.
#
# Why --environment is on srun, NOT on #SBATCH:
#   CSCS pyxis: "#SBATCH --environment" puts the BATCH SCRIPT itself inside the
#   container, which means srun is called from inside the container and fails to
#   load the host-OS HPE Slingshot switch plugin (libjson-c.so.5 missing from
#   container LD_LIBRARY_PATH) → srun: fatal: Can't find plugin for
#   switch/hpe_slingshot (observed: job 2407639). Batch script must run on host
#   OS; each srun step carries its own --environment=verl_env.
#
# Node layout (resolved from SLURM_JOB_NODELIST at runtime):
#   Node 0 (JUDGE_NODE): Qwen3-32B sglang server, tp=4, port 30100
#   Node 1 (RL_NODE):    VeRL GRPO training, 4-GPU FSDP
#
# Submit per arm:
#   sbatch --export=ARM=dynamath slurm/visual_math_rl_v3_2node.slurm
#   sbatch --export=ARM=mathvr   slurm/visual_math_rl_v3_2node.slurm

set -euo pipefail
ulimit -n 65535
echo "=== v3 2-node job started at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "=== Nodes: $SLURM_JOB_NODELIST ==="

BASE="/capstor/scratch/cscs/$USER/verl-apertus"
VERL="/users/$USER/verl-vision"
EMU3_SRC="/users/$USER/Emu3.5/src"

cd "$BASE"
mkdir -p "$BASE/logs"

export PYTHONPATH="${BASE}:${VERL}:${EMU3_SRC}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH:-/iopsstor/scratch/cscs/$USER}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
unset SSL_CERT_FILE

# --- resolve node assignments ---
JUDGE_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
RL_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -1)
echo "=== JUDGE_NODE=$JUDGE_NODE  RL_NODE=$RL_NODE"

# --- ARM / dataset selection ---
ARM="${ARM:-dynamath}"
echo "=== ARM = $ARM ==="

case "$ARM" in
  mathvr)
    TRAIN_PARQUET="$BASE/data_prep/visual_math_rl/mathvr_train.parquet"
    IDIST_VAL="$BASE/data_prep/visual_math_rl/mathvr_indist_eval.parquet"
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

# --- pre-flight checks (host OS — file existence only, no Python needed) ---
echo "=== Pre-flight Checks ==="
ERR=0
check() { if [ -e "$2" ]; then echo "  OK: $1"; else echo "  FAIL: $1 ($2)"; ERR=$((ERR+1)); fi; }
check "verl repo"               "$VERL/verl/__init__.py"
check "trainer yaml"            "$BASE/configs/visual_math_rl_grpo.yaml"
check "reward fn"               "$BASE/rewards/visual_math_rl_reward.py"
check "train parquet"           "$TRAIN_PARQUET"
check "in-dist val"             "$IDIST_VAL"
check "mathvision val"          "$MV_VAL"
check "mathverse val"           "$MS_VAL"
check "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"
[ $ERR -eq 0 ] || { echo "ERROR: $ERR pre-flight check(s) failed."; exit 1; }

# --- write judge URL before launching server (URL is deterministic from JUDGE_NODE) ---
JUDGE_PORT=30100
JUDGE_URL="http://${JUDGE_NODE}:${JUDGE_PORT}/v1"
URL_FILE="$BASE/slurm/judge_url.txt"

{
  echo "$JUDGE_URL"
  echo "# job_id=${SLURM_JOB_ID}  host=${JUDGE_NODE}  written=$(date -Iseconds)  model=Qwen/Qwen3-32B"
} > "$URL_FILE"
echo "=== Judge URL: $JUDGE_URL"
echo "=== URL written to: $URL_FILE"

cleanup() {
  echo "=== 2-node job ending — removing judge URL file ==="
  rm -f "$URL_FILE" 2>/dev/null || true
}
trap cleanup EXIT

# --- launch judge on JUDGE_NODE (background) ---
# sglang manages tp=4 GPU processes internally; --ntasks=1 is correct.
# --environment=verl_env on srun (not on #SBATCH) so batch script runs on host
# OS with SLURM plugins intact; container is invoked per srun step.
#
# env vars MUST be re-exported INSIDE the container via `bash -c`: the CSCS
# container engine does NOT forward the host shell's env across the
# --environment boundary (observed job 2414387: HF_HOME ignored, sglang
# downloaded Qwen3-32B to the container's ~/.cache = over-quota VAST home →
# "Disk quota exceeded"). Host values are interpolated at sbatch-render time,
# then re-exported in the container shell before python3.
#
# HF_HUB_OFFLINE=1: the full Qwen3-32B revision is pre-cached under
# $HF_HOME/hub (17/17 safetensors + config + fast tokenizer.json, verified).
# Offline => zero network, zero writes to ~/.cache, no quota risk; fails fast
# and clearly if anything is missing instead of silently re-downloading.
echo "=== Launching sglang Qwen3-32B on $JUDGE_NODE (tp=4, port=$JUDGE_PORT) ==="
srun --nodes=1 --ntasks=1 --nodelist="$JUDGE_NODE" --environment=verl_env \
  bash -c "
    export HF_HOME='$HF_HOME'
    export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export PYTHONUNBUFFERED=1
    python3 -m sglang.launch_server \
      --model-path Qwen/Qwen3-32B \
      --tp 4 \
      --host 0.0.0.0 \
      --port $JUDGE_PORT \
      --mem-fraction-static 0.85 \
      --trust-remote-code
  " &
JUDGE_PID=$!

# --- block until judge /health is ready (~15 min for Qwen3-32B) ---
# curl loop (host-OS curl) avoids needing container Python imports on the batch script.
export VISUAL_MATH_JUDGE_URL="$JUDGE_URL"
echo "=== Waiting for judge /health (timeout 1800s) ==="
DEADLINE=$((SECONDS + 1800))
until curl -sf --max-time 5 "http://${JUDGE_NODE}:${JUDGE_PORT}/health" > /dev/null 2>&1; do
  if [ $SECONDS -ge $DEADLINE ]; then
    echo "ERROR: judge at http://${JUDGE_NODE}:${JUDGE_PORT}/health not ready after 1800s"
    exit 1
  fi
  sleep 10
done
echo "=== Judge READY ==="

# --- RL training on RL_NODE ---
# VeRL hybrid_engine manages its own FSDP + sglang rollout processes internally.
TOTAL_STEPS="${TOTAL_STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
echo "=== Starting GRPO training on $RL_NODE (arm=$ARM, steps=$TOTAL_STEPS, save@$SAVE_FREQ, test@$TEST_FREQ) ==="

# Same in-container env re-export as the judge srun (host env does NOT cross the
# --environment boundary). PYTHONPATH is critical here: verl, Emu3.5, and the
# reward fn are resolved from it; without re-export the trainer can't import
# them. No HF_HUB_OFFLINE: the trainer loads the local Apertus checkpoint and
# HF_HOME points at roomy iopsstor, so an incidental small fetch can't hit a
# quota wall. Inner single quotes on data.val_files protect the [..] from glob
# while host vars still expand inside the double-quoted bash -c body.
srun --nodes=1 --ntasks=1 --nodelist="$RL_NODE" --environment=verl_env \
  bash -c "
    export HF_HOME='$HF_HOME'
    export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
    export PYTHONPATH='$PYTHONPATH'
    export PYTHONUNBUFFERED=1
    export HYDRA_FULL_ERROR=1
    export WANDB_MODE=offline
    export VISUAL_MATH_JUDGE_URL='$VISUAL_MATH_JUDGE_URL'
    python3 -m verl.trainer.main_ppo \
      --config-path='$BASE/configs' \
      --config-name=visual_math_rl_grpo \
      data.train_files='$TRAIN_PARQUET' \
      'data.val_files=[$IDIST_VAL,$MV_VAL,$MS_VAL]' \
      trainer.experiment_name='$EXPERIMENT' \
      trainer.total_epochs=10 \
      trainer.total_training_steps='$TOTAL_STEPS' \
      trainer.test_freq='$TEST_FREQ' \
      trainer.save_freq='$SAVE_FREQ' \
      $*
  "

echo "=== RL training complete at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "=== v3 2-node job done (arm=$ARM) ==="
# cleanup trap removes judge_url.txt; SLURM kills the judge srun automatically
