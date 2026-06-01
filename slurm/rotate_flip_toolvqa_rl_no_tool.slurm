#!/bin/bash
#SBATCH --job-name=rotflip_toolvqa_no_tool
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=10:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
trap 'status=$?; echo "=== rotflip_toolvqa_no_tool exited with status ${status} at $(date) ==="' EXIT
ulimit -n 65535

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

PROJECT="/users/$USER/capscratch/verl-apertus"
VERL="/users/$USER/capscratch/verl"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
BASE_CHECKPOINT_ROOT="/capstor/scratch/cscs/$USER/verl-apertus/checkpoints/apertus8b-grpo-rotate-flip-toolvqa-no-tool"
START_FRESH="${START_FRESH:-0}"
TRAINER_OVERRIDES=()

if [ "${1:-}" = "--fresh" ] || [ "${1:-}" = "fresh" ]; then START_FRESH=1; shift; fi
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$BASE_CHECKPOINT_ROOT}"
if [ "$START_FRESH" = "1" ]; then
    TRAINER_OVERRIDES+=("trainer.resume_mode=disable")
    TRAINER_OVERRIDES+=("trainer.default_local_dir=$CHECKPOINT_ROOT")
fi

cd "$PROJECT"
export PYTHONPATH="${PROJECT}:${VERL}:${EMU3_SRC}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH}/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-online}

echo "=== Pre-flight Checks ==="
test -e "$VERL/verl/__init__.py"
test -e "$PROJECT/configs/rotate_flip_toolvqa_rl_no_tool_grpo.yaml"
test -e "$PROJECT/rewards/toolvqa_rl_reward.py"
test -e "$PROJECT/data_prep/rotate_flip_toolvqa_rl_no_tool/train.parquet"
test -e "$PROJECT/data_prep/rotate_flip_toolvqa_rl_no_tool/val.parquet"

if [ "$START_FRESH" = "1" ]; then
    case "$CHECKPOINT_ROOT" in
        /capstor/scratch/cscs/$USER/verl-apertus/checkpoints/apertus8b-grpo-rotate-flip-toolvqa-no-tool*)
            rm -rf "$CHECKPOINT_ROOT" ;;
        *) echo "ERROR: refusing to remove unexpected checkpoint path: $CHECKPOINT_ROOT"; exit 1 ;;
    esac
fi

nvidia-smi --list-gpus
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT/configs" \
    --config-name=rotate_flip_toolvqa_rl_no_tool_grpo \
    "${TRAINER_OVERRIDES[@]}" \
    "$@"

echo "=== Saved checkpoints ==="
find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sort -V || true
echo "=== Job Complete ==="
