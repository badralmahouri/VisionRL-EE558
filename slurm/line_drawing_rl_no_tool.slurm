#!/bin/bash
#SBATCH --job-name=line_rl_no_tool
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
trap 'status=$?; echo "=== line_rl_no_tool exited with status ${status} at $(date) ==="' EXIT
ulimit -n 65535

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

PROJECT="/users/$USER/capscratch/verl-apertus"
VERL="/users/$USER/capscratch/verl"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
BASE_CHECKPOINT_ROOT="/capstor/scratch/cscs/$USER/verl-apertus/checkpoints/apertus8b-grpo-line-drawing-no-tool"
START_FRESH="${START_FRESH:-0}"
TRAINER_OVERRIDES=()

if [ "${1:-}" = "--fresh" ] || [ "${1:-}" = "fresh" ] || [ "${1:-}" = "start_from_top" ]; then
    START_FRESH=1
    shift
fi

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
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning:transformers,ignore::UserWarning:torch.distributed.fsdp.fully_sharded_data_parallel,ignore::UserWarning:ray._private.worker"
export RAY_DEDUP_LOGS=1
export RAY_BACKEND_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-online}

ERROR_COUNT=0
check_path() {
    local label="$1" path="$2"
    if [ -e "$path" ]; then
        echo "  OK: $label"
    else
        echo "  FAIL: $label ($path)"
        ((ERROR_COUNT++))
    fi
}

echo "=== Pre-flight Checks ==="
check_path "verl repo" "$VERL/verl/__init__.py"
check_path "trainer yaml" "$PROJECT/configs/line_drawing_rl_no_tool_grpo.yaml"
check_path "reward fn" "$PROJECT/rewards/cof_rl_reward.py"
check_path "train parquet" "$PROJECT/data_prep/line_drawing_rl_no_tool/train.parquet"
check_path "val parquet" "$PROJECT/data_prep/line_drawing_rl_no_tool/val.parquet"
check_path "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"

if [ $ERROR_COUNT -gt 0 ]; then
    echo "ERROR: $ERROR_COUNT pre-flight checks failed."
    exit 1
fi

if [ "$START_FRESH" = "1" ]; then
    echo "=== Fresh-start checkpoint cleanup ==="
    echo "Checkpoint root: $CHECKPOINT_ROOT"
    case "$CHECKPOINT_ROOT" in
        /capstor/scratch/cscs/$USER/verl-apertus/checkpoints/apertus8b-grpo-line-drawing-no-tool*)
            if [ -d "$CHECKPOINT_ROOT" ]; then
                echo "Removing existing checkpoint directory: $CHECKPOINT_ROOT"
                rm -rf "$CHECKPOINT_ROOT"
            else
                echo "No existing checkpoint directory to remove."
            fi
            ;;
        *)
            echo "ERROR: refusing to remove unexpected checkpoint path: $CHECKPOINT_ROOT"
            exit 1
            ;;
    esac
fi

echo "=== GPU Info ==="
nvidia-smi --list-gpus

echo "== Job started at $(date)"
echo "=== Starting no-tool line-drawing GRPO training ==="
echo "=== Trainer launch at $(date) ==="
echo "Start fresh: $START_FRESH"
echo "Checkpoint root: $CHECKPOINT_ROOT"
if [ ${#TRAINER_OVERRIDES[@]} -gt 0 ]; then
    printf 'Trainer fresh-start overrides:\n'
    printf '  %s\n' "${TRAINER_OVERRIDES[@]}"
fi
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT/configs" \
    --config-name=line_drawing_rl_no_tool_grpo \
    "${TRAINER_OVERRIDES[@]}" \
    "$@"


echo "=== Saved checkpoints ==="
if [ -d "$CHECKPOINT_ROOT" ]; then
    mapfile -t CHECKPOINT_STEPS < <(
        find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' \
            | sed 's/global_step_//' \
            | sort -n
    )

    if [ ${#CHECKPOINT_STEPS[@]} -gt 0 ]; then
        printf 'Checkpoint root: %s\n' "$CHECKPOINT_ROOT"
        printf 'Saved global steps: %s\n' "${CHECKPOINT_STEPS[*]}"
        printf 'For inference sweep, run:\n'
        printf 'CHECKPOINT_ROOT="%s" TEST_PARQUET="%s" OUTPUT_ROOT="%s" STEPS="%s" sbatch slurm/run_line_drawing_no_tool_inference_sweep.slurm\n' \
            "$CHECKPOINT_ROOT" \
            "$PROJECT/data_prep/line_drawing_rl_no_tool/test.parquet" \
            "$PROJECT/results/line_drawing_rl/no_tool_checkpoints" \
            "${CHECKPOINT_STEPS[*]}"
    else
        echo "No global_step_* checkpoints found under $CHECKPOINT_ROOT"
    fi
else
    echo "Checkpoint root does not exist: $CHECKPOINT_ROOT"
fi

echo "=== Job ended at $(date) ==="
echo "=== Job Complete ==="
