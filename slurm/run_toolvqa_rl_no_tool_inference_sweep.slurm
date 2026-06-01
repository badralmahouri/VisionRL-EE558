#!/bin/bash
#SBATCH --job-name=toolvqa_no_tool_infer
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
trap 'status=$?; echo "=== toolvqa_no_tool_infer exited with status ${status} at $(date) ==="' EXIT
ulimit -n 65535

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

PROJECT="/users/$USER/capscratch/verl-apertus"
VERL="/users/$USER/capscratch/verl"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/capstor/scratch/cscs/$USER/verl-apertus/checkpoints/apertus8b-grpo-toolvqa-no-tool}"
TEST_PARQUET="${TEST_PARQUET:-$PROJECT/data_prep/toolvqa_rl_no_tool/test.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT/results/toolvqa_rl/no_tool_checkpoints}"
STEPS="${STEPS:-10 20 30 40 50 60 70 80 90 100}"

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
export WANDB_MODE=${WANDB_MODE:-offline}

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
check_path "trainer yaml" "$PROJECT/configs/toolvqa_rl_no_tool_grpo.yaml"
check_path "test parquet" "$TEST_PARQUET"
check_path "checkpoint root" "$CHECKPOINT_ROOT"
check_path "accuracy bridge" "$PROJECT/evaluation/compute_toolvqa_rl_accuracy.py"
check_path "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"

if [ $ERROR_COUNT -gt 0 ]; then
    echo "ERROR: $ERROR_COUNT pre-flight checks failed."
    exit 1
fi

echo "=== GPU Info ==="
nvidia-smi --list-gpus

echo "== No-tool ToolVQA sweep started at $(date)"
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Test parquet:     $TEST_PARQUET"
echo "Output root:      $OUTPUT_ROOT"
echo "Steps:            $STEPS"
echo "Extra overrides:  $*"

echo "=== Requested checkpoint check ==="
PRESENT_STEPS=()
MISSING_STEPS=()
for STEP in $STEPS; do
    if [ -d "$CHECKPOINT_ROOT/global_step_${STEP}" ]; then
        PRESENT_STEPS+=("$STEP")
    else
        MISSING_STEPS+=("$STEP")
    fi
done
echo "Present steps: ${PRESENT_STEPS[*]:-none}"
echo "Missing steps: ${MISSING_STEPS[*]:-none}"
if [ ${#PRESENT_STEPS[@]} -eq 0 ]; then
    echo "ERROR: none of the requested checkpoints exist."
    exit 1
fi
if [ ${#MISSING_STEPS[@]} -gt 0 ]; then
    echo "WARNING: missing checkpoints will be skipped."
fi

COMPLETED_STEPS=()

for STEP in $STEPS; do
    CHECKPOINT="$CHECKPOINT_ROOT/global_step_${STEP}"
    RESULT_DIR="$OUTPUT_ROOT/global_step_${STEP}"
    ROLLOUT_DIR="$RESULT_DIR/rollouts"
    ACCURACY_JSON="$RESULT_DIR/accuracy.json"

    echo
    echo "=== Evaluating no-tool global_step_${STEP} at $(date) ==="
    echo ">>> START global_step_${STEP}"
    echo "Checkpoint: $CHECKPOINT"
    echo "Result dir: $RESULT_DIR"

    if [ ! -d "$CHECKPOINT" ]; then
        echo "Skipping missing checkpoint: $CHECKPOINT"
        continue
    fi

    rm -rf "$ROLLOUT_DIR"
    mkdir -p "$ROLLOUT_DIR"

    echo "=== [global_step_${STEP}] Starting rollout generation at $(date) ==="
    python3 -m verl.trainer.main_ppo \
        --config-path="$PROJECT/configs" \
        --config-name=toolvqa_rl_no_tool_grpo \
        data.val_files="$TEST_PARQUET" \
        trainer.val_only=true \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="$CHECKPOINT" \
        trainer.validation_data_dir="$ROLLOUT_DIR" \
        trainer.rollout_data_dir="$ROLLOUT_DIR" \
        trainer.logger='["console"]' \
        actor_rollout_ref.rollout.n=1 \
        "$@"

    echo "=== [global_step_${STEP}] Rollout generation finished at $(date) ==="
    echo "=== [global_step_${STEP}] Computing accuracy ==="
    python3 evaluation/compute_toolvqa_rl_accuracy.py \
        --rollout-dir "$ROLLOUT_DIR" \
        --output-dir "$RESULT_DIR"

    if [ -f "$ACCURACY_JSON" ]; then
        echo "=== [global_step_${STEP}] Accuracy written: $ACCURACY_JSON ==="
    else
        echo "WARNING: expected accuracy file was not created: $ACCURACY_JSON"
    fi

    COMPLETED_STEPS+=("$STEP")
    echo ">>> DONE global_step_${STEP} at $(date)"
done

echo "Completed steps: ${COMPLETED_STEPS[*]:-none}"
echo "== No-tool ToolVQA sweep ended at $(date)"
echo "=== Job Complete ==="
