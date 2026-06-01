#!/bin/bash
#SBATCH --job-name=apertus_rotate_flip_base
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=1:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=debug
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
trap 'status=$?; echo "=== apertus_rotate_flip_base exited with status ${status} at $(date) ==="' EXIT

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

BASE="/users/$USER/capscratch/verl-apertus"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
cd "$BASE"

export PYTHONPATH="${EMU3_SRC}:${BASE}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

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
check_path "Apertus config" "$BASE/configs/apertus.yaml"
check_path "test metadata" "$BASE/data_prep/rotate_flip_rl/test_metadata.jsonl"
check_path "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"
python3 -c "import einops" 2>/dev/null && echo "  OK: einops installed" || { echo "  FAIL: einops not installed"; ((ERROR_COUNT++)); }

if [ $ERROR_COUNT -gt 0 ]; then
    echo "ERROR: $ERROR_COUNT pre-flight checks failed."
    exit 1
fi

echo "=== GPU Info ==="
nvidia-smi --list-gpus

echo "=== Starting Apertus rotate-flip baseline inference ==="
python3 inference/run_rotate_flip_baseline.py \
    --model apertus \
    --config configs/apertus.yaml \
    --metadata data_prep/rotate_flip_rl/test_metadata.jsonl \
    --output-dir results/rotate_flip_rl/apertus_baseline \
    "$@"

echo "=== Evaluating Apertus rotate-flip baseline ==="
python3 evaluation/compute_rotate_flip_rl_accuracy.py \
    --rollout-file results/rotate_flip_rl/apertus_baseline/predictions.jsonl \
    --output-dir results/rotate_flip_rl/apertus_baseline \
    --metadata data_prep/rotate_flip_rl/test_metadata.jsonl

echo "=== Job Complete ==="
