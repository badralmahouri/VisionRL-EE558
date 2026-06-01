#!/bin/bash
#SBATCH --job-name=qwen_line_baseline
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
trap 'status=$?; echo "=== qwen_line_baseline exited with status ${status} at $(date) ==="' EXIT

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

BASE="/users/$USER/capscratch/verl-apertus"
cd "$BASE"

export PYTHONPATH="${BASE}:${PYTHONPATH:-}"
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
check_path "Qwen config" "$BASE/configs/qwen.yaml"
check_path "test metadata" "$BASE/data_prep/line_drawing_rl/test_metadata.jsonl"
python3 -c "from qwen_vl_utils import process_vision_info" 2>/dev/null && echo "  OK: qwen-vl-utils" || { echo "  FAIL: qwen-vl-utils not installed"; ((ERROR_COUNT++)); }

if [ $ERROR_COUNT -gt 0 ]; then
    echo "ERROR: $ERROR_COUNT pre-flight checks failed."
    exit 1
fi

echo "=== GPU Info ==="
nvidia-smi --list-gpus

echo "=== Starting Qwen line-drawing baseline inference ==="
python3 inference/run_line_drawing_baseline.py \
    --model qwen \
    --config configs/qwen.yaml \
    --metadata data_prep/line_drawing_rl/test_metadata.jsonl \
    --output-dir results/line_drawing_rl/qwen_baseline \
    "$@"

echo "=== Evaluating Qwen line-drawing baseline ==="
python3 evaluation/compute_accuracy.py \
    --predictions results/line_drawing_rl/qwen_baseline/predictions.jsonl \
    --output results/line_drawing_rl/qwen_baseline/accuracy.json \
    --mode mcq

echo "=== Job Complete ==="
