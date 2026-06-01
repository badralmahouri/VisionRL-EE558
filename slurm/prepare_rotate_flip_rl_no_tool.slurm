#!/bin/bash
#SBATCH --job-name=prepare_rotflip_no_tool
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=1:30:00
#SBATCH --environment=verl_env
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
trap 'status=$?; echo "=== prepare_rotflip_no_tool exited with status ${status} at $(date) ==="' EXIT

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

BASE="/users/$USER/capscratch/verl-apertus"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
cd "$BASE"
export PYTHONPATH="${EMU3_SRC}:${BASE}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONUNBUFFERED=1

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
check_path "raw rotate/flip data" "$BASE/data_prep/rotate_flip_rl/raw.jsonl"
check_path "test raw rotate/flip data" "$BASE/data_prep/rotate_flip_rl/test_raw.jsonl"
check_path "rotate/flip images dir" "$BASE/data_prep/rotate_flip_rl/images"
check_path "Apertus config" "$BASE/configs/apertus.yaml"
check_path "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"

if [ $ERROR_COUNT -gt 0 ]; then
    echo "ERROR: $ERROR_COUNT pre-flight checks failed."
    exit 1
fi

echo "=== GPU Info ==="
nvidia-smi --list-gpus

echo "=== Parsing no-tool rotate-flip RL data ==="
python3 data_prep/prepare_rotate_flip_rl_parse.py \
    --config configs/apertus.yaml \
    --input data_prep/rotate_flip_rl/raw.jsonl \
    --test-input data_prep/rotate_flip_rl/test_raw.jsonl \
    --image-root data_prep/rotate_flip_rl \
    --output-dir data_prep/rotate_flip_rl_no_tool \
    --no-tool \
    "$@"

echo "=== Job Complete ==="
