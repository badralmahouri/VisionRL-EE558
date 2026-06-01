#!/bin/bash
#SBATCH --job-name=prepare_line_drawing_rl
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

echo "=== Job started at $(date '+%Y-%m-%d %H:%M:%S') ==="

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
check_path "raw line data" "$BASE/data_prep/line_drawing_rl/raw.jsonl"
check_path "test raw line data" "$BASE/data_prep/line_drawing_rl/test_raw.jsonl"
check_path "line images dir" "$BASE/data_prep/line_drawing_rl/images"
check_path "Apertus config" "$BASE/configs/apertus.yaml"
check_path "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"

if [ $ERROR_COUNT -gt 0 ]; then
    echo "ERROR: $ERROR_COUNT pre-flight checks failed."
    echo "Hint: run python data_prep/prepare_line_drawing_rl_generate.py first."
    exit 1
fi

echo "=== GPU Info ==="
nvidia-smi --list-gpus

echo "=== Parsing line-drawing RL data ==="
python3 data_prep/prepare_line_drawing_rl_parse.py --config configs/apertus.yaml "$@"

echo "=== Job ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "=== Job Complete ==="
