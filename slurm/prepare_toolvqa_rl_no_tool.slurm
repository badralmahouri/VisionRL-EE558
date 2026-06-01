#!/bin/bash
#SBATCH --job-name=prepare_toolvqa_no_tool
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128000
#SBATCH --time=1:30:00
#SBATCH --environment=verl_env
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
trap 'status=$?; echo "=== prepare_toolvqa_no_tool exited with status ${status} at $(date) ==="' EXIT

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

PROJECT="/users/$USER/capscratch/verl-apertus"
EMU3_SRC="/users/$USER/capscratch/Emu3.5/src"
cd "$PROJECT"

export PYTHONPATH="${PROJECT}:${EMU3_SRC}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH}/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT/data_prep/toolvqa_rl_no_tool}"
KEEP_TOOLS="${KEEP_TOOLS:-TextToBbox RegionAttributeDescription CountGivenObject Plot}"

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
check_path "ToolVQA metadata" "$PROJECT/data_prep/toolvqa/metadata.jsonl"
check_path "ToolVQA images" "$PROJECT/data_prep/toolvqa/images"
check_path "parser" "$PROJECT/data_prep/prepare_toolvqa_rl_parse.py"
check_path "Apertus config" "$PROJECT/configs/apertus.yaml"
check_path "Emu3.5 vision_tokenizer" "$EMU3_SRC/vision_tokenizer/__init__.py"

if [ $ERROR_COUNT -gt 0 ]; then
    echo "ERROR: $ERROR_COUNT pre-flight checks failed."
    exit 1
fi

echo "=== GPU Info ==="
nvidia-smi --list-gpus

echo "== Job started at $(date)"
echo "=== Encoding ToolVQA no-tool RL parquet ==="
echo "Output dir: $OUTPUT_DIR"
echo "Keep tools filter: $KEEP_TOOLS"

python3 data_prep/prepare_toolvqa_rl_parse.py \
    --output-dir "$OUTPUT_DIR" \
    --no-tool \
    --keep-tools $KEEP_TOOLS \
    "$@"

echo "=== Job Complete ==="
