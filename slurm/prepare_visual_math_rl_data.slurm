#!/bin/bash
#SBATCH --job-name=vmath_data_prep
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=4:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Step 1 of plan: data prep. Downloads Math-VR-train, DynaMath_Sample,
# MathVision, MathVerse Vision-Only from HF; IBQ-encodes each image; writes
# parquets to data_prep/visual_math_rl/.
#
# Submit selection via env vars:
#   sbatch --export=STAGE=mathvr   slurm/prepare_visual_math_rl_data.slurm
#   sbatch --export=STAGE=dynamath slurm/prepare_visual_math_rl_data.slurm
#   sbatch --export=STAGE=eval     slurm/prepare_visual_math_rl_data.slurm
#   sbatch --export=STAGE=all      slurm/prepare_visual_math_rl_data.slurm  (sequential)
#
# Three stages can run in parallel as separate SLURM jobs (each needs 1 GPU
# for IBQ encoding).

set -euo pipefail
echo "=== Job started at $(date '+%Y-%m-%d %H:%M:%S') ==="

source /users/$USER/miniconda3/etc/profile.d/conda.sh
conda activate verl

BASE="/capstor/scratch/cscs/$USER/verl-apertus"
EMU3_SRC="/users/$USER/Emu3.5/src"
cd "$BASE"

export PYTHONPATH="${BASE}:${EMU3_SRC}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH:-/iopsstor/scratch/cscs/$USER}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONUNBUFFERED=1
unset SSL_CERT_FILE

STAGE="${STAGE:-all}"
echo "=== STAGE = $STAGE ==="

# --- pre-flight ---
ERR=0
check() { if [ -e "$2" ]; then echo "  OK: $1"; else echo "  FAIL: $1 ($2)"; ERR=$((ERR+1)); fi; }
check "apertus.yaml"              "$BASE/configs/apertus.yaml"
check "vq tokenizer dir"          "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Emu3.5-VisionTokenizer/config.yaml"
check "Emu3.5 vision_tokenizer"   "$EMU3_SRC/vision_tokenizer/__init__.py"
check "common helpers"            "$BASE/data_prep/visual_math_rl_common.py"
python3 -c "import datasets, transformers, pyarrow, PIL" 2>&1 || { echo "  FAIL: imports"; ERR=$((ERR+1)); }
[ $ERR -eq 0 ] || { echo "ERROR: $ERR pre-flight failed."; exit 1; }

nvidia-smi --list-gpus

run_mathvr() {
    echo "=== Stage: mathvr ==="
    python3 data_prep/prepare_visual_math_rl_mathvr.py \
        --out-dir "$BASE/data_prep/visual_math_rl" \
        --seed 42
}

run_dynamath() {
    echo "=== Stage: dynamath ==="
    python3 data_prep/prepare_visual_math_rl_dynamath.py \
        --out-dir "$BASE/data_prep/visual_math_rl" \
        --seed 42
}

run_eval() {
    echo "=== Stage: eval (mathvision-500 + mathverse-VO-500) ==="
    python3 data_prep/prepare_visual_math_rl_eval.py \
        --out-dir "$BASE/data_prep/visual_math_rl" \
        --seed 42
}

case "$STAGE" in
  mathvr)   run_mathvr ;;
  dynamath) run_dynamath ;;
  eval)     run_eval ;;
  all)      run_mathvr; run_dynamath; run_eval ;;
  *)        echo "ERROR: unknown STAGE=$STAGE (mathvr|dynamath|eval|all)"; exit 1 ;;
esac

echo ""
echo "=== Final listing ==="
ls -lh "$BASE/data_prep/visual_math_rl/"*.parquet 2>/dev/null || echo "(no parquets yet)"
echo ""
echo "=== Job ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
