#!/bin/bash
#SBATCH --job-name=prepare_lens_search
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=2:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Phase D.1 data prep — FVQA-only Lens-search SFT + RL parquets.
#
# Runs:
#   1. prepare_lens_search_sft.py → sft_train.parquet (~950), sft_val.parquet (~50),
#      sft_selected_data_ids.json
#   2. prepare_lens_search_rl.py  → rl_train.parquet (~3,856; excludes SFT data_ids)
#
# Total time: ~30-60 min for ~5k IBQ-encoded images on 1 GPU.

set -euo pipefail
echo "=== Job started at $(date '+%Y-%m-%d %H:%M:%S') ==="

source /users/$USER/miniconda3/etc/profile.d/conda.sh
conda activate verl

BASE="/capstor/scratch/cscs/$USER/verl-apertus"
cd "$BASE"

EMU3_SRC="/users/$USER/Emu3.5/src"
export PYTHONPATH="${EMU3_SRC}:${BASE}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH:-/iopsstor/scratch/cscs/$USER}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONUNBUFFERED=1
unset SSL_CERT_FILE

APERTUS_CHECKPOINT="/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF"
VQ_MODEL_PATH="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Emu3.5-VisionTokenizer"

mkdir -p "$BASE/logs" "$BASE/data_prep/lens_search"

# --- Pre-flight checks ---
echo ""
echo "=== Pre-flight ==="
ERR=0
check() { if [ -e "$2" ]; then echo "  OK: $1"; else echo "  FAIL: $1 ($2)"; ERR=$((ERR+1)); fi; }
check "Apertus tokenizer cfg"  "$APERTUS_CHECKPOINT/tokenizer_config.json"
check "VQ model"               "$VQ_MODEL_PATH/model.ckpt"
check "Emu3.5 src"             "$EMU3_SRC/vision_tokenizer/__init__.py"
check "configs/apertus.yaml"   "$BASE/configs/apertus.yaml"
check "lens_search_common"     "$BASE/data_prep/lens_search_common.py"
check "prepare_lens_search_sft.py" "$BASE/data_prep/prepare_lens_search_sft.py"
check "prepare_lens_search_rl.py"  "$BASE/data_prep/prepare_lens_search_rl.py"
python3 -c "import torch, transformers, pyarrow, einops" 2>&1 || { echo "  FAIL: imports"; ERR=$((ERR+1)); }
[ $ERR -gt 0 ] && { echo "ERROR: $ERR pre-flight failed."; exit 1; }

nvidia-smi --list-gpus

# --- Step 1: SFT prep (1k subset) ---
echo ""
echo "=== Step 1: SFT data prep (1k subset) ==="
T1=$(date +%s)
python3 data_prep/prepare_lens_search_sft.py \
    --n 1000 \
    --val-ratio 0.05 \
    --seed 42 \
    --config configs/apertus.yaml
T2=$(date +%s)
echo "  Step 1 elapsed: $((T2 - T1))s"

# --- Step 2: RL prep (remaining ~3,856 rows, excludes SFT data_ids) ---
echo ""
echo "=== Step 2: RL data prep (~3,856 remaining) ==="
T3=$(date +%s)
python3 data_prep/prepare_lens_search_rl.py \
    --config configs/apertus.yaml \
    --sft-selected-ids data_prep/lens_search/sft_selected_data_ids.json
T4=$(date +%s)
echo "  Step 2 elapsed: $((T4 - T3))s"

# --- Final summary ---
echo ""
echo "=== Outputs ==="
for f in sft_train.parquet sft_val.parquet sft_selected_data_ids.json rl_train.parquet sft_metadata.jsonl rl_metadata.jsonl; do
    p="$BASE/data_prep/lens_search/$f"
    if [ -e "$p" ]; then
        echo "  OK ($(du -h "$p" | awk '{print $1}'))  $p"
    else
        echo "  MISSING  $p"
    fi
done

echo ""
echo "=== Job ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
