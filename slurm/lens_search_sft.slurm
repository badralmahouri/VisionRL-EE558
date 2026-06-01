#!/bin/bash
#SBATCH --job-name=lens_sft
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=460000
#SBATCH --time=1:25:00
#SBATCH --environment=verl_env
#SBATCH --partition=debug
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Arm B SFT cold-start for Lens-search — D.7.3 (ToolVQA-based).
# 2,516 train + 132 val rows × 5 epochs × bs=32 ≈ 394 grad updates.
# Expected wall time: ~40-60 min on 4 GPUs.
# Uses the generic CoFSFTDataset (loss-mask is tool-name-agnostic).

set -euo pipefail
echo "=== Job started at $(date '+%Y-%m-%d %H:%M:%S') ==="

source /users/$USER/miniconda3/etc/profile.d/conda.sh
conda activate verl

BASE="/capstor/scratch/cscs/$USER/verl-apertus"
VERL="/users/$USER/verl-vision"
EMU3_SRC="/users/$USER/Emu3.5/src"

cd "$BASE"
export PYTHONPATH="${BASE}:${VERL}:${EMU3_SRC}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH:-/iopsstor/scratch/cscs/$USER}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
unset SSL_CERT_FILE

# Clean stale torch extension caches (known GH200 issue from cof_sft.slurm)
rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true

APERTUS_CHECKPOINT="/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF"

mkdir -p logs

# --- Pre-flight checks ---
echo ""
echo "=== Pre-flight ==="
ERR=0
check() { if [ -e "$2" ]; then echo "  OK: $1"; else echo "  FAIL: $1 ($2)"; ERR=$((ERR+1)); fi; }
check "Apertus ckpt"          "$APERTUS_CHECKPOINT/config.json"
check "verl repo"             "$VERL/verl/__init__.py"
check "SFT train parquet"     "$BASE/data_prep/lens_search/sft_toolvqa_train.parquet"
check "SFT val parquet"       "$BASE/data_prep/lens_search/sft_toolvqa_val.parquet"
check "Custom dataset class"  "$BASE/data_prep/cof_sft_dataset.py"
check "SFT config"            "$BASE/configs/lens_search_sft.yaml"
python3 -c "import verl, transformers" 2>&1 | head -3
if [ $ERR -gt 0 ]; then echo "ERROR: $ERR pre-flight failed."; exit 1; fi

nvidia-smi --list-gpus

# --- Launch SFT (single-node) ---
echo ""
echo "=== Starting Arm B SFT Training ==="
T0=$(date +%s)
torchrun \
    --standalone \
    --nproc-per-node=4 \
    -m verl.trainer.sft_trainer \
    --config-path="$BASE/configs" \
    --config-name=lens_search_sft
T1=$(date +%s)
echo ""
echo "=== SFT elapsed: $((T1 - T0))s ==="

# --- Output summary ---
echo "=== Output ckpt dir ==="
ls -la "$BASE/checkpoints/apertus8b-lens-search-sft-toolvqa/" 2>&1 | head -20

echo ""
echo "=== Job ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
