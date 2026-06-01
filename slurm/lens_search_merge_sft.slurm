#!/bin/bash
#SBATCH --job-name=lens_merge_sft
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=0:30:00
#SBATCH --environment=verl_env
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Merge the FSDP-sharded SFT checkpoint into a single HF-loadable directory
# so Arm B's GRPO can `AutoModelForCausalLM.from_pretrained(...)` it.
#
# Default source: latest global_step_* dir under apertus8b-lens-search-sft/
# Override via env var SFT_SRC_DIR.
# Default target: <source>_hf

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
export PYTHONUNBUFFERED=1
unset SSL_CERT_FILE

# Auto-discover latest global_step_* if SFT_SRC_DIR not set
if [ -z "${SFT_SRC_DIR:-}" ]; then
    CKPT_DIR="$BASE/checkpoints/apertus8b-lens-search-sft"
    LATEST=$(ls -1d "$CKPT_DIR"/global_step_* 2>/dev/null | grep -v '_hf$' | sort -t_ -k3 -n | tail -1)
    if [ -z "$LATEST" ]; then
        echo "ERROR: no global_step_* dir found under $CKPT_DIR"
        exit 1
    fi
    SFT_SRC_DIR="$LATEST"
fi
TARGET_DIR="${SFT_TARGET_DIR:-${SFT_SRC_DIR}_hf}"

echo "  SFT_SRC_DIR = $SFT_SRC_DIR"
echo "  TARGET_DIR  = $TARGET_DIR"

# Pre-flight
ERR=0
check() { if [ -e "$2" ]; then echo "  OK: $1"; else echo "  FAIL: $1 ($2)"; ERR=$((ERR+1)); fi; }
check "patched merger script" "$BASE/scripts/merge_lens_sft_fsdp_to_hf.py"
check "FSDP shards"           "$SFT_SRC_DIR/model_world_size_4_rank_0.pt"
check "HF config dir"         "$SFT_SRC_DIR/huggingface/config.json"
[ $ERR -gt 0 ] && { echo "ERROR: $ERR pre-flight failed."; exit 1; }

nvidia-smi --list-gpus

# Run the merger via our patched wrapper (handles replicated/zero-dim tensors;
# upstream verl FSDPModelMerger crashes on those, see scripts/merge_lens_sft_fsdp_to_hf.py).
echo ""
echo "=== Running FSDP -> HF merge (patched) ==="
T0=$(date +%s)
python3 "$BASE/scripts/merge_lens_sft_fsdp_to_hf.py" \
    --local_dir "$SFT_SRC_DIR" \
    --target_dir "$TARGET_DIR"
T1=$(date +%s)
echo ""
echo "=== Merge elapsed: $((T1 - T0))s ==="

# Sanity-check output
echo ""
echo "=== Output ==="
ls -la "$TARGET_DIR" 2>&1 | head -30
echo ""
SAFETENSORS_COUNT=$(ls "$TARGET_DIR"/*.safetensors 2>/dev/null | wc -l)
echo "  safetensors files: $SAFETENSORS_COUNT"

if [ "$SAFETENSORS_COUNT" -eq 0 ]; then
    echo "ERROR: no .safetensors files in $TARGET_DIR — merger may have silently failed"
    exit 1
fi

echo ""
echo "=== Job ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
