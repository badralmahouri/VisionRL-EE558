#!/bin/bash
#SBATCH --job-name=lens_ood_eval
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=01:25:00
#SBATCH --environment=verl_env
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Run lens-search OOD inference (no training).
# REQUIRED env vars (set via `sbatch --export=ALL,...`):
#   CKPT_PATH                       HF-merged ckpt dir
#   BENCH                           one of aokvqa | textvqa | infoseek
#   GOOGLE_APPLICATION_CREDENTIALS  path to service-account JSON (for live mode)
#
# Optional:
#   TOOL_MODE         live (default) or cached
#   LIMIT             N rows (default = all)
#   GCP_CALL_LIMIT    hard cap on tool calls (default 1000 for full bench)
#   OUTPUT            override output JSONL path

set -euo pipefail
ulimit -n 65535
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
unset SSL_CERT_FILE

export TOOL_MODE="${TOOL_MODE:-live}"
export LENS_TOOL_MODE="$TOOL_MODE"
export GCP_CALL_LIMIT="${GCP_CALL_LIMIT:-1000}"
LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then LIMIT_ARG="--limit $LIMIT"; fi
OUTPUT_ARG=""
if [ -n "${OUTPUT:-}" ]; then OUTPUT_ARG="--output $OUTPUT"; fi

# --- Pre-flight ---
echo ""
echo "=== Pre-flight ==="
ERR=0
check() { if [ -e "$2" ]; then echo "  OK: $1"; else echo "  FAIL: $1 ($2)"; ERR=$((ERR+1)); fi; }
check "verl repo"            "$VERL/verl/__init__.py"
check "lens_search_tool.py"  "$BASE/tools/lens_search_tool.py"
check "run_lens_ood_eval.py" "$BASE/inference/run_lens_ood_eval.py"
check "Emu3.5 tokenizer src" "$EMU3_SRC/vision_tokenizer/__init__.py"
check "ckpt"                 "${CKPT_PATH:-MISSING}/config.json"
check "OOD manifest"         "$BASE/data_prep/lens_baseline/${BENCH:-MISSING}/manifest.json"

if [ "$TOOL_MODE" = "live" ]; then
    if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] || [ ! -f "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
        echo "  FAIL: GOOGLE_APPLICATION_CREDENTIALS unset or missing (live mode)"
        ERR=$((ERR+1))
    else
        echo "  OK: GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS"
    fi
fi

if [ $ERR -gt 0 ]; then echo "ERROR: $ERR pre-flight failed."; exit 1; fi

echo ""
echo "  CKPT_PATH=$CKPT_PATH"
echo "  BENCH=$BENCH"
echo "  TOOL_MODE=$TOOL_MODE"
echo "  GCP_CALL_LIMIT=$GCP_CALL_LIMIT"
echo "  LIMIT=${LIMIT:-<all>}"

nvidia-smi --list-gpus

# --- Run ---
echo ""
echo "=== Running OOD inference ==="
T0=$(date +%s)
python3 -u inference/run_lens_ood_eval.py \
    --ckpt "$CKPT_PATH" \
    --bench "$BENCH" \
    --tool-mode "$TOOL_MODE" \
    --gcp-call-limit "$GCP_CALL_LIMIT" \
    $LIMIT_ARG $OUTPUT_ARG
T1=$(date +%s)
echo ""
echo "=== Eval elapsed: $((T1 - T0))s ==="
echo "=== Job ended at $(date '+%Y-%m-%d %H:%M:%S') ==="
