#!/bin/bash
#SBATCH --job-name=qwen_judge
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=8:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Persistent Qwen3-32B sglang judge server for v3 reward.
#
# Writes its discoverable URL to $BASE/slurm/judge_url.txt BEFORE launching
# the server. Downstream jobs (visual_math_rl_train.slurm) read that file and
# block on /health via inference.llm_judge_math.wait_for_server.
#
# Launch pattern cloned from slurm/lens_probe.slurm phase 3 (verified working
# with Qwen3-32B, tp=4, mem-fraction-static=0.85).
#
# Stays up for the SLURM time limit OR until scancel'd. Submit once per
# v3-experiment session; reusable across multiple RL runs.

set -euo pipefail
echo "=== Judge job started at $(date '+%Y-%m-%d %H:%M:%S') ==="

source /users/$USER/miniconda3/etc/profile.d/conda.sh
conda activate verl

BASE="/capstor/scratch/cscs/$USER/verl-apertus"
cd "$BASE"
mkdir -p "$BASE/logs"

export PYTHONUNBUFFERED=1
export HF_HOME="${SCRATCH:-/iopsstor/scratch/cscs/$USER}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
unset SSL_CERT_FILE

JUDGE_PORT=30100
JUDGE_HOST=$(hostname -f)
JUDGE_URL="http://${JUDGE_HOST}:${JUDGE_PORT}/v1"
URL_FILE="$BASE/slurm/judge_url.txt"

# Quick sanity: verify imports BEFORE writing URL (avoid stale URL on crash)
python3 -c "import sglang; print(f'  sglang {sglang.__version__}')"

# Write URL file with metadata; truncate-replace so dependent jobs always read
# the URL for THIS judge instance.
{
  echo "$JUDGE_URL"
  echo "# job_id=${SLURM_JOB_ID}  host=${JUDGE_HOST}  written=$(date -Iseconds)"
} > "$URL_FILE"
echo "=== Judge URL: $JUDGE_URL"
echo "=== URL written to: $URL_FILE"
nvidia-smi --list-gpus

cleanup() {
  echo "=== judge SLURM ending; sglang process tree will exit with shell ==="
  # Clear URL file so any future RL job sees the missing-server state plainly.
  rm -f "$URL_FILE" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Launching sglang Qwen3-32B server (tp=4, port=$JUDGE_PORT)"
# Foreground sglang — SLURM holds the allocation as long as this process runs.
# Boot takes ~15 min (Qwen3-32B ~64 GB bf16 sharded across 4 GPUs).
# All stdout/stderr captured by #SBATCH --output / --error.
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B \
    --tp 4 \
    --host 0.0.0.0 \
    --port $JUDGE_PORT \
    --mem-fraction-static 0.85 \
    --trust-remote-code
