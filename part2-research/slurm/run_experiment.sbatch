#!/bin/bash
#SBATCH --job-name=manifold
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# There is no --account line on purpose. Pass it when you submit (see submit.sh), so a run is never
# charged to an allocation by accident.
set -euo pipefail

CONFIG=${1:?usage: sbatch --account=<account> run_experiment.sbatch <config.yaml> [out.jsonl]}
OUT=${2:-results/$(basename "$CONFIG" .yaml)-${SLURM_JOB_ID}.jsonl}

cd "${SLURM_SUBMIT_DIR:-.}"
# Activate your Python environment however you do it, e.g. export MANIFOLD_ENV_ACTIVATE="source ~/envs/manifold/bin/activate"
[ -n "${MANIFOLD_ENV_ACTIVATE:-}" ] && eval "$MANIFOLD_ENV_ACTIVATE"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
python -m manifoldsearch.run "$CONFIG" --out "$OUT" --device cuda
