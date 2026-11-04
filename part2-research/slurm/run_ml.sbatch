#!/bin/bash
#SBATCH --job-name=manifold-ml
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Generic launcher for the machine-learning experiments: runs `python -m <module> <args...>` in the cluster env.
# No --account line on purpose: submit with  sbatch --account=<account> --job-name=manifold-<what> slurm/run_ml.sbatch <module> <args...>
# Needs MANIFOLD_ENV_ACTIVATE exported, e.g. "module load python/3.13.5; source /scratch/<you>/manifold/env/bin/activate".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[ -n "${MANIFOLD_ENV_ACTIVATE:-}" ] && eval "$MANIFOLD_ENV_ACTIVATE"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python -m "$@"
