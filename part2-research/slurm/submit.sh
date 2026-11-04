#!/bin/bash
# Submit one experiment to the cluster's GPU partition, or just show what would be submitted.
#
#   ACCOUNT=<slurm account> ./slurm/submit.sh experiments/configs/scaling.yaml          # prints the command only
#   ACCOUNT=<slurm account> ./slurm/submit.sh experiments/configs/scaling.yaml --go     # submits it
#
# ACCOUNT is required and has no default: check with your supervisor or course staff that the allocation may
# be used for this work before you set it. Run from part2-research/ (it creates logs/ and results/).
set -euo pipefail
CONFIG=${1:?usage: ACCOUNT=<account> submit.sh <config.yaml> [--go]}
: "${ACCOUNT:?set ACCOUNT to the slurm account this run may be charged to}"
mkdir -p logs results
CMD=(sbatch --account="$ACCOUNT" slurm/run_experiment.sbatch "$CONFIG")
if [ "${2:-}" = "--go" ]; then "${CMD[@]}"; else echo "dry run (add --go to submit):"; printf ' %q' "${CMD[@]}"; echo; fi
