#!/bin/bash
#SBATCH --job-name=voting
#SBATCH --partition=cpu
#SBATCH --account=pi_jensen_umass_edu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=scratch/exercises/logs/voting_%j.out
#SBATCH --error=scratch/exercises/logs/voting_%j.err

set -euo pipefail

PROJECT=/work/pi_jensen_umass_edu/svaidyanatha_umass_edu/hydra
source $PROJECT/.venv/bin/activate

python -u $PROJECT/exercises/voting.py

# Verify assertions passed
if ! grep -q "All assertions passed" scratch/exercises/logs/voting_${SLURM_JOB_ID}.out 2>/dev/null; then
    echo "FAIL: assertions did not pass" >&2
    exit 1
fi
echo "OK: voting.py assertions passed"
