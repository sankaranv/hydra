#!/bin/bash
#SBATCH --job-name=obcb
#SBATCH --partition=cpu
#SBATCH --account=pi_jensen_umass_edu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=scratch/exercises/logs/obcb_%j.out
#SBATCH --error=scratch/exercises/logs/obcb_%j.err

set -euo pipefail

PROJECT=/work/pi_jensen_umass_edu/svaidyanatha_umass_edu/hydra
source $PROJECT/.venv/bin/activate

python -u $PROJECT/exercises/obcb.py

if ! grep -q "All assertions passed" scratch/exercises/logs/obcb_${SLURM_JOB_ID}.out 2>/dev/null; then
    echo "FAIL: assertions did not pass" >&2
    exit 1
fi
echo "OK: obcb.py assertions passed"
