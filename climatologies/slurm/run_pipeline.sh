#!/bin/bash
# Build the job list, (re)generate the sbatch scripts from the given
# config, and submit the fragments array job followed by the combine job
# (chained with --dependency=afterok so combine only runs once every
# fragment succeeds).
#
# Usage: run_pipeline.sh --config config_12km.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate cmip6-utils

python build_job_list.py "$@"
python slurm/generate_sbatch.py "$@"

frag_job_id=$(sbatch --parsable slurm/submit_fragments.sbatch)
echo "submitted fragments array job: $frag_job_id"

combine_job_id=$(sbatch --parsable --dependency=afterok:"$frag_job_id" slurm/submit_combine.sbatch)
echo "submitted combine job: $combine_job_id (after $frag_job_id)"
