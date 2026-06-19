#!/bin/sh
#SBATCH --job-name=climatology_fragments
#SBATCH --partition=analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --array=0-426%12
#SBATCH --output=/beegfs/CMIP6/jdpaul3/climatologies/logs/slurm/fragment_%A_%a.out

echo Start slurm && date
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate cmip6-utils

python /import/home/jdpaul3/cmip6-utils/climatologies/compute_fragment.py --job-index $SLURM_ARRAY_TASK_ID --config /import/home/jdpaul3/cmip6-utils/climatologies/config.yaml
echo End slurm && date
