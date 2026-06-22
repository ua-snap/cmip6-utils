#!/bin/sh
#SBATCH --job-name=climatology_qc
#SBATCH --partition=analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/beegfs/CMIP6/jdpaul3/climatologies/logs/slurm/qc_%j.out

set -e
echo Start slurm && date
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate cmip6-utils

python /import/home/jdpaul3/cmip6-utils/climatologies/qc.py --config /import/home/jdpaul3/cmip6-utils/climatologies/config.yaml
echo End slurm && date
