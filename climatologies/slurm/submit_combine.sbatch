#!/bin/sh
#SBATCH --job-name=climatology_combine
#SBATCH --partition=analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/CMIP6/jdpaul3/climatologies/logs/slurm/combine_%j.out

set -e
echo Start slurm && date
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate cmip6-utils

python /import/home/jdpaul3/cmip6-utils/climatologies/combine.py --config /import/home/jdpaul3/cmip6-utils/climatologies/config.yaml
echo End slurm && date
