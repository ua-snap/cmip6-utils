#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=username@alaska.edu
#SBATCH --output=path/to/mirror_tests_slurm_%j.out
#SBATCH -p t1small

echo Start slurm && date

source /path/to/conda_init.sh
conda activate cmip6-utils

python -m pytest -v tests/test_mirror.py

echo End slurm && date
