#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$SLURM_EMAIL
#SBATCH --output=$TEST_OUT_DIR/mirror_tests_slurm_%j.out
#SBATCH -p t1small

echo Start slurm && date

source $CONDA_INIT
conda activate cmip6-utils

python -m pytest -v tests/test_mirror.py

echo End slurm && date
