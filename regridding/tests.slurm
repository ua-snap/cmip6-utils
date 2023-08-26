#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=username@alaska.edu
#SBATCH --output=/path/to/regridding_tests_slurm_%j.out
#SBATCH -p t1small

echo Start slurm && date

source /path/to/conda.sh
conda activate cmip6-utils

export SCRATCH_DIR=/path/to/cmip6_regridding
export EXAMPLE_REGRID_FILE=/path/to/example_regrid_file.nc

python -m pytest -v tests/test_regridding.py

echo End slurm && date
