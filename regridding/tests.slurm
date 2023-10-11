#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$SLURM_EMAIL
#SBATCH --output=regridding_tests_slurm_%j.out
#SBATCH -p t1small

echo Start slurm && date

source $CONDA_INIT
conda activate cmip6-utils

echo Running regridding tests
export EXAMPLE_REGRID_FILE=/beegfs/CMIP6/arctic-cmip6/CMIP6/ScenarioMIP/NCAR/CESM2/ssp370/r11i1p1f1/Amon/tas/gn/v20200528/tas_Amon_CESM2_ssp370_r11i1p1f1_gn_206501-210012.nc
export PYTHONWARNINGS="ignore"

python -m pytest -v tests/test_regridding.py

echo End slurm && date
