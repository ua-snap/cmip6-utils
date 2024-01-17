#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$SLURM_EMAIL
#SBATCH --output=time_dim_test_%j.out
#SBATCH -p t1small

echo Start slurm && date

source $CONDA_INIT
conda activate cmip6-utils

echo Checking for time dimension
export PYTHONWARNINGS="ignore"

#create csv for test results; will overwrite any csv from last test
#this filepath must match the time_dim_error_file variable from config
> $PROJECT_DIR/regridding/time_dim_test.csv

#run the test to populate the csv
python $PROJECT_DIR/regridding/time_dim_test.py

#print a message about number of results in the CSV
echo Number of files lacking time dimension: $(wc -l < $PROJECT_DIR/regridding/time_dim_test.csv)
echo See time_dim_test.csv for filepaths

echo Time dimension test completed

echo End slurm && date