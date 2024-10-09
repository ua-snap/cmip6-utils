#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition t2small
#SBATCH --job-name=resample_era5
#SBATCH --output=resample_era5.out 
#SBATCH --exclusive

echo Start slurm && date
source ../regridding/conda_init.sh
conda activate cil-xclim
time python resample_era5.py --era5_dir /beegfs/CMIP6/wrf_era5/04km --output_dir /beegfs/CMIP6/wrf_era5/4km_daily --no_clobber
