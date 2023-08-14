#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=username@alaska.edu
#SBATCH --output=/path/to/get_min_max_slurm_%j.out
#SBATCH -p t1small

echo Start slurm && date

source /path/to/conda.sh
conda activate cmip6-utils
export PYTHONWARNINGS="ignore"

# Get the absolute min/max values for each variable across all files.
for var in evspsbl huss pr prsn psl rlds rsds ta tas ua va
do
  timeout 20m /path/to/cmip6-utils/get_min_max.py -v $var
  while [ $? == 124 ]; do
    timeout 20m /path/to/cmip6-utils/get_min_max.py -v $var
    echo ""
  done
done

echo End slurm && date
