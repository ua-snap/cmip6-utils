"""Configuration for CMIP6 regridding"""

import os
from pathlib import Path


# path-based required env vars will throw error if None
# path to root of this repo, for constructing absolute paths to scripts
PROJECT_DIR = (
    Path(os.getenv("PROJECT_DIR"))
    if "PROJECT_DIR" in os.environ
    else Path("/home/rltorgerson/cmip6-utils/")
)
SCRATCH_DIR = (
    Path(os.getenv("SCRATCH_DIR"))
    if "SCRATCH_DIR" in os.environ
    else Path("/beegfs/CMIP6/rltorgerson/indicators/")
)
conda_init_script = (
    Path(os.getenv("CONDA_INIT"))
    if "CONDA_INIT" in os.environ
    else PROJECT_DIR.joinpath("indicators/conda_init.sh")
)
try:
    slurm_email = Path(os.getenv("SLURM_EMAIL"))
except TypeError:
    slurm_email = None

# this will probably not change between users
cmip6_dir = Path("/beegfs/CMIP6/arctic-cmip6/")
regrid_dir = Path("/beegfs/CMIP6/arctic-cmip6/regrid/")
slurm_dir = SCRATCH_DIR.parent.joinpath("slurm")

# path to script for regridding
indicators_script = PROJECT_DIR.joinpath("indicators/indicators.py")

indicator_tmp_fp = "{indicator}_{model}_{scenario}_indicator.nc"

sbatch_head_kwargs = {
    # slurm info (doesn't need to be hardcoded, but OK for now?)
    "partition": "t2small",
    "ncpus": 24,
    "conda_init_script": conda_init_script,
    "slurm_email": slurm_email,
}
