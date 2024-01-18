"""Configuration for CMIP6 regridding"""

import os
from pathlib import Path


# path-based required env vars will throw error if None
# path to root of this repo, for constructing absolute paths to scripts
# PROJECT_DIR = Path(os.getenv("PROJECT_DIR"))
# SCRATCH_DIR = Path(os.getenv("SCRATCH_DIR"))
conda_init_script = Path(os.getenv("CONDA_INIT"))
try:
    slurm_email = Path(os.getenv("SLURM_EMAIL"))
except TypeError:
    slurm_email = None

# this will probably not change between users
cmip6_dir = Path("/beegfs/CMIP6/arctic-cmip6/CMIP6")

indicator_tmp_fp = "{indicator}_{model}_{scenario}_indicator.nc"
