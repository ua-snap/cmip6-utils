"""Configuration for CMIP6 regridding"""

import os
from pathlib import Path


# path-based required env vars will throw error if None
# path to root of this repo, for constructing absolute paths to scripts
PROJECT_DIR = Path(os.getenv("PROJECT_DIR"))
SCRATCH_DIR = Path(os.getenv("SCRATCH_DIR"))
conda_init_script = Path(os.getenv("CONDA_INIT"))
try:
    slurm_email = Path(os.getenv("SLURM_EMAIL"))
except TypeError:
    slurm_email = None

# path to script for regridding
regrid_script = PROJECT_DIR.joinpath("regridding/regrid.py")

# slurm info (doesn't need to be hardcoded, but OK for now?)
partition = "t1small"
ncpus = 24

# this will probably not change between users
cmip6_dir = Path("/beegfs/CMIP6/arctic-cmip6/CMIP6")

# manifest file
manifest_fp = PROJECT_DIR.joinpath("transfers/llnl_manifest.csv")

# directory to write re-gridded files
regrid_dir = Path("/beegfs/CMIP6/arctic-cmip6/regrid")
regrid_dir.mkdir(exist_ok=True)

# paths to text files containing paths to regrid
regrid_batch_dir = SCRATCH_DIR.joinpath("regrid_batch")
regrid_batch_dir.mkdir(exist_ok=True)
# template name for batch files
batch_tmp_fn = "batch_{model}_{scenario}.txt"

# target regridding file - all files will be regridded to the grid in this file
target_grid_fp = cmip6_dir.joinpath("ScenarioMIP/NCAR/CESM2/ssp370/r11i1p1f1/Amon/tas/gn/v20200528/tas_Amon_CESM2_ssp370_r11i1p1f1_gn_206501-210012.nc")

# institution model strings (<institution>_<model>, from mirrored data) that we will be regridding
inst_models = [
    'NOAA-GFDL_GFDL-ESM4',
    'NIMS-KMA_KACE-1-0-G',
    'CNRM-CERFACS_CNRM-CM6-1-HR',
    'NCC_NorESM2-MM',
    'AS-RCEC_TaiESM1',
    'MOHC_HadGEM3-GC31-MM',
    'MOHC_HadGEM3-GC31-LL',
    'MIROC_MIROC6',
    'EC-Earth-Consortium_EC-Earth3-Veg',
    'NCAR_CESM2',
    'NCAR_CESM2-WACCM',
    'MPI-M_MPI-ESM1-2-LR'
]
