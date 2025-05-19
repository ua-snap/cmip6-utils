"""Configuration for CMIP6 regridding"""

from pathlib import Path

slurm_email = "uaf-snap-sys-team@alaska.edu"
cmip6_dir = Path("/beegfs/CMIP6/arctic-cmip6/")
regrid_dir = Path("/beegfs/CMIP6/arctic-cmip6/regrid/")
indicator_tmp_fp = "{indicator}_{model}_{scenario}_indicator.nc"
