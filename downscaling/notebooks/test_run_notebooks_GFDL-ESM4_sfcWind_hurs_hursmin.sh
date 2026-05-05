#!/bin/bash

# its recommended to use the cmip6-utils environment and a high-memory node to run these notebooks (e.g. "analysis")
# for example:
#
# srun --partition=analysis --pty /bin/bash
# export BASE_DIR=/beegfs/CMIP6/jdpaul3/cmip6_downscaled_llm_fixes_12km_new_vars/sfcWind_hurs_hursmin/
# cd /path/to/cmip6-utils/downscaling/notebooks
# conda activate cmip6-utils
# bash test_run_notebooks_GFDL-ESM4_sfcWind_hurs_hursmin.sh

# GFDL-ESM4	historical, ssp126, ssp245, ssp370, ssp585	sfcWind, hurs, hursmin
papermill downscaled_sfcWind.ipynb downscaled_sfcWind_GFDL-ESM4.ipynb -p models 'GFDL-ESM4' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 31.3
papermill downscaled_hurs.ipynb downscaled_hurs_GFDL-ESM4.ipynb -p models 'GFDL-ESM4' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'
papermill downscaled_hursmin.ipynb downscaled_hursmin_GFDL-ESM4.ipynb -p models 'GFDL-ESM4' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'
