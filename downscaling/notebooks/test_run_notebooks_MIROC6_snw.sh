#!/bin/bash

# its recommended to use the cmip6-utils environment and a high-memory node to run these notebooks (e.g. "analysis")
# for example:
#
# srun --partition=analysis --pty /bin/bash
# export BASE_DIR=/beegfs/CMIP6/jdpaul3/cmip6_downscaled_llm_fixes_12km_new_vars/snw/
# cd /path/to/cmip6-utils/downscaling/notebooks
# conda activate cmip6-utils
# bash test_run_notebooks_MIROC6_snw.sh

# MIROC6	historical, ssp126, ssp245, ssp370, ssp585
papermill downscaled_snw.ipynb downscaled_snw_MIROC6.ipynb -p models 'MIROC6' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 2540
