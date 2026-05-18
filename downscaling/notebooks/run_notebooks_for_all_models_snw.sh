#!/bin/bash

# its recommended to use the cmip6-utils environment and a high-memory node to run these notebooks (e.g. "analysis")
# for example:
#
# srun --partition=analysis --mem=250G --pty /bin/bash
# export BASE_DIR=/beegfs/CMIP6/jdpaul3/cmip6_downscaled_llm_fixes_12km_new_vars/snw/
# cd /path/to/cmip6-utils/downscaling/notebooks
# conda activate cmip6-utils
# bash run_notebooks_for_all_models_snw.sh

# CESM2	historical, ssp126, ssp245, ssp370, ssp585
papermill downscaled_snw.ipynb downscaled_snw_CESM2.ipynb -p models 'CESM2' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 3000

# GFDL-ESM4	historical, ssp126, ssp245, ssp370, ssp585
papermill downscaled_snw.ipynb downscaled_snw_GFDL-ESM4.ipynb -p models 'GFDL-ESM4' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 3000

# HadGEM3-GC31-LL	historical, ssp245
papermill downscaled_snw.ipynb downscaled_snw_HadGEM3-GC31-LL.ipynb -p models 'HadGEM3-GC31-LL' -p scenarios 'historical ssp245' -p threshold 3000

# MIROC6	historical, ssp126, ssp245, ssp370, ssp585
papermill downscaled_snw.ipynb downscaled_snw_MIROC6.ipynb -p models 'MIROC6' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 3000

# MPI-ESM1-2-HR	historical, ssp126, ssp245, ssp370, ssp585
papermill downscaled_snw.ipynb downscaled_snw_MPI-ESM1-2-HR.ipynb -p models 'MPI-ESM1-2-HR' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 3000

# MRI-ESM2-0	historical, ssp126, ssp245, ssp370, ssp585
papermill downscaled_snw.ipynb downscaled_snw_MRI-ESM2-0.ipynb -p models 'MRI-ESM2-0' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 3000
