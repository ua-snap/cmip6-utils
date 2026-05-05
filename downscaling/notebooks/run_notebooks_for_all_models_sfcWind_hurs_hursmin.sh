#!/bin/bash

# its recommended to use the cmip6-utils environment and a high-memory node to run these notebooks (e.g. "analysis")
# for example:
#
# srun --partition=analysis --pty /bin/bash
# export BASE_DIR=/beegfs/CMIP6/jdpaul3/cmip6_downscaled_llm_fixes_12km_new_vars/sfcWind_hurs_hursmin/
# cd /path/to/cmip6-utils/downscaling/notebooks
# conda activate cmip6-utils
# bash run_notebooks_for_all_models_sfcWind_hurs_hursmin.sh

# CESM2	historical, ssp126, ssp245, ssp370, ssp585	sfcWind, hurs
papermill downscaled_sfcWind.ipynb downscaled_sfcWind_CESM2.ipynb -p models 'CESM2' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 31.3
papermill downscaled_hurs.ipynb downscaled_hurs_CESM2.ipynb -p models 'CESM2' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'

# CNRM-CM6-1-HR	sfcWind: historical; hurs/hursmin: historical, ssp126, ssp245, ssp370, ssp585
papermill downscaled_sfcWind.ipynb downscaled_sfcWind_CNRM-CM6-1-HR.ipynb -p models 'CNRM-CM6-1-HR' -p scenarios 'historical' -p threshold 31.3
papermill downscaled_hurs.ipynb downscaled_hurs_CNRM-CM6-1-HR.ipynb -p models 'CNRM-CM6-1-HR' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'
papermill downscaled_hursmin.ipynb downscaled_hursmin_CNRM-CM6-1-HR.ipynb -p models 'CNRM-CM6-1-HR' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'

# EC-Earth3-Veg	sfcWind: historical; hurs: historical, ssp126, ssp370, ssp585; hursmin: historical, ssp126, ssp245, ssp370, ssp585
papermill downscaled_sfcWind.ipynb downscaled_sfcWind_EC-Earth3-Veg.ipynb -p models 'EC-Earth3-Veg' -p scenarios 'historical' -p threshold 31.3
papermill downscaled_hurs.ipynb downscaled_hurs_EC-Earth3-Veg.ipynb -p models 'EC-Earth3-Veg' -p scenarios 'historical ssp126 ssp370 ssp585'
papermill downscaled_hursmin.ipynb downscaled_hursmin_EC-Earth3-Veg.ipynb -p models 'EC-Earth3-Veg' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'

# GFDL-ESM4	historical, ssp126, ssp245, ssp370, ssp585	sfcWind, hurs
papermill downscaled_sfcWind.ipynb downscaled_sfcWind_GFDL-ESM4.ipynb -p models 'GFDL-ESM4' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 31.3
papermill downscaled_hurs.ipynb downscaled_hurs_GFDL-ESM4.ipynb -p models 'GFDL-ESM4' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'

# HadGEM3-GC31-LL	sfcWind/hurs: historical, ssp126, ssp245, ssp585
papermill downscaled_sfcWind.ipynb downscaled_sfcWind_HadGEM3-GC31-LL.ipynb -p models 'HadGEM3-GC31-LL' -p scenarios 'historical ssp126 ssp245 ssp585' -p threshold 31.3
papermill downscaled_hurs.ipynb downscaled_hurs_HadGEM3-GC31-LL.ipynb -p models 'HadGEM3-GC31-LL' -p scenarios 'historical ssp126 ssp245 ssp585'

# HadGEM3-GC31-MM	sfcWind/hurs: historical, ssp126, ssp585
papermill downscaled_sfcWind.ipynb downscaled_sfcWind_HadGEM3-GC31-MM.ipynb -p models 'HadGEM3-GC31-MM' -p scenarios 'historical ssp126 ssp585' -p threshold 31.3
papermill downscaled_hurs.ipynb downscaled_hurs_HadGEM3-GC31-MM.ipynb -p models 'HadGEM3-GC31-MM' -p scenarios 'historical ssp126 ssp585'

# KACE-1-0-G	sfcWind/hurs: historical, ssp126, ssp245, ssp370, ssp585; hursmin: historical, ssp585
papermill downscaled_sfcWind.ipynb downscaled_sfcWind_KACE-1-0-G.ipynb -p models 'KACE-1-0-G' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 31.3
papermill downscaled_hurs.ipynb downscaled_hurs_KACE-1-0-G.ipynb -p models 'KACE-1-0-G' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'
papermill downscaled_hursmin.ipynb downscaled_hursmin_KACE-1-0-G.ipynb -p models 'KACE-1-0-G' -p scenarios 'historical ssp585'

# MIROC6	historical, ssp126, ssp245, ssp370, ssp585	sfcWind, hurs, hursmin
papermill downscaled_sfcWind.ipynb downscaled_sfcWind_MIROC6.ipynb -p models 'MIROC6' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 31.3
papermill downscaled_hurs.ipynb downscaled_hurs_MIROC6.ipynb -p models 'MIROC6' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'
papermill downscaled_hursmin.ipynb downscaled_hursmin_MIROC6.ipynb -p models 'MIROC6' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'

# MPI-ESM1-2-HR	historical, ssp126, ssp245, ssp370, ssp585	sfcWind, hurs, hursmin
papermill downscaled_sfcWind.ipynb downscaled_sfcWind_MPI-ESM1-2-HR.ipynb -p models 'MPI-ESM1-2-HR' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 31.3
papermill downscaled_hurs.ipynb downscaled_hurs_MPI-ESM1-2-HR.ipynb -p models 'MPI-ESM1-2-HR' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'
papermill downscaled_hursmin.ipynb downscaled_hursmin_MPI-ESM1-2-HR.ipynb -p models 'MPI-ESM1-2-HR' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'

# MRI-ESM2-0	historical, ssp126, ssp245, ssp370, ssp585	sfcWind, hurs, hursmin
papermill downscaled_sfcWind.ipynb downscaled_sfcWind_MRI-ESM2-0.ipynb -p models 'MRI-ESM2-0' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 31.3
papermill downscaled_hurs.ipynb downscaled_hurs_MRI-ESM2-0.ipynb -p models 'MRI-ESM2-0' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'
papermill downscaled_hursmin.ipynb downscaled_hursmin_MRI-ESM2-0.ipynb -p models 'MRI-ESM2-0' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'

# NorESM2-MM	historical, ssp126, ssp245, ssp370, ssp585	sfcWind, hurs
papermill downscaled_sfcWind.ipynb downscaled_sfcWind_NorESM2-MM.ipynb -p models 'NorESM2-MM' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 31.3
papermill downscaled_hurs.ipynb downscaled_hurs_NorESM2-MM.ipynb -p models 'NorESM2-MM' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'

# TaiESM1	historical, ssp126, ssp245, ssp370, ssp585	sfcWind, hurs
papermill downscaled_sfcWind.ipynb downscaled_sfcWind_TaiESM1.ipynb -p models 'TaiESM1' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 31.3
papermill downscaled_hurs.ipynb downscaled_hurs_TaiESM1.ipynb -p models 'TaiESM1' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585'
