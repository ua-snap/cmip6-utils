#!/bin/bash

# its recommended to the cmip6-utils environment and a high-memory node to run these notebooks (e.g. "analysis")
# for example:
#
# srun --partition=analysis --pty /bin/bash
# export BASE_DIR=/beegfs/CMIP6/jdpaul3/cmip6_downscaled_llm_fixes_12km_all/cmip6_12km_downscaling/
# cd /path/to/cmip6-utils/downscaling/notebooks
# conda activate cmip6-utils
# bash run_notebooks_for_all_models.sh

#GFDL-ESM4    historical, ssp126, ssp245, ssp370, ssp585	pr, tasmax, tasmin
papermill downscaled_pr.ipynb downscaled_pr_GFDL-ESM4.ipynb -p models 'GFDL-ESM4' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 400
papermill downscaled_tasmax.ipynb downscaled_tasmax_GFDL-ESM4.ipynb -p models 'GFDL-ESM4' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 35
papermill downscaled_tasmin.ipynb downscaled_tasmin_GFDL-ESM4.ipynb -p models 'GFDL-ESM4' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold -55
papermill downscaled_dtr.ipynb downscaled_dtr_GFDL-ESM4.ipynb -p models 'GFDL-ESM4' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 40

#CESM2	historical, ssp126, ssp585	pr
papermill downscaled_pr.ipynb downscaled_pr_CESM2.ipynb -p models 'CESM2' -p scenarios 'historical ssp126 ssp585' -p threshold 400

#CNRM-CM6-1-HR	historical, ssp126, ssp585	pr, tasmax, tasmin
papermill downscaled_pr.ipynb downscaled_pr_CNRM-CM6-1-HR.ipynb -p models 'CNRM-CM6-1-HR' -p scenarios 'historical ssp126 ssp585' -p threshold 400
papermill downscaled_tasmax.ipynb downscaled_tasmax_CNRM-CM6-1-HR.ipynb -p models 'CNRM-CM6-1-HR' -p scenarios 'historical ssp126 ssp585' -p threshold 35
papermill downscaled_tasmin.ipynb downscaled_tasmin_CNRM-CM6-1-HR.ipynb -p models 'CNRM-CM6-1-HR' -p scenarios 'historical ssp126 ssp585' -p threshold -55
papermill downscaled_dtr.ipynb downscaled_dtr_CNRM-CM6-1-HR.ipynb -p models 'CNRM-CM6-1-HR' -p scenarios 'historical ssp126 ssp585' -p threshold 40

# E3SM-2-0	historical, ssp370	pr, tasmax, tasmin
papermill downscaled_pr.ipynb downscaled_pr_E3SM-2-0.ipynb -p models 'E3SM-2-0' -p scenarios 'historical ssp370' -p threshold 400
papermill downscaled_tasmax.ipynb downscaled_tasmax_E3SM-2-0.ipynb -p models 'E3SM-2-0' -p scenarios 'historical ssp370' -p threshold 35
papermill downscaled_tasmin.ipynb downscaled_tasmin_E3SM-2-0.ipynb -p models 'E3SM-2-0' -p scenarios 'historical ssp370' -p threshold -55
papermill downscaled_dtr.ipynb downscaled_dtr_E3SM-2-0.ipynb -p models 'E3SM-2-0' -p scenarios 'historical ssp370' -p threshold 40

# EC-Earth3-Veg	historical, ssp126, ssp370, ssp585	pr, tasmax, tasmin	ssp126 scenario is only available for the pr variable.
papermill downscaled_pr.ipynb downscaled_pr_EC-Earth3-Veg.ipynb -p models 'EC-Earth3-Veg' -p scenarios 'historical ssp126 ssp370 ssp585' -p threshold 400
papermill downscaled_tasmax.ipynb downscaled_tasmax_EC-Earth3-Veg.ipynb -p models 'EC-Earth3-Veg' -p scenarios 'historical ssp370 ssp585' -p threshold 35
papermill downscaled_tasmin.ipynb downscaled_tasmin_EC-Earth3-Veg.ipynb -p models 'EC-Earth3-Veg' -p scenarios 'historical ssp370 ssp585' -p threshold -55
papermill downscaled_dtr.ipynb downscaled_dtr_EC-Earth3-Veg.ipynb -p models 'EC-Earth3-Veg' -p scenarios 'historical ssp370 ssp585' -p threshold 40

# HadGEM3-GC31-LL	historical, ssp126, ssp245, ssp585	pr, tasmax, tasmin
papermill downscaled_pr.ipynb downscaled_pr_HadGEM3-GC31-LL.ipynb -p models 'HadGEM3-GC31-LL' -p scenarios 'historical ssp126 ssp245 ssp585' -p threshold 400
papermill downscaled_tasmax.ipynb downscaled_tasmax_HadGEM3-GC31-LL.ipynb -p models 'HadGEM3-GC31-LL' -p scenarios 'historical ssp126 ssp245 ssp585' -p threshold 35
papermill downscaled_tasmin.ipynb downscaled_tasmin_HadGEM3-GC31-LL.ipynb -p models 'HadGEM3-GC31-LL' -p scenarios 'historical ssp126 ssp245 ssp585' -p threshold -55
papermill downscaled_dtr.ipynb downscaled_dtr_HadGEM3-GC31-LL.ipynb -p models 'HadGEM3-GC31-LL' -p scenarios 'historical ssp126 ssp245 ssp585' -p threshold 40

# HadGEM3-GC31-MM	historical, ssp126, ssp585	pr, tasmax, tasmin
papermill downscaled_pr.ipynb downscaled_pr_HadGEM3-GC31-MM.ipynb -p models 'HadGEM3-GC31-MM' -p scenarios 'historical ssp126 ssp585' -p threshold 400
papermill downscaled_tasmax.ipynb downscaled_tasmax_HadGEM3-GC31-MM.ipynb -p models 'HadGEM3-GC31-MM' -p scenarios 'historical ssp126 ssp585' -p threshold 35
papermill downscaled_tasmin.ipynb downscaled_tasmin_HadGEM3-GC31-MM.ipynb -p models 'HadGEM3-GC31-MM' -p scenarios 'historical ssp126 ssp585' -p threshold -55
papermill downscaled_dtr.ipynb downscaled_dtr_HadGEM3-GC31-MM.ipynb -p models 'HadGEM3-GC31-MM' -p scenarios 'historical ssp126 ssp585' -p threshold 40

# KACE-1-0-G	historical, ssp126, ssp245, ssp370, ssp585	pr, tasmax, tasmin
papermill downscaled_pr.ipynb downscaled_pr_KACE-1-0-G.ipynb -p models 'KACE-1-0-G' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 400
papermill downscaled_tasmax.ipynb downscaled_tasmax_KACE-1-0-G.ipynb -p models 'KACE-1-0-G' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 35
papermill downscaled_tasmin.ipynb downscaled_tasmin_KACE-1-0-G.ipynb -p models 'KACE-1-0-G' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold -55
papermill downscaled_dtr.ipynb downscaled_dtr_KACE-1-0-G.ipynb -p models 'KACE-1-0-G' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 40

# MIROC6	historical, ssp126, ssp245, ssp370, ssp585	pr, tasmax, tasmin
papermill downscaled_pr.ipynb downscaled_pr_MIROC6.ipynb -p models 'MIROC6' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 400
papermill downscaled_tasmax.ipynb downscaled_tasmax_MIROC6.ipynb -p models 'MIROC6' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 35
papermill downscaled_tasmin.ipynb downscaled_tasmin_MIROC6.ipynb -p models 'MIROC6' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold -55
papermill downscaled_dtr.ipynb downscaled_dtr_MIROC6.ipynb -p models 'MIROC6' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 40

# MPI-ESM1-2-HR	historical, ssp126, ssp245, ssp370, ssp585	pr, tasmax, tasmin
papermill downscaled_pr.ipynb downscaled_pr_MPI-ESM1-2-HR.ipynb -p models 'MPI-ESM1-2-HR' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 400
papermill downscaled_tasmax.ipynb downscaled_tasmax_MPI-ESM1-2-HR.ipynb -p models 'MPI-ESM1-2-HR' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 35
papermill downscaled_tasmin.ipynb downscaled_tasmin_MPI-ESM1-2-HR.ipynb -p models 'MPI-ESM1-2-HR' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold -55
papermill downscaled_dtr.ipynb downscaled_dtr_MPI-ESM1-2-HR.ipynb -p models 'MPI-ESM1-2-HR' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 40

# MRI-ESM2-0	historical, ssp126, ssp245, ssp370, ssp585	pr, tasmax, tasmin
papermill downscaled_pr.ipynb downscaled_pr_MRI-ESM2-0.ipynb -p models 'MRI-ESM2-0' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 400
papermill downscaled_tasmax.ipynb downscaled_tasmax_MRI-ESM2-0.ipynb -p models 'MRI-ESM2-0' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 35
papermill downscaled_tasmin.ipynb downscaled_tasmin_MRI-ESM2-0.ipynb -p models 'MRI-ESM2-0' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold -55
papermill downscaled_dtr.ipynb downscaled_dtr_MRI-ESM2-0.ipynb -p models 'MRI-ESM2-0' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 40

# NorESM2-MM	historical, ssp126, ssp245, ssp370, ssp585	pr, tasmax, tasmin
papermill downscaled_pr.ipynb downscaled_pr_NorESM2-MM.ipynb -p models 'NorESM2-MM' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 400
papermill downscaled_tasmax.ipynb downscaled_tasmax_NorESM2-MM.ipynb -p models 'NorESM2-MM' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 35
papermill downscaled_tasmin.ipynb downscaled_tasmin_NorESM2-MM.ipynb -p models 'NorESM2-MM' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold -55
papermill downscaled_dtr.ipynb downscaled_dtr_NorESM2-MM.ipynb -p models 'NorESM2-MM' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 40

# TaiESM1	historical, ssp126, ssp245, ssp370, ssp585	pr, tasmax, tasmin
papermill downscaled_pr.ipynb downscaled_pr_TaiESM1.ipynb -p models 'TaiESM1' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 400
papermill downscaled_tasmax.ipynb downscaled_tasmax_TaiESM1.ipynb -p models 'TaiESM1' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 35
papermill downscaled_tasmin.ipynb downscaled_tasmin_TaiESM1.ipynb -p models 'TaiESM1' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold -55
papermill downscaled_dtr.ipynb downscaled_dtr_TaiESM1.ipynb -p models 'TaiESM1' -p scenarios 'historical ssp126 ssp245 ssp370 ssp585' -p threshold 40



