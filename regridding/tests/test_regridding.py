"""Test the existence of, and values within, regridded files.

Usage:

# All lat/lon grid coordinates will be compared against the target regrid file in config.py

cd regridding
python -m pytest tests/test_regridding.py
"""

import json
import pytest
import numpy as np
import os
import xarray as xr
from multiprocessing import Pool
from pathlib import Path
from config import target_grid_fp, regrid_dir, variables
from regrid import open_and_crop_dataset, prod_lat_slice


min_max_buffer_percent = 0.001

# The lists of models, scenarios, variables, and intervals have been copied from
# the CMIP6 scope document

models = [
    "ACCESS-CM2",
    "CESM2",
    "CNRM-CM6-1-HR",
    "EC-Earth3-Veg",
    "GFDL-ESM4",
    "HadGEM3-GC31-LL",
    "HadGEM3-GC31-MM",
    "KACE-1-0-G",
    "MIROC6",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NorESM2-MM",
    "E3SM",
]

scenarios = [
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585",
]

# Missing variables have been commented out for now.
# TODO: Uncomment these variables once they are regridded.


intervals = [
    "Amon",
    "day",
]

# TODO: Need to add tests for "Time-independent 2-d land surface data" ("orog",
# "sftlf", "sftof") also once the expected directory structure is known.


def validate_grid(args):
    regrid_fp, target_lat_arr, target_lon_arr = args
    try:
        regrid_ds = xr.open_dataset(regrid_fp)
        assert np.all(regrid_ds["lat"].values == target_lat_arr)
        assert np.all(regrid_ds["lon"].values == target_lon_arr)
        assert regrid_ds["time"].encoding["calendar"] == "noleap"
    except:
        assert False
    assert True


def generate_min_max_range(min_max_fp):
    """Generate the minimum and maximum ranges from the files in min_max_dir"""
    min_max_range = {}
    var_id = min_max_fp.name.split(".json")[0]

    with open(min_max_fp, "r") as infile:
        min_max_vals = json.load(infile)

    min_vals = []
    max_vals = []
    for k, v in min_max_vals.items():
        min_vals.append(float(v["min"]))
        max_vals.append(float(v["max"]))

    min_max_range[var_id] = [min(min_vals), max(max_vals)]

    return min_max_range


def validate_min_max_nan(args):
    fp, var_id, min_max_range = args

    if var_id in [
        "hus",
        "ua",
        "va",
        "ta",
    ]:
        nan_threshold = 0.2
    elif var_id in [
        "prsn",
        "snd",
    ]:
        nan_threshold = 0.6
    elif var_id in ["tos", "mrsos"]:
        nan_threshold = 0.7
    elif var_id in ["mrro"]:
        nan_threshold = 0.8
    elif var_id in [
        "sithick",
        "snw",
    ]:
        nan_threshold = 0.9
    else:
        nan_threshold = 0.1

    min = min_max_range[var_id][0]
    max = min_max_range[var_id][1]

    try:
        regrid_ds = xr.open_dataset(fp)
    except:
        assert False

    values = regrid_ds[var_id].values
    difference = abs(max - min)
    difference_buffer = difference * min_max_buffer_percent
    outliers = values[(values < min) | (values > max)]

    # Exclude outliers that are just a tiny bit off of the expected min/max.
    outliers = outliers[
        ~np.isclose(outliers, min, rtol=difference_buffer)
        & ~np.isclose(outliers, max, rtol=difference_buffer)
    ]

    within_range = len(outliers) == 0
    percent_nan = np.count_nonzero(np.isnan(values)) / values.size

    # Test fails for some variables when testing percent_nan lower than 0.6.
    assert within_range and percent_nan < nan_threshold


def validate_variable(variable, ncpus):
    regrid_fps = list(regrid_dir.glob(f"*/*/*/*/{variable}_*.nc"))
    # derive this dynamically from files in tmp/
    min_max_range = generate_min_max_range(Path("tmp").joinpath(f"{variable}.json"))
    args = [(fp, variable, min_max_range) for fp in regrid_fps]
    results = []
    with Pool(ncpus) as pool:
        list(pool.imap_unordered(validate_min_max_nan, args))


# def test_file_existence():
#     for model in models:
#         for scenario in scenarios:
#             for variable in variables:
#                 for interval in intervals:
#                     regrid_fp = regrid_dir.joinpath(f"{model}/{scenario}/{interval}/{variable}/")
#                     files = regrid_fp.glob(f"{variable}_{interval}_{model}_{scenario}_regrid_*.nc")
#                     if len(list(files)) == 0:
#                         print(f"Directory does not exist or has no files: {regrid_fp}")
#                         assert False
#     assert True


dst_ds = open_and_crop_dataset(target_grid_fp, prod_lat_slice)
target_lat_arr = dst_ds["lat"].values
target_lon_arr = dst_ds["lon"].values
ncpus = 24
var_ids = list(variables.keys())


@pytest.mark.parametrize("var_id", var_ids)
def test_grid_match(var_id):
    # dst_ds = xr.open_dataset(test_grid_fp).sel(lat=slice(50, 90))
    regrid_fps = list(regrid_dir.glob(f"**/{var_id}*.nc"))
    args = [(fp, target_lat_arr, target_lon_arr) for fp in regrid_fps]
    with Pool(ncpus) as pool:
        list(pool.imap_unordered(validate_grid, args))


# @pytest.mark.parametrize("var_id", var_ids)
# def test_variable(var_id):
#     validate_variable(var_id)
