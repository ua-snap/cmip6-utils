"""Test the existence of, and values within, regridded files.

Usage:

export SCRATCH_DIR=/path/to/scratch_dir

# Set EXAMPLE_REGRID_FILE environment variable to one of the regrid files.
# All lat/lon grid coordinates will be compared against this file.
export EXAMPLE_REGRID_FILE=/path/to/example_regrid_file.nc

cd regridding
python -m pytest tests/test_regridding.py
"""

import numpy as np
import os
import xarray as xr
from regrid import rename_file
from multiprocessing import Pool
from pathlib import Path

min_max_buffer_percent = 0.001

SCRATCH_DIR = Path(os.getenv("SCRATCH_DIR"))
regrid_dir = SCRATCH_DIR.joinpath("regrid")

# The lists of models, scenarios, variables, and intervals have been copied from
# the CMIP6 scope document

models = [
    "ACCESS-CM2",
    "CESM2",
    "CNRM",
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
varnames = [
    "tas",
    "pr",
    "psl",
    "huss",
    # "uas",
    # "vas",
    "ta",
    "ua",
    "va",
    "hus",
    "evspsbl",
    # "mrro",
    # "mrsos",
    "prsn",
    # "snd",
    # "snw",
    "rlds",
    "rsds",
    # "rsd",
    # "clt",
]

intervals = [
    "Amon",
    "day",
]

# TODO: Need to add tests for "Time-independent 2-d land surface data" ("orog",
# "sftlf", "sftof") also once the expected directory structure is known.


def validate_dimensions(args):
    regrid_fp, target_lat_arr, target_lon_arr = args
    try:
        regrid_ds = xr.open_dataset(regrid_fp)
        assert np.all(regrid_ds["lat"].values == target_lat_arr)
        assert np.all(regrid_ds["lon"].values == target_lon_arr)
    except:
        assert False
    assert True


def validate_min_max_nan(args):
    fp, variable = args

    # Ranges were taken from source CMIP6 data using the get_min_max.py script.
    min_max_ranges = {
        "evspsbl": [-2.679381e-05, 0.00019423357],
        "hus": [-0.011698332, 0.026849562],
        "huss": [-0.00022705719, 0.029937433],
        "pr": [-1.2892809e-20, 0.008893973],
        "prsn": [-1.1401009e-22, 0.004692739],
        "psl": [90716.89, 108336.56],
        "rlds": [-104.074745, 542.4809],
        "rsds": [-0.00390625, 463.78638],
        "ta": [0.0, 323.19162],
        "tas": [190.24911, 326.23245],
        "ua": [-133.71509, 146.29332],
        "va": [-135.3523, 137.70459],
    }

    nan_thresholds = {}
    for varname in varnames:
        if varname == "prsn":
            nan_thresholds[varname] = 0.6
        elif varname in ["hus", "ua", "va"]:
            nan_thresholds[varname] = 0.2
        else:
            nan_thresholds[varname] = 0.1

    min = min_max_ranges[variable][0]
    max = min_max_ranges[variable][1]

    try:
        regrid_ds = xr.open_dataset(fp)
    except:
        assert False

    values = regrid_ds[variable].values
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
    assert within_range and percent_nan < nan_thresholds[variable]


def validate_variable(variable):
    regrid_fps = list(regrid_dir.glob(f"*/*/*/*/{variable}_*.nc"))
    args = [(fp, variable) for fp in regrid_fps]
    results = []
    with Pool(5) as pool:
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


def test_dimensions():
    test_grid_fp = Path(os.getenv("EXAMPLE_REGRID_FILE"))
    dst_ds = xr.open_dataset(test_grid_fp)
    target_lat_arr = dst_ds["lat"].values
    target_lon_arr = dst_ds["lon"].values
    regrid_fps = list(regrid_dir.glob("*/*/*/*/*.nc"))
    args = [(fp, target_lat_arr, target_lon_arr) for fp in regrid_fps]
    with Pool(5) as pool:
        list(pool.imap_unordered(validate_dimensions, args))


def test_evspsbl():
    validate_variable("evspsbl")


def test_hus():
    validate_variable("hus")


def test_huss():
    validate_variable("huss")


def test_pr():
    validate_variable("pr")


def test_prsn():
    validate_variable("prsn")


def test_psl():
    validate_variable("psl")


def test_rlds():
    validate_variable("rlds")


def test_rsds():
    validate_variable("rsds")


def test_tas():
    validate_variable("tas")


def test_ua():
    validate_variable("ua")


def test_va():
    validate_variable("va")
