"""Test that the "cropped" files (those that were not actually regridded) all have the correct grid and calendar.

Usage:
    python -m pytest tests/test_cropping.py

Note - this is an extra utility script in case the cropping is done separately from the regridding. 
As it is currently set up, the cropping script will write to the same regridding output directory, 
 and running the normal test suite will validate the grids of these cropped files. However, if
 the cropping was done after the regridded files have been moved out, the regrid completion
 tests will fail. So you can simply use this script instead of running the full test suite.
"""

from multiprocessing import Pool
import pytest
from config import target_grid_fp, variables, regrid_dir
from regrid import open_and_crop_dataset, prod_lat_slice
from tests.test_regridding import validate_grid

dst_ds = open_and_crop_dataset(target_grid_fp, prod_lat_slice)
target_lat_arr = dst_ds["lat"].values
target_lon_arr = dst_ds["lon"].values
ncpus = 24
var_ids = list(variables.keys())


@pytest.mark.parametrize("var_id", var_ids)
def test_grid_match(var_id):
    regrid_fps = list(regrid_dir.glob(f"**/{var_id}*.nc"))
    args = [(fp, target_lat_arr, target_lon_arr) for fp in regrid_fps]
    with Pool(ncpus) as pool:
        list(pool.imap_unordered(validate_grid, args))
