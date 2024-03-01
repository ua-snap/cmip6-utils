"""Test the existence of, and values within, daily temperature range files.
It would make sense to supply the same arguments as given to the slurm job, right?
1. Check that all files exist and can open with xarray.
2. Read in the data and do a range check (min / max) to ensure all values are within reasonable bounds

Example usage:
    python -m pytest tests/test_cmip6_dtr.py --tasmax_dir /import/beegfs/CMIP6/arctic-cmip6/regrid/CESM2/ssp585/day/tasmax --tasmin_dir /import/beegfs/CMIP6/arctic-cmip6/regrid/CESM2/ssp585/day/tasmin --output_dir /import/beegfs/CMIP6/kmredilla/dtr_processing/netcdf/CESM2/ssp585/dtr
"""

import pytest
from pathlib import Path
import numpy as np
import xarray as xr
from multiprocessing import cpu_count
from dask.distributed import LocalCluster
from config import expected_value_ranges


@pytest.fixture
def dtr_cli_args(request) -> tuple:
    """
    Fixture to accept the tasmax, tasmin, and output directories from the command line.
    :param request: pytest request object
    Done using this guide https://pytest-with-eric.com/pytest-advanced/pytest-addoption/

    :return: Tuple of length and number of alpha num characters
    """
    # Taking passed args for length and number of chars
    tasmax_dir = Path(request.config.getoption("--tasmax_dir"))
    tasmin_dir = Path(request.config.getoption("--tasmin_dir"))
    output_dir = Path(request.config.getoption("--output_dir"))

    yield tasmax_dir, tasmin_dir, output_dir


def dtr_files_exist(output_dir):
    dtr_fps = list(output_dir.glob("dtr*.nc"))
    return len(dtr_fps) > 0


def test_file_existence(dtr_cli_args):
    """Test that the expected files are present"""
    tasmax_dir, tasmin_dir, output_dir = dtr_cli_args
    if not dtr_files_exist(output_dir):
        pytest.skip("No DTR files found.")

    # I think a for loop is a good bet because pytest might print variable values where asserts fail?
    # we aren't testing the three-way correspondence between directories but this is tested in the worker script
    tasmax_fps = list(tasmax_dir.glob("tasmax*.nc"))
    assert len(tasmax_fps) > 0
    for tasmax_fp in tasmax_fps:
        tasmax_fn = tasmax_fp.name
        tasmin_fp = tasmin_dir.joinpath(tasmax_fn.replace("tasmax", "tasmin"))
        assert tasmin_fp.exists()
        # this should give us the DTR analog file name
        dtr_fp = output_dir.joinpath(tasmax_fp.name.replace("tasmax", "dtr"))
        assert dtr_fp.exists()

    n_tasmax_fps = len(tasmax_fps)
    n_tasmin_fps = len(list(tasmin_dir.glob("tasmin*.nc")))
    n_dtr_fps = len(list(output_dir.glob("dtr*.nc")))
    assert n_tasmin_fps == n_tasmax_fps == n_dtr_fps


def test_file_structure(dtr_cli_args):
    """Test that the files open and look as expected"""
    tasmax_dir, tasmin_dir, output_dir = dtr_cli_args
    if not dtr_files_exist(output_dir):
        pytest.skip("No DTR files found.")

    dtr_fps = list(output_dir.glob("dtr*.nc"))

    for fp in dtr_fps:
        with xr.open_dataset(fp) as ds:
            assert "dtr" in ds.data_vars
            assert list(ds.dims) == ["time", "lat", "lon"]


def test_value_range(dtr_cli_args):
    """Ensure all data values fall within expected range"""
    tasmax_dir, tasmin_dir, output_dir = dtr_cli_args
    if not dtr_files_exist(output_dir):
        pytest.skip("No DTR files found.")

    dtr_fps = list(output_dir.glob("dtr*.nc"))

    # start up a dask client for the final range check
    with LocalCluster(
        n_workers=int(0.9 * cpu_count()),
        memory_limit="4GB",
    ) as cluster:
        with xr.open_mfdataset(dtr_fps) as ds:
            assert ds["dtr"].max() < expected_value_ranges["dtr"]["maximum"]
            assert ds["dtr"].min() >= expected_value_ranges["dtr"]["minimum"]
            # chceks that files aren't all the same value
            assert ~np.all(ds["dtr"].values[0, 0, 0] == ds["dtr"].values)
