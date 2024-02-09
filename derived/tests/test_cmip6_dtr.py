"""Test the existence of, and values within, daily temperature range files.
It would make sense to supply the same arguments as given to the slurm job, right?
1. Check that all files exist and can open with xarray.
2. Read in the data and do a range check (min / max) to ensure all values are within reasonable bounds

Example usage:
    python -m pytest tests/test_cmip6_dtr.py
"""

import xarray as xr
from dask.distributed import LocalCluster
from cmip6_dtr import parse_args, dtr_tmp_fn
from config import expected_value_ranges

# get same inputs
tasmax_dir, tasmin_dir, output_dir = parse_args()


def test_file_existence():
    """Test that the expected files are present"""
    # I think a for look is a good bet because pytest might print variable values where asserts fail?
    # we aren't testing the three-way correspondence between directories but this is tested in the worker script
    tasmax_fps = list(tasmax_dir.glob("tasmax*.nc"))
    for tasmax_fp in tasmax_fps:
        tasmax_fn = tasmax_fp.name
        tasmin_fp = tasmin_dir.joinpath(tasmax_fn.repalce("tasmax", "tasmin"))
        assert tasmin_fp.exists()
        # this should give us the DTR analog file name
        dtr_fp = output_dir.joinpath(tasmax_fp.name.replace("tasmax", "dtr"))
        assert dtr_fp.exists()

    n_tasmax_fps = len(tasmax_fps)
    n_tasmin_fps = len(list(tasmin_dir.glob("tasmin*.nc")))
    n_dtr_fps = len(list(output_dir.glob("dtr*.nc")))
    assert n_tasmin_fps == n_tasmax_fps == n_dtr_fps


def test_file_structure():
    """Test that the files open and look as expected"""
    for fp in output_dir.glob("dtr*.nc"):
        with xr.open_dataset(fp) as ds:
            assert "dtr" in ds.data_vars
            assert all(ds.dims == ["time", "lat", "lon"])


def test_value_range():
    """Ensure all data values fall within expected range"""
    # start up a dask client for the final range check
    with LocalCluster(
        n_workers=int(0.9 * cpu_count()),
        memory_limit="4GB",
    ) as cluster:
        with xr.open_mfdataset(output_dir.glob("dtr*.nc")) as ds:
            assert ds["dtr"].max() < expected_value_ranges["dtr"]["maximum"]
            assert ds["dtr"].min() < expected_value_ranges["dtr"]["minimum"]
