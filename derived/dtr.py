"""Script for making daily / diurnal temperature range (dtr) data from tmax and tmin data.
Here, tmax and tmin refer to the daily maximum and minimum temperature data,
but not necessarily "tas" or temperature at surface - other temperature variables should work as well.
This script is designed to work with any gridded daily tmax and tmin data, not just CMIP6. E.g., it can be used for ERA5 data.
The only requirement is that the gridded daily data is in a flat file structure in the input directories,
and can be opened with xarray.open_mfdataset.

Example usage:
    python cmip6_dtr.py --tmax_dir /import/beegfs/CMIP6/arctic-cmip6/regrid/GFDL-ESM4/historical/day/tasmax --tmin_dir /import/beegfs/CMIP6/arctic-cmip6/regrid/GFDL-ESM4/historical/day/tasmin --dtr_tmp_fn dtr_GFDL-ESM4_historical_{start_date}_{end_date}.nc --target_dir /import/beegfs/CMIP6/snapdata/dtr_processing/netcdf/GFDL-ESM4/historical/day/dtr
"""

import argparse
import logging
from pathlib import Path
from dask.distributed import Client
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def get_tmax_tmin_fps(tmax_dir, tmin_dir):
    """Helper function for getting tasmax and tasmin filepaths. Put in function for checking prior to slurming.
    Assumes that all files in the input directories are the target input files.
    """
    tmax_fps = list(tmax_dir.glob("*"))
    tmin_fps = list(tmin_dir.glob(f"*"))

    return tmax_fps, tmin_fps


def get_var_id(ds):
    """Get the variable id from the dataset attributes.
    This is a helper function for getting the variable id from the dataset attributes.
    """
    if "variable_id" in ds.attrs.keys():
        var_id = ds.attrs["variable_id"]
        assert var_id in ds.data_vars, f"{var_id} not in {ds.data_vars}"
    else:
        assert len(ds.data_vars) == 1, "Dataset must have exactly one variable"
        var_id = list(ds.data_vars.keys())[0]
    return var_id


def get_start_end_dates(ds):
    """Get the start and end dates from the dataset attributes."""
    start_date = ds.time.min().dt.strftime("%Y%m%d").values.item()
    end_date = ds.time.max().dt.strftime("%Y%m%d").values.item()
    return start_date, end_date


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tmax_dir",
        type=str,
        help="Directory containing daily maximum temperature data saved by year (and nothing else)",
        required=True,
    )
    parser.add_argument(
        "--tmin_dir",
        type=str,
        help="Directory containing daily minimum temperature data saved by year (and nothing else)",
        required=True,
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        help="Directory for writing daily temperature range data",
        required=True,
    )
    parser.add_argument(
        "--dtr_tmp_fn",
        type=str,
        help="Template filename for the daily temperature range data",
        required=True,
    )
    args = parser.parse_args()

    return (
        Path(args.tmax_dir),
        Path(args.tmin_dir),
        Path(args.target_dir),
        args.dtr_tmp_fn,
    )


if __name__ == "__main__":
    tmax_dir, tmin_dir, target_dir, dtr_tmp_fn = parse_args()

    # assumes all files in one dir have corresponding file in the other
    tmax_fps, tmin_fps = get_tmax_tmin_fps(tmax_dir, tmin_dir)

    target_dir.mkdir(exist_ok=True)

    with Client(n_workers=4, threads_per_worker=6) as client:
        with xr.open_mfdataset(tmax_fps, engine="h5netcdf", parallel=True) as tmax_ds:
            tmax_var_id = get_var_id(tmax_ds)
            with xr.open_mfdataset(
                tmin_fps, engine="h5netcdf", parallel=True
            ) as tmin_ds:
                tmin_var_id = get_var_id(tmin_ds)
                dtr = tmax_ds[tmax_var_id] - tmin_ds[tmin_var_id]
                units = tmax_ds[tmax_var_id].attrs["units"]
                assert units == tmin_ds[tmin_var_id].attrs["units"]

        dtr.name = "dtr"
        dtr.attrs = {
            "long_name": "Daily temperature range",
            "units": units,
        }
        # replace any negative values (tasmax - tasmin < 0) with 0.0000999
        # using this number instead of zero gives us a way of estimating what spots were tweaked
        # include the isnull() check so we don't replace nan's
        dtr = dtr.where((dtr.isnull() | (dtr >= 0)), 0.0000999)

        # the list here at the end is just making sure we have a matching dim order
        dtr_ds = dtr.to_dataset()[list(tmax_ds[tmax_var_id].dims)]
        dtr_ds.attrs = {k: v for k, v in tmax_ds.attrs.items() & tmin_ds.attrs.items()}

        # write
        for year in dtr_ds.time.dt.year.unique():
            year_ds = dtr_ds.sel(time=str(year))
            start_date, end_date = get_start_end_dates(year_ds)
            output_fp = target_dir.joinpath(
                dtr_tmp_fn.format(start_date=start_date, end_date=end_date)
            )
            logging.info(
                f"Writing {year_ds.dtr.name} for {start_date} to {end_date} to {output_fp}"
            )
            year_ds.to_netcdf(output_fp)
