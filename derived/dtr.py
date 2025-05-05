"""Script for making daily / diurnal temperature range (dtr) data from tmax and tmin data.
Here, tmax and tmin refer to the daily maximum and minimum temperature data,
but not necessarily "tas" or temperature at surface - other temperature variables should work as well.
This script is designed to work with any gridded daily tmax and tmin data, not just CMIP6. E.g., it can be used for ERA5 data.
The only requirement is that the gridded daily data is in a flat file structure in the input directories,
and can be opened with xarray.open_mfdataset.

Example usage:
    python dtr.py \
        --tmax_dir /import/beegfs/CMIP6/arctic-cmip6/regrid/GFDL-ESM4/historical/day/tasmax \
        --tmin_dir /import/beegfs/CMIP6/arctic-cmip6/regrid/GFDL-ESM4/historical/day/tasmin \
        --output_dir /import/beegfs/CMIP6/snapdata/dtr_processing/netcdf/GFDL-ESM4/historical/day/dtr \
        --dtr_tmp_fn dtr_GFDL-ESM4_historical_{start_date}_{end_date}.nc

    or ERA5 e.g.:

    python dtr.py \
        --tmax_dir /import/beegfs/CMIP6/arctic-cmip6/daily_era5_4km_3338/netcdf/t2max \
        --tmin_dir /import/beegfs/CMIP6/arctic-cmip6/daily_era5_4km_3338/netcdf/t2min \
        --output_dir /import/beegfs/CMIP6/snapdata/dtr_processing/era5_dtr \
        --dtr_tmp_fn dtr_{year}_4km_3338.nc


    python dtr.py \
        --tmax_dir /center1/CMIP6/kmredilla/daily_era5_4km_3338/netcdf/t2max \
        --tmin_dir /center1/CMIP6/kmredilla/daily_era5_4km_3338/netcdf/t2min \
        --output_dir /import/beegfs/CMIP6/kmredilla/dtr_processing/era5_dtr/dtr \
        --dtr_tmp_fn dtr_{year}_4km_3338.nc
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import xarray as xr
import string
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


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
        "--output_dir",
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
        Path(args.output_dir),
        args.dtr_tmp_fn,
    )


def get_tmax_tmin_fps(tmax_dir, tmin_dir):
    """Helper function for getting tasmax and tasmin filepaths. Put in function for checking prior to slurming.
    Assumes that all files in the input directories are the target input files.
    """
    tmax_fps = list(tmax_dir.glob("*"))
    tmin_fps = list(tmin_dir.glob("*"))

    assert (
        len(tmax_fps) > 0
    ), f"No tasmax files found in the input directory, in {tmax_dir}"
    assert (
        len(tmin_fps) > 0
    ), f"No tasmin files found in the input directory, in {tmin_dir}"
    assert len(tmax_fps) == len(
        tmin_fps
    ), f"Number of tmax and tmin files must be the same. tmax: {len(tmax_fps)} files in {tmax_dir}, tmin: {len(tmin_fps)} files in {tmin_dir}"

    return tmax_fps, tmin_fps


def get_var_id(ds):
    """Get the variable id from the dataset attributes.
    This is a helper function for getting the variable id from the dataset attributes.
    """
    if "variable_id" in ds.attrs.keys():
        var_id = ds.attrs["variable_id"]
        assert var_id in ds.data_vars, f"{var_id} not in {ds.data_vars}"
    else:
        valid_vars = [var for var in ds.data_vars if set(ds[var].dims) == set(ds.dims)]
        assert (
            len(valid_vars) == 1
        ), f"Dataset must have exactly one variable indexed by all dimensions. Found: {valid_vars}"
        var_id = valid_vars[0]

    return var_id


def get_start_end_dates(ds):
    """Get the start and end dates from the dataset attributes."""
    start_date = ds.time.min().dt.strftime("%Y%m%d").values.item()
    end_date = ds.time.max().dt.strftime("%Y%m%d").values.item()
    return start_date, end_date


def extract_format_keys(template):
    """Extract keys from a string to be formatted."""
    formatter = string.Formatter()
    return list(
        set([key for _, key, _, _ in formatter.parse(template) if key is not None])
    )


def make_output_filepath(output_dir, dtr_tmp_fn, start_date, end_date):
    """Make the output file path from the template and start and end dates."""
    keys = extract_format_keys(dtr_tmp_fn)

    if "start_date" in keys:
        start_date = datetime.strptime(start_date, "%Y%m%d").year
    if "end_date" in keys:
        end_date = datetime.strptime(end_date, "%Y%m%d").year

    if keys == ["start_date", "end_date"]:
        output_dir.joinpath(dtr_tmp_fn.format(start_date=start_date, end_date=end_date))
        output_fp = output_dir.joinpath(
            dtr_tmp_fn.format(start_date=start_date, end_date=end_date)
        )
    elif keys == ["year"]:
        start_year = datetime.strptime(start_date, "%Y%m%d").year
        end_year = datetime.strptime(end_date, "%Y%m%d").year
        if start_year != end_year:
            raise ValueError(
                f"Start and end dates must be in the same year for template {dtr_tmp_fn}"
            )
        output_fp = output_dir.joinpath(dtr_tmp_fn.format(year=start_year))
    else:
        raise ValueError(
            f"Template DTR filename, {dtr_tmp_fn}, must have either start_date and end_date or year as keys"
        )

    return output_fp


if __name__ == "__main__":
    tmax_dir, tmin_dir, output_dir, dtr_tmp_fn = parse_args()

    # assumes all files in one dir have corresponding file in the other
    tmax_fps, tmin_fps = get_tmax_tmin_fps(tmax_dir, tmin_dir)

    with xr.open_mfdataset(
        tmax_fps, engine="h5netcdf", parallel=True, chunks="auto"
    ) as tmax_ds:
        with xr.open_mfdataset(
            tmin_fps, engine="h5netcdf", parallel=True, chunks="auto"
        ) as tmin_ds:
            tmax_var_id = get_var_id(tmax_ds)
            tmin_var_id = get_var_id(tmin_ds)
            dtr = tmax_ds[tmax_var_id] - tmin_ds[tmin_var_id]
            dtr.persist()

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
    dtr_ds = dtr.to_dataset().transpose(*list(tmax_ds[tmax_var_id].dims))
    dtr_ds.attrs = {k: v for k, v in tmax_ds.attrs.items() & tmin_ds.attrs.items()}
    # give this a variable_id attribute for consistency (helps with e.g. regridding with regrid.py)
    dtr_ds.attrs["variable_id"] = "dtr"

    # write
    for year in np.unique(dtr_ds.time.dt.year):
        year_ds = dtr_ds.sel(time=str(year))
        start_date, end_date = get_start_end_dates(year_ds)
        output_fp = make_output_filepath(output_dir, dtr_tmp_fn, start_date, end_date)
        logging.info(
            f"Writing {year_ds.dtr.name} for {start_date} to {end_date} to {output_fp}"
        )
        year_ds.to_netcdf(output_fp)
