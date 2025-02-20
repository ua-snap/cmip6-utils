"""Reample (aggregate) the ERA5 from hourly to daily data.
This is done for maximum and minimum temperature and total precipitation.

Example usage:
    python resample_era5.py --era5_dir /beegfs/CMIP6/wrf_era5/04km --output_dir /beegfs/CMIP6/kmredilla/era5_4km_daily/

"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
from dask.distributed import LocalCluster, Client


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Aggregate the ERA5 data.")
    parser.add_argument(
        "--era5_dir",
        type=Path,
        required=True,
        help="The directory containing the ERA5 data.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="The directory to save the aggregated data.",
    )
    parser.add_argument(
        "--fn_str",
        type=str,
        default="era5_wrf_dscale_4km_{date}.nc",
        help="The filename string for the input data.",
    )
    parser.add_argument(
        "--no_clobber",
        action="store_true",
        default=False,
        help="Do not overwrite existing files in rasda_dir.",
    )

    args = parser.parse_args()

    return Path(args.era5_dir), Path(args.output_dir), args.fn_str, args.no_clobber


def get_year_filepaths(era5_dir, year, fn_str):
    """Get all of the filepaths for a single year of ERA5 data."""
    fps = sorted(era5_dir.joinpath(f"{year}").glob(fn_str.format(date="*")))
    return fps


def get_last_15_days_flepaths(era5_dir, year, fn_str):
    """Get the last 15 days' worth files of the supplied year"""
    last_15 = [
        (datetime(year, 12, 31) - timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(14, -1, -1)
    ]
    fps = [
        era5_dir.joinpath(f"{year}/{fn_str.format(date=ymd_str)}")
        for ymd_str in last_15
    ]
    return fps


def get_first_15_days_flepaths(era5_dir, year, fn_str):
    """Get the first 15 days' worth files of the supplied year"""
    first_15 = [
        (datetime(year, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(15)
    ]
    fps = [
        era5_dir.joinpath(f"{year}/{fn_str.format(date=ymd_str)}")
        for ymd_str in first_15
    ]
    return fps


def get_drop_vars(fp):
    """Get the variables to exclude from the open_mfdataset() call using a single sample file."""

    ds = xr.open_dataset(fp)
    drop_vars = [
        x
        for x in list(ds.variables)
        # these are the variables we want to keep
        if x
        not in ["T2", "rainnc", "Time", "south_north", "west_east", "XLONG", "XLAT"]
    ]
    ds.close()

    return drop_vars


def open_dataset(fps, drop_vars):
    """Open a batch of hourly files and return the dataset"""
    era5_ds = xr.open_mfdataset(
        fps, drop_variables=drop_vars, engine="h5netcdf", parallel=True
    )

    return era5_ds


def resample(era5_ds, agg_var):
    """Resample an hourly dataset to daily resolution using method provided"""
    agg_var_lut = get_agg_var_lut(agg_var)
    var_id, agg_func = agg_var_lut["var_id"], agg_var_lut["agg_func"]

    agg_da = (
        era5_ds[var_id]
        .resample(Time="1D")
        .apply(agg_func)
        .to_dataset(name=agg_var)
        .rename({"XLONG": "lon", "XLAT": "lat", "Time": "time"})
    )

    era5_ds.close()
    del era5_ds

    return agg_da


def get_agg_var_lut(agg_var):
    """Look up table for the aggregation function for each variable"""
    lut = {
        "t2min": {"var_id": "T2", "agg_func": lambda x: x.min(dim="Time")},
        "t2max": {"var_id": "T2", "agg_func": lambda x: x.max(dim="Time")},
        "pr": {"var_id": "rainnc", "agg_func": lambda x: x.sum(dim="Time")},
    }

    return lut[agg_var]


def agg_files_exist(year, agg_vars, output_dir, fn_str):
    """Check if the aggregated files already exist"""
    file_exist_accum = []
    for agg_var in agg_vars:
        # fps should be groupd by year so we only need one
        out_fp = output_dir.joinpath(fn_str.format(date=f"{agg_var}_{year}"))
        file_exist_accum.append(out_fp.exists())

    all_files_exist = all(file_exist_accum)

    return all_files_exist


def check_no_clobber(no_clobber, year, agg_vars, output_dir, fn_str):
    """Check if the no_clobber flag is set and if the files already exist"""
    if no_clobber and agg_files_exist(year, agg_vars, output_dir, fn_str):
        print(f"Resampled files for {year} already exist, skipping")
        return True
    else:
        print(f"Missing some resampled files for {year}, processing")
        return False


def write_data(agg_da, output_dir, fn_str, agg_var, year):
    """Write the resampled data to disk"""
    out_fp = output_dir.joinpath(fn_str.format(date=f"{agg_var}_{year}"))
    agg_da.to_netcdf(out_fp)

    return out_fp


def open_and_resample(fps, drop_vars, agg_vars, output_dir, year):
    """Open the dataset and aggregate the data for each variable"""
    # only need to open the dataset once
    era5_ds = open_dataset(fps, drop_vars)
    for agg_var in agg_vars:
        agg_da = resample(era5_ds, agg_var)
        out_fp = write_data(agg_da, output_dir, fn_str, agg_var, year)
        print(year, agg_var, f"done, writen to {out_fp}")


def resample_full_years(
    era5_dir, output_dir, fn_str, full_years, drop_vars, agg_vars, no_clobber
):
    """Resample the full years of data"""
    for year in full_years:
        if check_no_clobber(no_clobber, year, agg_vars, output_dir, fn_str):
            continue
        fps = get_year_filepaths(era5_dir, year, fn_str)
        open_and_resample(fps, drop_vars, agg_vars, output_dir, year)


def resample_partial_years(
    era5_dir, output_dir, fn_str, full_years, drop_vars, agg_vars, no_clobber
):
    """Resample the first and last 15 days of the data"""
    first_year = full_years[0] - 1
    if not check_no_clobber(no_clobber, first_year, agg_vars, output_dir, fn_str):
        fps = get_last_15_days_flepaths(era5_dir, first_year, fn_str)
        open_and_resample(fps, drop_vars, agg_vars, output_dir, first_year)

    last_year = full_years[-1] + 1
    if not check_no_clobber(no_clobber, last_year, agg_vars, output_dir, fn_str):
        fps = get_first_15_days_flepaths(era5_dir, last_year, fn_str)
        open_and_resample(fps, drop_vars, agg_vars, output_dir, last_year)


def main(era5_dir, output_dir, fn_str, no_clobber):
    # the full years we want are 1985 to 2014
    full_years = list(range(1985, 2015))
    # want these variables at daily resolution
    agg_vars = ["t2min", "t2max", "pr"]
    # list of variables to exclude from the open_mfdataset() call
    drop_vars = get_drop_vars(
        era5_dir.joinpath(f"1995/{fn_str.format(date='1995-01-01')}")
    )

    resample_full_years(
        era5_dir, output_dir, fn_str, full_years, drop_vars, agg_vars, no_clobber
    )
    resample_partial_years(
        era5_dir, output_dir, fn_str, full_years, drop_vars, agg_vars, no_clobber
    )


if __name__ == "__main__":
    era5_dir, output_dir, fn_str, no_clobber = parse_args()

    # The threads_per_worker=1 seems to have really helped with open_mfdataset()
    cluster = LocalCluster(threads_per_worker=1)
    client = Client(cluster)

    main(era5_dir, output_dir, fn_str, no_clobber)
