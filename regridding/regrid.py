"""Script for regridding a batch of files listed in a text file

Note - this script first crops the dataset to the panarctic domain of 50N and up. 
"""

import argparse
import calendar
import random
import time
from pathlib import Path
import cftime
import numpy as np
import pandas as pd
import xesmf as xe
import xarray as xr

# ignore serializationWarnings from xarray for datasets with multiple FillValues
import warnings

warnings.filterwarnings("ignore", category=xr.SerializationWarning)


# define the production latitude domain slice
prod_lat_slice = slice(50, 90)


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-b",
        dest="regrid_batch_fp",
        type=str,
        help="Batch file containing filepaths to be regridded",
        required=True,
    )
    parser.add_argument(
        "-d", dest="dst_fp", type=str, help="Destination grid filepath", required=True
    )
    parser.add_argument(
        "-o",
        dest="out_dir",
        type=str,
        help="Path to directory where regridded data should be written",
        required=True,
    )
    parser.add_argument(
        "--no-clobber",
        dest="no_clobber",
        action="store_true",
        default=False,
        help="Do not regrid a file if the regridded file already exists in out_dir",
    )
    args = parser.parse_args()

    return args.regrid_batch_fp, args.dst_fp, Path(args.out_dir), args.no_clobber


def init_regridder(src_ds, dst_ds):
    """Initialize the regridder object for a single dataset. All source files listed in the batch files should have the same grid, and so we should only need to initiate this for a single file.

    Args:
        src_ds (xarray.DataSet): Source dataset for initializing a regridding object. This should have the same grid as all files in the batch file being worked on.
        dst_ds (xarray.DataSet): Destination dataset for initializing a regridding object. This should be a cropped version of the pipeline's target grid dataset.

    Returns:
        regridder (xesmf.Regridder): a regridder object
    """
    regridder = xe.Regridder(src_ds, dst_ds, "bilinear", unmapped_to_nan=True)

    return regridder


def parse_cmip6_fp(fp):
    """Pull some data attributes from an ESGF CMIP6 filepath.

    Args:
        fp (pathlib.Path): CMIP6 path mirrored from ESGF

    Returns:
        attr_di (dict): dict of modeling-relevant attributes of the filepath and filename
    """
    model, scenario, variant, frequency, variable_id, grid_type = fp.parts[-8:-2]
    timeframe = fp.name.split("_")[-1].split(".nc")[0]

    attr_di = {
        "model": model,
        "scenario": scenario,
        "variant": variant,
        "frequency": frequency,
        "variable_id": variable_id,
        "grid_type": grid_type,
        "timeframe": timeframe,
    }

    return attr_di


def generate_regrid_filepath(fp, out_dir):
    """Generates the name for a regridded file using info parsed from source filepath

    Args:
        fp (str): CMIP6 filepath to generate name for
        out_dir (pathlib.Path): path to the root of the output directory for regridded files

    Returns:
        new_fn (str): new filename string
    """
    fp_attrs = parse_cmip6_fp(fp)

    # drop the variant and grid type from existing filename to get the regrid filename
    fn = fp.name.replace(f"_{fp_attrs['variant']}_", "_").replace(
        f"_{fp_attrs['grid_type']}_", "_regrid_"
    )
    # construct the output filepath
    regrid_fp = out_dir.joinpath(
        *(fp_attrs[k] for k in ["model", "scenario", "frequency", "variable_id"])
    ).joinpath(fn)

    return regrid_fp


def generate_single_year_filename(original_fp, da):
    # take everything preceding the original daterange component of filename
    nodate_fn_str = original_fp.split(".nc")[0].split("_")[:-1].join("_")
    year_fn_str = [
        pd.to_datetime(da.time.values[i]).strftime("%Y%m%d") for i in [0, -1]
    ].join("-")
    out_fp = original_fp.parent.joinpath(f"{nodate_fn_str}_{year_fn_str}.nc")

    return out_fp


# def fix_hour(ts):
#     """Fix the hour in a cftime object if they are 0 instead of 12 (standard is 12 for most all frequencies)"""
#     s = pd.Series(ts)
#     if np.any(s.dt.hour != 12):
#         new_ts = np.datetime64([f"{t.year}-{t.month}-{t.day}T12:00:00" for t in s])
#     else:
#         new_ts = ts

#     return new_ts


def open_and_crop_dataset(fp, lat_slice):
    """Open the connection to a dataset and crop it to a panarctic domain of 50N and up.

    Args:
        fp (pathlib.Path): path to file to be opened as xarray.Dataset and cropped to a panarctic extent
        lat_slice (slice): slice object for cropping latitude dimension of dataset to

    Returns:
        src_ds (xarray.Dataset): xarray Dataset (chunked with Dask) cropped to a panarctic domain
    """
    # we are cropping the dataset using the .sel method as we do not need to regrid the entire grid,
    #  only the part that will evetually wind up in the dataset.
    try:
        src_ds = xr.open_dataset(fp, chunks={"time": 100}).sel(lat=lat_slice)
    except ValueError:
        print(fp)

    return src_ds


def generate_random_date_indices(year):
    """Get the random date indices for a given year (only using year to have consistent seed)"""
    random.seed(year)
    ridx_list = []
    for i in range(5):
        ridx = random.randrange(0 + i * 72, 72 + i * 72)
        if ridx == 359:
            ridx -= 1
        ridx_list.append(ridx)

    return ridx_list


def dayfreq_add_bnds(out_ds, ds):
    """Add the bnds variables back to a daily frequency dataset after converting from 360day to gregorian"""
    lower_bnds = out_ds.time.dt.date.values.astype("datetime64[ns]")
    upper_bnds = lower_bnds + pd.Timedelta("1d")
    new_bnds = [[l, u] for l, u in zip(lower_bnds, upper_bnds)]
    time_bnds = xr.DataArray(
        new_bnds,
        dims=["time", "bnds"],
        coords=dict(time=(["time"], out_ds.time.values)),
    )
    time_bnds.encoding = ds.time_bnds.encoding
    time_bnds.encoding["calendar"] = "gregorian"
    time_bnds.attrs = ds.time_bnds.attrs
    out_ds = out_ds.assign(
        time_bnds=time_bnds, lat_bnds=ds.lat_bnds, lon_bnds=ds.lon_bnds
    )

    return out_ds


def dayfreq_360day_to_gregorian(ds):
    """convert a 360 day calendar time axis on a daily dataset to gregorian numpy.datetime64 by selecting random dates (from different chunks of the year, following the method in https://doi.org/10.31223/X5M081) to insert a new slice as the mean between two adjacent time slices"""
    ts = ds.time.values
    var_id = ds.attrs["variable_id"]
    start_year = ts[0].year
    end_year = ts[-1].year
    # make sure we are indeed working with a timeseries on a valid 360 day calendar
    assert (end_year - start_year + 1) * 360 == len(ts)

    # we will split, compute means, and combine on the random dates selected
    # iterate over years, compute the dates to do this for
    year_da_list = []
    for year in range(start_year, end_year + 1):
        year_da = ds[var_id].sel(time=slice(f"{year}-01-01", f"{year}-12-30"))
        sub_das = []
        prev_idx = 0
        ridx_list = generate_random_date_indices(year)
        for ridx, i in zip(ridx_list, range(len(ridx_list))):
            # for each date, we need a mean of the two adjacent dates
            # append chunk between previous random index (or 0) and next random index
            sub_das.append(
                year_da.isel(time=slice(prev_idx, ridx)).assign_coords(
                    time=np.arange(prev_idx + i, ridx + i)
                )
            )
            new_idx = ridx + i
            new_slice = (
                year_da.isel(time=slice(ridx, ridx + 2))
                .mean(dim="time")
                .expand_dims(time=1)
                .assign_coords(time=[new_idx])
            )
            sub_das.append(new_slice)
            prev_idx = ridx

        sub_das.append(
            year_da.isel(time=slice(prev_idx, 360)).assign_coords(
                time=np.arange(prev_idx + i + 1, 365)
            )
        )

        year_greg_da = xr.concat(sub_das, dim="time")

        if calendar.isleap(year):
            # note, slice is upper-bound exclusive with .isel, but is inclusive with .sel
            leap_slice = (
                year_greg_da.sel(time=slice(58, 59))
                .mean(dim="time")
                .expand_dims(time=1)
                .assign_coords(time=[59])
            )
            pre_leap_da = year_greg_da.sel(time=slice(0, 58))
            post_leap_da = year_greg_da.sel(time=slice(59, 364)).assign_coords(
                time=np.arange(60, 366)
            )
            year_greg_da = xr.concat(
                [pre_leap_da, leap_slice, post_leap_da], dim="time"
            )

        year_greg_da = year_greg_da.assign_coords(
            # ensuring hour used is consistent here as well (1200)
            time=pd.date_range(
                f"{year}-01-01 12:00:00", f"{year}-12-31 12:00:00"
            ).to_numpy()
        )
        year_da_list.append(year_greg_da)

    new_greg_da = xr.concat(year_da_list, dim="time")
    out_ds = new_greg_da.to_dataset()
    out_ds.attrs = ds.attrs
    out_ds.time.encoding = ds.time.encoding
    out_ds.time.encoding["calendar"] = "gregorian"
    out_ds.time.attrs = ds.time.attrs
    # add the bnds variables back in
    out_ds = dayfreq_add_bnds(out_ds, ds)

    return out_ds


def Amonfreq_cftime_to_gregorian(ds):
    """Convert the calendar of a cftime dataset to gregorian"""
    assert type(ds.time.values[0]) in [
        cftime._cftime.Datetime360Day,
        cftime._cftime.DatetimeNoLeap,
    ]
    assert ds.attrs["frequency"] == "Amon"
    new_times = pd.to_datetime(
        [f"{t.year}-{t.month}-15T12:00:00" for t in ds.time.values]
    )
    out_ds = ds.assign_coords(time=new_times)
    out_ds.time.encoding = ds.encoding
    out_ds.time.encoding["calendar"] = "gregorian"
    out_ds.time.attrs = ds.attrs

    return out_ds


def dayfreq_noleap_to_gregorian(ds):
    ts = ds.time.values
    var_id = ds.attrs["variable_id"]
    start_year = ts[0].year
    end_year = ts[-1].year
    # make sure we are indeed working with a timeseries on a valid 365 day noleap calendar
    assert (end_year - start_year + 1) * 365 == len(ts)

    # we will split, compute means, and combine on the random dates selected
    # iterate over years, compute the dates to do this for
    year_da_list = []
    for year in range(start_year, end_year + 1):
        year_da = ds[var_id].sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        if calendar.isleap(year):
            # note, slice is upper-bound exclusive with .isel, but is inclusive with .sel
            leap_slice = (
                year_greg_da.sel(time=slice(f"{year}-02-28", f"{year}-03-01"))
                .mean(dim="time")
                .expand_dims(time=1)
                .assign_coords(time=f"{year}-02-29")
            )
            pre_leap_da = year_greg_da.sel(time=slice(f"{year}-01-01", f"{year}-02-28"))
            post_leap_da = year_greg_da.sel(
                time=slice(f"{year}-03-01", f"{year}-12-31")
            )
            year_greg_da = xr.concat(
                [pre_leap_da, leap_slice, post_leap_da], dim="time"
            )
            year_da_list.append(year_greg_da)
        else:
            year_da_list.append(year_da)

    new_greg_da = xr.concat(year_da_list, dim="time")
    out_ds = new_greg_da.to_dataset()
    out_ds.attrs = ds.attrs
    out_ds.time.encoding = ds.time.encoding
    out_ds.time.encoding["calendar"] = "gregorian"
    out_ds.time.attrs = ds.time.attrs
    # add the bnds variables back in
    out_ds = dayfreq_add_bnds(out_ds, ds)

    return ds


def fix_time_and_write(ds, out_fp):
    """Fix the time dimension of a regridded dataset if needed; write dataset, splitting by appropriate time chunks if needed."""
    if ds.attrs["frequency"] == "day":
        if isinstance(ds.time.values[0], cftime._cftime.Datetime360Day):
            ds = dayfreq_360day_to_gregorian(ds)
        elif isinstance(ds.time.values[0], cftime._cftime.DatetimeNoLeap):
            pass  # TO-DO: handle conversion from cftime._cftime.DatetimeNoLeap
            ds = dayfreq_noleap_to_gregorian(ds)
    elif ds.attrs["frequency"] == "Amon":
        ds = Amonfreq_cftime_to_gregorian(ds)

    # now write out by appropriate time chunks
    if ds.attrs["freq"] == "day":
        out_fps = []
        # write out by year for daily data
        for year, da in ds.groupby("time.year"):
            year_out_fp = generate_single_year_filename(out_fp, da)
            year_ds = da.to_dataset
            year_ds = ds.attrs
            year_ds.to_netcdf(year_out_fp)
            out_fps.append(year_out_fp)
    elif ds.attrs["freq"] == "Amon":
        pass  # TO-DO: handle month splitting if needed

    return out_fps


def regrid_dataset(fp, regridder, out_fp, lat_slice):
    """Regrid a dataset using a regridder object initiated using the target grid with a latitude domain of 50N and up.

    Args:
        fp (pathlib.Path): path to file to be regridded
        regridder (xesmf.Regridder): regridder object initialized on source dataset that has the same grid as dataset as read from fp
        out_fp (pathlib.Path): Path to output regridded file
        lat_slice (slice): slice object for cropping latitude dimension of dataset to

    Returns:
        out_fp (pathlib.Path): Path to output regridded file (just to return something)
    """
    # open the source dataset
    # open the "exteneded" latitude domain so the regridding effort includes the minimum production latitude.
    src_ds = open_and_crop_dataset(fp, lat_slice=lat_slice)

    regrid_task = regridder(src_ds, keep_attrs=True)
    regrid_ds = regrid_task.compute()

    out_fps = fix_time_and_write(regrid_ds, out_fp)

    # # subset to the production latitude slice after regridding.
    # regrid_ds.to_netcdf(out_fp)

    return out_fps


if __name__ == "__main__":
    # parse args
    regrid_batch_fp, dst_fp, out_dir, no_clobber = parse_args()

    # get the paths of files to regrid from the batch file
    with open(regrid_batch_fp) as f:
        lines = f.readlines()
    src_fps = [Path(line.replace("\n", "")) for line in lines]

    # open destination dataset for regridding to.
    dst_ds = open_and_crop_dataset(dst_fp, lat_slice=prod_lat_slice)
    # do the same for one of the source datasets to configure the regridder object
    # defining an "extended" latitude slice, so that grids encoompass the entire
    #  production latitude extent before regridding (e.g. a grid will have domain [49.53, 90] instead of [50.75, 90],
    #  so this is probably always going to give just one more "row" of grid cells for interpolation.)
    ext_lat_slice = slice(49, 90)
    src_init_ds = open_and_crop_dataset(src_fps[0], lat_slice=ext_lat_slice)

    # use one of the source files to be regridded and the destination grid file to create a regridder object
    regridder = init_regridder(src_init_ds, dst_ds)

    # now iterate over files in batch file and run the regridding
    print(f"Regridding {len(src_fps)} files", flush=True)
    tic = time.perf_counter()

    results = []
    for fp in src_fps:
        out_fp = generate_regrid_filepath(fp, out_dir)
        # make sure the parent dirs exist
        out_fp.parent.mkdir(exist_ok=True, parents=True)

        if no_clobber:
            if not out_fp.exists():
                results.append(regrid_dataset(fp, regridder, out_fp, ext_lat_slice))
            else:
                continue
        else:
            print("clobber!")
            results.append(regrid_dataset(fp, regridder, out_fp, ext_lat_slice))

    print(
        f"done, {len(results)} files regridded in {np.round((time.perf_counter() - tic) / 60, 1)}m"
    )
