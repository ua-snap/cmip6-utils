"""Script for regridding a batch of files listed in a text file

Note - this script first crops the dataset to the panarctic domain of 50N and up.
"""

import argparse
import random
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
import cftime
import numpy as np
import pandas as pd
import xesmf as xe
import xarray as xr
from pyproj import CRS
from xclim.core import units

# project
from config import variables, landsea_variables

# ignore serializationWarnings from xarray for datasets with multiple FillValues
warnings.filterwarnings("ignore", category=xr.SerializationWarning)


def parse_args():
    """Parse some command line arguments

    Returns
    -------
    regrid_batch_fp : str
        Batch file containing filepaths to be regridded
    dst_fp : str
        Destination grid filepath
    src_sftlf_fp : str
        Path to sftlf file to use for land masking source
    dst_sftlf_fp : str
        Path to sftlf file to use for land masking destination/target
    out_dir : pathlib.Path
        Path to directory where regridded data should be written
    interp_method : str
        Interpolation method to use for regridding
    no_clobber : bool
        Do not overwrite existing regidded files
    rasdafy : bool
        Do some Rasdaman-specific tweaks to the data
    """
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
        "--src_sftlf_fp",
        type=str,
        help="Path to sftlf file to use for land masking source",
        required=False,
    )
    parser.add_argument(
        "--dst_sftlf_fp",
        type=str,
        help="Path to sftlf file to use for land masking destination/target",
        required=False,
    )
    parser.add_argument(
        "-o",
        dest="out_dir",
        type=str,
        help="Path to directory where regridded data should be written",
        required=True,
    )
    parser.add_argument(
        "--interp_method",
        type=str,
        help="Interpolation method to use for regridding",
    )
    parser.add_argument(
        "--rasdafy",
        action="store_true",
        help="Do some Rasdaman-specific tweaks to the data",
    )
    parser.add_argument(
        "--no-clobber",
        action="store_true",
        help="Do not overwrite existing regidded files",
    )
    args = parser.parse_args()

    return (
        Path(args.regrid_batch_fp),
        args.dst_fp,
        args.src_sftlf_fp,
        args.dst_sftlf_fp,
        Path(args.out_dir),
        args.interp_method,
        args.rasdafy,
        args.no_clobber,
    )


def init_regridder(src_ds, dst_ds, interp_method):
    """Initialize the regridder object for a single dataset.
    All source files listed in the batch files should have the same grid,
    and so we should only need to initiate this for a single file.

    Parameters
    ----------
    src_ds : xarray.Dataset
        Source dataset for initializing a regridding object.
        This should have the same grid as all files in the batch file being worked on.
    dst_ds : xarray.Dataset
        Destination dataset for initializing a regridding object.
    interp_method : str
        Interpolation method to use for regridding.
        Options are 'bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch'

    Returns:
        regridder (xesmf.Regridder): a regridder object
    """
    # cache existing encoding / attrs
    lon_enc = dst_ds["lon"].encoding
    lon_attrs = dst_ds["lon"].attrs
    # convert to -180 to 180 lon coords, and reapply encoding / attrs
    dst_ds["lon"] = (dst_ds["lon"] + 180) % 360 - 180
    dst_ds["lon"].encoding = lon_enc
    dst_ds["lon"].attrs = lon_attrs
    # probably doesn't matter but technically correct after adjustment
    dst_ds["lon"].attrs["valid_max"] = 180
    dst_ds["lon"].attrs["valid_min"] = -180
    # sort
    if len(dst_ds.lon.dims) == 1:
        dst_ds = dst_ds.sortby(dst_ds.lon, ascending=True)
    # initialize the regridder which now contains standard -180 to 180 longitude values
    regridder = xe.Regridder(
        src_ds,
        dst_ds,
        interp_method,
        unmapped_to_nan=True,
        periodic=True,
        ignore_degenerate=True,
    )

    return regridder


def parse_cmip6_fp(fp):
    """Pull some data attributes/identifiers from a CMIP6 filepath.

    Parameters
    ----------
    fp : pathlib.Path
        CMIP6 path mirrored from ESGF

    Returns
    -------
    attr_di : dict
        dict of modeling-relevant attributes of the filepath and filename
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
    """Generates the name for a regridded file using info parsed from source filepath.

    Parameters
    ----------
    fp : str
        CMIP6 filepath to generate name for
    out_dir : pathlib.Path
        path to the root of the output directory for regridded files

    Returns
    -------
    regrid_fp : pathlib.Path
        path for a regridded file that would be generated from the input CMIP6 filepath
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


def fix_hour_in_time_dim(ds):
    """Fix the hour in a time dimension. Some datasets have hours that are not 12:00:00.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to fix the time dimension of

    Returns
    -------
    ds : xarray.Dataset
        Dataset with time dimension fixed to 12:00:00
    """
    if np.any(ds.time.dt.hour != 12):
        new_ts = pd.to_datetime(
            [
                f"{year}-{month}-{day}T12:00:00"
                for year, month, day in zip(
                    ds.time.dt.year, ds.time.dt.month, ds.time.dt.day
                )
            ]
        )
        ds = ds.assign_coords(time=new_ts)

    return ds


def generate_random_date_indices(year):
    """Get the random date indices for a given year (only using year to have consistent seed).

    Parameters
    ----------
    year : int
        Year to generate random date indices for

    Returns
    -------
    ridx_list : list
        List of random date indices for the given year
    """
    random.seed(year)
    ridx_list = []
    for i in range(5):
        ridx = random.randrange(0 + i * 72, 72 + i * 72)
        if ridx == 359:
            ridx -= 1
        ridx_list.append(ridx)

    return ridx_list


def dayfreq_360day_to_noleap(ds):
    """convert a 360 day calendar time axis on a daily dataset to noleap numpy.datetime64
    by selecting random dates (from different chunks of the year, following the method in
    https://doi.org/10.31223/X5M081) to insert a new slice as the mean between two adjacent
    time slices.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to convert the time axis of

    Returns
    -------
    out_ds : xarray.Dataset
        Dataset with time axis converted from 360-day to noleap
    """
    ts = ds.time.values
    var_id = ds.attrs["variable_id"]
    start_year = ts[0].year
    end_year = ts[-1].year

    # okay if there are coordinates indexed by time that are NOT time, just be like "no"
    # this happens with e.g. a grid mapping variable, such as spatial_ref, created
    # by rioxrarray. These do not need to be indexed by time and, the regridding may convert
    # them to an actual coordinate instead of a data variable.
    # taking mean over the time dim below causes issues.
    assert all(
        ["time" not in ds.coords[coord].dims for coord in ds.coords if coord != "time"]
    ), "Non-time coordinates indexed by time detected. Check for time-indexed grid mapping variable in target dataset"

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

        year_noleap_da = xr.concat(sub_das, dim="time")
        # ensuring hour used is consistent here (1200)
        dates = pd.date_range(f"{year}-01-01T12:00:00", f"{year}-12-31T12:00:00")
        # drop any possible leap day created by date_range
        noleap_dates = dates[~((dates.month == 2) & (dates.day == 29))].to_numpy()
        year_noleap_da = year_noleap_da.assign_coords(time=noleap_dates)
        year_da_list.append(year_noleap_da)

    new_noleap_da = xr.concat(year_da_list, dim="time")
    out_ds = new_noleap_da.to_dataset()
    out_ds.attrs = ds.attrs
    out_ds.time.encoding = ds.time.encoding
    out_ds.time.encoding["calendar"] = "noleap"
    out_ds.time.encoding["units"] = "days since 1950-01-01"
    out_ds.time.attrs = ds.time.attrs

    return out_ds


def dayfreq_gregorian_to_noleap(ds):
    """Convert gregorian calendar time axis to noleap.
    (Pretty sure there is an xarray function to do this.)

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to convert the time axis of

    Returns
    -------
    out_ds : xarray.Dataset
        Dataset with time axis converted from gregorian to noleap
    """
    out_ds = ds.sel(time=~((ds.time.dt.day == 29) & (ds.time.dt.month == 2)))
    del out_ds.time.encoding["dtype"]  # let this be assigned?
    out_ds.time.encoding["calendar"] = "noleap"
    # Run this function just to ensure consistent hour values
    out_ds = fix_hour_in_time_dim(out_ds)

    return out_ds


def generate_single_year_filename(original_fp, year_ds):
    """Generate a filename for a single year's worth of data. Used for splitting regridded data by year.

    Parameters
    ----------
    original_fp : pathlib.Path
        Original (regridded) filepath to generate a new filepath from
    year_ds : xarray.Dataset
        Dataset for a single year to generate a filename for

    Returns
    -------
    out_fp : pathlib.Path
        Path for writing a single year's worth of regridded data corresponding to year_ds
    """
    # take everything preceding the original daterange component of filename
    nodate_fn_str = "_".join(original_fp.name.split(".nc")[0].split("_")[:-1])
    time_bnds = [year_ds.time.values[i] for i in [0, -1]]
    # want these as datetime object to use strftime
    if isinstance(time_bnds[0], np.datetime64):
        time_bnds = [pd.to_datetime(tb) for tb in time_bnds]
    if "day" in year_ds.attrs["frequency"]:
        year_fn_str = "-".join([tb.strftime("%Y%m%d") for tb in time_bnds])
    elif "mon" in year_ds.attrs["frequency"]:
        # drop date for monthly data
        year_fn_str = "-".join([tb.strftime("%Y%m") for tb in time_bnds])

    out_fp = original_fp.parent.joinpath(f"{nodate_fn_str}_{year_fn_str}.nc")

    return out_fp


def Amonfreq_fix_time(out_ds, src_ds):
    """Fix the time dimension of a regridded monthly dataset to ensure that the
    day of month used is 15 and not 14 or 16.

    Parameters
    ----------
    out_ds : xarray.Dataset
        Dataset to fix the time dimension of
    src_ds : xarray.Dataset
        Source dataset used to create out_ds. For passing attributes

    Returns
    -------
    out_ds : xarray.Dataset
        Dataset with time dimension fixed to 15th of the
    """
    if type(out_ds.time.values[0]) in [
        cftime._cftime.Datetime360Day,
    ]:
        new_times = pd.to_datetime(
            [f"{t.year}-{t.month}-15T12:00:00" for t in out_ds.time.values],
            format="%Y-%m-%dT%H:%M:%S",
        )
    else:
        if not np.all(out_ds.time.dt.day.values == 15):
            new_times = pd.to_datetime(
                [
                    f"{year.item()}-{month.item()}-15T12:00:00"
                    for year, month in zip(out_ds.time.dt.year, out_ds.time.dt.month)
                ],
                format="%Y-%m-%dT%H:%M:%S",
            )
        else:
            new_times = out_ds.time.values

    out_ds = out_ds.assign_coords(time=new_times)
    out_ds.time.encoding = src_ds.time.encoding
    out_ds.time.encoding["calendar"] = "noleap"
    out_ds.time.encoding["units"] = "days since 1950-01-01"
    out_ds.time.attrs = src_ds.time.attrs

    return out_ds


def get_time_res_days(ds):
    """Get the temporal resolution of a dataset in days from the time variable directly.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to get the temporal resolution of

    Returns
    -------
    res_days : int
        Temporal resolution of the dataset in days
    """
    if type(ds.time.values[0]) in [
        cftime._cftime.Datetime360Day,
        cftime._cftime.DatetimeNoLeap,
    ]:
        # have seen some datasets with a weird first time values
        #  (e.g. DatetimeNoLeap(1849, 12, 31, 23, 44, 59, 999993, has_year_zero=True))
        #  so use the 2nd and 3rd indices
        res_days = (ds.time.values[2] - ds.time.values[1]).days
    elif isinstance(ds.time.values[0], np.datetime64):
        res_days = (ds.time.values[2] - ds.time.values[1]).astype("timedelta64[D]")
    else:
        print(f"Unrecognized time type: {type(ds.time.values[0])}")

    return res_days


def check_is_dayfreq(ds):
    """Function to make sure a "day" frequency dataset is actually daily. Some are mis-labelled.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to check the frequency of

    Returns
    -------
    is_dayfreq : bool
        Whether the dataset is actually daily
    """
    return get_time_res_days(ds) == 1


def check_is_monfreq(ds):
    """Function to make sure a "monthly" frequency dataset is actually monthly. Some are mis-labelled.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to check the frequency of

    Returns
    -------
    is_dayfreq : bool
        Whether the dataset is actually monthly
    """
    return get_time_res_days(ds) in [28, 29, 30, 31]


def fix_time(out_ds, src_ds):
    """Fix the time dimension of a regridded dataset if needed;

    Parameters
    ----------
    out_ds : xarray.Dataset
        Dataset to fix the time dimension of
    src_ds : xarray.Dataset
        Source dataset used to create out_ds (for fixing time axis)

    Returns
    -------
    out_fps : list
        List of filepaths written to
    """
    if check_is_dayfreq(out_ds):
        # make sure we assign correct daily frequency type
        out_ds.attrs["frequency"] = [
            s for s in variables[out_ds.attrs["variable_id"]]["table_ids"] if "day" in s
        ][0]
        if isinstance(out_ds.time.values[0], cftime._cftime.Datetime360Day):
            out_ds = dayfreq_360day_to_noleap(out_ds)
        elif isinstance(out_ds.time.values[0], np.datetime64):
            out_ds = dayfreq_gregorian_to_noleap(out_ds)
        else:
            assert isinstance(
                out_ds.time.values[0], cftime._cftime.DatetimeNoLeap
            ), f"Unrecognized time type: {type(out_ds.time.values[0])}"

    elif check_is_monfreq(out_ds):
        # make sure we assign correct monthly frequency type
        out_ds.attrs["frequency"] = [
            s for s in variables[out_ds.attrs["variable_id"]]["table_ids"] if "mon" in s
        ][0]
        out_ds = Amonfreq_fix_time(out_ds, src_ds)

    # make sure the time axis is unlimited (this means it is a "record dimension" in netCDF parlance)
    out_ds.encoding["unlimited_dims"] = ["time"]
    # just ensuring we have a consistent time units value
    out_ds.time.encoding["units"] = "days since 1950-01-01"
    # some variations of this calendar are called 365_day.
    # We will ensure they are all "noleap" for consistency.
    out_ds.time.encoding["calendar"] = "noleap"
    # also ensure time axis encoding dtype is removed,
    # sometimes the incorrect dtype is assigned? perhaps at regridding?
    try:
        del out_ds.time.encoding["dtype"]
    except KeyError:
        pass

    # make sure bnds variables are out, we probably don't need for this dataset
    # just makes things simpler.
    # not sure if they are always/never/sometimes kept through regridding
    for bnd_var in ["bnds", "lat_bnds", "lon_bnds", "time_bnds"]:
        if bnd_var in out_ds:
            out_ds = out_ds.drop_vars(bnd_var)

    return out_ds


def write_regridded_files(out_ds, out_fp):
    """Write a regridded dataset to files for each year.

    Parameters
    ----------
    out_ds : xarray.Dataset
        Dataset to write to a file
    out_fp : pathlib.Path
        Filepath to write the dataset to

    Returns
    -------
    out_fp : pathlib.Path
        Filepath written to
    """
    # write out everything (monthly and daily freqs) by year
    out_fps = []
    for year, year_ds in out_ds.groupby("time.year"):
        if year_ds.time.shape[0] == 1:
            # skip weird files where first time value is last day of a year
            continue
        if year < 1950:
            # skip any years before 1950
            continue
        year_out_fp = generate_single_year_filename(out_fp, year_ds)
        # Make sure we are writing the time dimension as noleap
        assert year_ds.time.encoding["calendar"] == "noleap"

        year_ds.to_netcdf(year_out_fp)
        out_fps.append(year_out_fp)

    [print(f"{fp} done") for fp in out_fps]

    return out_fp


def get_var_id(ds):
    """Get the CMIP6 variable ID from a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to get the variable ID from

    Returns
    -------
    var_id : str
        Variable ID of the dataset
    """
    # assumes we only have one data variable
    var_ids = [var_id for var_id in list(ds.data_vars) if var_id in variables]
    assert len(var_ids) != 0, "No variable ID found in Dataset."
    assert len(var_ids) == 1, f"More than one variable ID found: {var_ids}."

    return var_ids[0]


def check_src_landsea(ds):
    """Check if the source dataset is a land/sea dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to check if it is a land/sea dataset

    Returns
    -------
    is_landsea : bool
        Whether the dataset is a land/sea dataset
    """
    var_id = get_var_id(ds)
    return var_id in landsea_variables


def regrid_sftlf_landmask(sftlf_fp, target_ds, threshold):
    """Derive a landmask (land == 1) from an sftlf file and regrid it to a target dataset.
    Not necessarily the target in the global scope of this script, but it may be.
    This is done because some sftlf files are not on the same grid as the dataset they are needed for.
    This is being included because I am not sure what is better - regridding the sftlf file or regridding the mask derived from it.

    Parameters
    ----------
    sftlf_fp : str
        Path to sftlf file to derive landmask from
    target_ds : xarray.Dataset
        Dataset to regrid the landmask to
    threshold : float
        Threshold for land/sea mask (0-100)

    Returns
    -------
    target_landmask : xarray.DataArray
        Landmask regridded to the target dataset
    """
    sftlf_ds = xr.open_dataset(sftlf_fp)
    landmask = sftlf_ds["sftlf"] > threshold
    var_id = get_var_id(target_ds)

    target_regridder = xe.Regridder(
        landmask,
        target_ds[var_id],
        method="nearest_s2d",
        unmapped_to_nan=True,
    )
    target_landmask = target_regridder(landmask, keep_attrs=True)

    return target_landmask


def check_src_nanmask(src_init_ds, dst_landmask):
    """Check if the source dataset has NaNs present representing land or sea.
    We would expect the NaN percentage to be fairly close to that of the dst_landmask.

    Parameters
    ----------
    src_init_ds : xarray.Dataset
        Source dataset to check for NaNs representing land/sea
    dst_landmask : xarray.DataArray
        Landmask for the destination dataset

    Returns
    -------
    good_nanmask : bool
        Whether the NaN mask for the source dataset is good
    """
    var_id = get_var_id(src_init_ds)
    nan_perc = (
        src_init_ds[var_id].isel(time=0).isnull().sum()
        / src_init_ds[var_id].isel(time=0).size
    )

    # raise warning if there are NaNs and the NaN percentage is not between 0.3 and 0.4
    # we have seen examples of "bad" nanmasks where there are many land pixels masked
    # (e.g. mrro for KACE-1-0-G)
    if landsea_variables[var_id] == "sea":
        dst_nan_perc = dst_landmask.sum() / dst_landmask.size
    elif landsea_variables[var_id] == "land":
        dst_nan_perc = (~dst_landmask).sum() / dst_landmask.size

    # I know this is a pretty big range but we have seen valid-looking nanmasks
    # with ~20% difference in coverage between expected and not
    good_nanmask = (dst_nan_perc - 0.2) < nan_perc < (dst_nan_perc + 0.2)

    if not (good_nanmask & src_init_ds[var_id].isnull().any()):
        print(
            (
                f"NaN percentage for {var_id} in source dataset ({var_id}) is {nan_perc:.2f}."
                f"Expected to be close to {dst_nan_perc:.2f} based on the destination landmask."
            )
        )
    return good_nanmask


def prep_for_landsea(src_init_ds, dst_ds, src_sftlf_fp, dst_sftlf_fp):
    """Prepare a land/sea dataset for regridding by adding masks to the source and target datasets.
    Mask for the source dataset is created from an sftlf file if it is available and if it has
    the same dimensions as the source dataset.

    Parameters
    ----------
    src_init_ds : xarray.Dataset
        Source dataset to add a mask to
    dst_ds : xarray.Dataset
        Destination dataset to add a mask to
    src_sftlf_fp : str
        Path to sftlf file for the source dataset
    dst_sftlf_fp : str
        Path to sftlf file for the destination dataset

    Returns
    -------
    src_init_ds, dst_ds : tuple[xarray.Dataset, xarray.Dataset]
        Source and destination datasets with mask added
    """
    # get the variable ID of the source dataset
    var_id = get_var_id(src_init_ds)
    assert check_src_landsea(
        src_init_ds
    ), "Variable ID of source dataset is not a land/sea variable"

    # set the threshold for land/sea area percentage if derived from sftlf file
    threshold = 0
    # use sftlf file for destination dataset always (assumes the target grid dataset is a non-land/sea variable)
    dst_landmask = regrid_sftlf_landmask(dst_sftlf_fp, dst_ds, threshold)
    src_has_mask = check_src_nanmask(src_init_ds, dst_landmask)

    # use a mask value of 1 for land and 0 for sea, but switch if it's a sea variable
    mask_val, nan_val = 1, 0
    if landsea_variables[var_id] == "sea":
        mask_val, nan_val = nan_val, mask_val

    # add mask to destination dataset
    dst_ds["mask"] = xr.where(dst_landmask, mask_val, nan_val)

    if src_has_mask:
        print("Source dataset has NaNs representing land/sea, will use this as mask")
        # we should always be using 0 as the null value replacement
        #   (second argument) and 1 as the mask value (third argument)
        #   when deriving from the source dataset NaNs
        src_init_ds["mask"] = xr.where(src_init_ds[var_id].isel(time=0).isnull(), 0, 1)
    else:
        if src_sftlf_fp is None:
            print(
                (
                    "Source dataset does not have NaNs representing land/sea, and no sftlf file provided for source. "
                    "Will use sftlf file for destination dataset."
                )
            )
            src_sftlf_fp = dst_sftlf_fp
        else:
            print(
                "Source dataset does not have NaNs representing land/sea, will use supplied source sftlf file."
            )

        src_landmask = regrid_sftlf_landmask(src_sftlf_fp, src_init_ds, threshold)
        src_init_ds["mask"] = xr.where(src_landmask, mask_val, nan_val)

    return src_init_ds, dst_ds


def apply_wgs84(ds):
    """Function to add spatial_ref coordinate, CRS attributes, and CRS encodings to make CF-compliant metadata for the WGS84 CRS.

    Parameters
    ----------
    ds : xarray.Dataset
        Regridded dataset to add WGS84 CRS info to. Should be on -180 to 180 longitude scale.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with WGS84 CRS info added
    """
    try:
        # Try to access an existing spatial_ref coordinate.
        # If this doesn't raise an exception, the dataset probably has CRS info and should be returned as-is.
        # If this fails, the dataset probably has no CF-compliant CRS info and we will add it.
        spatial_ref_coord_ = ds.spatial_ref
        return ds

    except:

        # get CF-compliant crs attribute dict
        cf_crs = CRS.from_epsg(4326).to_cf()

        try:
            # create a spatial_ref coordinate, which is an empty array but has the CF-compliant crs attribute dict
            ds = ds.assign_coords({"spatial_ref": ([], np.array(0), cf_crs)})

            # add a second attribute "spatial_ref" identical to "crs_wkt" (this is redundant, but matches test rioxarray output)
            ds["spatial_ref"].attrs["spatial_ref"] = cf_crs["crs_wkt"]

            # manually link spatial_ref attributes to the data variable via "grid_mapping" encoding
            # assumes dataset will only have one data variable!
            var_id = get_var_id(ds)
            ds[var_id].encoding["grid_mapping"] = "spatial_ref"
            return ds

        except:
            return ds


def convert_units(ds):
    """Convert units of a dataset to more useful units. Hardcoded in this function for now.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to convert the units of

    Returns
    -------
    ds : xarray.Dataset
        Dataset with units converted to more useful units
    """
    # make sure units are more useful
    var_id = get_var_id(ds)

    if var_id in ["pr", "prsn", "snw"]:
        # precip
        ds[var_id] = units.convert_units_to(ds[var_id], "mm")

    elif var_id in ["tas", "tasmax", "tasmin"]:
        # temperature
        ds[var_id] = units.convert_units_to(ds[var_id], "degC")

    return ds


def rasdafy_dataset(ds):
    """Apply some tweaks to the data that make things better for Rasdaman ingestion.
    We want to make sure the axes are ordered (time, lon, lat), that the time axis is "unlimited",
    and that the latitutde axis is in decreasing order.
    We will also make units are more useful (e.g. mm precip instead of kg m-2 s-1).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to prepare for Rasdaman ingestion

    Returns
    -------
    ds : xarray.Dataset
        Dataset prepared for Rasdaman ingestion
    """
    # drop bnds dimension if it exists
    for bnd_dim in ["bnds", "nbnd", "lat_bnds", "lon_bnds", "time_bnds"]:
        ds = ds.drop_dims(bnd_dim, errors="ignore")

    var_id = get_var_id(ds)
    ds = ds.drop_dims([dim for dim in ds.dims if dim not in ["time", "lat", "lon"]])[
        [var_id]
    ]

    # make sure the latitude dim is in decreasing order
    if "lat" in ds.dims:
        if ds.lat.values[0] < ds.lat.values[-1]:
            ds = ds.sel(lat=slice(None, None, -1))

        # make sure the dims are ordered (time, lon, lat) for Rasdaman
        # (if present, using lat presence as a proxy for both)
        ds = ds.transpose("time", "lon", "lat")

    ds = convert_units(ds)

    return ds


def fix_attrs(ds):
    """Fix some attributes of the dataset to make

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to fix the attributes of

    Returns
    -------
    ds : xarray.Dataset
        Dataset with attributes fixed
    """
    # make sure longitude min and max attributes are set correctly
    if "lon" in ds.dims:
        ds["lon"].attrs["valid_max"] = 180
        ds["lon"].attrs["valid_min"] = -180

    # fix interpolation method attributes (could be remnants from GCM that don't match)
    var_id = get_var_id(ds)
    if "interp_method" in ds[var_id].attrs:
        # save parent interp method if present
        ds[var_id].attrs["parent_interp_method"] = ds[var_id].attrs["interp_method"]

    ds[var_id].attrs["interp_method"] = ds.attrs["regrid_method"]
    del ds.attrs["regrid_method"]

    if "grid" in ds.attrs:
        ds.attrs["parent_grid"] = ds.attrs["grid"]
        ds.attrs["grid"] = "0.9x1.25 finite volume grid (43x288 latxlon)"

    if "grid_label" in ds.attrs:
        ds.attrs["parent_grid_label"] = ds.attrs["grid_label"]
        del ds.attrs["grid_label"]

    # make sure standard names are consistent
    ds[var_id].attrs["long_name"] = variables[var_id]["name"]

    return ds


def write_retry_batch_file(regrid_batch_dir, errs):
    """Append each item in a list of filepaths to a text file. Lines are appended to the file if it already exists.
    If a collection of batch files are being simultaneously processed by this regrid.py script via multiple slurm jobs,
    a single text file will be generated that lists all files that failed the regridding process and can be retried.

    Parameters
    ----------
    regrid_batch_dir : str
        Directory containing the regridding batch files

    errs : list
        List of filepaths that failed the regridding process
    """
    retry_fn = Path(regrid_batch_dir).joinpath("batch_retry.txt")
    with open(retry_fn, "a") as f:
        for fp in errs:
            f.write(f"{fp}\n")


def parse_output_filename_times_from_file(fp):
    """Parse a date range in format YYYYMM-YYYYMM or YYYYMMDD-YYYYMMDD.

    Originally this function relied only on the timeframe string in the filenames.
    But that is not gauranteed to be correct so this function was refactored to rely on the time variable.

    Returns a list of strings representing times that should be used in all the output
    files of a given source file, since they are saved by year.

    Parameters
    ----------
    fp : pathlib.Path
        Path to file to parse the time range from

    Returns
    -------
    timerange_strings : list
        List of strings representing the time range of the dataset in the file
    """
    with xr.open_dataset(fp) as ds:
        if check_is_dayfreq(ds):
            # calendars get converted to noleap so start / end days of year are consistent
            start_day = "01"
            end_day = "31"
        elif check_is_monfreq(ds):
            start_day = end_day = ""

        years = np.unique(ds.time.dt.year.values)

        # first rule of CMIP6 - don't assume anything
        # instead of assuming all files have all months, just iterate
        timerange_strings = []
        for year in years:
            months = ds.time.sel(time=f"{year}").dt.month.values
            start_month = str(months[0]).zfill(2)
            end_month = str(months[-1]).zfill(2)
            tr_str = f"{year}{start_month}{start_day}-{year}{end_month}{end_day}"
            timerange_strings.append(tr_str)

    return timerange_strings


def regrid_dataset(fp, regridder, out_fp, src_mask=None, rasdafy=False):
    """Regrid a dataset using a regridder object initiated using the target grid with
    a latitude domain of 50N and up.

    Parameters
    ----------
    fp : pathlib.Path
        Path to file to be regridded
    regridder : xesmf.Regridder
        Regridder object initialized on source dataset that has the same grid as dataset as read from fp
    out_fp : pathlib.Path
        Path to write regridded file to
    src_mask : xarray.DataArray
        Mask for the source dataset, if available
    rasdafy : bool
        Whether to apply some tweaks to the dataset to make it better for Rasdaman ingestion

    Returns
    -------
    out_fp : pathlib.Path
        Path to output regridded file
    """
    # open the source dataset
    src_ds = xr.open_dataset(fp)
    # imposing this restriction will help with datasets that have just tons of
    # daily data in a single file, far more than we need e.g. back to 1850
    src_ds = src_ds.sel(time=slice("1950", "2100"))

    # add mask if not none
    if src_mask is not None:
        src_ds["mask"] = src_mask

    regrid_task = regridder(src_ds, keep_attrs=True)
    regrid_ds = regrid_task.compute()

    # if the variable is a fixed frequency variable, just write it as is without any time modifications
    # this should never occur because only daily and monthly frequency data should be regridded,
    # but technically possible if the prefect parameters ever include "fx", "Ofx", or "orog" frequencies
    if not any(fixed_freq_var in str(out_fp) for fixed_freq_var in ["sftlf", "sftof"]):
        regrid_ds = fix_time(regrid_ds, src_ds)

    # add CRS info
    if "lon" in regrid_ds.dims:
        regrid_ds = apply_wgs84(regrid_ds)

    # rasdafy the dataset
    if rasdafy:
        regrid_ds = rasdafy_dataset(regrid_ds)

    # fix attributes
    regrid_ds = fix_attrs(regrid_ds)
    # write
    out_fp = write_regridded_files(regrid_ds, out_fp)

    return out_fp


if __name__ == "__main__":
    # parse args
    (
        regrid_batch_fp,
        dst_fp,
        src_sftlf_fp,
        dst_sftlf_fp,
        out_dir,
        interp_method,
        rasdafy,
        no_clobber,
    ) = parse_args()

    # get the paths of files to regrid from the batch file
    with open(regrid_batch_fp) as f:
        lines = f.readlines()
    src_fps = [Path(line.replace("\n", "")) for line in lines]

    # cannot open / crop if dataset is irregular in lat / lon.
    # I think we should shift to simply regridding to the destination grid,
    # which we should ensure has the domain we want.
    dst_ds = xr.open_dataset(dst_fp, chunks={"time": 100})

    # open first source file for initializing regridder
    src_init_ds = xr.open_dataset(src_fps[0])

    # if sftlf_fp is provided, assume it is needed for masking
    if src_sftlf_fp:
        assert (
            dst_sftlf_fp
        ), "If source sftlf file is provided, destination sftlf file must also be provided"
    # assume dst_sftlf_fp will ALWAYS be provided if we are dealing with masked variables
    if dst_sftlf_fp:
        src_init_ds, dst_ds = prep_for_landsea(
            src_init_ds,
            dst_ds,
            src_sftlf_fp,
            dst_sftlf_fp,
        )

    # use one of the source files to be regridded and the destination grid file to create a regridder object
    regridder = init_regridder(src_init_ds, dst_ds, interp_method)

    # now iterate over files in batch file and run the regridding
    print(f"Regridding {len(src_fps)} files", flush=True)
    tic = time.perf_counter()

    results = []
    errs = []
    no_clobbers = []

    for fp in src_fps:

        out_fp = generate_regrid_filepath(fp, out_dir)
        # make sure the parent dirs exist
        out_fp.parent.mkdir(exist_ok=True, parents=True)

        # remove date from filename about to be regridded
        #  and list all existing files created from the source file being examined
        nodate_out_fn = "_".join(out_fp.name.split(".nc")[0].split("_")[:-1])
        existing_fps = list(out_fp.parent.glob(f"{nodate_out_fn}*.nc"))

        # get a list of yearly time ranges from the multi-year source filename
        expected_filename_time_ranges = parse_output_filename_times_from_file(fp)

        # search existing filenames for the time range strings
        # if all time range strings are found in existing filenames, and no_clobber=True, then skip regridding
        if (
            all(
                [
                    any(time_str in fp.name for fp in existing_fps)
                    for time_str in expected_filename_time_ranges
                ]
            )
            and no_clobber
        ):
            no_clobbers.append(str(fp))
        else:
            src_mask = None
            if "mask" in src_init_ds.data_vars:
                src_mask = src_init_ds["mask"]
            try:
                results.append(regrid_dataset(fp, regridder, out_fp, src_mask, rasdafy))
            except Exception as e:
                errs.append(str(fp))
                print(f"\nFILE NOT REGRIDDED: {fp}\n     Errors printed below:\n")
                traceback.print_exc()

    print(
        f"Regridding done, {len(results)} files regridded in {np.round((time.perf_counter() - tic) / 60, 1)}m"
    )

    if len(results) < len(src_fps):
        print("\nThe following files were NOT regridded because:\n")
        print("PROCESSING ERROR:", "\nPROCESSING ERROR: ".join(errs))
        if no_clobber and len(no_clobbers) > 0:
            print("\nThe following files were NOT regridded because:\n")
            print("OVERWRITE ERROR:", "\nOVERWRITE ERROR: ".join(no_clobbers))

    # if any filepaths failed to regrid due to errors, add them to a "batch_retry.txt" file to be optionally retried
    if len(errs) > 0:
        regrid_batch_dir = regrid_batch_fp.parent
        write_retry_batch_file(regrid_batch_dir, errs)
