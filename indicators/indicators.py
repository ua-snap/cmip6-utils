"""Script for computing a set of indicators for a given set of files.
This means a set of indicators that share the same source variable or variables, and for a given model and scenario.
Handling of missing data (e.g. a model-scenario-variable combination that does not exist) should be done outside of this script.

Usage:
    python indicators.py --indicators rx1day --model CESM2 --scenario ssp585 --input_dir /beegfs/CMIP6/kmredilla/cmip6_regridding/regrid --out_dir /beegfs/CMIP6/kmredilla/indicators
"""

import argparse
from pathlib import Path
import cftime
import numpy as np
import xarray as xr
import xclim.indices as xci
from xclim.indicators import atmos
from xclim.core.calendar import percentile_doy
from xclim.core.units import convert_units_to, to_agg_units
from xclim.indices.generic import threshold_count
import datetime

from config import *
from luts import *


def fix_pr_units(pr):
    """Fix precipitation units to mm/day if needed"""
    if pr.attrs["units"] == "mm":
        # assumes we are only dealing with daily data
        # this is just a check to make sure we have daily data by diff'ing the days
        assert np.sum(np.diff(pr.time.dt.day) == 1) / len(pr.time) > 0.9
        pr.attrs["units"] = "mm/day"
    return pr


def rx1day(pr):
    """'Max 1-day precip' - the max daily precip value recorded for a year.

    Args:
        pr (xarray.DataArray): daily total precip values

    Returns:
        Max 1-day precip for each year
    """
    pr = fix_pr_units(pr)
    out = xci.max_n_day_precipitation_amount(pr, freq="YS")
    out.attrs["units"] = "mm"

    return out


def su(tasmax):
    """'Summer days' - the number of days with tasmax above 25 C

    Args:
        tasmax (xarray.DataArray): daily maximum temperature values for a year

    Returns:
        Number of summer days for each year
    """
    return xci.tx_days_above(tasmax, "25 degC", freq="YS")


def dw(tasmin):
    """'Deep winter days' - the number of days with tasmin below -30 C

    Args:
        tasmin (xarray.DataArray): daily minimum temperature values for a year

    Returns:
        Number of deep winter days for each year
    """
    return xci.tn_days_below(tasmin, thresh="-30 degC", freq="YS")


def ftc(tasmax, tasmin):
    """'Freeze-thaw days' or 'Daily freeze thaw cycles' in xclim.
    The number of days with a freeze-thaw cycle.
    Here, a freeze-thaw cycle is defined as a day where maximum daily temperature is
    above 0°C a given threshold and minimum daily temperature is at or below 0°C

    Args:
        tasmax (xarray.DataArray): daily maximum temperature values for a year
        tasmin (xarray.DataArray): daily minimum temperature values for a year

    Returns:
        Number of freeze-thaw days for each year
    """
    ftc = atmos.daily_freezethaw_cycles(
        tasmin,
        tasmax,
        thresh_tasmin="0 degC",
        thresh_tasmax="0 degC",
        op_tasmin="<=",
        op_tasmax=">",
    )

    # change units from "days" to "d" to prevent decode_cf issues when opening ftc output later on
    # change dtype from float to integer
    ftc.attrs["units"] = "d"
    ftc = ftc.astype(np.int64)

    return ftc


def take_sorted(arr, axis, idx):
    """Helper function for the 'hot day' and 'cold day' indices to slice a numpy array after sorting it. Done in favor of fixed, []-based indexing.

    Args:
        arr (numpy.ndarray): array
        axis (int): axis to sort and slice according to
        idx (int): index value to slice arr at across all other axes

    Returns:
        array of values at position idx of arr sorted along axis
    """
    # np.sort defaults to ascending...
    return np.take(np.sort(arr, axis), idx, axis)


def hd(tasmax):
    """'Hot Day' - the 6th hottest day of the year

    Args:
        tasmax (xarray.DataArray): daily maximum temperature values for a year

    Returns:
        Hot Day values for each year
    """

    def func(tasmax):
        # hd is simply "6th hottest" day
        return tasmax.reduce(take_sorted, dim="time", idx=-6)

    out = tasmax.resample(time="1Y").map(func)
    out.attrs["units"] = "C"
    out.attrs["comment"] = "'hot day': 6th hottest day of the year"

    return out


def cd(tasmin):
    """'Cold Day' - the 6th coldest day of the year

    Args:
        tasmin (xarray.DataArray): daily minimum temperature values

    Returns:
        Cold Day values for each year
    """

    def func(tasmin):
        # cd is simply "6th coldest" day
        return tasmin.reduce(take_sorted, dim="time", idx=5)

    out = tasmin.resample(time="1Y").map(func)
    out.attrs["units"] = "C"
    out.attrs["comment"] = "'cold day': 6th coldest day of the year"

    return out


def rx5day(pr):
    """'Max 5-day precip' - the max 5-day precip value recorded for a year.

    Args:
        pr (xarray.DataArray): daily total precip values

    Returns:
        Max 5-day precip for each year
    """
    pr = fix_pr_units(pr)
    out = xci.max_n_day_precipitation_amount(pr, 5, freq="YS")
    out.attrs["units"] = "mm"

    return out


def wsdi(tasmax, hist_da):
    """'Warm spell duration index' - Annual count of occurrences of at least 5 consecutive days with daily max T above 90th percentile of historical values for the date

    Args:
        tasmax (xarray.DataArray): daily maximum temperature values
        hist_da (xarray.DataArray): historical daily maximum temperature values

    Returns:
        Warm spell duration index for each year
    """
    tasmax_per = percentile_doy(hist_da, per=90).sel(percentiles=90)

    return xci.warm_spell_duration_index(
        tasmax, tasmax_per, window=6, freq="YS"
    ).drop_vars("percentiles")


def csdi(tasmin, hist_da):
    """'Cold spell duration index' - Annual count of occurrences of at least 5 consecutive days with daily min T below 10th percentile of historical values for the date

    Args:
        tasmin (xarray.DataArray): daily minimum temperature values for a year
        hist_da (xarray.DataArray): historical daily minimum temperature values

    Returns:
        Cold spell duration index for each year
    """
    tasmin_per = percentile_doy(hist_da, per=10).sel(percentiles=10)
    return xci.cold_spell_duration_index(
        tasmin, tasmin_per, window=6, freq="YS"
    ).drop_vars("percentiles")


def r10mm(pr):
    """'Heavy precip days' - number of days in a year with over 10mm of precip

    Args:
        pr (xarray.DataArray): daily total precip values

    Returns:
        Number of heavy precip days for each year
    """
    # code based on xclim.indices._threshold.tg_days_above
    pr = fix_pr_units(pr)
    thresh = "10 mm/day"
    thresh = convert_units_to(thresh, pr)
    f = threshold_count(pr, ">", thresh, freq="YS")
    return to_agg_units(f, pr, "count")


def cwd(pr):
    """'Consecutive wet days' - number of the most consecutive days with precip > 1 mm

    Args:
        pr (xarray.DataArray): daily total precip values

    Returns:
        Max number of consecutive wet days for each year
    """
    pr = fix_pr_units(pr)
    return xci.maximum_consecutive_wet_days(pr, thresh=f"1 mm/day", freq="YS")


def cdd(pr):
    """'Consecutive dry days' - number of the most consecutive days with precip < 1 mm

    Args:
        pr (xarray.DataArray): daily total precip values

    Returns:
        Max number of consecutive dry days for each year
    """
    pr = fix_pr_units(pr)
    return xci.maximum_consecutive_dry_days(pr, thresh=f"1 mm/day", freq="YS")


def convert_times_to_years(time_da):
    """Convert the time values in a time axis (DataArray) to integer year values. Handles cftime types and numpy.datetime64."""
    if time_da.values.dtype == np.dtype("<M8[ns]"):
        # just a double check that we have nanosecond precision since we will divide by 1e9 to get seconds
        assert len(str(time_da.values[0])) == 29
        cftimes = [
            cftime.num2date(t / 1e9, "seconds since 1970-01-01")
            for t in time_da.values.astype(int)
        ]
    elif isinstance(
        time_da.values[0],
        cftime._cftime.Datetime360Day,
    ) or isinstance(
        time_da.values[0],
        cftime._cftime.DatetimeNoLeap,
    ):
        cftimes = time_da.values

    years = [t.year for t in cftimes]

    return years


def compute_indicator(da, idx, coord_labels, kwargs={}):
    """Summarize a DataArray according to a specified index / aggregation function

    Args:
        da (xarray.DataArray): the DataArray object containing the base variable data to be summarized according to aggr
        idx (str): String corresponding to the name of the indicator to compute (assumes value is equal to the name of the corresponding global function)
        coord_labels (dict): dict with model and scenario as keys for labeling resulting xarray dataset coordinates.
        kwargs (dict): additional arguments for the index function being called

    Returns:
        A new data array with dimensions year, latitude, longitude, in that order containing the summarized information
    """
    # dask array must be computed here in order to change nodata values
    new_da = (
        globals()[idx](da, **kwargs)
        # .transpose("time", "lat", "lon")
        # .reset_coords(["longitude", "latitude", "height"], drop=True)
    ).compute()
    new_da.name = idx
    # get the nodata mask from first time slice
    nodata = np.broadcast_to(np.isnan(da.sel(time=da["time"].values[0])), new_da.shape)
    # remask, because xclim switches nans to 0
    # xclim is inconsistent about the types returned.
    if new_da.dtype in [np.int32, np.int64]:
        new_da.values[nodata] = -9999
    else:
        new_da.values[nodata] = np.nan

    new_dims = list(coord_labels.keys())
    new_da = new_da.assign_coords(coord_labels).expand_dims(new_dims)
    # convert the time dimension to integer years instead of CF time objects
    years = convert_times_to_years(new_da.time)
    new_da = new_da.rename({"time": "year"}).assign_coords({"year": years})

    return new_da


def run_compute_indicators(fp_di, indicators, coord_labels, kwargs={}):
    """Open connections to data files for a particular model variable, scenario, and model and compute all requested indicators.

    Args:
        fp_di (path-like): Dict of paths to the files for the variables required for creating the indicators variables in indicators
        indicators (list): indicators to derive using data in provided filepaths
        coord_labels (dict): dict with model and scenario as keys for labeling resulting xarray dataset coordinates.

    Returns:
        summary_das (tuple): tuple of the form (da, index, scenario, model), where da is a DataArray with dimensions of year (summary year), latitude (lat) and longitude (lon)
    """
    out = []
    for idx in indicators:
        if idx in ["ftc"]:
            with xr.open_mfdataset(fp_di["tasmin"]) as tasmin_ds:
                with xr.open_mfdataset(fp_di["tasmax"]) as tasmax_ds:
                    kwargs = {"tasmin": tasmin_ds["tasmin"]}
                    out.append(
                        compute_indicator(
                            da=tasmax_ds["tasmax"],
                            idx=idx,
                            coord_labels=coord_labels,
                            kwargs=kwargs,
                        )
                    )

        elif idx in ["wsdi", "csdi"]:
            # get normal years of the variable data as single dataset
            hist_fp_di = find_var_files_and_create_fp_dict(
                model, "historical", idx_varid_lu[idx], input_dir
            )
            hist_fps = []
            for year in normal_years:
                for fp in hist_fp_di[
                    idx_varid_lu[idx][0]
                ]:  # 0 index since there is only one var used in these indicators
                    if f"regrid_{year}" in fp.name:
                        hist_fps.append(fp)

            with xr.open_mfdataset(
                hist_fps,
                chunks="auto",
                parallel=True,
            ) as hist_ds:

                # open all years of the the variable data as a single dataset
                with xr.open_mfdataset(
                    fp_di[
                        idx_varid_lu[idx][0]
                    ],  # 0 index since there is only one var used in these indicators
                    chunks="auto",
                    parallel=True,
                ) as var_ds:

                    if "height" in var_ds.coords:
                        var_ds = var_ds.drop_vars("height")

                    kwargs = {
                        "hist_da": hist_ds[idx_varid_lu[idx][0]]
                    }  # 0 index since there is only one var used in these indicators
                    out.append(
                        compute_indicator(
                            da=var_ds[
                                idx_varid_lu[idx][0]
                            ],  # 0 index since there is only one var used in these indicators
                            idx=idx,
                            coord_labels=coord_labels,
                            kwargs=kwargs,
                        )
                    )

        else:
            # this will assume there is only one input variable
            assert len(fp_di.keys()) == 1
            var_id = list(fp_di.keys())[0]
            with xr.open_mfdataset(fp_di[var_id]) as ds:
                out.append(
                    compute_indicator(
                        da=ds[var_id],
                        idx=idx,
                        coord_labels=coord_labels,
                        kwargs=kwargs,
                    )
                )

    return out


def build_attrs(
    indicator, scenario, model, start_year, end_year, lat_min, lat_max, lon_min, lon_max
):
    """Build standardized attribute dictionarys for computed indicator datasets. This function uses lookup tables imported from indicators/luts.py.

    Args:
        indicator (str): indicator id
        model (str): model id
        scenario (str): scenario id
        start_year (str): first year of dataset
        end_year (str): last year of dataset
        lat_min (str): minimum latitude of dataset
        lat_max (str): maximum latitude of dataset
        lon_min (str): minimum longitude of dataset
        lon_max (str): maximum longitude of dataset

    Returns:
        global_attrs, var_coord_attrs (tuple): tuple of global and variable/coordinate attribute dictionarys
    """
    # test units to determine NA value TODO: revisit this if there are any other integer units besides "days"
    if units_lu[indicator] != "d":
        fill_value = "NaN"
    else:
        fill_value = "-9999"

    # build global attribute dict for the whole dataset:
    global_attrs = {
        "title": f"{indicator_lu[indicator]['title']}, {start_year}-{end_year}: {model}-{scenario}",
        "author": "Scenarios Network for Alaska and Arctic Planning (SNAP), International Arctic Research Center, University of Alaska Fairbanks",
        "creation_date": datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "email": "uaf-snap-data-tools@alaska.edu",
        "website": "https://uaf-snap.org/",
        "scenario": f"{scenario}",
        "model": f"{model}",
        "institution": f"{model_lu[model]['institution']}",
        "institution_name": f"{model_lu[model]['institution_name']}",
    }

    # but attribute dict for individual coordinates and variables:
    var_coord_attrs = {
        "lat": {
            "name": "latitude",
            "units": "degrees north",
            "lat_max": f"{lat_max}",
            "lat_min": f"{lat_min}",
        },
        "lon": {
            "name": "longitude",
            "units": "degrees east",
            "lon_max": f"{lon_max}",
            "lon_min": f"{lon_min}",
        },
        "year": {
            "start_year": start_year,
            "end_year": end_year,
        },
        "scenario": {"description": "Forcing scenario used to drive model"},
        "model": {"description": "Source model of input climate data"},
        indicator: {
            "long_name": indicator_lu[indicator]["long_name"],
            "units": f"{units_lu[indicator]}",
            "description": indicator_lu[indicator]["description"],
        },
    }
    return global_attrs, var_coord_attrs, fill_value


def find_and_replace_attrs(idx_ds, model, scenario, **kwargs):
    """Replace original indicator dataset attributes with standardized attribute dictionarys.
    This function does a simple check to make sure all original variables/coordinates are in the standardized attribute dict,
    and drops the 'height' coordinate, if it exists.

    Args:
        idx_ds (xarray.Dataset): computed indicator dataset with original attributes
        model (str): model id
        scenario (str): scenario id
        kwargs (dict): basic kwargs dictionary to supply model and scenario

    Returns:
        idx_ds (xarray.Dataset): computed indicator dataset with standardized attributes
    """

    ds_vars = list(idx_ds.variables)
    idx = list(idx_ds.data_vars)[0]

    # remove height coord if it exists
    if "height" in ds_vars:
        ds_vars.remove("height")
        idx_ds = idx_ds.reset_coords(names="height", drop=True)

    # get dataset values and build attrs
    start_year, end_year, lat_min, lat_max, lon_min, lon_max = (
        idx_ds.year.values.min().astype(str),
        idx_ds.year.values.max().astype(str),
        idx_ds.lat.values.min().astype(str),
        idx_ds.lat.values.max().astype(str),
        idx_ds.lon.values.min().astype(str),
        idx_ds.lon.values.max().astype(str),
    )
    global_attrs, var_coord_attrs, fill_value = build_attrs(
        idx,
        scenario,
        model,
        start_year=start_year,
        end_year=end_year,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )
    new_vars = list(var_coord_attrs.keys())
    # need to add this one in as it should always be present in inputs
    if "spatial_ref" in idx_ds.variables:
        new_vars.append("spatial_ref")

    # test for presence of all original ds vars (excluding height) in the new attrs
    if False in [i in new_vars for i in ds_vars]:
        print(
            "Not all original dataset variables (excluding height) are accounted for in new standardized variables! Process aborted."
        )
        raise Exception(
            "Not all original dataset variables (excluding height) are accounted for in new standardized variables! Process aborted."
        )
    else:
        # replace global attrs
        idx_ds.attrs = global_attrs
        # replace variable and coordinate attributes
        for var in var_coord_attrs:
            idx_ds[var].attrs = var_coord_attrs[var]

        # make sure _FillValue is set in .encoding
        idx_ds[idx].encoding["_FillValue"] = fill_value
    return idx_ds


def check_varid_indicator_compatibility(indicators, var_ids):
    """Check that all of the indicators to be processed use the same variables"""
    try:
        assert all([idx_varid_lu[idx] == var_ids for idx in indicators])
    except AssertionError:
        raise Exception(
            f"Incompatible variables ({var_ids}) and indicators ({indicators}) encountered."
        )


def find_var_files_and_create_fp_dict(model, scenario, var_ids, input_dir):
    """Check that input files exist in the input directory. Output a dictionary of filepaths."""

    # Lookup correct "day" frequency for each var id by searching for substring - this should grab "0day" and "SIday"
    # We build dicts for frequency and filepath to allow for possibility of more than one variable
    freq_di = {
        var_id: [i for i in varid_freqs[var_id] if "day" in i] for var_id in var_ids
    }

    fp_di = {
        var_id: list(
            input_dir.joinpath(
                f"{model}/{scenario}/{freq_di[var_id][0]}/{var_id}"
            ).glob("*.nc")
        )
        for var_id in var_ids
    }
    # Check if there are files found in the input directory for each variable needed
    # List variables that are missing files
    missing_var_ids = []
    for k in fp_di:
        if len(fp_di[k]) == 0:
            missing_var_ids.append(k)
    for var_id in missing_var_ids:
        print(f"File not found in input directory: {fp_di[var_id]}. Process aborted.")
        raise Exception(
            f"File not found in input directory: {fp_di[var_id]}. Process aborted."
        )

    return fp_di


def generate_base_kwargs(model, scenario, indicators, var_ids, input_dir):
    """Function for creating some kwargs for the run_compute_indicators function.
    Contains a validation routine to ensure input files exist, and attempts to copy them from the backup directory if they don't.

    Args:
        model (str): model name as used in filepaths
        scenario (str): scenario name as used in filepaths
        indicators (list): indicators to derive using data in provided filepaths
        var_ids (list): list of CMIP6 variable IDs needed for that
        input_dir (pathlib.Path): path to main directory containing regridded files
    """
    check_varid_indicator_compatibility(indicators, var_ids)

    fp_di = find_var_files_and_create_fp_dict(model, scenario, var_ids, input_dir)

    coord_labels = dict(
        scenario=scenario,
        model=model,
    )
    kwargs = dict(
        fp_di=fp_di,
        indicators=indicators,
        coord_labels=coord_labels,
    )

    return kwargs


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indicators",
        type=str,
        help="' '-separated list of indicators to compute, in quotes",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="name of model, as used in filepaths",
        required=True,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="name of scenario, as used in filepaths",
        required=True,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to input directory having filepath structure <model>/<scenario>/day/<variable ID>/<files>",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to directory where indicators data should be written",
        required=True,
    )
    parser.add_argument(
        "--no-clobber",
        action="store_true",
        default=False,
        help="Do not overwrite files if they exists in out_dir",
    )
    args = parser.parse_args()

    return (
        args.indicators.split(" "),
        args.model,
        args.scenario,
        Path(args.input_dir),
        Path(args.out_dir),
        args.no_clobber,
    )


if __name__ == "__main__":

    (
        indicators,
        model,
        scenario,
        input_dir,
        out_dir,
        no_clobber,
    ) = parse_args()
    # this is the part where we get all variable IDs for all indicators
    # assuming the first indicator uses the same as all others
    var_ids = idx_varid_lu[indicators[0]]
    kwargs = generate_base_kwargs(
        model=model,
        scenario=scenario,
        indicators=indicators,
        var_ids=var_ids,
        input_dir=input_dir,
    )

    indicators_ds = xr.merge(run_compute_indicators(**kwargs))

    # write each indicator to its own file for now
    out_fps_to_validate = []
    for idx in indicators_ds.data_vars:
        out_fp = out_dir.joinpath(
            model,
            scenario,
            idx,
            indicator_tmp_fp.format(indicator=idx, model=model, scenario=scenario),
        )
        # ensure this nested path exists
        out_fp.parent.mkdir(exist_ok=True, parents=True)
        # convert indicator array into its own dataset
        idx_ds = indicators_ds[idx].to_dataset()
        # standardize the attributes
        idx_ds_out = find_and_replace_attrs(idx_ds, model, scenario, **kwargs)
        # write
        idx_ds_out.to_netcdf(out_fp)
        # add filepath to list for validation
        out_fps_to_validate.append(out_fp)
