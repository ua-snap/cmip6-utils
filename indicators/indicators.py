"""Script for computing a set of indicators for a given set of files. 
This means a set of indicators that share the same source variable or variables, and for a given model and scenario.
Handling of missing data (e.g. a model-scenario-variable combination that does not exist) should be done outside of this script.

Usage: 
    python indicators.py --indicators rx1day --model CESM2 --scenario ssp585 --input_dir /center1/CMIP6/kmredilla/cmip6_regridding/regrid --backup_dir /beegfs/CMIP6/arctic-cmip6/regrid --out_dir /center1/CMIP6/kmredilla/indicators
"""

import argparse
from pathlib import Path
import cftime
import numpy as np
import xarray as xr
import xclim.indices as xci
import shutil
import sys
import datetime

# needed for other indicators but not yet
# from xclim.core.calendar import percentile_doy
# from xclim.core.units import convert_units_to, to_agg_units
# from xclim.indices.generic import threshold_count
from xclim.indicators import atmos  # , icclim
from config import *
from luts import idx_varid_lu


def rx1day(pr):
    """'Max 1-day precip' - the max daily precip value recorded for a year.

    Args:
        pr (xarray.DataArray): daily total precip values

    Returns:
        Max 1-day precip for each year
    """
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
        tasmin (xarray.DataArray): daily maximum temperature values for a year

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
        tasmin (xarray.DataArray): daily maximum temperature values for a year

    Returns:
        Number of freeze-thaw days for each year
    """
    return atmos.daily_freezethaw_cycles(
        tasmin,
        tasmax,
        thresh_tasmin="0 degC",
        thresh_tasmax="0 degC",
        op_tasmin="<=",
        op_tasmax=">",
    )


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
    new_da = (
        globals()[idx](da, **kwargs)
        # .transpose("time", "lat", "lon")
        # .reset_coords(["longitude", "latitude", "height"], drop=True)
    )
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


def check_varid_indicator_compatibility(indicators, var_ids):
    """Check that all of the indicators to be processed use the same variables"""
    try:
        assert all([idx_varid_lu[idx] == var_ids for idx in indicators])
    except AssertionError:
        raise Exception(
            f"Incompatible variables ({var_ids}) and indicators ({indicators}) encountered."
        )


def validate_outputs(indicators, out_fps_to_validate):
    """Run some validations on a list of output filepaths."""
    # first validate that indicator input arguments are reflected in the number of output files, and their filenames.
    if len(indicators) == len(out_fps_to_validate):
        fp_inds = [fp.parts[-1].split("_")[0] for fp in out_fps_to_validate]
        if set(fp_inds).issubset(indicators):
            print("Success: File written for each indicator.")
        else:
            print("Fail: Missing indicator files. Check output directory.")
    else:
        print(
            "Fail: Number of indicators and number of output files not equal. Possible missing indicator files, check output directory."
        )

    # validate that files were modified in the last 10 minutes
    # this might be useful info if we are overwriting existing indicator files, to make sure we have actually created a new file
    # may need to be adjusted based on real processing times
    for fp in out_fps_to_validate:
        mod_time = datetime.datetime.fromtimestamp(Path(fp).stat().st_mtime)
        elapsed = datetime.datetime.now() - mod_time
        if elapsed.seconds < 600:
            print(f"Success: File {str(fp)} was modified in last 10 minutes.")
        else:
            print(
                f"Fail: File {str(fp)} was modified over 10 minutes ago. If you are trying to overwrite an existing indicator file, it may not have worked."
            )

    # next validate that xarray can open each one of the output files.
    for fp in out_fps_to_validate:
        try:
            xr.open_dataset(fp)
            print("Success: File could be opened by xarray.")
        except:
            print("Fail: File could not be opened by xarray.")


def find_var_files_and_create_fp_dict(model, scenario, var_ids, input_dir, backup_dir):
    """Check that input files exist in the input directory. If not, check the backup directory. Output a dictionary of filepaths."""
    # TO-DO: the frequency, currently "day", is hard-coded, although for future indicators
    #  that rely on variables in other domains (e.g. ocean) that have other frequencies
    #  (e.g. "Oday") this will fail. Perhaps use a lookup table to identify variables that don't use "day"
    #  and run if/then routine to use correct frequency
    frequency = "day"

    # We build a dict to allow for possibility of more than one variable
    fp_di = {
        var_id: list(
            input_dir.joinpath(f"{model}/{scenario}/{frequency}/{var_id}").glob("*.nc")
        )
        for var_id in var_ids
    }
    # Check if there are files found in the input directory for each variable needed
    # List variables that are missing files
    missing_var_ids = []
    for k in fp_di:
        if len(fp_di[k]) == 0:
            missing_var_ids.append(k)
    # If there are variables with missing files, check the backup directory
    # Again we build a dict to allow for possibility of more than one variable with missing files
    if len(missing_var_ids) > 0:
        bu_fp_di = {
            var_id: list(
                backup_dir.joinpath(f"{model}/{scenario}/{frequency}/{var_id}").glob(
                    "*.nc"
                )
            )
            for var_id in missing_var_ids
        }
        # List variables that are missing files
        bu_missing_var_ids = []
        for k in bu_fp_di:
            if len(bu_fp_di[k]) == 0:
                bu_missing_var_ids.append(k)
        # If there are still variables with missing files, throw error that lists the missing files
        if len(bu_missing_var_ids) > 0:
            raise Exception(
                f"Fail: No files found in input directory or backup directory for model: {model}, scenario: {scenario}, frequency: {frequency}, variable(s): {bu_missing_var_ids}"
            )
        # If the files are found in backup directory, attempt to copy the entire tree to the input directory
        #  and add the filepaths to the filepath dictionary output
        else:
            for var_id in missing_var_ids:
                print(
                    f"No files found in input directory for variable: {var_id}, attempting to copy from backup directory..."
                )
                try:
                    shutil.copytree(
                        backup_dir.joinpath(f"{model}/{scenario}/{frequency}/{var_id}"),
                        input_dir.joinpath(f"{model}/{scenario}/{frequency}/{var_id}"),
                    )
                    fp_di[var_id] = list(
                        input_dir.joinpath(
                            f"{model}/{scenario}/{frequency}/{var_id}"
                        ).glob("*.nc")
                    )
                    no_files_copied = len(fp_di[var_id])
                    if no_files_copied > 0:
                        print(
                            f"Success: {str(no_files_copied)} files successfully copied to input directory for variable: {var_id}."
                        )
                    else:
                        print(
                            f"Fail: No files copied to input directory for variable: {var_id}."
                        )
                except:
                    raise Exception(
                        f"Fail: Could not copy files from backup directory to input directory for variable {var_id}. Processing aborted."
                    )

    return fp_di


def generate_base_kwargs(model, scenario, indicators, var_ids, input_dir, backup_dir):
    """Function for creating some kwargs for the run_compute_indicators function.
    Contains a validation routine to ensure input files exist, and attempts to copy them from the backup directory if they don't.

    Args:
        model (str): model name as used in filepaths
        scenario (str): scenario name as used in filepaths
        indicators (list): indicators to derive using data in provided filepaths
        var_ids (list): list of CMIP6 variable IDs needed for that
        input_dir (pathlib.Path): path to main directory containing regridded files
        backup_dir (pathlib.Path): path to backup directory containing regridded files
    """
    check_varid_indicator_compatibility(indicators, var_ids)

    fp_di = find_var_files_and_create_fp_dict(
        model, scenario, var_ids, input_dir, backup_dir
    )

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
        "--backup_dir",
        type=str,
        help="Path to backup input directory having filepath structure <model>/<scenario>/day/<variable ID>/<files>",
        required=True,
        default=str(cmip6_dir.parent.joinpath("regrid")),
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
        Path(args.backup_dir),
        Path(args.out_dir),
        args.no_clobber,
    )


if __name__ == "__main__":
    (
        indicators,
        model,
        scenario,
        input_dir,
        backup_dir,
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
        backup_dir=backup_dir,
    )

    indicators_ds = xr.merge(run_compute_indicators(**kwargs)).compute()
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
        # write
        indicators_ds[idx].to_dataset().to_netcdf(out_fp)
        # add filepath to list for validation
        out_fps_to_validate.append(out_fp)

    # validate outputs
    validate_outputs(indicators, out_fps_to_validate)
