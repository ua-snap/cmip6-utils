"""Module for QC functions for regridded data.
"""

import concurrent.futures
from pathlib import Path
import cftime
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pandas.errors import OutOfBoundsDatetime
from regrid import (
    generate_regrid_filepath,
    parse_output_filename_times_from_file,
    convert_units,
    parse_cmip6_fp,
    prod_lat_slice,
    open_and_crop_dataset,
    get_var_id,
)
from config import model_inst_lu, prod_scenarios


def get_source_fps_from_batch_files(regrid_batch_dir):
    """Get all of the source filepaths from the batch files in the regrid batch directory."""
    source_fps = []
    for fp in regrid_batch_dir.glob("*.txt"):
        with open(fp) as f:
            source_fps.extend([Path(line.strip()) for line in f])

    return source_fps


def extract_params_from_src_filepath(src_fp):
    """Assumes this is a source CMIP6 filepath."""
    # .parts mapping for source: 7: model, 8: scenario, 10: frequency, 11: variable
    params = parse_cmip6_fp(src_fp)

    # the frequency could be something like Amon or Eday or whatever,
    #  but we only use "day" or "mon" for the regridding
    if "mon" in params["frequency"]:
        params["frequency"] = "mon"
    else:
        # it should be daily if not monthly, just double check it
        assert "day" in params["frequency"]
        params["frequency"] = "day"

    return params


def summarize_slurm_out_files(slurm_dir):
    """Read all .out files in the slurm directory, and summarize overwrite/processing errors.
    Write processing errors to the qc error file.
    Return another list with all file paths that were not processed, to be ignored from subsequent QC steps.
    """
    overwrite_lines = []
    error_lines = []
    fps_to_ignore = []

    for out_file in slurm_dir.glob("*.out"):
        with open(out_file, "r") as f:
            for line in f:
                if line.startswith("OVERWRITE ERROR") and line.endswith(".nc\n"):
                    overwrite_lines.append(line)
            for line in f:
                if line.startswith("PROCESSING ERROR") and line.endswith(".nc\n"):
                    error_lines.append(line)
                    fps_to_ignore.append(Path(line.split(" ")[-1].split("\n")[0]))
    if len(overwrite_lines) > 0:
        print(
            f"Warning: {len(overwrite_lines)} source files were not regridded because their output files already exist. The existing output files will be QC'd here anyway."
        )
    if len(error_lines) > 0:
        print(
            f"Error: {len(error_lines)} source files were not regridded due to processing errors. There are no outputs to QC. Check qc/qc_error.txt for source file paths."
        )
        _ = [print(line) for line in error_lines]

    return fps_to_ignore


def generate_expected_regrid_fps(src_fp, regrid_dir):
    """Generate expected regrid filepaths from a source filepath."""
    # build expected base file path from the source file path
    expected_base_fp = generate_regrid_filepath(src_fp, regrid_dir)
    base_timeframe = expected_base_fp.name.split("_")[-1].split(".nc")[0]
    # get a list of yearly time range strings from the multi-year source filename
    expected_filename_time_ranges = parse_output_filename_times_from_file(src_fp)
    # replace the timeframe in the base file path with the yearly time ranges, and add to expected_fps list
    expected_regrid_fps = []
    for yearly_timeframe in expected_filename_time_ranges:
        expected_fp = str(expected_base_fp).replace(base_timeframe, yearly_timeframe)
        expected_regrid_fps.append(Path(expected_fp))

    return expected_regrid_fps


def generate_regrid_fps_from_params(models, scenarios, vars, freqs, regrid_dir):
    """Expecting lists of models, scenarios, vars, and freqs (as supplied to the notebook)."""
    regrid_fps = []
    for model in models.split():
        for scenario in scenarios.split():
            for var in vars.split():
                for freq in freqs.split():
                    regrid_fps += list(
                        regrid_dir.glob(f"{model}/{scenario}/*{freq}/{var}/*.nc")
                    )
    return regrid_fps


def make_qc_file(output_directory):
    """Make a qc_directory and qc_error.txt file to save results."""
    qc_dir = output_directory.joinpath("qc")
    qc_dir.mkdir(exist_ok=True)
    error_file = qc_dir.joinpath("qc_error.txt")
    with open(error_file, "w") as e:
        pass
    return error_file


def file_min_max(fp):
    """Get file min and max values within the Arctic latitudes."""
    try:
        try:
            # using the h5netcdf engine because it seems faster and might help prevent pool hanging
            src_ds = xr.open_dataset(fp, engine="h5netcdf")
        except:
            # this seems to have only failed due to some files (KACE model) being written in netCDF3 format
            src_ds = xr.open_dataset(fp)

        var_id = get_var_id(src_ds)

        # handle regridded data being flipped
        if src_ds.lat[0] > src_ds.lat[-1]:
            lat_slicer = slice(90, 49)
        else:
            lat_slicer = slice(49, 90)

        src_ds_slice = src_ds.sel(lat=lat_slicer)

        src_ds_slice = convert_units(src_ds_slice)

        src_min, src_max = float(src_ds_slice[var_id].min()), float(
            src_ds_slice[var_id].max()
        )
        return {"file": str(fp), "min": src_min, "max": src_max}
    except:
        return {"file": None, "min": None, "max": None}


def subsample_files(fps, min_qc=20, max_qc=75):
    """Get a random sample of files for QC."""
    pct = 10
    pct_count = round(len(fps) * (pct / 100))
    if len(fps) <= min_qc:
        qc_files = fps
    elif pct_count >= max_qc:
        qc_files = random.sample(fps, max_qc)
    else:
        qc_files = random.sample(fps, pct_count)

    return qc_files


def compare_expected_to_existing_and_check_values(
    regrid_dir,
    regrid_batch_dir,
    slurm_dir,
    vars,
    freqs,
    models,
    scenarios,
    fps_to_ignore,
):
    """Iterate through model / scenario/ frequency/ variable combos, comparing data from expected file paths to existing file paths.
    If all expected files exist, check their values against source files.
    Writes error messages to qc error file, and returns a list of fps with errors for printing a summary message.
    """
    # set up lists to collect error text
    source_files_missing_regrids = []
    ds_errors = []
    value_errors = []

    src_fps = get_source_fps_from_batch_files(regrid_batch_dir)

    fps_to_ignore = summarize_slurm_out_files(slurm_dir)
    for fp in fps_to_ignore:
        if fp in src_fps:
            src_fps.remove(fp)

    existing_regrid_fps = generate_regrid_fps_from_params(
        models, scenarios, vars, freqs, regrid_dir
    )

    # we can subsample more files here because we don't need to review them visually
    subsample_files(existing_regrid_fps, max_qc=1000)

    # create dicts of min/max values for each regridded file and each source file
    regrid_min_max = {}
    src_min_max = {}

    # using multiprocessing, populate the dicts with min/max values for all regridded files and source files
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(file_min_max, existing_regrid_fps))

    # populate min/max dict / store dataset errors
    for result in results:
        regrid_min_max[result["file"]] = {
            "min": result["min"],
            "max": result["max"],
        }

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(file_min_max, src_fps))

    # populate min/max dict
    for result in results:
        src_min_max[result["file"]] = {
            "min": result["min"],
            "max": result["max"],
        }

    # create a list of expected regridded file paths from the source file paths
    for src_fp in src_fps:
        expected_regrid_fps = generate_expected_regrid_fps(src_fp, regrid_dir)

        # search existing files for the expected files, and if not found add text to appropriate error list
        # if all are found, run the final QC step to compare values

        if not all([fp in existing_regrid_fps for fp in expected_regrid_fps]):
            source_files_missing_regrids.append(str(src_fp))

        # only want to run the summaries on source files that have all expected regrids
        # call min/max from src dict
        src_min, src_max = (
            src_min_max[str(src_fp)]["min"],
            src_min_max[str(src_fp)]["max"],
        )
        # iterate thru expected filepaths
        for regrid_fp in expected_regrid_fps:
            # check if in keys, if not then the file did not open in file_min_max()
            if str(regrid_fp) in regrid_min_max.keys():
                # compare values
                regrid_min, regrid_max = (
                    regrid_min_max[str(regrid_fp)]["min"],
                    regrid_min_max[str(regrid_fp)]["max"],
                )
                if (src_max >= regrid_min >= src_min) and (
                    src_max >= regrid_max >= src_min
                ):
                    pass
                else:
                    value_errors.append(str(regrid_fp))
            else:
                ds_errors.append(str(regrid_fp))

    return ds_errors, value_errors


def get_matching_time_filepath(fps, test_date):
    """Find a file from a given list of raw CMIP6 filepaths that conatins the test date within the timespan in the filename."""
    matching_fps = []
    for fp in fps:
        start_str, end_str = fp.name.split(".nc")[0].split("_")[-1].split("-")
        start_str = f"{start_str}01" if len(start_str) == 6 else start_str
        # end date should be constructed as the end of month for monthly data
        #  (and should always be December??)
        end_str = f"{end_str}31" if len(end_str) == 6 else end_str
        format_str = "%Y%m%d"
        try:
            start_dt = pd.to_datetime(start_str, format=format_str)
            # it should be OK if end date is
            end_dt = pd.to_datetime(end_str, format=format_str)
        except OutOfBoundsDatetime:
            # we should not be regridding files with time values that cause this (2300 etc)
            continue

        if start_dt <= test_date < end_dt:
            matching_fps.append(fp)

    # there should only be one
    assert (
        len(matching_fps) == 1
    ), f"Test date {test_date} matched {len(matching_fps)} files (from {fps})."

    return matching_fps[0]


def generate_cmip6_filepath_from_regrid_filename(fn, cmip6_dir):
    """Get the path to the original CMIP6 filename from a regridded file name.

    Because the original CMIP6 filenames were split up during the processing,
    this method finds the original filename based on matching all possible attributes,
    then testing for inclusion of regrid file start date within the date range formed by the CMIP6 file timespan.
    """
    var_id, freq, model, scenario, _, timespan = fn.split(".nc")[0].split("_")
    institution = model_inst_lu[model]
    experiment_id = "ScenarioMIP" if scenario in prod_scenarios else "CMIP"
    # Construct the original CMIP6 filepath from the filename.
    # Need to use glob because of the "grid type" filename attribute that we do not have a lookup for.
    var_dir = cmip6_dir.joinpath(f"{experiment_id}/{institution}/{model}/{scenario}")
    glob_str = f"*/{freq}/{var_id}/*/*/{var_id}_{freq}_{model}_{scenario}_*.nc"
    candidate_fps = list(var_dir.glob(glob_str))

    assert (
        candidate_fps
    ), f"No files found for regridded file {fn} in {var_dir} with {glob_str}."

    start_str = timespan.split("-")[0]
    format_str = "%Y%m" if len(start_str) == 6 else "%Y%m%d"
    start_dt = pd.to_datetime(start_str, format=format_str)
    cmip6_fp = get_matching_time_filepath(candidate_fps, start_dt)

    return cmip6_fp


def plot_comparison(regrid_fp, cmip6_dir):
    """For a given regridded file, find the source file and plot side by side."""
    src_fp = generate_cmip6_filepath_from_regrid_filename(regrid_fp.name, cmip6_dir)
    src_ds = open_and_crop_dataset(src_fp, lat_slice=prod_lat_slice)
    # entire plotting function is inside this try block
    # if the dataset cannot be opened, just print a message instead of an error
    try:
        regrid_ds = xr.open_dataset(regrid_fp)
    except:
        print(f"Regridded dataset could not be opened: {regrid_fp}")

    # lat axis is flipped in regrid files
    src_lat_slice = slice(55, 75)
    regrid_lat_slice = slice(75, 55)
    lon_slice_src = slice(200, 240)
    lon_slice_regrid = slice(-160, -120)
    time_val = regrid_ds.time.values[0]
    var_id = src_ds.attrs["variable_id"]
    assert get_var_id(src_ds) == var_id, "Variable ID mismatch"
    assert get_var_id(regrid_ds) == var_id, "Variable ID mismatch"

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle(
        f"Variable: {var_id}     Model: {src_ds.attrs['source_id']}     Scenario: {src_ds.attrs['experiment_id']}"
    )

    # now, there are multiple possible time formats for the source dataset.
    # convert the chosen time value to that matching format for subsetting.
    sel_method = None
    if isinstance(src_ds.time.values[0], cftime._cftime.Datetime360Day):
        # It seems like monthly data use 16 for the day
        src_hour = src_ds.time.dt.hour[0]
        src_time = cftime.Datetime360Day(
            year=time_val.year,
            month=time_val.month,
            day=time_val.day,
            hour=src_hour,
        )
    elif isinstance(
        src_ds.time.values[0], pd._libs.tslibs.timestamps.Timestamp
    ) or isinstance(src_ds.time.values[0], np.datetime64):
        src_hour = src_ds.time.dt.hour[0].values.item()
        src_time = pd.to_datetime(
            f"{time_val.year}-{time_val.month}-{time_val.day}T{src_hour}:00:00"
        )
    else:
        if time_val not in src_ds.time.values:
            src_hour = src_ds.time.dt.hour[0]
            src_time = cftime.DatetimeNoLeap(
                year=time_val.year,
                month=time_val.month,
                day=time_val.day,
                hour=src_hour,
            )
        else:
            src_time = time_val
    if src_time not in src_ds.time.values:
        print(f"Sample timestamp not found in source file ({src_fp}). Using nearest.")
        # probably safe to just use nearest method in any event
        # since there can be incorrectly labeled frequency attributes
        sel_method = "nearest"
        if src_ds.attrs["frequency"] == "mon":
            # We expect that the file will be monthly if the source time chosen is not actually in the dataset
            # This is because we make the time values consistent in the regridded files,
            # and monthly source files might have used e.g. 16th day
            pass
        else:
            # OK this happens and there is nothing we can do about the source data not having consistent attributes.
            # Don't need to fail. Just print a message.
            print("Expected monthly file but frequency attribute does not match.")

    # ensure units are consistent with regridded dataset
    src_ds = convert_units(src_ds)

    # get a vmin and vmax from src dataset to use for both plots, if a map
    try:
        vmin = (
            src_ds[var_id]
            .sel(time=src_time, method=sel_method)
            .sel(lat=src_lat_slice, lon=lon_slice_src)
            .values.min()
        )
        vmax = (
            src_ds[var_id]
            .sel(time=src_time, method=sel_method)
            .sel(lat=src_lat_slice, lon=lon_slice_src)
            .values.max()
        )
    except:
        print("Error getting vmin and vmax values from source data.")

    try:  # maps
        src_ds[var_id].sel(time=src_time, method=sel_method).sel(
            lat=src_lat_slice, lon=lon_slice_src
        ).plot(ax=axes[0], vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Source dataset (timestamp: {src_time})")
        regrid_ds[var_id].sel(time=time_val).sel(
            lat=regrid_lat_slice,
            lon=lon_slice_regrid,
            # explitictly set the x axis to be the standard longitude for regridded data
            #  because grid is (confusingly) oriented time, lon, lat for rasdaman
        ).plot(ax=axes[1], vmin=vmin, vmax=vmax, x="lon")
        axes[1].set_title(f"Regridded dataset (timestamp: {time_val})")
        axes[1].set_xlabel("longitude [standard]")
        plt.show()

    except:  # histograms
        src_ds[var_id].sel(time=src_time, method=sel_method).sel(
            lat=src_lat_slice, lon=lon_slice_src
        ).plot(ax=axes[0])
        axes[0].set_title(f"Source dataset (timestamp: {src_time})")
        regrid_ds[var_id].sel(time=time_val).sel(
            lat=regrid_lat_slice, lon=lon_slice_regrid
        ).plot(ax=axes[1])
        axes[1].set_title(f"Regridded dataset (timestamp: {time_val})")

    plt.show()
