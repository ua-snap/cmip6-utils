"""Script for performing QC checks on files produced via the regridding pipeline and their individual slurm job output files.
This script uses the regridding batch files as a "to-do list" of files to check.
QC errors are written to {output_directory}/qc/qc_error.txt, and are summarized in print statements. 

Usage:
    python qc.py --output_directory /center1/CMIP6/jdpaul3/regrid/cmip6_regridding --vars 'pr ta' --freqs 'mon day'
"""

import argparse
from multiprocessing import Pool, set_start_method
import xarray as xr
from datetime import datetime
from pathlib import Path
from regrid import (
    generate_regrid_filepath,
    parse_output_filename_times_from_file,
    convert_units,
    parse_cmip6_fp,
)


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


def compare_expected_to_existing_and_check_values(
    regrid_dir,
    regrid_batch_dir,
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
