"""Script for performing QC checks on files produced via the regridding pipeline and their individual slurm job output files.
This script uses the regridding batch files as a "to-do list" of files to check.
QC errors are written to {output_directory}/qc/qc_error.txt, and are summarized in print statements. 

Usage:
    python qc.py --output_directory /center1/CMIP6/jdpaul3/regrid/cmip6_regridding --vars 'pr ta'
"""

import argparse
import xarray as xr
from pathlib import Path
from regrid import (
    generate_regrid_filepath,
    parse_cmip6_fp,
    parse_output_filename_times_from_timeframe,
)


def get_source_fps_from_batch_files(regrid_batch_dir, var):
    """For a given variable, use the batch files to get filenames of all source files that should have been regridded."""
    # Return a list of all files inside *.txt that start with variable name
    source_fps = []
    for fp in regrid_batch_dir.glob("*.txt"):
        with open(fp) as f:
            # Return a list of all the files inside f with a basename that starts with args.variable.
            source_fps.extend(
                [
                    Path(line.strip())
                    for line in f
                    if Path(line.strip()).name.startswith(var + "_")
                ]
            )
    return source_fps


def summarize_slurm_out_files(slurm_dir, error_file):
    """Read all .out files in the slurm directory, and summarize overwrite/processing errors.
    Write processing errors to the qc error file.
    Return another list with all file paths that were not processed, to be ignored from subsequent QC steps."""
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
            f"Warning: {len(overwrite_lines)} files were not regridded because they already exist, but will be QC'd here anyway."
        )
    if len(error_lines) > 0:
        print(
            f"Error: {len(error_lines)} files were not regridded due to processing errors. These files will cannot be QC'd. Check qc/qc_error.txt for file paths."
        )
        with open(error_file, "a") as e:
            e.write(
                "The following files had errors during the regridding process and do not have output files to QC:\n\n"
            )
            e.write(("\n".join(error_lines)))
            e.write("\n")
    return fps_to_ignore


def compare_expected_to_existing_and_check_values(
    regrid_dir, regrid_batch_dir, vars, fps_to_ignore, error_file
):
    """Iterate through variables, comparing expected file paths to existing file paths.
    If all expected files exist, check their values against source files.
    Writes error messages to qc error file, and returns a list of fps with errors for printing a summary message."""
    output_errors = []
    ds_errors = []
    value_errors = []

    for var in vars.split():
        # get existing files for the variable
        existing_fps = list(regrid_dir.glob(f"**/{var}/**/*.nc"))
        # list all source file paths found in the batch files, and ignore the ones that had processing errors identified in QC1
        var_src_fps = get_source_fps_from_batch_files(regrid_batch_dir, var)
        for fp in fps_to_ignore:
            if fp in var_src_fps:
                var_src_fps.remove(fp)
        # create a list of expected regridded file paths from the source file paths
        # remove the files that we know have errors from QC1
        for src_fp in var_src_fps:
            # build expected base file path from the source file path
            expected_base_fp = generate_regrid_filepath(src_fp, regrid_dir)
            base_timeframe = expected_base_fp.name.split("_")[-1].split(".nc")[0]
            # get a list of yearly time range strings from the multi-year source filename
            expected_filename_time_ranges = parse_output_filename_times_from_timeframe(
                parse_cmip6_fp(src_fp)["timeframe"]
            )
            # replace the timeframe in the base file path with the yearly time ranges, and add to expected_fps list
            expected_fps = []
            for yearly_timeframe in expected_filename_time_ranges:
                expected_fp = str(expected_base_fp).replace(
                    base_timeframe, yearly_timeframe
                )
                expected_fps.append(Path(expected_fp))

            # search existing files for the expected files, and if not found add to error list
            # if all are found, run the final QC step to compare values
            if all([fp in existing_fps for fp in expected_fps]):
                src_ds = src_ds = xr.open_dataset(src_fp)
                src_min, src_max = float(src_ds[var].min()), float(src_ds[var].max())
                for regrid_fp in expected_fps:
                    ds_error, value_error = check_for_reasonable_values(
                        src_fp, src_min, src_max, regrid_fp, var
                    )
                    if ds_error != []:
                        ds_errors.append(ds_error)
                    if value_error != []:
                        value_errors.append(value_error)
            else:
                output_errors.append(fp)

        # write all errors to qc_error.txt
        with open(error_file, "a") as e:
            if output_errors != []:
                e.write(
                    "Could not find all expected regridded output files for the following source files:\n\n"
                )
                e.write(("\n".join(map(str, output_errors))))
                e.write("\n")
            if ds_errors != []:
                e.write(
                    "Could not open datasets for the following regridded files:\n\n"
                )
                e.write(("\n".join(map(str, ds_errors))))
                e.write("\n")
            if value_errors != []:
                e.write(
                    "Values outside source range for the following regridded files:\n\n"
                )
                e.write(("\n".join(map(str, value_errors))))
                e.write("\n")
    return output_errors, ds_errors, value_errors


def check_for_reasonable_values(src_fp, src_min, src_max, regrid_fp, var):
    """Comparing source file min/max values to regridded file min/max values.
    All regridded values should fall between source min/max values. This also checks if regridded datasets open.
    Returns a tuple of error lists to add to the categorized error lists."""
    ds_error = []
    value_error = []
    try:
        regrid_ds = xr.open_dataset(regrid_fp)
        regrid_min, regrid_max = float(regrid_ds[var].min()), float(
            regrid_ds[var].max()
        ) 
        if (src_max >= regrid_min >= src_min) and (src_max >= regrid_max >= src_min):
            pass
        else:
            value_error.append(src_fp)
    except:
        ds_error.append(src_fp)
  
    return ds_error, value_error


def make_qc_file(output_directory):
    """Make a qc_directory and qc_error.txt file to save results."""
    qc_dir = output_directory.joinpath("qc")
    qc_dir.mkdir(exist_ok=True)
    error_file = qc_dir.joinpath("qc_error.txt")
    with open(error_file, "w") as e:
        pass
    return error_file


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_directory",
        type=str,
        help="Path to directory where all regridded data was written",
        required=True,
    )
    parser.add_argument(
        "--vars",
        type=str,
        help="list of variables",
        required=True,
    )
    args = parser.parse_args()
    return Path(args.output_directory), args.vars


if __name__ == "__main__":

    output_directory, vars = parse_args()
    regrid_dir = output_directory.joinpath("regrid")
    regrid_batch_dir = output_directory.joinpath("regrid_batch")
    slurm_dir = output_directory.joinpath("slurm", "regrid")
    error_file = make_qc_file(output_directory)

    print("QC process started...")

    # check slurm files
    fps_to_ignore = summarize_slurm_out_files(slurm_dir, error_file)

    # check if all expected files exist, check for dataset opening & reasonable values
    (
        output_errors,
        ds_errors,
        value_errors,
    ) = compare_expected_to_existing_and_check_values(
        regrid_dir, regrid_batch_dir, vars, fps_to_ignore, error_file
    )

    # print summary messages
    error_count = len(output_errors) + len(ds_errors) + len(value_errors)
    print(f"QC process complete: {error_count} errors found.")
    if len(output_errors) > 0:
        print(
            f"Errors found when looking for expected output files. {len(output_errors)} files are missing expected outputs. See {str(error_file)} for error log."
        )
    if len(ds_errors) > 0:
        print(
            f"Errors in opening some datasets. {len(ds_errors)} files could not be opened. See {str(error_file)} for error log."
        )
    if len(value_errors) > 0:
        print(
            f"Errors in dataset values. {len(value_errors)} files have regridded values outside of source file range. See {str(error_file)} for error log."
        )