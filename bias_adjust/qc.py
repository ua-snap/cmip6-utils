"""
Usage:
    python qc.py --output_directory /beegfs/CMIP6/crstephenson/bias_adjust --vars 'pr ta' --freqs 'mon day'
"""

import argparse
import multiprocessing
import numpy as np
import xarray as xr
from pathlib import Path


def make_qc_file(output_directory):
    """Make a qc_directory and qc_error.txt file to save results."""
    qc_dir = output_directory.joinpath("qc")
    qc_dir.mkdir(exist_ok=True)
    error_file = qc_dir.joinpath("qc_error.txt")
    with open(error_file, "w") as e:
        pass
    return error_file


def get_file_paths(bias_adjust_dir, models, scenarios, vars, freqs, error_file):
    """Get list of all bias adjusted files."""
    for model in models.split():
        for scenario in scenarios.split():
            for var_id in vars.split():
                for freq in freqs.split():
                    fps = list(bias_adjust_dir.glob(f"{model}/{scenario}/{freq}/{var_id}/*.nc"))
    return fps


def check_nodata(fp, ds, e):
    nodata_count = np.count_nonzero(np.isnan(ds.to_array()))
    if nodata_count > 0:
        e.write(f"Error: {fp} has unexpected nodata pixels.\n")


def check_bbox(fp, ds, e):
    lat = ds["lat"]
    lon = ds["lon"]
    min_lat = lat.min().values
    max_lat = lat.max().values
    min_lon = lon.min().values
    max_lon = lon.max().values
    if min_lon < 0 or max_lon > 360 or min_lat < 50 or max_lat > 90:
        e.write(f"Error: {fp} has bbox [{min_lon}, {max_lon}, {min_lat}, {max_lat}] outside of expected range.\n")


def inspect_files(fps, error_file):
    with open(error_file, "a") as e:
        for fp in fps:
            try:
                ds = xr.open_dataset(fp)
            except Exception as e:
                e.write(f"Error: {fp} could not be opened.\n")
                continue
            check_nodata(fp, ds, e)
            check_bbox(fp, ds, e)


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
    parser.add_argument(
        "--freqs",
        type=str,
        help="list of frequencies used in generating batch files",
        required=True,
    )
    parser.add_argument(
        "--models",
        type=str,
        help="list of models used in generating batch files",
        required=True,
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="list of scenarios used in generating batch files",
        required=True,
    )
    args = parser.parse_args()
    return (
        Path(args.output_directory),
        args.vars,
        args.freqs,
        args.models,
        args.scenarios,
    )


if __name__ == "__main__":

    output_directory, vars, freqs, models, scenarios = parse_args()
    bias_adjust_dir = output_directory.joinpath("netcdf")
    slurm_dir = output_directory.joinpath("slurm")
    error_file = make_qc_file(output_directory)

    print("QC process started...")

    fps = get_file_paths(bias_adjust_dir, models, scenarios, vars, freqs, error_file)
    inspect_files(fps, error_file)

    # # print summary messages
    # error_count = len(output_errors) + len(ds_errors) + len(value_errors)
    # print(f"QC process complete: {error_count} errors found.")
    # if len(output_errors) > 0:
    #     print(
    #         f"Errors found when looking for expected output files. {len(output_errors)} files are missing expected outputs. See {str(error_file)} for error log."
    #     )
    # if len(ds_errors) > 0:
    #     print(
    #         f"Errors in opening some datasets. {len(ds_errors)} files could not be opened. See {str(error_file)} for error log."
    #     )
    # if len(value_errors) > 0:
    #     print(
    #         f"Errors in dataset values. {len(value_errors)} files have regridded values outside of source file range. See {str(error_file)} for error log."
    #     )
