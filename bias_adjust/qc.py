"""
Usage:
    python qc.py --output_directory /beegfs/CMIP6/crstephenson/bias_adjust --vars 'pr tasmax' --freqs 'mon day'
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path
from luts import expected_value_ranges


def make_qc_file(output_directory):
    """Make a qc_directory and qc_error.txt file to save results."""
    qc_dir = output_directory.joinpath("qc")
    qc_dir.mkdir(exist_ok=True)
    error_file = qc_dir.joinpath("qc_error.txt")
    open(error_file, "w") # Clear file if it exists.
    return error_file


def get_file_paths(bias_adjust_dir, model, scenario, var_id, freq):
    return list(bias_adjust_dir.glob(f"{model}/{scenario}/{freq}/{var_id}/*.nc"))


def valid_bbox(ds):
    lat = ds["lat"]
    lon = ds["lon"]
    min_lat = lat.min().values
    max_lat = lat.max().values
    min_lon = lon.min().values
    max_lon = lon.max().values
    if min_lon < 0 or max_lon > 360 or min_lat < 50 or max_lat > 90:
        return False
    return True


def valid_nodata(ds):
    nodata_count = np.count_nonzero(np.isnan(ds.to_array()))
    if nodata_count > 0:
        return False
    return True


def valid_values(var_id, ds):
    min_val = expected_value_ranges[var_id]["minimum"]
    max_val = expected_value_ranges[var_id]["maximum"]
    if ds[var_id].min() < min_val or ds[var_id].max() > max_val:
        return False
    return True


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

    bbox_errors = []
    nodata_errors = []
    value_errors = []

    print("QC process started...")

    for model in models.split():
        for scenario in scenarios.split():
            for var_id in vars.split():
                for freq in freqs.split():
                    fps = get_file_paths(bias_adjust_dir, model, scenario, var_id, freq)
                    for fp in fps:
                        try:
                            ds = xr.open_dataset(fp)
                        except Exception as e:
                            e.write(f"Error: {fp} could not be opened.\n")
                            continue
                        if not valid_bbox(ds):
                            bbox_errors.append(fp)
                        if not valid_nodata(ds):
                            nodata_errors.append(fp)
                        if not valid_values(var_id, ds):
                            value_errors.append(fp)

    with open(error_file, "a") as e:
        e.write(f"Files with BBOX errors:\n")
        e.write("\n".join(str(bbox_errors)) + "\n")
        e.write(f"Files with nodata errors:\n")
        e.write("\n".join(str(nodata_errors)) + "\n")
        e.write(f"Files with value range errors:\n")
        e.write("\n".join(str(value_errors)) + "\n")

    error_count = len(bbox_errors) + len(nodata_errors) + len(value_errors)
    print(f"QC process complete. {error_count} errors found.")
