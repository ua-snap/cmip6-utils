"""
Usage:
    python qc.py --sim_dir /import/beegfs/CMIP6/arctic-cmip6/regrid --output_dir /beegfs/CMIP6/crstephenson --models 'GFDL-ESM4' --scenarios 'ssp126 ssp585' --vars 'tasmax pr' --freqs 'day'
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path
from luts import expected_value_ranges

global sim_dir, output_dir


def make_qc_file():
    """Make a qc_directory and qc_error.txt file to save results"""
    qc_dir = output_dir.joinpath("bias_adjust/qc")
    qc_dir.mkdir(exist_ok=True)
    error_file = qc_dir.joinpath("qc_error.txt")
    open(error_file, "w")  # Clear file if it exists.
    return error_file


def get_file_paths(model, scenario, var_id, freq, adjusted=True):
    """Get all file paths for a given model, scenario, variable, and frequency"""
    # Iterate over years to make sure files are sequential order for sim vs. adj comparisons.
    years = range(2015, 2101)
    files = []
    if adjusted:
        bias_adjust_dir = output_dir.joinpath("bias_adjust/netcdf")
        bias_adjust_dir.joinpath(f"{model}/{scenario}/{freq}/{var_id}")
        for year in years:
            files += list(
                bias_adjust_dir.glob(
                    f"{model}/{scenario}/{freq}/{var_id}/*{year}0101*.nc"
                )
            )
    else:
        for year in years:
            files += list(
                sim_dir.glob(f"{model}/{scenario}/{freq}/{var_id}/*{year}0101*.nc")
            )
    return files


def valid_bbox(adj_ds):
    """Check if the bounding box is within the expected range"""
    lat = adj_ds["lat"]
    lon = adj_ds["lon"]
    min_lat = lat.min().values
    max_lat = lat.max().values
    min_lon = lon.min().values
    max_lon = lon.max().values
    if min_lon < 0 or max_lon > 360 or min_lat < 50 or max_lat > 90:
        return False
    return True


def valid_nodata(adj_ds):
    """Check if there are any nodata values in the dataset"""
    nodata_count = np.count_nonzero(np.isnan(adj_ds.to_array()))
    if nodata_count > 0:
        return False
    return True


def valid_values(var_id, adj_ds):
    """Check if the values are within the expected range"""
    min_val = expected_value_ranges[var_id]["minimum"]
    max_val = expected_value_ranges[var_id]["maximum"]
    if adj_ds[var_id].min() < min_val or adj_ds[var_id].max() > max_val:
        return False
    return True


def valid_deltas(var_id, sim_ds, adj_ds):
    """Check if the sim vs. adj deltas are below the allowed maximum"""
    deltas = sim_ds[var_id] - adj_ds[var_id]
    max_allowed_delta = expected_value_ranges[var_id]["delta_maximum"]
    max_found_delta = deltas.max().squeeze().values
    if max_found_delta > max_allowed_delta:
        return False
    return True


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim_dir",
        type=str,
        help="Path to directory where all regridded data was written",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory where all bias-adjusted data was written",
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
        Path(args.sim_dir),
        Path(args.output_dir),
        args.vars,
        args.freqs,
        args.models,
        args.scenarios,
    )


if __name__ == "__main__":
    sim_dir, output_dir, vars, freqs, models, scenarios = parse_args()
    slurm_dir = output_dir.joinpath("slurm")
    error_file = make_qc_file()

    bbox_errors = []
    nodata_errors = []
    value_errors = []
    delta_errors = []

    print("QC process started...")

    for var_id in vars.split():
        for freq in freqs.split():
            for model in models.split():
                for scenario in scenarios.split():
                    sim_fps = get_file_paths(
                        model, scenario, var_id, freq, adjusted=False
                    )
                    adj_fps = get_file_paths(model, scenario, var_id, freq)
                    for sim_fp, adj_fp in zip(sim_fps, adj_fps):
                        try:
                            sim_ds = xr.open_dataset(sim_fp)
                        except Exception as e:
                            e.write(f"Error: {sim_fp} could not be opened.\n")
                            continue
                        try:
                            adj_ds = xr.open_dataset(adj_fp)
                        except Exception as e:
                            e.write(f"Error: {adj_fp} could not be opened.\n")
                            continue

                        if not valid_bbox(adj_ds):
                            bbox_errors.append(str(adj_fp))
                        if not valid_nodata(adj_ds):
                            nodata_errors.append(str(adj_fp))
                        if not valid_values(var_id, adj_ds):
                            value_errors.append(str(adj_fp))
                        if not valid_deltas(var_id, sim_ds, adj_ds):
                            delta_errors.append(str(adj_fp))

    with open(error_file, "a") as e:
        if bbox_errors:
            e.write(f"Files with BBOX errors:\n")
            e.write("\n".join(bbox_errors) + "\n\n")
        if nodata_errors:
            e.write(f"Files with nodata errors:\n")
            e.write("\n".join(nodata_errors) + "\n\n")
        if value_errors:
            e.write(f"Files with value range errors:\n")
            e.write("\n".join(value_errors) + "\n\n")
        if delta_errors:
            e.write(f"Files with max delta errors:\n")
            e.write("\n".join(delta_errors) + "\n\n")

    error_count = (
        len(bbox_errors) + len(nodata_errors) + len(value_errors) + len(delta_errors)
    )
    print(f"QC process complete. {error_count} errors found.")
