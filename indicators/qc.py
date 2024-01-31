"""Script for performing QC checks on files produced via indicators.py and their individual slurm job output files.
This script uses the {out_dir}/qc/qc.csv file produced in the indicators/slurm.py script as a "to-do list" of files to check.
QC errors are written to {out_dir}/qc/qc_error.txt. 

Usage:
    python qc.py --out_dir /beegfs/CMIP6/jdpaul3/scratch
"""

import argparse
import pandas as pd
import os
import xarray as xr
import numpy as np
from pathlib import Path
from luts import units_lu, ranges_lu, idx_varid_lu, varid_freqs


def check_nodata(idx, output_fp, ds, in_dir):
    """Check for no data equivalence between inputs and outputs. 
    Parse the filename to find indicator/model/scenario combo and locate appropriate input file(s).
    Assumes that yearly .nc input files have the same no data extent, and uses the first file to define no data cells.
    Outputs True/False."""

    ###########################################
    #TODO: add filename parsing / input lookup!

    #get list of variables used to compute indicator
    vars = idx_varid_lu[idx]
    print(vars)
    input_nodata_arrays = []
    for var in vars:
        #parse output filepath and use parts to build input dir
        freq = varid_freqs[var][0]
        model, scenario = Path(output_fp).parts[7], Path(output_fp).parts[8]
        path_to_input_dir = in_dir.joinpath(model, scenario, freq, var)
        #get first .nc file from input dir
        first_input_file = [f for f in path_to_input_dir.glob('**/*.nc')][0]
        in_ds = xr.open_dataset(first_input_file)
        #get True/False array of nodata values from input; these should also be no data values in output
        input_nodata_array = np.broadcast_to(np.isnan(in_ds[var].sel(time=in_ds["time"].values[0])), ds[idx].shape)
        input_nodata_arrays.append(input_nodata_array)

    results = []
    for i in input_nodata_arrays:
        #check if the actual output no data values match input no data array
        #use dtypes to choose between -9999 and np.nan
        if ds[idx].dtype in [np.int32, np.int64]:
            output_nodata = ds[idx].values == -9999
            if np.array_equal(output_nodata, i) == False:
                results.append(False)
            else:
                results.append(True)
        else:
            output_nodata = np.isnan(ds[idx].values)
            if np.array_equal(output_nodata, i) == False:
                results.append(False)
            else:
                results.append(True)

    return results

    

def qc_by_row(row, error_file, in_dir):
    # set up list to collect error strings
    error_strings = []

    # QC 1: does the slurm job output file exist? And does it show success message?
    if os.path.isfile(row[2]) == False:
        error_strings.append(f"ERROR: Expected job output file {row[2]} not found.")
        job_output_exists = None
    else:
        job_output_exists = True
        with open(row[2], "r") as o:
            lines = o.read().splitlines()
            if len(lines) > 0:
                if not lines[-1] == "Job Completed":
                    error_strings.append(
                        f"ERROR: Slurm job not completed. See {row[2]}."
                    )
                if not lines[-1] == "Job Completed":
                    error_strings.append(
                        f"ERROR: Slurm job not completed. See {row[2]}."
                    )
            else:
                error_strings.append(f"ERROR: Slurm job output is empty. See {row[2]}.")

    # QC 2: does the indicator .nc file exist?
    if os.path.isfile(row[1]) == False:
        error_strings.append(f"ERROR: Expected indicator file {row[1]} not found.")
        indicator_output_exists = None
    else:
        indicator_output_exists = True

    if job_output_exists is not None and indicator_output_exists is not None:

        # QC 3: do the indicator string, indicator .nc filename, and indicator variable name in dataset match?
        qc_indicator_string = row[0]
        fp = Path(row[1])
        fp_indicator_string = fp.parts[-1].split("_")[0]

        try:  # also checks that the dataset opens
            ds = xr.open_dataset(fp)
            ds_indicator_string = list(ds.data_vars)[0]
        except:
            error_strings.append(f"ERROR: Could not open dataset: {row[1]}.")
            ds = None

        if ds is not None:
            if not fp_indicator_string == ds_indicator_string == qc_indicator_string:
                error_strings.append(
                    f"ERROR: Mismatch of indicator strings found between parameter: {qc_indicator_string}, filename: {row[1]}, and dataset variable: {ds_indicator_string}."
                )

        # skip the final QC steps if the file could not be opened
        if ds is not None:

            # QC 4: do the unit attributes in the first year data array match expected values in the lookup table?

            ds_units = ds[ds_indicator_string].attrs["units"]
            lu_units = units_lu[qc_indicator_string]

            if not ds[ds_indicator_string].attrs["units"] == units_lu[qc_indicator_string]:
                error_strings.append(
                    f"ERROR: Mismatch of unit dictionary found between dataset and lookup table in filename: {row[1]}. {ds_units} and {lu_units}."
                )

            # QC 5: do the files contain reasonable values as defined in the lookup table?
            min_val = ranges_lu[qc_indicator_string]["min"]
            max_val = ranges_lu[qc_indicator_string]["max"]

            if ((ds[ds_indicator_string].values < min_val) & (ds[ds_indicator_string].values != -9999)).any():
                error_strings.append(
                    f"ERROR: Minimum values outside range in dataset: {row[1]}."
                )
            if (ds[ds_indicator_string].values > max_val).any():
                error_strings.append(
                    f"ERROR: Maximum values outside range in dataset: {row[1]}."
                )

            # QC 6: do the nodata cells in the output match nodata cells in the input?
            if False in check_nodata(row[0], row[1], ds, in_dir):
                error_strings.append(
                    f"ERROR: No data cells in input variable(s) do not match no data cells in output dataset: {row[1]}."
                )

    # Log the errors: write any errors into the error file
    if len(error_strings)>0:
        with open(error_file, "a") as e:
            e.write(("\n".join(error_strings)))
            e.write("\n")

    return len(error_strings)


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to directory where indicators data, QC files, and slurm files were written",
        required=True,
    )

    parser.add_argument(
        "--in_dir",
        type=str,
        help="Path to directory containing source input files",
        required=True,
    )

    args = parser.parse_args()

    return Path(args.out_dir), Path(args.in_dir)


if __name__ == "__main__":

    out_dir, in_dir = parse_args()

    # build qc file path from out_dir argument and load qc file;
    # first column is indicator name, second column is indicators .nc filepath, third column is slurm job output filepath
    qc_file = out_dir.joinpath("qc", "qc.csv")
    df = pd.read_csv(qc_file, header=None)
    # build error file path from SCRATCH_DIR and create error file
    error_file = out_dir.joinpath("qc", "qc_error.txt")
    with open(error_file, "w") as e:
        pass

    print("QC process started...")

    error_count = 0
    for _index, row in df.iterrows():
        row_errs = qc_by_row(row, error_file, in_dir)
        error_count = error_count + row_errs

    print(
        f"QC process complete: {str(error_count)} errors found. See {str(error_file)} for error log."
    )

