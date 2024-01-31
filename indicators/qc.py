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
from luts import units_lu, ranges_lu


def qc_by_row(row, error_file):
    # set up list to collect error strings
    error_strings = []

    # QC 1: does the slurm job output file exist? And does it show success message?
    if os.path.isfile(row[2]) == False:
        error_strings.append(f"ERROR: Expected job output file {row[2]} not found.")
    else:
        with open(row[2], "r") as o:
            lines = o.read().splitlines()
            if len(lines) > 0:
                if not lines[-1] == "Job Completed":
                    error_strings.append(
                        f"ERROR: Slurm job not completed. See {row[2]}."
                    )
            else:
                error_strings.append(f"ERROR: Slurm job output is empty. See {row[2]}.")

    # QC 2: does the indicator .nc file exist?
    if os.path.isfile(row[1]) == False:
        error_strings.append(f"ERROR: Expected indicator file {row[1]} not found.")

    # QC 3: do the indicator string, indicator .nc filename, and indicator variable name in dataset match?
    qc_indicator_string = row[0]
    fp = Path(row[1])
    fp_indicator_string = fp.parts[-1].split("_")[0]

    try:  # also checks that the dataset opens
        ds = xr.open_dataset(fp)
        ds_indicator_string = list(ds.data_vars)[0]
    except:
        error_strings.append(f"ERROR: Could not open dataset: {row[1]}.")
        ds_indicator_string = "None"
        ds = None

    if not fp_indicator_string == ds_indicator_string == qc_indicator_string:
        error_strings.append(
            f"ERROR: Mismatch of indicator strings found between parameter: {qc_indicator_string}, filename: {row[1]}, and dataset variable: {ds_indicator_string}."
        )

    # skip the final QC steps if the file could not be opened
    if ds is not None:
        # QC 4: do the unit attributes in the first year data array match expected values in the lookup table?
        if not ds[ds_indicator_string].attrs == units_lu[qc_indicator_string]:
            error_strings.append(
                f"ERROR: Mismatch of unit dictionary found between dataset and lookup table in filename: {row[1]}."
            )

        # QC 5: do the files contain reasonable values as defined in the lookup table?
        min_val = ranges_lu[qc_indicator_string]["min"]
        max_val = ranges_lu[qc_indicator_string]["max"]

        if (ds[ds_indicator_string].values < min_val).any():
            error_strings.append(
                f"ERROR: Minimum values outside range in dataset: {row[1]}."
            )
        if (ds[ds_indicator_string].values > max_val).any():
            error_strings.append(
                f"ERROR: Maximum values outside range in dataset: {row[1]}."
            )

    # Log the errors: write any errors into the error file
    with open(error_file, "a") as e:
        e.write(("\n".join(error_strings)))

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

    args = parser.parse_args()

    return Path(args.out_dir)


if __name__ == "__main__":
    out_dir = parse_args()

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
        row_errs = qc_by_row(row, error_file)
        error_count = error_count + row_errs

    print(
        f"QC process complete: {str(error_count)} errors found. See {str(error_file)} for error log."
    )
