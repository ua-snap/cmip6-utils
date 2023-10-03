#!/usr/bin/env python
import argparse
import numpy as np
from multiprocessing import Pool, set_start_method
import xarray as xr
from pathlib import Path
import tqdm
import os
import json


# Check if files is openable using xarray.
def file_min_max(args):
    file, variable = args
    src_ds = xr.open_dataset(file)

    # Data was sliced from 49 to 90 degrees latitude before regridding.
    # Use this same slice to determine min and max values for regrid testing.
    src_ds = src_ds.sel(lat=slice(49, 90))
    values = src_ds[variable].values

    min = np.nanmin(values)
    max = np.nanmax(values)

    return {"file": str(file), "min": min, "max": max}


def main(args):
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    # Return a list of all files inside *.txt that start with variable name.
    source_fps = []
    for fp in regrid_batch_dir.glob("*.txt"):
        with open(fp) as f:
            # Return a list of all the files inside f with a basename that starts with args.variable.
            source_fps.extend(
                [
                    Path(line.strip())
                    for line in f
                    if Path(line.strip()).name.startswith(args.variable + "_")
                ]
            )

    json_file = f"tmp/{args.variable}.json"

    if not os.path.isfile(json_file):
        with open(json_file, "w") as outfile:
            json.dump({}, outfile)

    with open(json_file, "r") as infile:
        completed_files = json.load(infile)

    # Get a new list of files that have not been completed yet.
    source_fps = [fp for fp in source_fps if not str(fp) in completed_files.keys()]

    print(f"{len(source_fps)} files left to process.")

    file_args = [(file, args.variable) for file in source_fps]

    # Split the file_args list into a list of 5 item chunks.
    arg_chunks = [file_args[i : i + 5] for i in range(0, len(file_args), 5)]

    for arg_chunk in arg_chunks:
        print(
            f"Processing chunk {arg_chunks.index(arg_chunk) + 1} of {len(arg_chunks)}."
        )
        with Pool(5) as p:
            results = list(
                tqdm.tqdm(
                    p.imap_unordered(file_min_max, arg_chunk), total=len(arg_chunk)
                )
            )

        for result in results:
            completed_files[result["file"]] = {
                "min": str(result["min"]),
                "max": str(result["max"]),
            }

        with open(json_file, "w") as outfile:
            json.dump(completed_files, outfile, indent=4)

    # Get an array of the mins of all items in completed_files.
    file_mins = np.array(
        [float(completed_files[key]["min"]) for key in completed_files.keys()]
    )
    file_maxes = np.array(
        [float(completed_files[key]["max"]) for key in completed_files.keys()]
    )

    min = np.nanmin(file_mins)
    max = np.nanmax(file_maxes)

    print(args.variable)
    print(f"Minimum value: {min}")
    print(f"Maximum value: {max}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to extract CMIP6 variable min/max values across all source files."
    )
    parser.add_argument(
        "-v",
        "--variable",
        action="store",
        dest="variable",
        type=str,
        help="climate variable corresponding to CMIP6 source directory (e.g., tas, pr, huss)",
        required=True,
    )

    set_start_method("spawn")
    args = parser.parse_args()
    _ = main(args)
