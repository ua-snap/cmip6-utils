#!/usr/bin/env python
"""
Script to create a target grid file by extracting the first time slice from an ERA5 NetCDF file.

This script takes a NetCDF file with a time dimension and extracts only the first time step,
preserving all variables, coordinates, and encoding settings.
"""

import argparse
import sys
from pathlib import Path
import xarray as xr


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract first time slice from a NetCDF file to create a target grid file."
    )
    parser.add_argument(
        "input_file", type=str, help="Input NetCDF file containing time series data"
    )
    parser.add_argument(
        "output_file", type=str, help="Output NetCDF file for the single time slice"
    )

    return parser.parse_args()


def create_target_grid_file(input_file: Path, output_file: Path) -> None:
    """
    Extract the first time slice from input file and save to output file.

    Parameters
    ----------
    input_file : Path
        Path to input NetCDF file with time dimension
    output_file : Path
        Path to output NetCDF file

    Raises
    ------
    ValueError
        If input file has no time dimension or time dimension is empty
    """
    print(f"Opening input file: {input_file}")

    # Open the dataset
    ds = xr.open_dataset(input_file)

    # Check if 'time' dimension exists
    if "time" not in ds.dims:
        ds.close()
        raise ValueError(
            f"Error: Input file does not have a 'time' dimension. "
            f"Available dimensions: {list(ds.dims.keys())}"
        )

    # Check if time dimension has at least one element
    time_size = ds.dims["time"]
    if time_size == 0:
        ds.close()
        raise ValueError("Error: Time dimension is empty (size = 0).")

    print(f"Time dimension size: {time_size}")
    print(f"Extracting first time slice...")

    # Extract first time slice (index 0)
    ds_slice = ds.isel(time=0)

    # Get encoding from original dataset to preserve compression settings
    # We need to handle this carefully for each variable
    # Valid encoding keys for netCDF4 backend
    valid_encoding_keys = {
        "complevel",
        "contiguous",
        "dtype",
        "zlib",
        "least_significant_digit",
        "fletcher32",
        "shuffle",
        "_FillValue",
        "chunksizes",
    }

    encoding = {}
    for var in ds_slice.variables:
        if var in ds.variables:
            # Copy encoding from original dataset, keeping only valid keys
            orig_encoding = ds[var].encoding
            var_encoding = {
                key: value
                for key, value in orig_encoding.items()
                if key in valid_encoding_keys
            }

            # Handle chunksizes specially: need to match the dimensions after slicing
            if (
                "chunksizes" in var_encoding
                and var_encoding["chunksizes"] is not None
                and "time" in ds[var].dims
            ):
                # Original variable had time dimension, but sliced variable doesn't
                # Need to remove the chunk size for the time dimension
                orig_dims = ds[var].dims
                sliced_dims = ds_slice[var].dims

                if len(sliced_dims) < len(orig_dims):
                    # Time dimension was removed, adjust chunksizes
                    time_dim_index = orig_dims.index("time")
                    chunksizes = list(var_encoding["chunksizes"])
                    chunksizes.pop(time_dim_index)
                    var_encoding["chunksizes"] = tuple(chunksizes)

            encoding[var] = var_encoding

    print(f"Variables in output: {list(ds_slice.data_vars.keys())}")
    print(f"Coordinates in output: {list(ds_slice.coords.keys())}")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to output file
    print(f"Saving to: {output_file}")
    ds_slice.to_netcdf(output_file, encoding=encoding, mode="w")

    # Close datasets
    ds.close()
    ds_slice.close()

    print("Successfully created target grid file!")


def main():
    """Main execution function."""
    args = parse_args()

    # Convert paths to Path objects
    input_file = Path(args.input_file).resolve()
    output_file = Path(args.output_file).resolve()

    # Validate input file exists
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    if not input_file.is_file():
        print(f"Error: Input path '{input_file}' is not a file.")
        sys.exit(1)

    print("=" * 80)
    print("Create Target Grid File - Extract First Time Slice")
    print("=" * 80)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print("=" * 80)

    try:
        create_target_grid_file(input_file, output_file)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
