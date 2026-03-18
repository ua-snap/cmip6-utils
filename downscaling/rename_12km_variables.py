#!/usr/bin/env python
"""
Script to rename ERA5 variables in directory names, filenames, and NetCDF files.

This script:
- Renames directories containing variables
- Renames filenames to use new variable names
- Renames the actual variable inside NetCDF files using xarray
- Copies files to a new output directory (does not modify in place)

Example usage:
    python rename_12km_variables.py \
        /import/beegfs/CMIP6/jdpaul3/wrf_era5_12km_daily \
        /import/beegfs/CMIP6/jdpaul3/wrf_era5_12km_daily/for_downscaling \
        --old t2_min t2_max rainnc_sum \
        --new t2min t2max pr
"""

import argparse
import sys
from pathlib import Path
import xarray as xr
from typing import List, Dict


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Rename ERA5 variables in directories, filenames, and NetCDF files."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Top-level input directory containing variable subdirectories",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory where renamed files will be saved",
    )
    parser.add_argument(
        "--old",
        nargs="+",
        required=True,
        help="List of old variable names (e.g., t2_min t2_max rainnc_sum)",
    )
    parser.add_argument(
        "--new",
        nargs="+",
        required=True,
        help="List of new variable names (e.g., t2min t2max pr)",
    )

    args = parser.parse_args()

    # Validate that old and new lists have the same length
    if len(args.old) != len(args.new):
        parser.error(
            f"--old and --new must have the same number of variables. "
            f"Got {len(args.old)} old and {len(args.new)} new."
        )

    return args


def create_variable_mapping(old_vars: List[str], new_vars: List[str]) -> Dict[str, str]:
    """Create a dictionary mapping old variable names to new ones."""
    return dict(zip(old_vars, new_vars))


def rename_variable_in_file(
    input_file: Path, output_file: Path, old_var: str, new_var: str
) -> None:
    """
    Rename a variable in a NetCDF file and save to output location.

    Parameters
    ----------
    input_file : Path
        Path to input NetCDF file
    output_file : Path
        Path to output NetCDF file
    old_var : str
        Old variable name to rename
    new_var : str
        New variable name

    Raises
    ------
    ValueError
        If the expected variable name is not found in the file
    """
    # Open the dataset
    ds = xr.open_dataset(input_file)

    # Check if the old variable exists in the dataset
    if old_var not in ds.data_vars:
        available_vars = list(ds.data_vars)
        ds.close()
        raise ValueError(
            f"Expected variable '{old_var}' not found in {input_file}. "
            f"Available variables: {available_vars}"
        )

    # Rename the variable
    ds_renamed = ds.rename({old_var: new_var})

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to output file (overwrite if exists)
    ds_renamed.to_netcdf(output_file, mode="w")

    # Close datasets
    ds.close()
    ds_renamed.close()

    print(f"  Processed: {input_file.name} -> {output_file.name}")


def process_variable(
    input_dir: Path, output_dir: Path, old_var: str, new_var: str
) -> None:
    """
    Process all files for a given variable.

    Parameters
    ----------
    input_dir : Path
        Top-level input directory
    output_dir : Path
        Top-level output directory
    old_var : str
        Old variable name
    new_var : str
        New variable name
    """
    # Construct paths
    var_input_dir = input_dir / old_var
    var_output_dir = output_dir / new_var

    # Check if input directory exists
    if not var_input_dir.exists():
        print(f"Warning: Directory {var_input_dir} does not exist. Skipping {old_var}.")
        return

    if not var_input_dir.is_dir():
        print(f"Warning: {var_input_dir} is not a directory. Skipping {old_var}.")
        return

    # Find all NetCDF files in the variable directory
    nc_files = sorted(var_input_dir.glob("*.nc"))

    if not nc_files:
        print(f"Warning: No .nc files found in {var_input_dir}. Skipping {old_var}.")
        return

    print(f"\nProcessing variable: {old_var} -> {new_var}")
    print(f"  Input directory: {var_input_dir}")
    print(f"  Output directory: {var_output_dir}")
    print(f"  Found {len(nc_files)} NetCDF files")

    # Process each file
    for nc_file in nc_files:
        # Generate new filename by replacing old variable name with new one
        new_filename = nc_file.name.replace(old_var, new_var)
        output_file = var_output_dir / new_filename

        try:
            rename_variable_in_file(nc_file, output_file, old_var, new_var)
        except Exception as e:
            print(f"  ERROR processing {nc_file.name}: {e}")
            sys.exit(1)


def main():
    """Main execution function."""
    args = parse_args()

    # Convert paths to Path objects
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Validate input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: Input path '{input_dir}' is not a directory.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create variable mapping
    var_mapping = create_variable_mapping(args.old, args.new)

    print("=" * 80)
    print("ERA5 Variable Renaming Script")
    print("=" * 80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nVariable mappings:")
    for old, new in var_mapping.items():
        print(f"  {old:15} -> {new}")
    print("=" * 80)

    # Process each variable
    for old_var, new_var in var_mapping.items():
        process_variable(input_dir, output_dir, old_var, new_var)

    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
