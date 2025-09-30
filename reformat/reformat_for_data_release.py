#!/usr/bin/env python3
"""
Reformat CMIP6 downscaling outputs for data release.

This script processes zarr files by:
1. Reordering variables to follow CF conventions (time, y, x, ensemble)
2. Adding ensemble dimension with <model_variant_scenario> format derived from the filename and config.py
3. Cleaning dataset attributes to keep only contact, creation_date, and history

Usage:
    python reformat_for_data_release.py <adjusted_dir> <output_dir>

Arguments:
    adjusted_dir: Path to directory containing input zarr files
    output_dir: Path to directory where reformatted files will be saved
"""

import argparse
import sys
import xarray as xr
from pathlib import Path

# local imports
# go up one directory to get the config file
sys.path.append("..")
from transfers.config import prod_variant_lu


def reorder_vars(ds):
    """Reorder variables to match CF conventions (time, y, x, ensemble)."""
    return ds.transpose("time", "y", "x", "ensemble")


def get_ensemble_str_from_filename(file, prod_variant_lu):
    """
    Extract ensemble string from filename.
    
    Filename format: <variable>_<model>_<scenario>_adjusted.zarr
    Returns: <model>_<variant>_<scenario>
    """
    model = file.name.split("_")[1]
    scenario = file.name.split("_")[2]
    try:
        variant = prod_variant_lu[model]
    except KeyError:
        print(f"Warning: model {model} not found in variant lookup table, setting variant to 'unknown'!")
        variant = "unknown"  # default if not found ... we can potentially look for this later on to flag issues?
    ensemble_str = f"{model}_{variant}_{scenario}"
    return ensemble_str


def add_ensemble_dim(ds, ensemble_str):
    """Add ensemble dimension to dataset."""
    ds = ds.expand_dims({"ensemble": [ensemble_str]})
    return ds


def clean_attrs(ds):
    """Drop attributes that are not 'contact', 'creation_date', or 'history'."""
    attrs_to_keep = ["contact", "creation_date", "history"]
    attrs_to_drop = [attr for attr in ds.attrs if attr not in attrs_to_keep]
    ds.attrs = {attr: ds.attrs[attr] for attr in attrs_to_keep if attr in ds.attrs}
    return ds


def process_file(file, prod_variant_lu):
    """Wrapper function to apply all processing steps to a file."""
    ds = xr.open_zarr(file, chunks="auto")
    ds = clean_attrs(ds)
    ensemble_str = get_ensemble_str_from_filename(file, prod_variant_lu)
    ds = add_ensemble_dim(ds, ensemble_str)
    ds = reorder_vars(ds)
    return ds


def main():
    """Main function to process all zarr files in the input directory."""
    parser = argparse.ArgumentParser(
        description="Reformat CMIP6 downscaling outputs for data release",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("adjusted_dir", type=Path, 
                       help="Path to directory containing input zarr files")
    parser.add_argument("output_dir", type=Path,
                       help="Path to directory where reformatted files will be saved")
    
    args = parser.parse_args()
    
    adjusted_dir = args.adjusted_dir
    output_dir = args.output_dir
    
    # Validate input directory exists
    if not adjusted_dir.exists():
        print(f"Error: Input directory {adjusted_dir} does not exist!")
        sys.exit(1)
    
    # List zarr files
    files = list(adjusted_dir.glob("**/*.zarr"))
    files.sort()
    
    if not files:
        print(f"No zarr files found in {adjusted_dir}")
        sys.exit(0)
    
    print(f"Found {len(files)} zarr files to reformat.")
    
    # Process datasets and save to new zarr files
    # Creating output directory if it doesn't exist
    success_count = 0
    failure_count = 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        print(f"Processing {file}...")
        try:
            ds = process_file(file, prod_variant_lu)
            print(f"File reformatted successfully!")
            success_count += 1
            
            output_file = output_dir / file.name
            ds.to_zarr(output_file, mode="w")
            ds.close()
            print(f"Saved to {output_file}")
            
        except Exception as e:
            print(f"File could not be reformatted! Error: {e}")
            failure_count += 1
    
    print(f"\n\nProcessing complete. Success: {success_count}, Failure: {failure_count}")


if __name__ == "__main__":
    main()