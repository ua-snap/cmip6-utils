#!/usr/bin/env python
"""
Script to create a target grid file by extracting the first time slice from an ERA5 NetCDF file.

This script takes a NetCDF file with a time dimension and extracts only the first time step,
preserving all variables, coordinates, and encoding settings. Also adds lon/lat coordinates
if they don't exist (for projected coordinate systems).

Example usage:
    python create_12km_target_grid_file.py \
        /beegfs/CMIP6/jdpaul3/wrf_era5_12km_daily/for_downscaling/t2max/t2max_2014_daily_era5_12km_3338.nc \
        /beegfs/CMIP6/jdpaul3/wrf_era5_12km_daily/for_downscaling/era5_12km_target_slice.nc
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer


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


def add_lonlat_coordinates(ds):
    """
    Add lon/lat coordinates to a dataset if they don't exist.

    For projected coordinate systems (x/y in meters), this computes
    2D lon/lat arrays from the x/y coordinates using the CRS information.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with x/y coordinates and spatial_ref CRS info

    Returns
    -------
    ds : xarray.Dataset
        Dataset with lon/lat coordinates added (if they didn't exist)
    """
    # Skip if lon/lat already exist
    if "lon" in ds.coords or "lon" in ds.data_vars:
        print("  lon/lat coordinates already present, skipping...")
        return ds

    # Check if we have x/y projected coordinates
    if "x" not in ds.dims or "y" not in ds.dims:
        print("  No x/y dimensions found, skipping lon/lat generation...")
        return ds

    # Get CRS information from spatial_ref (check both coords and data_vars)
    crs = None
    spatial_ref = None

    if "spatial_ref" in ds.coords:
        spatial_ref = ds.coords["spatial_ref"]
    elif "spatial_ref" in ds.data_vars:
        spatial_ref = ds.data_vars["spatial_ref"]

    if spatial_ref is not None:
        try:
            # Try to get crs_wkt from attributes
            crs_wkt = spatial_ref.attrs.get("crs_wkt")
            if crs_wkt:
                print(f"  Found crs_wkt in spatial_ref attributes")
                crs = CRS.from_wkt(crs_wkt)
            else:
                print(f"  spatial_ref found but no crs_wkt attribute")
                print(f"  Available attributes: {list(spatial_ref.attrs.keys())}")
        except Exception as e:
            print(f"  Error parsing CRS: {e}")
    else:
        print("  No spatial_ref variable found in dataset")

    if crs is None:
        print("  Warning: No CRS information found, cannot compute lon/lat")
        return ds

    print(f"  Found CRS: {crs.name}")
    print(f"  Computing lon/lat from x/y coordinates...")

    # Create 2D meshgrid from x and y coordinates
    x_coords = ds.x.values
    y_coords = ds.y.values
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Transform from projected coordinates to geographic (lon/lat)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_grid, lat_grid = transformer.transform(x_grid, y_grid)

    # Add as coordinates with proper dimensions and attributes
    ds = ds.assign_coords(
        {
            "lon": (
                ("y", "x"),
                lon_grid,
                {
                    "_FillValue": np.nan,
                    "long_name": "longitude",
                    "standard_name": "longitude",
                    "units": "degrees_east",
                },
            ),
            "lat": (
                ("y", "x"),
                lat_grid,
                {
                    "_FillValue": np.nan,
                    "long_name": "latitude",
                    "standard_name": "latitude",
                    "units": "degrees_north",
                },
            ),
        }
    )

    print(f"  ✓ Added lon/lat coordinates (shape: {lon_grid.shape})")
    print(f"    Longitude range: [{lon_grid.min():.2f}, {lon_grid.max():.2f}]")
    print(f"    Latitude range: [{lat_grid.min():.2f}, {lat_grid.max():.2f}]")

    return ds


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

    # Preserve spatial_ref if it exists (it might get dropped during isel)
    spatial_ref_var = None
    if "spatial_ref" in ds.coords:
        spatial_ref_var = ds.coords["spatial_ref"]
        print("  Found spatial_ref in coordinates")
    elif "spatial_ref" in ds.data_vars:
        spatial_ref_var = ds.data_vars["spatial_ref"]
        print("  Found spatial_ref in data variables")

    # Extract first time slice (index 0)
    ds_slice = ds.isel(time=0)

    # Drop the scalar time coordinate/variable completely
    # This ensures target grid file has no time dimension/coordinate
    if "time" in ds_slice.coords:
        print("  Dropping scalar time coordinate...")
        ds_slice = ds_slice.drop_vars("time")

    # Restore spatial_ref if it was dropped and we had saved it
    if spatial_ref_var is not None and "spatial_ref" not in ds_slice:
        print("  Restoring spatial_ref after time slice")
        ds_slice = ds_slice.assign_coords({"spatial_ref": spatial_ref_var})

    # Add lon/lat coordinates if they don't exist
    print("Checking for lon/lat coordinates...")
    ds_slice = add_lonlat_coordinates(ds_slice)

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
        # Skip time if it still exists somehow (should have been dropped)
        if var == "time":
            continue

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
        elif var in ["lon", "lat"]:
            # These were added by add_lonlat_coordinates, use simple encoding
            encoding[var] = {"dtype": "float64"}

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
