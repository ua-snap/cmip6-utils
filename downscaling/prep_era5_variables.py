#!/usr/bin/env python
"""
Script to rename ERA5 variables in directory names, filenames, and NetCDF files,
and convert temperature variables from Celsius to Kelvin.

This script:
- Renames directories containing variables
- Renames filenames to use new variable names
- Renames the actual variable inside NetCDF files using xarray
- Converts temperature variables (t2_min, t2_max) from Celsius to Kelvin
- Copies files to a new output directory (does not modify in place)

Works with ERA5 outputs from the WRF ERA5 curation pipeline at any resolution
(4km, 12km, etc.).

Legacy mode (--legacy):
    In addition to renaming and unit conversion, regrids all output files to
    exactly match the legacy 4km ERA5 grid defined by the default target grid
    file (default_target_grid_files/era5_4km_default_target_grid.nc). Use this
    when preparing 4km ERA5 inputs for a downscaling run that uses the default
    4km target grid. Legacy mode is only applicable to 4km data.

    The regrid uses bilinear (linear) interpolation over the x/y projected
    coordinates. The spatial_ref (CRS) coordinate is copied from the target
    grid file so that all output files share identical grid metadata.

    To override the default legacy grid file, pass --legacy-grid-file.

Example usage (standard):
    python prep_era5_variables.py \
        /import/beegfs/CMIP6/jdpaul3/wrf_era5_daily \
        /import/beegfs/CMIP6/jdpaul3/wrf_era5_daily/for_downscaling \
        --old t2_min t2_max rainnc_sum \
        --new t2min t2max pr \
        --celsius-to-kelvin t2_min t2_max

Example usage (legacy mode):
    python prep_era5_variables.py \
        /import/beegfs/CMIP6/jdpaul3/wrf_era5_daily \
        /import/beegfs/CMIP6/jdpaul3/wrf_era5_daily/for_downscaling \
        --old t2_min t2_max rainnc_sum \
        --new t2min t2max pr \
        --celsius-to-kelvin t2_min t2_max \
        --legacy
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Set

_DEFAULT_LEGACY_GRID_FILE = (
    Path(__file__).parent / "default_target_grid_files" / "era5_4km_default_target_grid.nc"
)

CELSIUS_TO_KELVIN_OFFSET = 273.15


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Rename ERA5 variables and optionally convert temperature units (C to K)."
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
    parser.add_argument(
        "--celsius-to-kelvin",
        nargs="+",
        default=[],
        metavar="VAR",
        help="List of old variable names (from --old) to convert from Celsius to Kelvin",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        default=False,
        help=(
            "Enable legacy mode: regrid output files to exactly match the legacy 4km ERA5 "
            "target grid. Only applicable to 4km data."
        ),
    )
    parser.add_argument(
        "--legacy-grid-file",
        type=str,
        default=str(_DEFAULT_LEGACY_GRID_FILE),
        metavar="FILE",
        help=(
            "Path to the legacy target grid NetCDF file used when --legacy is set. "
            f"Defaults to {_DEFAULT_LEGACY_GRID_FILE}"
        ),
    )

    args = parser.parse_args()

    if len(args.old) != len(args.new):
        parser.error(
            f"--old and --new must have the same number of variables. "
            f"Got {len(args.old)} old and {len(args.new)} new."
        )

    invalid = set(args.celsius_to_kelvin) - set(args.old)
    if invalid:
        parser.error(
            f"--celsius-to-kelvin variables not found in --old: {sorted(invalid)}"
        )

    return args


def create_variable_mapping(old_vars: List[str], new_vars: List[str]) -> Dict[str, str]:
    """Create a dictionary mapping old variable names to new ones."""
    return dict(zip(old_vars, new_vars))


def regrid_to_legacy_grid(ds: xr.Dataset, target_ds: xr.Dataset) -> xr.Dataset:
    """
    Regrid a dataset to exactly match the legacy 4km ERA5 target grid.

    Uses bilinear (linear) interpolation over the projected x/y coordinates.
    The spatial_ref (CRS) coordinate is copied from target_ds so that all
    output files share identical grid metadata.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to regrid.
    target_ds : xr.Dataset
        Target grid dataset (e.g. era5_4km_default_target_grid.nc).

    Returns
    -------
    xr.Dataset
        Dataset regridded to match target_ds x/y coordinates.
    """
    ds_regridded = ds.interp_like(target_ds, method="linear")

    # Carry over spatial_ref from the target grid so CRS metadata is exact
    for loc in ("coords", "data_vars"):
        if "spatial_ref" in getattr(target_ds, loc):
            target_spatial_ref = getattr(target_ds, loc)["spatial_ref"]
            if "spatial_ref" not in ds_regridded.coords and "spatial_ref" not in ds_regridded.data_vars:
                ds_regridded = ds_regridded.assign_coords({"spatial_ref": target_spatial_ref})
            break

    return ds_regridded


def print_legacy_grid_comparison(sample_output_file: Path, target_ds: xr.Dataset) -> None:
    """
    Print a side-by-side comparison of grid metadata between a sample output file
    and the legacy target grid.

    Checks x/y dimensions, coordinate ranges, resolution, and CRS (crs_wkt).
    Called once per variable after all files for that variable have been written.

    Parameters
    ----------
    sample_output_file : Path
        Path to a representative output NetCDF file to inspect.
    target_ds : xr.Dataset
        The legacy target grid dataset.
    """
    _OK = "OK"
    _MISMATCH = "*** MISMATCH ***"
    _TOL = 0.01  # metres — effectively exact for EPSG:3338 projected coords

    ds_out = xr.open_dataset(sample_output_file)

    col_w = 28  # width of each value column
    label_w = 22

    def _row(label, out_val, tgt_val, match):
        status = _OK if match else _MISMATCH
        print(
            f"    {label:<{label_w}} {str(out_val):<{col_w}} {str(tgt_val):<{col_w}} {status}"
        )

    print(f"\n  Legacy grid comparison  (sample: {sample_output_file.name})")
    print(f"  {'─' * (label_w + col_w * 2 + 10)}")
    print(
        f"    {'Attribute':<{label_w}} {'Output':<{col_w}} {'Target':<{col_w}} Status"
    )
    print(f"    {'─' * (label_w + col_w * 2 + 8)}")

    all_ok = True

    # --- Dimensions ---
    for dim in ("x", "y"):
        out_size = ds_out.dims.get(dim, "missing")
        tgt_size = target_ds.dims.get(dim, "missing")
        match = out_size == tgt_size
        all_ok = all_ok and match
        _row(f"dim {dim}", out_size, tgt_size, match)

    # --- Coordinate ranges and resolution ---
    for coord in ("x", "y"):
        if coord in ds_out.coords and coord in target_ds.coords:
            out_arr = ds_out[coord].values
            tgt_arr = target_ds[coord].values
            out_min, out_max = float(out_arr.min()), float(out_arr.max())
            tgt_min, tgt_max = float(tgt_arr.min()), float(tgt_arr.max())

            min_match = abs(out_min - tgt_min) < _TOL
            max_match = abs(out_max - tgt_max) < _TOL
            all_ok = all_ok and min_match and max_match
            _row(f"{coord} min (m)", f"{out_min:.4f}", f"{tgt_min:.4f}", min_match)
            _row(f"{coord} max (m)", f"{out_max:.4f}", f"{tgt_max:.4f}", max_match)

            if len(out_arr) > 1 and len(tgt_arr) > 1:
                out_res = float(np.mean(np.diff(out_arr)))
                tgt_res = float(np.mean(np.diff(tgt_arr)))
                res_match = abs(out_res - tgt_res) < _TOL
                all_ok = all_ok and res_match
                _row(f"{coord} resolution (m)", f"{out_res:.4f}", f"{tgt_res:.4f}", res_match)

    # --- CRS (crs_wkt from spatial_ref) ---
    def _get_crs_wkt(ds):
        for loc in ("coords", "data_vars"):
            if "spatial_ref" in getattr(ds, loc):
                return getattr(ds, loc)["spatial_ref"].attrs.get("crs_wkt")
        return None

    out_crs = _get_crs_wkt(ds_out)
    tgt_crs = _get_crs_wkt(target_ds)
    if tgt_crs is not None:
        crs_match = out_crs == tgt_crs
        all_ok = all_ok and crs_match
        out_crs_display = "present" if out_crs else "missing"
        tgt_crs_display = "present"
        if not crs_match:
            out_crs_display = f"differs ({out_crs_display})"
        _row("CRS (crs_wkt)", out_crs_display, tgt_crs_display, crs_match)

    print(f"    {'─' * (label_w + col_w * 2 + 8)}")
    if all_ok:
        print("  Result: all grid attributes match the legacy target grid.")
    else:
        print("  Result: *** one or more grid attributes do NOT match — check above ***")

    ds_out.close()


def rename_variable_in_file(
    input_file: Path,
    output_file: Path,
    old_var: str,
    new_var: str,
    convert_to_kelvin: bool = False,
    target_grid: Optional[xr.Dataset] = None,
) -> None:
    """
    Rename a variable in a NetCDF file, optionally convert C to K, optionally
    regrid to the legacy target grid, and save to output location.

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
    convert_to_kelvin : bool
        If True, add 273.15 to convert the variable from Celsius to Kelvin
    target_grid : xr.Dataset, optional
        If provided, regrid the output to exactly match this grid (legacy mode).

    Raises
    ------
    ValueError
        If the expected variable name is not found in the file
    """
    ds = xr.open_dataset(input_file)

    if old_var not in ds.data_vars:
        available_vars = list(ds.data_vars)
        ds.close()
        raise ValueError(
            f"Expected variable '{old_var}' not found in {input_file}. "
            f"Available variables: {available_vars}"
        )

    if convert_to_kelvin:
        ds[old_var] = ds[old_var] + CELSIUS_TO_KELVIN_OFFSET
        ds[old_var].attrs["units"] = "K"

    ds_renamed = ds.rename({old_var: new_var})

    if target_grid is not None:
        ds_renamed = regrid_to_legacy_grid(ds_renamed, target_grid)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds_renamed.to_netcdf(output_file, mode="w")

    ds.close()
    ds_renamed.close()

    notes = []
    if convert_to_kelvin:
        notes.append("C -> K")
    if target_grid is not None:
        notes.append("regridded to legacy grid")
    note_str = f" ({', '.join(notes)})" if notes else ""
    print(f"  Processed: {input_file.name} -> {output_file.name}{note_str}")


def process_variable(
    input_dir: Path,
    output_dir: Path,
    old_var: str,
    new_var: str,
    convert_to_kelvin: bool = False,
    target_grid: Optional[xr.Dataset] = None,
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
    convert_to_kelvin : bool
        If True, convert temperature values from Celsius to Kelvin
    target_grid : xr.Dataset, optional
        If provided, regrid each output file to exactly match this grid (legacy mode).
    """
    var_input_dir = input_dir / old_var
    var_output_dir = output_dir / new_var

    if not var_input_dir.exists():
        print(f"Warning: Directory {var_input_dir} does not exist. Skipping {old_var}.")
        return

    if not var_input_dir.is_dir():
        print(f"Warning: {var_input_dir} is not a directory. Skipping {old_var}.")
        return

    nc_files = sorted(var_input_dir.glob("*.nc"))

    if not nc_files:
        print(f"Warning: No .nc files found in {var_input_dir}. Skipping {old_var}.")
        return

    notes = []
    if convert_to_kelvin:
        notes.append("C -> K")
    if target_grid is not None:
        notes.append("legacy regrid")
    note_str = f" ({', '.join(notes)})" if notes else ""
    print(f"\nProcessing variable: {old_var} -> {new_var}{note_str}")
    print(f"  Input directory: {var_input_dir}")
    print(f"  Output directory: {var_output_dir}")
    print(f"  Found {len(nc_files)} NetCDF files")

    first_output_file: Optional[Path] = None
    for nc_file in nc_files:
        new_filename = nc_file.name.replace(old_var, new_var)
        output_file = var_output_dir / new_filename

        try:
            rename_variable_in_file(
                nc_file,
                output_file,
                old_var,
                new_var,
                convert_to_kelvin=convert_to_kelvin,
                target_grid=target_grid,
            )
            if first_output_file is None:
                first_output_file = output_file
        except Exception as e:
            print(f"  ERROR processing {nc_file.name}: {e}")
            sys.exit(1)

    if target_grid is not None and first_output_file is not None:
        print_legacy_grid_comparison(first_output_file, target_grid)


def main():
    """Main execution function."""
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: Input path '{input_dir}' is not a directory.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    var_mapping = create_variable_mapping(args.old, args.new)
    kelvin_vars: Set[str] = set(args.celsius_to_kelvin)

    # Load legacy target grid once; reused across all files
    target_grid: Optional[xr.Dataset] = None
    if args.legacy:
        legacy_grid_file = Path(args.legacy_grid_file).resolve()
        if not legacy_grid_file.exists():
            print(f"Error: Legacy grid file '{legacy_grid_file}' does not exist.")
            sys.exit(1)
        target_grid = xr.open_dataset(legacy_grid_file)

    print("=" * 80)
    print("ERA5 Variable Preparation Script")
    print("=" * 80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    if target_grid is not None:
        print(f"Legacy mode:      ON  ({Path(args.legacy_grid_file).resolve()})")
    print(f"\nVariable mappings:")
    for old, new in var_mapping.items():
        conversion = " [C -> K]" if old in kelvin_vars else ""
        print(f"  {old:15} -> {new}{conversion}")
    print("=" * 80)

    for old_var, new_var in var_mapping.items():
        process_variable(
            input_dir,
            output_dir,
            old_var,
            new_var,
            convert_to_kelvin=(old_var in kelvin_vars),
            target_grid=target_grid,
        )

    if target_grid is not None:
        target_grid.close()

    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
