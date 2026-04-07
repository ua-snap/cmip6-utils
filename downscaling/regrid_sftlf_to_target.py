"""Script to regrid an sftlf (land fraction) file to a target grid resolution.

This is used in cascade regridding to create land masks for intermediate grids.
Uses nearest-neighbor interpolation to preserve sharp land/ocean boundaries.

Example usage:
    python regrid_sftlf_to_target.py \
        --source_sftlf /path/to/source_sftlf.nc \
        --target_grid /path/to/target_grid.nc \
        --output_sftlf /path/to/output_sftlf.nc
"""

import argparse
import logging
import xarray as xr
import xesmf as xe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_sftlf",
        type=str,
        help="Path to source sftlf file to regrid",
        required=True,
    )
    parser.add_argument(
        "--target_grid",
        type=str,
        help="Path to target grid file (provides target resolution)",
        required=True,
    )
    parser.add_argument(
        "--output_sftlf",
        type=str,
        help="Path to write regridded sftlf file",
        required=True,
    )
    args = parser.parse_args()
    return args.source_sftlf, args.target_grid, args.output_sftlf


def regrid_sftlf(source_sftlf_path, target_grid_path, output_sftlf_path):
    """Regrid sftlf file to target grid resolution.

    Args:
        source_sftlf_path (str): Path to source sftlf file
        target_grid_path (str): Path to target grid file
        output_sftlf_path (str): Path to output regridded sftlf file
    """
    logger.info(f"Loading source sftlf from {source_sftlf_path}")
    source_ds = xr.open_dataset(source_sftlf_path)

    logger.info(f"Loading target grid from {target_grid_path}")
    target_ds = xr.open_dataset(target_grid_path)

    # Extract sftlf variable
    if "sftlf" not in source_ds:
        raise ValueError(f"Variable 'sftlf' not found in {source_sftlf_path}")

    sftlf_var = source_ds["sftlf"]

    # If sftlf has a time dimension, drop it
    if "time" in sftlf_var.dims:
        logger.info("Dropping time dimension from sftlf")
        sftlf_var = sftlf_var.isel(time=0, drop=True)

    logger.info(f"Source sftlf shape: {sftlf_var.shape}")
    logger.info(
        f"Target grid shape: lat={target_ds.dims.get('lat', target_ds.dims.get('y'))}, "
        f"lon={target_ds.dims.get('lon', target_ds.dims.get('x'))}"
    )

    # Use nearest-neighbor to preserve sharp land/ocean boundaries
    # nearest_s2d means "nearest source to destination"
    logger.info("Creating regridder with nearest_s2d method...")
    regridder = xe.Regridder(
        sftlf_var,
        target_ds,
        method="nearest_s2d",
        unmapped_to_nan=False,  # Keep all values, don't introduce NaNs
        ignore_degenerate=True,
    )

    logger.info("Regridding sftlf to target grid...")
    regridded_sftlf = regridder(sftlf_var, keep_attrs=True)

    # Convert to dataset and preserve attributes
    output_ds = regridded_sftlf.to_dataset(name="sftlf")

    # Ensure proper attributes
    output_ds["sftlf"].attrs.update(
        {
            "long_name": "percentage_of_the_grid_cell_occupied_by_land_including_lakes",
            "standard_name": "land_area_fraction",
            "units": "%",
            "valid_min": 0.0,
            "valid_max": 100.0,
        }
    )

    # Add global attributes
    output_ds.attrs.update(
        {
            "title": "Regridded land fraction (sftlf)",
            "source": f"Regridded from {source_sftlf_path}",
            "regridding_method": "nearest_s2d",
        }
    )

    logger.info(f"Writing regridded sftlf to {output_sftlf_path}")
    output_ds.to_netcdf(output_sftlf_path)

    logger.info(f"✓ Successfully created regridded sftlf file")
    logger.info(f"  Output shape: {regridded_sftlf.shape}")
    logger.info(
        f"  Land fraction range: [{float(regridded_sftlf.min()):.2f}, {float(regridded_sftlf.max()):.2f}]%"
    )
    logger.info(
        f"  Land pixels (>0%%): {int((regridded_sftlf > 0).sum())} / {int(regridded_sftlf.size)}"
    )


if __name__ == "__main__":
    source_sftlf, target_grid, output_sftlf = parse_args()
    regrid_sftlf(source_sftlf, target_grid, output_sftlf)
