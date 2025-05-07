"""This script creates a grid with 0.5 degree resolution on 0-360 degree longitude for a domain slightly larger than the 4km ERA5 WRF data

# ERA5 extent is slightly smaller than this: (-177, 54, -128, 73)
# so we will create a grid with 0.5 degree resolution on 0-360 degree longitude

Example usage:
    python make_intermediate_target_grid_file.py \
        --src_file /beegfs/CMIP6/arctic-cmip6/CMIP6/ScenarioMIP/NCAR/CESM2/ssp370/r11i1p1f1/Amon/tas/gn/v20200528/tas_Amon_CESM2_ssp370_r11i1p1f1_gn_206501-210012.nc \
        --out_file /center1/CMIP6/kmredilla/cmip6_4km_downscaling/intermediate_target.nc
"""

import argparse
import logging
import xarray as xr
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src_file",
        type=str,
        help="Path to use as source for creating intermediate target grid",
        required=True,
    )
    parser.add_argument(
        "--out_file",
        type=str,
        help="Path to write intermediate target grid file for cascade regridding",
        required=True,
    )
    args = parser.parse_args()

    return (
        args.src_file,
        args.out_file,
    )


def get_num(min_val, max_val, step):
    """Get the value needed for num param of numpy.linspace
    Args:
        min_val (float): minimum value
        max_val (float): maximum value
        step (float): step size

    Returns:
        int: number of values needed for linspace
    """
    return int((max_val - min_val) / step) + 1


def create_intermediate_target_grid(src_file, out_file):
    """Create intermediate target grid for regridding
    Args:
        src_file (str): path to input file
        out_file (str): path to output file
    """
    # hardcoded values for just larger than the 4km ERA5 WRF data
    min_lon, max_lon = 183, 232
    lon_num = get_num(min_lon, max_lon, 0.5)
    min_lat, max_lat = 54, 73
    lat_num = get_num(min_lat, max_lat, 0.5)

    new_lon = np.linspace(min_lon, max_lon, lon_num)
    new_lat = np.linspace(min_lat, max_lat, lat_num)

    ds = xr.open_dataset(src_file)
    # just a catch to help ensure we have a typical CMIP6 file with lon values in increasing order (i.e. 0 - 360)
    assert (
        ds.lon.values[0] < ds.lon.values[-1]
    ), "Longitude values are not in increasing order"
    mid_res_ds = ds.isel(time=0, drop=True).interp(
        lat=new_lat, lon=new_lon, method="linear"
    )
    del mid_res_ds.encoding["unlimited_dims"]

    logger.info(
        f"Creating intermediate target grid file at {out_file} with {lon_num} lon and {lat_num} lat points at {0.5} degree resolution"
    )
    mid_res_ds.to_netcdf(out_file)


if __name__ == "__main__":
    src_file, out_file = parse_args()
    create_intermediate_target_grid(src_file, out_file)
