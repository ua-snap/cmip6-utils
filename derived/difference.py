"""Script for computing the difference between two variables and saving it as a new variable.
This will be used primarily for computing daily temperature range (dtr) from tmax and tmin data, 
and for computing tasmin from tmax and dtr data.
This script is designed to work with zarr stores as inputs. 

Example usage:
    python difference.py \
        --minuend_store /center1/CMIP6/kmredilla/cmip6_4km_downscaling/adjusted/tasmax_GFDL-ESM4_historical_adjusted.zarr \
        --subtrahend_store /center1/CMIP6/kmredilla/cmip6_4km_downscaling/adjusted/dtr_GFDL-ESM4_historical_adjusted.zarr \
        --output_store /center1/CMIP6/kmredilla/cmip6_4km_downscaling/derived/tasmin/tasmin_GFDL-ESM4_historical_adjusted.zarr \
        --new_var_id tasmin
"""

import argparse
import logging
import shutil
from pathlib import Path
import numpy as np
import xarray as xr
import string
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minuend_store",
        type=str,
        help="Directory containing 'minued' data (that from which subtrahend is subtracted)",
        required=True,
    )
    parser.add_argument(
        "--subtrahend_store",
        type=str,
        help="Directory containing 'subtrahend' data (that which is subtracted from minuend)",
        required=True,
    )
    parser.add_argument(
        "--output_store",
        type=str,
        help="Directory for writing difference data.",
        required=True,
    )
    parser.add_argument(
        "--new_var_id",
        type=str,
        help="New variable id for the resulting difference data",
    )
    args = parser.parse_args()

    return (
        Path(args.minuend_store),
        Path(args.subtrahend_store),
        Path(args.output_store),
        args.new_var_id,
    )


def get_var_id(ds):
    """Get the variable id from the dataset attributes.
    This is a helper function for getting the variable id from the dataset attributes.
    """
    if "variable_id" in ds.attrs.keys():
        var_id = ds.attrs["variable_id"]
        assert var_id in ds.data_vars, f"{var_id} not in {ds.data_vars}"
    else:
        valid_vars = [var for var in ds.data_vars if set(ds[var].dims) == set(ds.dims)]
        assert (
            len(valid_vars) == 1
        ), f"Dataset must have exactly one variable indexed by all dimensions. Found: {valid_vars}"
        var_id = valid_vars[0]

    return var_id


if __name__ == "__main__":
    minuend_store, subtrahend_store, output_store, new_var_id = parse_args()

    with xr.open_dataset(minuend_store, engine="zarr") as minu_ds:
        with xr.open_dataset(subtrahend_store, engine="zarr") as subtr_ds:
            minu_var_id = get_var_id(minu_ds)
            subtr_var_id = get_var_id(subtr_ds)
            minuend = minu_ds[minu_var_id]
            diff = minuend - subtr_ds[subtr_var_id]

    # get units and make sure they are the same between both inputs
    units = minuend.attrs["units"]
    assert units == subtr_ds[subtr_var_id].attrs["units"]

    diff.name = new_var_id
    diff.attrs = {
        "units": units,
    }
    diff.encoding = minuend.encoding

    # the list here at the end is just making sure we have a matching dim order
    diff_ds = diff.to_dataset().transpose(*list(minuend.dims))
    diff_ds.attrs = {k: v for k, v in minu_ds.attrs.items() & subtr_ds.attrs.items()}
    # give this a variable_id attribute for consistency (helps with e.g. regridding with regrid.py)
    diff_ds.attrs["variable_id"] = new_var_id

    logging.info(f"Writing { diff_ds.attrs["variable_id"]} to {output_store}")
    if output_store.exists():
        logging.info(f"Deleting existing {output_store}")
        shutil.rmtree(output_store)

    diff_ds.to_zarr(output_store)
