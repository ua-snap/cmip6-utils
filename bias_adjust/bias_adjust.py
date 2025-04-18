"""Script for bias adjusting a given model and scenario. Uses a pre-trained quantile mapping adjustment object.

Example usage:
    python bias_adjust.py --train_path /center1/CMIP6/kmredilla/bias_adjustment_testing/trained_qdm_pr_GFDL-ESM4.zarr --sim_path /center1/CMIP6/kmredilla/zarr_bias_adjust_inputs/pr_GFDL-ESM4_historical.zarr --adj_path /center1/CMIP6/kmredilla/cmip6_4km_3338_downscaled/pr_GFDL-ESM4_historical_adj.zarr
"""

import argparse
import datetime
import logging
import shutil

# import multiprocessing as mp
from itertools import product
from pathlib import Path
import xarray as xr
import dask
from dask.distributed import Client
from xclim import sdba

# from luts import jitter_under_lu
from train_qm import get_var_id


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def add_global_attrs(ds, src_fp):
    """Add global attributes to a new adjusted dataset

    Args:
        ds (xarray.Dataset): dataset of a adjusted data
        src_fp (path-like): path to file where source data was pulled from

    Returns:
        xarray.Dataset with updated global attributes
    """
    with xr.open_dataset(src_fp) as src_ds:
        # create new global attributes
        new_attrs = {
            "history": "File was processed by Scenarios Network for Alaska and Arctic Planning (SNAP) using xclim",
            "contact": "uaf-snap-data-tools@alaska.edu",
            "Conventions": "CF-1.7",
            "creation_date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            # experimental idea - store attributes from source file in an attribute for reference
            # use a string as netcdf cannot be serialized if this is a dict
            # (it can be reconstructed with eval() if desired later)
            "parent_attributes": str(src_ds.attrs),
        }

    ds.attrs = new_attrs

    return ds


def drop_non_coord_vars(ds):
    """Function to drop all coordinates from xarray dataset which are not coordinate variables, i.e. which are not solely indexed by a dimension of the same name

    Args:
        ds (xarray.Dataset): dataset to drop non-coordinate-variables from

    Returns:
        ds (xarray.Dataset): dataset with only dimension coordinates
    """
    coords_to_drop = [coord for coord in ds.coords if ds[coord].dims != (coord,)]
    # some datasets have a variables such as spatial_ref that are indexed by time and should be dropped.
    vars_to_drop = [var for var in ds.data_vars if len(ds[var].dims) < 3]
    ds = ds.drop_vars(coords_to_drop + vars_to_drop)

    return ds


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_path",
        type=str,
        help="Path to trained quantile mapping adjustment netcdf file",
        required=True,
    )
    parser.add_argument(
        "--sim_path",
        type=str,
        help="Path to model data to be adjusted",
        required=True,
    )
    parser.add_argument(
        "--adj_path",
        type=str,
        help="Path to write adjusted data",
        required=True,
    )

    args = parser.parse_args()

    return (
        Path(args.train_path),
        Path(args.sim_path),
        Path(args.adj_path),
    )


def validate_sim_source(train_ds, sim_ds):
    logging.info("Validating source_id")
    assert train_ds.attrs["source_id"] == sim_ds.attrs["source_id"]
    logging.info(
        "Simulated data source (model) validated, sim trained adjustment "
        f"({train_ds.attrs['source_id']}) matches sim ({sim_ds.attrs['source_id']})"
    )


if __name__ == "__main__":
    train_path, sim_path, adj_path = parse_args()

    # fewer workers and more threads is better for non-GIL like Numpy etc
    with Client(n_workers=4, threads_per_worker=6) as client:
        # open connection to trained QM dataset
        train_ds = xr.open_zarr(train_path).chunk({"x": 50, "y": 50})
        qm = sdba.QuantileDeltaMapping.from_dataset(train_ds)

        # create a dataset containing all projected data to be adjusted
        # not adjusting historical, no need for now
        # need to rechunk this one too, same reason as for training data
        sim_ds = xr.open_zarr(sim_path)
        validate_sim_source(train_ds, sim_ds)

        var_id = get_var_id(sim_ds)

        scen = qm.adjust(
            sim_ds[var_id],
            extrapolation="constant",
            interp="nearest",
        )
        scen_ds = scen.to_dataset(name=var_id)
        logging.info(f"Running adjustment and writing to {adj_path}")

        if adj_path.exists():
            logging.info(f"Adjusted data store exists, removing ({adj_path}).")
            shutil.rmtree(adj_path, ignore_errors=True)

        logging.info(f"Writing adjusted data to {adj_path}")
        scen_ds.to_zarr(adj_path)
