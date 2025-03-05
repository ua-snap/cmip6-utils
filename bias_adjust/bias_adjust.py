"""Script for bias adjusting a given model and scenario. Uses a pre-trained quantile mapping adjustment object.

Usage:
    python bias_adjust.py --train_path /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/trained/tasmax_GFDL-ESM4_trained.zarr --sim_path /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/optimized_inputs/tasmax_day_GFDL-ESM4_ssp245.zarr --adj_path /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/det_testing/tasmax_GFDL-ESM4_ssp245_{det_config}.zarr

    # for detrend testing
    python bias_adjust.py --train_path /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/trained/tasmax_GFDL-ESM4_trained.zarr --sim_path /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/optimized_inputs/tasmax_day_GFDL-ESM4_historical.zarr --adj_path /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/det_testing/tasmax_GFDL-ESM4_historical_{det_config}.zarr --det_config det0 --region Fairbanks
"""

import argparse
import datetime
import logging
import re
import shutil

# import multiprocessing as mp
from itertools import product
from pathlib import Path
import xarray as xr
import dask
from dask.distributed import Client
from xclim import sdba
from xclim.sdba.detrending import NoDetrend, LoessDetrend, MeanDetrend, PolyDetrend
from config import ref_tmp_fn, cmip6_tmp_fn, train_tmp_fn
from luts import jitter_under_lu
from train_qm import get_var_id


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# for detrend testing
# remove when done
detrend_configs = {
    "det0": NoDetrend(group="time.dayofyear"),
    "det1": LoessDetrend(
        group="time.dayofyear", d=0, niter=1, f=0.2, weights="tricube"
    ),
    "det2": MeanDetrend(group="time.dayofyear"),
    "det3": PolyDetrend(group="time.dayofyear", degree=1),
    "det4": PolyDetrend(group="time.dayofyear", degree=2),
    "det5": PolyDetrend(group="time.dayofyear", degree=3),
}

regions = {
    "Fairbanks": {"x": slice(2.5e5, 5e5), "y": slice(1.75e6, 1.5e6)},
    "MatSu": {"x": slice(0.5e5, 3e5), "y": slice(1.35e6, 1.1e6)},
    "Yakutat": {"x": slice(6.5e5, 9e5), "y": slice(1.25e6, 1e6)},
}


def extract_values_from_format(format_str, formatted_str, keys):
    """Extract values from a formatted string using the original format string"""
    # Create a regex pattern from the format string
    pattern = re.sub(r"{(\w+)}", r"(?P<\1>.+)", format_str)
    match = re.match(pattern, formatted_str)
    if match:
        return {key: match.group(key) for key in keys}
    else:
        raise ValueError(f"Keys {keys} not found in the formatted string")


def validate_train_fp(train_fp, model, scenario, var_id):
    """Validate the train_fp by checking the attributes in the filename against the script parameters"""
    train_fp_attrs = extract_values_from_format(
        train_tmp_fn, train_fp.name, ["method", "var_id", "model", "scenario"]
    )
    if (
        train_fp_attrs["model"] != model
        or train_fp_attrs["scenario"] != scenario
        or train_fp_attrs["var_id"] != var_id
    ):
        raise ValueError(
            f"Model, scenario, or var_id in train_fp does not match the input values"
        )
    return train_fp


def generate_adjusted_filepaths(adj_dir, var_ids, models, scenarios, years):
    """Generate the adjusted filepaths. Args are lists to allow multiple combinations

    Args:
        adj_dir (pathlib.Path): path to parent output directory
        var_ids (list): list of variable IDs (str)
        models (list): list of models (str)
        scenarios (list): list of scenarios (str)
        years (list): list of years because the data will be written to one file per year

    Returns:
        list of adjusted filepaths
    """
    tmp_fn = "{var_id}_day_{model}_{scenario}_adjusted_{year}0101-{year}1231.nc"
    adj_fps = [
        adj_dir.joinpath(
            model,
            scenario,
            "day",
            var_id,
            tmp_fn.format(model=model, scenario=scenario, var_id=var_id, year=year),
        )
        for model, scenario, var_id, year in product(models, scenarios, var_ids, years)
    ]

    return adj_fps


def generate_cmip6_fp(input_dir, model, scenario, var_id, year):
    """Get the filepath to a cmip6 file in input_dir given the attributes

    Args:
        input_dir (pathlib.Path): path to the input directory
        model (str): model being processed
        scenario (str): scenario being processed
        var_id (str): variable ID being processed
        year (int/str): year being processed

    Returns:
        src_fp (pathlib.Path): Path to the source CMIP6 file
    """
    src_fp = input_dir.joinpath(
        model,
        scenario,
        "day",
        var_id,
        cmip6_tmp_fn.format(var_id=var_id, model=model, scenario=scenario, year=year),
    )

    return src_fp


def get_sim_fps(input_dir, model, scenario, var_id, start_year, end_year):
    """Get the filepaths to the simulated data to be adjusted"""
    sim_years = list(range(start_year, end_year + 1))
    sim_fps = [
        generate_cmip6_fp(input_dir, model, scenario, var_id, year)
        for year in sim_years
    ]
    return sim_fps


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


def parse_sim_filename(sim_tmp_fn, sim_path):
    """Parse the filename of a simulated data file to get the variable ID, model, and scenario"""

    sim_fp_attrs = extract_values_from_format(
        sim_tmp_fn, sim_path.name, ["var_id", "model", "scenario"]
    )
    return sim_fp_attrs["var_id"], sim_fp_attrs["model"], sim_fp_attrs["scenario"]


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
    parser.add_argument(
        "--det_config",
        type=str,
        help="Detrending configuration",
        default="det0",
    )
    parser.add_argument(
        "--region",
        type=str,
        help="region for subsetting",
        default=None,
    )
    args = parser.parse_args()

    return (
        Path(args.train_path),
        Path(args.sim_path),
        args.adj_path,
        args.det_config,
        args.region,
    )


if __name__ == "__main__":
    train_path, sim_path, adj_path, det_config, region = parse_args()
    if region:
        sample_sel = regions[region]
    else:
        sample_sel = {}

    # suggestion from dask for ignoring large chunk warnings
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        # fewer workers and more threads is better for non-GIL like Numpy etc
        with Client(n_workers=4, threads_per_worker=6) as client:
            # open connection to trained QM dataset
            train_ds = xr.open_zarr(train_path).sel(**sample_sel)
            qm = sdba.DetrendedQuantileMapping.from_dataset(train_ds)
            # Create the detrending object
            det = detrend_configs[det_config]

            # create a dataset containing all projected data to be adjusted
            # not adjusting historical, no need for now
            # need to rechunk this one too, same reason as for training data
            sim_ds = xr.open_zarr(sim_path).sel(**sample_sel)
            var_id = get_var_id(sim_ds)

            scen = qm.adjust(
                sim_ds[var_id],
                extrapolation="constant",
                interp="nearest",
                detrend=det,
            )
            scen_ds = scen.to_dataset(name=var_id)
            logging.info(f"Running adjustment and loading into memory")
            # scen_ds.load()

            adj_path = Path(adj_path.format(det_config=det_config, region=region))
            if adj_path.exists():
                logging.info(f"Adjusted data store exists, removing ({adj_path}).")
                shutil.rmtree(adj_path, ignore_errors=True)

            logging.info(f"Writing adjusted data to {adj_path}")
            scen_ds.to_zarr(adj_path)
