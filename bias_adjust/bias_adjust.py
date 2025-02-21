"""Script for bias adjusting a given model and scenario. Uses a pre-trained quantile mapping adjustment object.

Usage:
    python bias_adjust.py --train_fp /import/beegfs/CMIP6/kmredilla/bias_adjust/trained/qdm_pr_GFDL-ESM4_ssp585.nc --var_id pr --model GFDL-ESM4 --scenario ssp585 --sim_dir /import/beegfs/CMIP6/kmredilla/cmip6_regridding/regrid --adj_dir /import/beegfs/CMIP6/kmredilla/bias_adjust/adjusted
"""

import argparse
import datetime
import re
import multiprocessing as mp
from itertools import product
from pathlib import Path
import numpy as np
import xarray as xr
import dask
from dask.distributed import Client
from xclim import sdba
from xclim.sdba.detrending import LoessDetrend
from config import ref_tmp_fn, cmip6_tmp_fn, train_tmp_fn
from luts import sim_ref_var_lu, varid_adj_kind_lu, jitter_under_lu


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


def drop_height(ds):
    """Function to drop the height variable from an xarray.Dataset if present.

    Args:
        ds (xarray.Dataset): dataset to drop variable from if present

    Returns:
        ds (xarray.Dataset): dataset with no height variable
    """
    try:
        ds = ds.drop_vars("height")
    except ValueError:
        ds = ds

    try:
        ds = ds.drop_vars("spatial_ref")
    except ValueError:
        ds = ds

    return ds


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_fp",
        type=str,
        help="Path to trained quantile mapping adjustment netcdf file",
        required=True,
    )
    parser.add_argument(
        "--var_id",
        type=str,
        help="Variable ID to adjust",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to adjust",
        required=True,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario to adjust",
        required=True,
    )
    parser.add_argument(
        "--sim_start_year",
        type=str,
        help="Starting year of simulated data to be adjusted",
        required=True,
    )
    parser.add_argument(
        "--sim_end_year",
        type=str,
        help="Ending year of simulated data to be adjusted",
        required=True,
    )
    parser.add_argument(
        "--sim_dir",
        type=str,
        help="Path to directory of simulated data files to be adjusted, with filepath structure <model>/<scenario>/day/<variable ID>/<files>",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        help="Path to directory of reference data with filepath structure <variable ID>/<files>",
    )
    parser.add_argument(
        "--adj_dir",
        type=str,
        help="Path to adjusted data output directory, for data only",
    )
    args = parser.parse_args()

    return (
        Path(args.train_fp),
        args.var_id,
        args.model,
        args.scenario,
        sim_start_year,
        sim_end_year,
        Path(args.sim_dir),
        Path(args.reference_dir),
        Path(args.adj_dir),
    )


if __name__ == "__main__":
    (
        train_fp,
        var_id,
        model,
        scenario,
        sim_start_year,
        sim_end_year,
        sim_dir,
        reference_dir,
        adj_dir,
    ) = parse_args()

    train_fp = validate_train_fp(train_fp, model, scenario, var_id)

    sim_fps = get_sim_fps(
        sim_dir, model, scenario, var_id, sim_start_year, sim_end_year
    )

    # get all remaining projected files
    sim_start_year = 2023
    # sim_end_year = 2030
    sim_end_year = 2100
    sim_years = list(range(sim_start_year, sim_end_year + 1))
    sim_fps = [
        generate_cmip6_fp(sim_dir, model, scenario, var_id, year) for year in sim_years
    ]

    kind = varid_adj_kind_lu[var_id]

    # suggestion from dask for ignoring large chunk warnings
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        # messed around with the dask config a lot. This works but generates lots of GC warnings. Best I found though.
        with Client(n_workers=20, threads_per_worker=1) as client:
            # open connection to trained QM dataset
            train_ds = xr.open_dataset(train_fp)
            dqm = sdba.QuantileDeltaMapping.from_dataset(train_ds)

            # Create the detrending object
            det = LoessDetrend(
                group="time.dayofyear", d=0, niter=1, f=0.2, weights="tricube"
            )

            # create a dataset containing all projected data to be adjusted
            # not adjusting historical, no need for now
            # need to rechunk this one too, same reason as for training data
            sim_ds = xr.open_mfdataset(sim_fps)
            sim_ds = drop_height(sim_ds)
            sim = sim_ds[var_id]
            sim.data = sim.data.rechunk({0: -1, 1: 15, 2: 15})

            scen = (
                # dqm.adjust(sim, extrapolation="constant", interp="nearest", detrend=det)
                dqm.adjust(sim, extrapolation="constant", interp="nearest")
            )
            # doing the computation here seems to help with performance
            scen.load()

    # now write the adjusted data to disk by year
    for year in sim_years:
        adj_fp = generate_adjusted_filepaths(
            adj_dir, [var_id], [model], [scenario], [year]
        )[0]
        # ensure dir exists before writing
        adj_fp.parent.mkdir(exist_ok=True, parents=True)
        # re-naming back to var_id and ensuring dataset has same dim ordering as
        #  underlying data array (although this might not matter much)
        out_ds = scen.sel(time=str(year)).to_dataset(name=var_id)[
            ["time", "lat", "lon", var_id]
        ]
        # get the source CMIP6 data file used for the attributes
        src_fp = generate_cmip6_fp(sim_dir, model, scenario, var_id, year)
        out_ds = add_global_attrs(out_ds, src_fp)
        out_ds.to_netcdf(adj_fp)

        print(year, "done", end=", ")
