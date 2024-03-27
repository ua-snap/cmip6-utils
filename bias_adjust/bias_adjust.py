"""

Usage:
    python bias_adjust.py --var_id pr --model GFDL-ESM4 --scenario ssp585 --input_dir /import/beegfs/CMIP6/kmredilla/cmip6_regridding/regrid --reference_dir /beegfs/CMIP6/arctic-cmip6/era5/daily_regrid --output_dir /import/beegfs/CMIP6/kmredilla/bias_adjust/netcdf
"""

import argparse
import datetime
import multiprocessing as mp
from itertools import product
from pathlib import Path
import numpy as np
import xarray as xr
import dask
from dask.distributed import Client
from xclim import sdba
from xclim.sdba.detrending import LoessDetrend
from luts import sim_ref_var_lu, varid_adj_kind_lu, jitter_under_lu


def generate_adjusted_filepaths(output_dir, var_ids, models, scenarios, years):
    """Generate the adjusted filepaths. Args are lists to allow multiple combinations

    Args:
        output_dir (pathlib.Path): path to parent output directory
        var_ids (list): list of variable IDs (str)
        models (list): list of models (str)
        scenarios (list): list of scenarios (str)
        years (list): list of years because the data will be written to one file per year

    Returns:
        list of adjusted filepaths
    """
    tmp_fn = "{var_id}_day_{model}_{scenario}_adjusted_{year}0101-{year}1231.nc"
    adj_fps = [
        output_dir.joinpath(
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


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--input_dir",
        type=str,
        help="Path to directory of simulated data files to be adjusted, with filepath structure <model>/<scenario>/day/<variable ID>/<files>",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        help="Path to directory of reference data with filepath structure <variable ID>/<files>",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to working directory, where outputs and ancillary files will be written",
    )
    parser.add_argument(
        "--no-clobber",
        action="store_true",
        default=False,
        help="Do not overwrite files if they exists in out_dir",
    )
    args = parser.parse_args()

    return (
        args.var_id,
        args.model,
        args.scenario,
        Path(args.input_dir),
        Path(args.reference_dir),
        Path(args.output_dir),
        args.no_clobber,
    )


if __name__ == "__main__":
    (
        var_id,
        model,
        scenario,
        input_dir,
        reference_dir,
        output_dir,
        no_clobber,
    ) = parse_args()

    # get reference files
    ref_var_id = sim_ref_var_lu[var_id]
    ref_start_year = 1993
    ref_end_year = 2022
    ref_years = list(range(ref_start_year, ref_end_year + 1))
    ref_fps = [
        reference_dir.joinpath(ref_var_id).joinpath(
            f"era5_daily_regrid_{ref_var_id}_{year}.nc"
        )
        for year in ref_years
    ]

    cmip6_tmp_fn = "{var_id}_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"

    # get modeled historical files
    hist_start_year = 1993
    hist_end_year = 2014
    hist_years = list(range(hist_start_year, hist_end_year + 1))
    hist_fps = [
        generate_cmip6_fp(input_dir, model, "historical", var_id, year)
        for year in hist_years
    ]

    # get modeled projected files for the remainder of reference period not available in modeled historical
    # these are the years we will be including from ScenarioMIP projections to match the reference period
    sim_ref_start_year = 2015
    sim_ref_end_year = 2022
    sim_ref_years = list(range(sim_ref_start_year, sim_ref_end_year + 1))
    sim_ref_fps = [
        generate_cmip6_fp(input_dir, model, scenario, var_id, year)
        for year in sim_ref_years
    ]

    # get all remaining projected files
    sim_start_year = 2023
    # sim_end_year = 2030
    sim_end_year = 2100
    sim_years = list(range(sim_start_year, sim_end_year + 1))
    sim_fps = [
        generate_cmip6_fp(input_dir, model, scenario, var_id, year)
        for year in sim_years
    ]

    kind = varid_adj_kind_lu[var_id]

    # suggestion from dask for ignoring large chunk warnings
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        # messed around with the dask config a lot. This works but generates lots of GC warnings. Best I found though.
        with Client(n_workers=20, threads_per_worker=1) as client:
            # open connection to data
            hist_ds = xr.open_mfdataset(hist_fps + sim_ref_fps)
            # convert calendar to noleap to match CMIP6
            ref_ds = xr.open_mfdataset(ref_fps).convert_calendar("noleap")

            # need to select ERA5 or ERA5T if present
            if "expver" in ref_ds.variables:
                ref_ds = ref_ds.sel(expver=1).drop_vars("expver")

            # need to re-chunk the data - cannot have multiple chunks along the adjustment dimension (time)
            ref = ref_ds[ref_var_id]
            hist = hist_ds[var_id]
            ref.data = ref.data.rechunk({0: -1, 1: 30, 2: 30})
            hist.data = hist.data.rechunk({0: -1, 1: 30, 2: 30})

            if var_id == "pr":
                # need to set the correct compatible precipitation units for ERA5 if precip
                ref.attrs["units"] = "m d-1"

            # ensure data does not have zeros, depending on variable
            if var_id in jitter_under_lu.keys():
                jitter_under_thresh = jitter_under_lu[var_id]
                ref = sdba.processing.jitter_under_thresh(
                    ref, thresh=jitter_under_thresh
                )
                hist = sdba.processing.jitter_under_thresh(
                    hist, thresh=jitter_under_thresh
                )
            print("jitter done")

            train_kwargs = dict(
                ref=ref,
                hist=hist,
                nquantiles=50,
                group="time.dayofyear",
                window=31,
                kind=kind,
            )
            if var_id == "pr":
                # do the adapt frequency thingy for precipitation data
                train_kwargs.update(adapt_freq_thresh="1 mm d-1")

            dqm = sdba.DetrendedQuantileMapping.train(**train_kwargs)

            # Create the detrending object
            det = LoessDetrend(
                group="time.dayofyear", d=0, niter=1, f=0.2, weights="tricube"
            )

            # create a dataset containing all modeled data to be adjusted
            # need to rechunk this one too, same reason as for training data
            proj_ds = xr.open_mfdataset(hist_fps + sim_ref_fps + sim_fps)
            sim = proj_ds[var_id]
            sim.data = sim.data.rechunk({0: -1, 1: 30, 2: 30})

            # free up memory? dunno if this helps
            del ref
            del hist

            scen = (
                dqm.adjust(sim, extrapolation="constant", interp="nearest", detrend=det)
                # in testing, adjusted outputs are oriented lat, lon, time for some reason
                .transpose("time", "lat", "lon")
            )
            # doing the computation here seems to help with performance
            scen.load()

    adj_years = ref_years + sim_years
    # now write the adjusted data to disk by year
    for year in adj_years:
        adj_fp = generate_adjusted_filepaths(
            output_dir, [var_id], [model], [scenario], [year]
        )[0]
        # ensure dir exists before writing
        adj_fp.parent.mkdir(exist_ok=True, parents=True)
        # re-naming back to var_id and ensuring dataset has same dim ordering as
        #  underlying data array (although this might not matter much)
        out_ds = scen.sel(time=str(year)).to_dataset(name=var_id)[
            ["time", "lat", "lon", var_id]
        ]
        # get the source CMIP6 data file used for the attributes
        src_scenario = "historical" if year < 2015 else scenario
        src_fp = generate_cmip6_fp(input_dir, model, src_scenario, var_id, year)
        out_ds = add_global_attrs(out_ds, src_fp)
        out_ds.to_netcdf(adj_fp)

        print(year, "done", end=", ")
