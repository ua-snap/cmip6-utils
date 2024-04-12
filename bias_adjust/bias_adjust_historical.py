"""Script to adjust historical data for a given model. Will simply omit the ScenarioMIP years and shift the overlapping historical years back. 

Usage:
    python bias_adjust.py --var_id pr --model GFDL-ESM4 --input_dir /import/beegfs/CMIP6/kmredilla/cmip6_regridding/regrid --reference_dir /beegfs/CMIP6/arctic-cmip6/era5/daily_regrid --adj_dir /import/beegfs/CMIP6/kmredilla/bias_adjust/netcdf
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
from bias_adjust import (
    generate_adjusted_filepaths,
    generate_cmip6_fp,
    add_global_attrs,
    drop_height,
)
from config import ref_tmp_fn, cmip6_tmp_fn
from luts import sim_ref_var_lu, varid_adj_kind_lu, jitter_under_lu


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
        "--adj_dir",
        type=str,
        help="Path to adjusted data output directory, for data only",
    )
    args = parser.parse_args()

    return (
        args.var_id,
        args.model,
        Path(args.input_dir),
        Path(args.reference_dir),
        Path(args.adj_dir),
    )


if __name__ == "__main__":
    (
        var_id,
        model,
        input_dir,
        reference_dir,
        adj_dir,
    ) = parse_args()

    scenario = "historical"

    # get reference files
    ref_var_id = sim_ref_var_lu[var_id]
    # We will call the "hist" years those which are the same as the reference years
    #  which we will be using for training. We will actually adjust further back than
    #  this, back to 1950.
    ref_start_year = 1981
    ref_end_year = 2014
    ref_years = list(range(ref_start_year, ref_end_year + 1))
    ref_fps = [
        reference_dir.joinpath(ref_var_id).joinpath(
            ref_tmp_fn.format(ref_var_id=ref_var_id, year=year)
        )
        for year in ref_years
    ]

    # get the modeled historical file we will be training with (hist_ref)
    hist_ref_fps = [
        generate_cmip6_fp(input_dir, model, scenario, var_id, year)
        for year in ref_years
    ]

    # get modeled historical files that we will be adjusting
    hist_start_year = 1950
    hist_end_year = 2014
    hist_years = list(range(hist_start_year, hist_end_year + 1))
    hist_fps = [
        generate_cmip6_fp(input_dir, model, scenario, var_id, year)
        for year in hist_years
    ]

    kind = varid_adj_kind_lu[var_id]

    # suggestion from dask for ignoring large chunk warnings
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        # messed around with the dask config a lot. This works but generates lots of GC warnings. Best I found though.
        with Client(n_workers=20, threads_per_worker=1) as client:
            # open connection to data
            hist_ref_ds = xr.open_mfdataset(hist_ref_fps)
            # convert calendar to noleap to match CMIP6
            ref_ds = xr.open_mfdataset(ref_fps).convert_calendar("noleap")

            # height variable is causing some issues in some cases and we don't need
            #  it for the bias-adjusted stuff, just drop it.
            hist_ref_ds = drop_height(hist_ref_ds)
            ref_ds = drop_height(ref_ds)

            # need to re-chunk the data - cannot have multiple chunks along the adjustment dimension (time)
            ref = ref_ds[ref_var_id]
            hist = hist_ref_ds[var_id]
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

            # create a dataset containing all historical data to be adjusted
            # calling this sim to denote that these are all the simulated data we want to adjust
            # need to rechunk this one too, same reason as for training data
            sim_ds = xr.open_mfdataset(hist_fps)
            sim_ds = drop_height(sim_ds)
            sim = sim_ds[var_id]
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

    adj_years = hist_years
    # now write the adjusted data to disk by year
    for year in adj_years:
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
        src_fp = generate_cmip6_fp(input_dir, model, scenario, var_id, year)
        out_ds = add_global_attrs(out_ds, src_fp)
        out_ds.to_netcdf(adj_fp)

        print(year, "done", end=", ")
print()
