"""Script to train a quantile mapping adjustment for a given model. Uses fixed historical reference years for training.

Usage:
    python train_qm.py --var_id tasmax --model GFDL-ESM4 --input_dir /import/beegfs/CMIP6/kmredilla/cmip6_4km_3338/regrid --reference_dir /beegfs/CMIP6/kmredilla/downscaling/era5_3338 --train_dir /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted/trained
"""

import argparse
import multiprocessing as mp
from pathlib import Path
import numpy as np
import xarray as xr
import dask
from dask.distributed import Client
from xclim import sdba
from xclim.sdba.detrending import LoessDetrend
from bias_adjust import (
    generate_cmip6_fp,
    drop_height,
)
from config import ref_tmp_fn, train_tmp_fn
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
        "--train_dir",
        type=str,
        help="Path to adjusted data output directory, for data only",
    )
    args = parser.parse_args()

    return (
        args.var_id,
        args.model,
        Path(args.input_dir),
        Path(args.reference_dir),
        Path(args.train_dir),
    )


if __name__ == "__main__":
    (
        var_id,
        model,
        input_dir,
        reference_dir,
        train_dir,
    ) = parse_args()

    scenario = "historical"

    # get reference files
    ref_var_id = sim_ref_var_lu[var_id]
    # We will call the "hist" years those which are the same as the reference years
    #  which we will be using for training.
    ref_start_year = 1984
    ref_end_year = 2014
    ref_years = list(range(ref_start_year, ref_end_year + 1))
    ref_fps = [
        reference_dir.joinpath(ref_tmp_fn.format(ref_var_id=ref_var_id, year=year))
        for year in ref_years
    ]

    # get the modeled historical file we will be training with (hist_ref)
    hist_ref_fps = [
        generate_cmip6_fp(input_dir, model, scenario, var_id, year)
        for year in ref_years
    ]

    # get modeled historical files that we will be adjusting
    hist_start_year = 1984
    hist_end_year = 2014
    hist_years = list(range(hist_start_year, hist_end_year + 1))
    hist_fps = [
        generate_cmip6_fp(input_dir, model, scenario, var_id, year)
        for year in hist_years
    ]

    kind = varid_adj_kind_lu[var_id]

    # suggestion from dask for ignoring large chunk warnings
    with dask.config.set(
        **{
            "array.slicing.split_large_chunks": False,
            "temporary_directory": "/beegfs/CMIP6/kmredilla/tmp",
        }
    ):
        # messed around with the dask config a lot. This works but generates lots of GC warnings. Best I found though.
        with Client(n_workers=10, threads_per_worker=1) as client:
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

            ref.data = ref.data.rechunk({0: -1, 1: 15, 2: 15})
            hist.data = hist.data.rechunk({0: -1, 1: 15, 2: 15})

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

            # dqm = sdba.DetrendedQuantileMapping.train(**train_kwargs)
            qdm = sdba.QuantileDeltaMapping.train(**train_kwargs)

            train_dir.mkdir(exist_ok=True)
            train_fp = train_dir.joinpath(
                train_tmp_fn.format(var_id=var_id, model=model, scenario=scenario)
            )

            print(f"Writing QDM object to {train_fp}")
            qdm.ds.to_netcdf(train_fp)
