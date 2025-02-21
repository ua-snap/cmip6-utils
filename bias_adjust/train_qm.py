"""Script to train a quantile mapping adjustment for a given model. Uses fixed historical reference years for training.

Usage:
    python train_qm.py --method qdm --var_id tasmax --model GFDL-ESM4 --start_year 1984 --end_year 2014 --hist_dir /import/beegfs/CMIP6/kmredilla/cmip6_4km_3338/regrid --reference_dir /beegfs/CMIP6/kmredilla/downscaling/era5_3338 --train_dir /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted/trained
"""

import argparse
from pathlib import Path
import xarray as xr
import dask
from dask.distributed import Client
from xclim import sdba
from bias_adjust import (
    generate_cmip6_fp,
    drop_height,
)
from config import ref_tmp_fn, train_tmp_fn
from luts import sim_ref_var_lu, varid_adj_kind_lu, jitter_under_lu


def validate_args(args):
    """Validate the supplied command line args."""
    if args.method.lower() != "qdm":
        raise ValueError(f"Method {method} not recognized. Only 'qdm' is supported.")

    args.hist_dir = Path(args.hist_dir)
    args.reference_dir = Path(args.reference_dir)
    args.train_dir = Path(args.train_dir)
    if not args.hist_dir.exists():
        raise FileNotFoundError(f"Directory {args.hist_dir} not found.")
    if not any(args.hist_dir.glob("*/" * 4 + "*.nc")):
        raise FileNotFoundError(
            f"No .nc files found in the expected directory structure under {args.hist_dir}"
        )
    if not args.reference_dir.glob("*.nc"):
        raise FileNotFoundError(
            f"No .nc files found in the reference data directory, {args.reference_dir}"
        )
    if not args.train_dir.parent.exists():
        raise FileNotFoundError(
            (
                f"Parent directory of requested training outputs directory, {args.train_dir.parent},"
                " does not exist, and needs to for this script to run."
            )
        )

    try:
        args.start_year = int(args.start_year)
    except ValueError:
        raise ValueError(f"start_year must be an integer, got {args.start_year}")
    try:
        args.end_year = int(args.end_year)
    except ValueError:
        raise ValueError(f"end_year must be an integer, got {args.end_year}")
    if not (1950 <= args.start_year < args.end_year <= 2014):
        raise ValueError(
            (
                f"start_year and end_year must be between 1950 and 2014, with start_year < end_year."
                f" got {args.start_year} and {args.end_year}"
            )
        )
    return args


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        type=str,
        help="Quantile Mapping method to use",
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
        "--start_year",
        type=str,
        help="Starting year of historical simulation and reference data to train on",
        required=True,
    )
    parser.add_argument(
        "--end_year",
        type=str,
        help="Ending year of historical simulated and reference data to train on",
        required=True,
    )
    parser.add_argument(
        "--hist_dir",
        type=str,
        help="Path to directory of simulated data files to be adjusted, with filepath structure <model>/<scenario>/day/<variable ID>/<files>",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        help="Path to directory of reference data with flat file structure containing all variables and years",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        help="Path to adjusted data output directory, for data only",
    )
    args = parser.parse_args()

    args = validate_args(args)

    return (
        args.method,
        args.var_id,
        args.model,
        args.start_year,
        args.end_year,
        args.hist_dir,
        args.reference_dir,
        args.train_dir,
    )


if __name__ == "__main__":
    (
        method,
        var_id,
        model,
        start_year,
        end_year,
        hist_dir,
        reference_dir,
        train_dir,
    ) = parse_args()

    scenario = "historical"

    # get reference files
    # assume we will have the same start and end years for both reference and historical data
    ref_var_id = sim_ref_var_lu[var_id]
    ref_years = list(range(start_year, end_year + 1))
    ref_fps = [
        reference_dir.joinpath(ref_tmp_fn.format(ref_var_id=ref_var_id, year=year))
        for year in ref_years
    ]
    # get the modeled historical file we will be training with (hist_ref)
    hist_ref_fps = [
        generate_cmip6_fp(hist_dir, model, scenario, var_id, year) for year in ref_years
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
            if method == "qdm":
                qm_train = sdba.QuantileDeltaMapping.train(**train_kwargs)

            train_dir.mkdir(exist_ok=True)
            train_fp = train_dir.joinpath(
                train_tmp_fn.format(
                    method=method, var_id=var_id, model=model, scenario=scenario
                )
            )

            print(f"Writing QDM object to {train_fp}")
            qm_train.ds.to_netcdf(train_fp)
