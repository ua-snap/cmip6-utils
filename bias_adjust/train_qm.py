"""Script to train a quantile mapping adjustment for a given model. Uses fixed historical reference years for training.

Usage:
    python train_qm.py --method qdm --sim_path /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/optimized_inputs/tasmax_day_GFDL-ESM4_historical.zarr --ref_path /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/optimized_inputs/era5_t2max.zarr
"""

import argparse
import shutil
from pathlib import Path

# import icclim
import xarray as xr
import dask
from dask.distributed import Client
from xclim import sdba
from config import ref_tmp_fn, train_tmp_fn
from luts import sim_ref_var_lu, varid_adj_kind_lu, jitter_under_lu


def validate_args(args):
    """Validate the supplied command line args."""

    args.sim_path = Path(args.sim_path)
    args.ref_path = Path(args.ref_path)
    args.train_path = Path(args.train_path)
    if not args.sim_path.exists():
        raise FileNotFoundError(f"Zarr store {args.sim_path} not found.")
    if not args.ref_path.exists():
        raise FileNotFoundError(f"Zarr store {args.ref_path} not found.")
    if not args.train_path.parent.exists():
        raise FileNotFoundError(
            f"Parent directory of requested training outputs directory, {args.train_path.parent},"
            " does not exist, and needs to for this script to run."
        )

    return args


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim_path",
        type=str,
        help="path to zarr store of historical simulation data",
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        help="path to zarr store of reference data",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        help="Path to write trained QM object to",
    )
    args = parser.parse_args()

    args = validate_args(args)

    return (
        # args.method,
        args.sim_path,
        args.ref_path,
        args.train_path,
    )


def get_var_id(ds):
    """Get the variable ID from the dataset."""
    if len(ds.data_vars) > 1:
        raise ValueError("More than one variable found in dataset.")
    var_id = [v for v in ds.data_vars][0]
    return var_id


def ensure_matching_time_coords(hist_ds, ref_ds):
    """Ensure that the time coordinates of two datasets match."""
    if all(ref_ds.time.values == hist_ds.time.values) is False:
        # if the first dates match, and the hours don't, we can just fix that by using the hour used in the historical data
        if (
            ref_ds.time.values[0].strftime("%Y-%m-%d")
            == hist_ds.time.values[0].strftime("%Y-%m-%d")
        ) and (ref_ds.time.size == hist_ds.time.size):
            if ref_ds.time.values.min().hour != hist_ds.time.values.min().hour:
                # just make the hour the same for one of them
                start_time = ref_ds.time.values.min()
                end_time = ref_ds.time.values.max()
                use_hour = hist_ds.time.values.min().hour
                new_ref_times = xr.cftime_range(
                    f"{start_time.year}-{str(start_time.month).zfill(2)}-{str(start_time.day).zfill(2)} {use_hour}:00:00",
                    f"{end_time.year}-{str(end_time.month).zfill(2)}-{str(end_time.day).zfill(2)} {use_hour}:00:00",
                    freq="D",
                    calendar="noleap",
                )
                ref_ds = ref_ds.assign(time=new_ref_times)
                assert all(
                    ref_ds.time.values == hist_ds.time.values
                ), "Hist and ref time values do not match after adjusting hours"

    return hist_ds, ref_ds


if __name__ == "__main__":
    (
        # method,
        sim_path,
        ref_path,
        train_path,
    ) = parse_args()
    with dask.config.set(
        **{
            # "array.slicing.split_large_chunks": False,
            "temporary_directory": "/beegfs/CMIP6/kmredilla/tmp",
        }
    ):
        # I *think* most Chinook nodes will have 28 or more CPUs,
        # so these should be safe n_workers and threads_per_worker values
        with Client(n_workers=4, threads_per_worker=6) as client:
            # open connection to data
            hist_ds = xr.open_zarr(sim_path)
            # convert calendar to noleap to match CMIP6
            ref_ds = xr.open_zarr(ref_path).convert_calendar("noleap")
            hist_ds, ref_ds = ensure_matching_time_coords(hist_ds, ref_ds)

            var_id = get_var_id(hist_ds)
            ref_var_id = sim_ref_var_lu[var_id]
            kind = varid_adj_kind_lu[var_id]

            if var_id == "pr":
                # need to set the correct compatible precipitation units for ERA5 if precip
                ref_ds[ref_var_id].attrs["units"] = "m d-1"

            # ensure data does not have zeros, depending on variable
            if var_id in jitter_under_lu.keys():
                jitter_under_thresh = jitter_under_lu[var_id]
                ref_ds[ref_var_id] = sdba.processing.jitter_under_thresh(
                    ref_ds[ref_var_id], thresh=jitter_under_thresh
                )
                hist_ds[var_id] = sdba.processing.jitter_under_thresh(
                    hist_ds[var_id], thresh=jitter_under_thresh
                )
            print("jitter done")
            train_kwargs = dict(
                ref=ref_ds[ref_var_id],
                hist=hist_ds[var_id],
                nquantiles=50,
                group="time.dayofyear",
                window=31,
                kind=kind,
            )
            if var_id == "pr":
                # do the adapt frequency thingy for precipitation data
                train_kwargs.update(adapt_freq_thresh="1 mm d-1")

            qm_train = sdba.DetrendedQuantileMapping.train(**train_kwargs)

            print(f"Writing QDM object to {train_path}")
            if train_path.exists():
                shutil.rmtree(train_path, ignore_errors=True)
            qm_train.ds.to_zarr(train_path)
