"""Script to profile the precipitation adjustment for a single model. It will iterate over a pre-defined set of parameter values and run the adjustment + summary for each, for a given model. 

Example usage:
    python profile_pr.py  --model_dir /beegfs/CMIP6/kmredilla/cmip6_regridding/regrid/MIROC6 --ref_dir /import/beegfs/CMIP6/arctic-cmip6/era5/daily_regrid --results_file /beegfs/CMIP6/kmredilla/bias_adjust/profiling/profiling_data/MIROC6_profiling_results.pkl
"""

# module for re-using code for bias adjustment exploratory data analysis

import pickle
import warnings
import argparse
from pathlib import Path
import dask
from dask.distributed import Client
from xclim.sdba.detrending import LoessDetrend
from baeda import *

chunk_size = 15


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to directory of simulated data files for a single model to be profiled, with filepath structure <model>/<scenario>/day/<variable ID>/<files>",
    )
    parser.add_argument(
        "--ref_dir",
        type=str,
        help="Path to directory of reference data with filepath structure <variable ID>/<files>",
    )
    parser.add_argument(
        "--ju_thresh",
        type=str,
        help="Jitter-under threshold for the adjustment.",
    )
    parser.add_argument(
        "--af_thresh",
        type=str,
        help="Frequency adaptation threshold for the adjustment.",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="Path to write the resulting profiling data to. Will be a pickled list of results.",
    )
    args = parser.parse_args()

    return (
        Path(args.model_dir),
        Path(args.ref_dir),
        Path(args.results_file),
        args.ju_thresh,
        args.af_thresh,
    )


def run_profile(model_dir, ref, det, ju_thresh, af_thresh, var_id="pr"):
    results = []
    hist = get_hist(
        model_dir, var_id, chunks=dict(lat_chunk=chunk_size, lon_chunk=chunk_size)
    )
    with warnings.catch_warnings():
        # getting warnings aboutlarge graph size, but have not figured out a way
        #  to silence them here using the dask methods (e.g. dask.config.set)
        warnings.simplefilter("ignore")
        scen = run_adjust(ref, hist, det, ju_thresh, af_thresh)

    results = {
        "ju_thresh": ju_thresh,
        "adapt_freq_thresh": af_thresh,
        "results": summarize(scen),
    }

    print(
        f"Adjustment with jitter-under thresh: {ju_thresh}, frequency adapt thresh: {af_thresh} done",
        flush=True,
    )

    return results


if __name__ == "__main__":
    (
        model_dir,
        ref_dir,
        results_file,
        ju_thresh,
        af_thresh,
    ) = parse_args()

    try:
        results_fn_model = results_file.name.split("_")[0]
        assert model_dir.name == results_fn_model
    except AssertionError:
        raise ValueError(
            f"Model directory name {model_dir.name} does not match results file prefix {results_fn_model}."
        )

    # we have to make some big chunks and this will silence a warning about that
    dask.config.set(
        **{
            "array.slicing.split_large_chunks": False,
        },
    )

    client = Client(n_workers=24, threads_per_worker=1)

    # get the reference data
    ref_var_id = "tp"
    ref_start_year = 1993
    ref_end_year = 2022
    ref_fps = get_era5_fps(ref_dir, ref_var_id, ref_start_year, ref_end_year)
    ref_ds = (
        xr.open_mfdataset(ref_fps)
        .convert_calendar("noleap")
        .sel(expver=1)
        .drop_vars("expver")
    )
    ref = get_rechunked_da(
        ref_ds, ref_var_id, lat_chunk=chunk_size, lon_chunk=chunk_size
    )
    ref.attrs["units"] = "m d-1"
    print("Reference dataset loaded.", flush=True)

    # Create the detrending object for reuse in adjustments
    det = LoessDetrend(group="time.dayofyear", d=0, niter=1, f=0.2, weights="tricube")
    results = run_profile(model_dir, ref, det, ju_thresh, af_thresh, var_id="pr")

    print(f"\nDone. Writing results to {results_file}.")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
