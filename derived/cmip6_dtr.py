"""Script for making daily / diurnal temperature range (dtr) data to be used for processing CMIP6 data.
Note, ERA5 dtr is processed in the ERA5 prep notebook.

Example usage:
    python cmip6_dtr.py --tasmax_dir /import/beegfs/CMIP6/arctic-cmip6/regrid/GFDL-ESM4/historical/day/tasmax --tasmin_dir /import/beegfs/CMIP6/arctic-cmip6/regrid/GFDL-ESM4/historical/day/tasmin --output_dir /import/beegfs/CMIP6/snapdata/dtr_processing/netcdf/GFDL-ESM4/historical/day/dtr
"""

import argparse
from pathlib import Path
import dask
from dask.distributed import LocalCluster
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import xarray as xr
from config import dtr_tmp_fn
from slurm_dtr import get_tmax_tmin_fps


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasmax_dir",
        type=str,
        help="Directory containing daily maximum temperature data saved by year (and nothing else)",
        required=True,
    )
    parser.add_argument(
        "--tasmin_dir",
        type=str,
        help="Directory containing daily minimum temperature data saved by year (and nothing else)",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory for writing daily temperature range data",
        required=True,
    )
    args = parser.parse_args()

    return (Path(args.tasmax_dir), Path(args.tasmin_dir), Path(args.output_dir))


if __name__ == "__main__":
    tasmax_dir, tasmin_dir, output_dir = parse_args()
    # keep getting a bunch of errors about unable to remove lock files on Chinook
    dask.config.set({"distributed.worker.use-file-locking": False})

    # assumes all files in one dir have corresponding file in the other
    tasmax_fps, tasmin_fps = get_tmax_tmin_fps(tasmax_dir, tasmin_dir)

    output_dir.mkdir(exist_ok=True)

    with LocalCluster(
        n_workers=int(0.9 * cpu_count()),
        memory_limit="4GB",
    ) as cluster:
        with xr.open_mfdataset(tasmax_fps) as tasmax_ds:
            with xr.open_mfdataset(tasmin_fps) as tasmin_ds:
                dtr = tasmax_ds["tasmax"] - tasmin_ds["tasmin"]
                units = tasmax_ds["tasmax"].attrs["units"]
                assert units == tasmin_ds["tasmin"].attrs["units"]

        dtr.name = "dtr"
        dtr.attrs = {
            "long_name": "Daily temperature range",
            "units": units,
        }
        # replace any negative values (tasmax - tasmin < 0) with 0.0000999
        # using this number instead of zero gives us a way of estimating what spots were tweaked
        # include the isnull() check so we don't replace nan's
        dtr = dtr.where((dtr.isnull() | (dtr >= 0)), 0.0000999)

        # the list here at the end is just making sure we have the time - lat - lon dimensional order
        dtr_ds = dtr.to_dataset()[["time", "lat", "lon", "dtr"]]
        dtr_ds.attrs = {
            k: v for k, v in tasmax_ds.attrs.items() & tasmin_ds.attrs.items()
        }

        # getting info for saving and some extra checks
        start_year = dtr_ds.time.data[0].year
        end_year = dtr_ds.time.data[-1].year
        years = list(range(start_year, end_year + 1))
        assert len(years) == len(tasmax_fps) == len(tasmin_fps)
        model, scenario = [dtr_ds.attrs[a] for a in ["source_id", "experiment_id"]]
        # lol sry
        assert all(
            [
                (model == tasmax_fp.name.split("_")[2] == tasmin_fp.name.split("_")[2])
                for tasmax_fp, tasmin_fp in zip(tasmax_fps, tasmin_fps)
            ]
        )
        assert all(
            [
                (
                    scenario
                    == tasmax_fp.name.split("_")[3]
                    == tasmin_fp.name.split("_")[3]
                )
                for tasmax_fp, tasmin_fp in zip(tasmax_fps, tasmin_fps)
            ]
        )

        # write
        for year in years:
            dtr_ds.sel(time=str(year)).to_netcdf(
                output_dir.joinpath(
                    dtr_tmp_fn.format(model=model, scenario=scenario, year=year)
                )
            )
