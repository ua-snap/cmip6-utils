"""Generate tasmin data files for all pairs of tasmax and dtr files in the supplied input directories.
Assumes a directory structure of <model>/<scenario>/day/<var_id> where both tasmax and dtr exist as <var_id> values.

Usage:
    python process_adjusted_tasmin.py --input_dir /beegfs/CMIP6/kmredilla/bias_adjust/netcdf
"""

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr


def create_and_write_tasmin(tasmax_fps, tasmin_dir):
    """Create the tasmin dataset from tasmax filepaths and tasmin filepaths.
    Writes to tasmin dir using the same filename structure as tasmax_fps.
    """
    # opening and processing in serial might be a little less efficient but it is more straightforward
    for tasmax_fp in tasmax_fps:
        dtr_fp = tasmax_fp.parent.parent.joinpath(
            "dtr", tasmax_fp.name.replace("tasmax_", "dtr_")
        )
        assert dtr_fp.exists()

        dtr_ds = xr.open_dataset(dtr_fp)
        tasmax_ds = xr.open_dataset(tasmax_fp)
        check_time = 100

        assert tasmax_ds.isel(time=check_time).time == dtr_ds.isel(time=check_time).time

        tasmin_da = tasmax_ds["tasmax"] - dtr_ds["dtr"]
        tasmin_da.name = "tasmin"
        tasmin_da.attrs = tasmax_ds["tasmax"].attrs.copy()
        tasmin_da.attrs.update(
            long_name="Daily Minimum Near-Surface Air Temperature",
            comment="minimum near-surface air temperature, computed from tasmax minus daily temperature range.",
        )
        tasmin_ds = tasmin_da.to_dataset()

        tasmin_ds.attrs = tasmax_ds.attrs
        tasmin_ds.attrs.update(
            creation_date=datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        )

        tasmin_fn = tasmax_fp.name.replace("tasmax", "tasmin")
        tasmin_fp = tasmin_dir.joinpath(tasmin_fn)

        tasmin_ds.to_netcdf(tasmin_fp)


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to directory of adjusted tasmax data, with filepath structure <model>/<scenario>/day/<variable ID>/<files>",
    )
    args = parser.parse_args()

    return Path(args.input_dir)


if __name__ == "__main__":
    input_dir = parse_args()
    # dir structure should be model/scenario/day/var_id
    for scen_dir in input_dir.glob("*/*"):
        dtr_dir = scen_dir.joinpath("day", "dtr")
        tasmax_dir = scen_dir.joinpath("day", "tasmax")
        tasmin_dir = scen_dir.joinpath("day", "tasmin")

        if dtr_dir.exists() and tasmax_dir.exists():
            tasmin_dir.mkdir(exist_ok=True)
            tasmax_fps = tasmax_dir.glob("tasmax*.nc")
            create_and_write_tasmin(tasmax_fps, tasmin_dir)

        print(scen_dir.parent.name, scen_dir.name, "done")
