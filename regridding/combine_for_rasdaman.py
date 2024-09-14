"""Script to combine regridded CMIP6 data for ingestion into rasdaman. 
This is just a simple wrapper for xarray's open_mfdataset function. 
It is assumed that the data has already been regridded and is stored in the following 
directory structure: <model>/<scenario>/<frequency (table ID)>/<variable ID>/<filename>. 
The script will combine all files for all models, scenarios, and the supplied table_id / frequency and the supplied variables into a single xarray dataset and write to disk.
"""

import argparse
import xarray as xr
from pathlib import Path
from config import var_group_id_lu


def get_var_ids(var_group_id):
    """Var_group_id is a string that contains the variable group ID, e.g. v1, v2, etc.
    This is just a mapping that will use the config.py file."""

    var_ids = var_group_id_lu[var_group_id]

    return var_ids


def get_files(var_ids, model, scenario, table_id, regrid_dir):
    """Note, we are using the "table_id" which is specific to CMIP6 and
    contains the frequency information in it but is how the data are organized.
    """
    fps = []

    for var_id in var_ids:
        fps.extend(regrid_dir.glob(f"{model}/{scenario}/{table_id}/{var_id}/*.nc"))

    return fps


def open_and_combine(var_group_id, model, scenario, table_id, regrid_dir, rasda_dir):
    """Rasda_dir is the directory where the combined dataset will be written to disk."""
    var_ids = get_var_ids(var_group_id)
    fps = get_files(var_ids, model, scenario, table_id, regrid_dir)
    ds = xr.open_mfdataset(
        fps,
        coords="all",
        compat="override",
        preprocess=lambda x: x.drop_vars(["spatial_ref", "height"], errors="ignore"),
    )

    ds.to_netcdf(rasda_dir.joinpath(f"{model}_{scenario}_{table_id}_{var_group_id}.nc"))


def parse_args():

    parser = argparse.ArgumentParser(
        description="Combine regridded CMIP6 data for ingestion into rasdaman."
    )
    parser.add_argument(
        "var_group_id", type=str, help="Variable group ID, one of v1_1, v1_2."
    )
    parser.add_argument("table_id", type=str, help="CMIP6 table ID.")
    parser.add_argument(
        "regrid_dir", type=str, help="Directory where regridded data is stored."
    )
    parser.add_argument(
        "rasda_dir",
        type=str,
        help="Directory where combined data will be written to disk.",
    )

    args = parser.parse_args()

    return (
        args.var_group_id,
        args.table_id,
        Path(args.regrid_dir),
        Path(args.rasda_dir),
    )


if __name__ == "__main__":

    var_group_id, table_id, regrid_dir, rasda_dir = parse_args()

    fps = []
    for year in years:
        for var_id in var_ids:
            fps.extend(
                regrid_dir.glob(
                    f"{model}/{scenario}/Amon/{var_id}/*{year}01-{year}12.nc"
                )
            )

    ds = xr.open_mfdataset(
        fps,
        coords="all",
        compat="override",
        preprocess=lambda x: x.drop_vars(["spatial_ref", "height"], errors="ignore"),
    )
