"""Generate a netcdf dataset of DOY summaries for subsequent quality control of bias adjusted outputs.

Example usage:
python generate_doy_summaries.py --working_dir /import/beegfs/CMIP6/crstephenson --sim_dir /import/beegfs/CMIP6/arctic-cmip6/regrid  --var_id pr --model GFDL-ESM4 --scenario ssp585

Output summary files will be written to working_dir/<config.output_dir_name>/qc/doy_summaries/*.nc
"""

import argparse
from pathlib import Path
import xarray as xr
from xclim import units
from config import (
    output_dir_name,
    qc_dir_name,
    doy_summary_tmp_fn,
    doy_summary_dir_name,
)
from bias_adjust import generate_cmip6_fp
from slurm import get_directories

PR_UNITS = "kg m-2 s-1"


def get_sim_fps(sim_dir, adj_dir, model, scenario, var_id, years):
    """Return list of both original and adjusted simulated (model) filepaths"""

    sim_fps = [
        generate_cmip6_fp(sim_dir, model, scenario, var_id, year) for year in years
    ]

    adj_fps = sorted(
        list(adj_dir.joinpath(model, scenario, "day", var_id).glob("*.nc"))
    )

    return sim_fps, adj_fps


def add_dims(ds, kind, scenario, var_id, model):
    return ds.assign_coords(
        kind=kind, scenario=scenario, var_id=var_id, model=model
    ).expand_dims(["kind", "scenario", "var_id", "model"])


def force_precip_flux(da):
    """Just ensures precip data has standardized units"""
    da = units.convert_units_to(da, PR_UNITS)

    return da


def add_units(ds, units):
    for var_id in ["min", "mean", "max"]:
        ds[var_id].attrs["units"] = units

    return ds


def open_and_extract_stats(fps, dim_kwargs):
    with xr.open_mfdataset(fps) as ds:
        # just load all the data at once if possible? should be only couple of GBs max
        da = ds[dim_kwargs["var_id"]].load()

        if dim_kwargs["var_id"] == "pr":
            da = force_precip_flux(da)

        doy_da = da.groupby("time.dayofyear")
        stat_ds = xr.merge(
            [
                doy_da.min().rename("min"),
                doy_da.mean().rename("mean"),
                doy_da.max().rename("max"),
            ]
        )

    stat_ds = add_units(stat_ds, da.attrs["units"])
    stat_ds = add_dims(stat_ds, **dim_kwargs)
    del da

    if "height" in stat_ds.coords:
        stat_ds = stat_ds.drop_vars("height")
    if "spatial_ref" in stat_ds.coords:
        stat_ds = stat_ds.drop_vars("spatial_ref")

    return stat_ds


def parse_args():
    """Parse some arguments"""
    # parameters cell
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
        "--working_dir",
        type=str,
        help="Working directory, which will be used to derive the adjusted output directory from config",
        required=True,
    )
    parser.add_argument(
        "--sim_dir",
        type=str,
        help="Directory of model output, likely (regridded output)",
        required=True,
    )
    args = parser.parse_args()

    return (
        args.var_id,
        args.model,
        args.scenario,
        Path(args.working_dir),
        Path(args.sim_dir),
    )


if __name__ == "__main__":
    (var_id, model, scenario, working_dir, sim_dir) = parse_args()

    if scenario == "historical":
        years = list(range(1951, 2014 + 1))
    else:
        years = list(range(2015, 2100 + 1))

    working_dir = Path(working_dir)
    sim_dir = Path(sim_dir)
    adj_dir = working_dir.joinpath("/" + output_dir_name + "/netcdf")
    cmip6_tmp_fn = "{var_id}_day_{model}_{scenario}_adjusted_{year}0101-{year}1231.nc"

    output_dir, adj_dir = get_directories(working_dir, output_dir_name)
    log_dir = "."

    models = [fp.name for fp in sim_dir.glob("*")]

    out_dir = working_dir.joinpath(output_dir_name, qc_dir_name, doy_summary_dir_name)
    out_dir.mkdir(exist_ok=True, parents=True)

    sim_fps, adj_fps = get_sim_fps(sim_dir, adj_dir, model, scenario, var_id, years)
    if len(adj_fps) == 0:
        print("No adjusted data found for ", model, scenario, var_id)
        exit(1)

    dim_kwargs = dict(scenario=scenario, var_id=var_id, model=model)
    dim_kwargs.update(kind="sim")
    proj_sim_ds = open_and_extract_stats(sim_fps, dim_kwargs)
    out_fp = out_dir.joinpath(doy_summary_tmp_fn.format(**dim_kwargs))
    proj_sim_ds.to_netcdf(out_fp)
    print(model, scenario, var_id, "sim done")

    dim_kwargs.update(kind="adj")
    proj_adj_ds = open_and_extract_stats(adj_fps, dim_kwargs)
    out_fp = out_dir.joinpath(doy_summary_tmp_fn.format(**dim_kwargs))
    proj_adj_ds.to_netcdf(out_fp)
    print(model, scenario, var_id, "adj done")
