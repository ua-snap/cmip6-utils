"""Generate a netcdf dataset of DOY summaries for subsequent quality control of bias adjusted outputs.

Example usage:
    python generate_doy_summaries.py --working_dir /import/beegfs/CMIP6/kmredilla --sim_dir /import/beegfs/CMIP6/kmredilla/cmip6_regridding/regrid

Output summary file will be written to working_dir/<config.output_dir_name>/qc/doy_summaries.nc
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


def get_sim_fps(model, scenario, var_id):
    """Return list of both original and adjusted simulated (model) filepaths"""
    years = hist_years if scenario == "historical" else proj_years
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
    """ "Just ensures precip data has standardized units"""
    da = units.convert_units_to(da, "kg m-2 s-1")

    return da


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
        Path(args.working_dir),
        Path(args.sim_dir),
    )


if __name__ == "__main__":
    working_dir, sim_dir = parse_args()
    # some other useful global variables

    hist_start_year = 1951
    hist_end_year = 2014
    hist_years = list(range(hist_start_year, hist_end_year + 1))

    proj_start_year = 2015
    proj_end_year = 2100
    proj_years = list(range(proj_start_year, proj_end_year + 1))

    working_dir = Path(working_dir)
    sim_dir = Path(sim_dir)
    cmip6_tmp_fn = "{var_id}_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"

    output_dir, adj_dir = get_directories(working_dir, output_dir_name)
    log_dir = "."

    models = [fp.name for fp in sim_dir.glob("*")]

    out_dir = working_dir.joinpath(output_dir_name, qc_dir_name, doy_summary_dir_name)
    out_dir.mkdir(exist_ok=True, parents=True)
    # just iterate and write results for each combination
    for model in models:
        for var_id in ["pr", "tasmin", "tasmax"]:
            for scenario in ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]:

                sim_fps, adj_fps = get_sim_fps(model, scenario, var_id)
                if len(adj_fps) == 0:
                    print("No adjusted data found for ", model, scenario, var_id)
                    continue

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
