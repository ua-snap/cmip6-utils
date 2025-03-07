"""Resample (aggregate) and reproject the WRF ERA5 from hourly to daily data in EPSG:3338.
This is done for maximum, mean, and minimum temperature, and total precipitation.

Example usage:
    python resample_and_regrid_era5.py --era5_dir /beegfs/CMIP6/wrf_era5/04km --output_dir /beegfs/CMIP6/kmredilla/daily_era5_4km_3338 --year 1965 --geo_file /beegfs/CMIP6/wrf_era5/geo_em.d02.nc
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import xarray as xr
import rioxarray
from dask.distributed import LocalCluster, Client
from pyproj import CRS, Transformer, Proj


OUT_FN_STR = "{var_id}_{year}_era5_4km_3338.nc"


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Aggregate the ERA5 data.")
    parser.add_argument(
        "--era5_dir",
        type=Path,
        required=True,
        help="The directory containing the ERA5 data.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="The directory to save the aggregated data.",
    )
    parser.add_argument(
        "--geo_file",
        type=str,
        help="Path to WRF geo_em file for projection information",
        required=True,
    )
    parser.add_argument(
        "--year",
        type=str,
        help="Year to process",
    )
    parser.add_argument(
        "--fn_str",
        type=str,
        default="era5_wrf_dscale_4km_{date}.nc",
        help="The filename string for the input data.",
    )
    parser.add_argument(
        "--no_clobber",
        action="store_true",
        default=False,
        help="Do not overwrite existing files in rasda_dir.",
    )

    args = parser.parse_args()

    return (
        Path(args.era5_dir),
        Path(args.output_dir),
        Path(args.geo_file),
        args.year,
        args.fn_str,
        args.no_clobber,
    )


def get_year_filepaths(era5_dir, year, fn_str):
    """Get all of the filepaths for a single year of ERA5 data."""
    fps = sorted(era5_dir.joinpath(f"{year}").glob(fn_str.format(date="*")))
    return fps


def get_drop_vars(fp):
    """Get the variables to exclude from the open_mfdataset() call using a single sample file."""

    ds = xr.open_dataset(fp)
    drop_vars = [
        x
        for x in list(ds.variables)
        # these are the variables we want to keep
        if x
        not in ["T2", "rainnc", "Time", "south_north", "west_east", "XLONG", "XLAT"]
    ]
    ds.close()

    return drop_vars


def open_dataset(fps, drop_vars):
    """Open a batch of hourly files and return the dataset"""
    era5_ds = xr.open_mfdataset(
        fps, drop_variables=drop_vars, engine="h5netcdf", parallel=True
    )

    return era5_ds


def resample(era5_ds, agg_var):
    """Resample an hourly dataset to daily resolution using method provided"""
    agg_var_lut = get_agg_var_lut(agg_var)
    var_id, agg_func = agg_var_lut["var_id"], agg_var_lut["agg_func"]

    agg_ds = (
        era5_ds[var_id]
        .resample(Time="1D")
        .apply(agg_func)
        .to_dataset(name=agg_var)
        .rename({"Time": "time"})
    )
    agg_ds[agg_var].attrs["units"] = era5_ds[var_id].attrs["units"]

    era5_ds.close()
    del era5_ds

    return agg_ds


def get_agg_var_lut(agg_var):
    """Look up table for the aggregation function for each variable"""
    lut = {
        "t2min": {"var_id": "T2", "agg_func": lambda x: x.min(dim="Time")},
        "t2": {"var_id": "T2", "agg_func": lambda x: x.mean(dim="Time")},
        "t2max": {"var_id": "T2", "agg_func": lambda x: x.max(dim="Time")},
        "pr": {"var_id": "rainnc", "agg_func": lambda x: x.sum(dim="Time")},
    }

    return lut[agg_var]


def agg_files_exist(year, agg_vars, output_dir, fn_str):
    """Check if the aggregated files already exist"""
    file_exist_accum = []
    for agg_var in agg_vars:
        # fps should be groupd by year so we only need one
        out_fp = get_output_filepath(output_dir, agg_var, year)
        file_exist_accum.append(out_fp.exists())

    all_files_exist = all(file_exist_accum)

    return all_files_exist


def check_no_clobber(no_clobber, year, agg_vars, output_dir, fn_str):
    """Check if the no_clobber flag is set and if the files already exist"""
    if no_clobber and agg_files_exist(year, agg_vars, output_dir, fn_str):
        logging.info(f"Resampled files for {year} already exist, skipping")
        return True
    else:
        logging.info(f"Missing some resampled files for {year}, processing")
        return False


def get_output_filepath(output_dir, agg_var, year):
    """Get the output filepath for the resampled data"""
    return output_dir.joinpath(agg_var, OUT_FN_STR.format(year=year, var_id=agg_var))


def write_data(agg_da, output_dir, agg_var, year):
    """Write the resampled data to disk"""
    out_fp = get_output_filepath(output_dir, agg_var, year)
    agg_da.to_netcdf(out_fp)

    return out_fp


def open_resample_regrid(
    fps,
    drop_vars,
    agg_vars,
    output_dir,
    year,
    grid_kwargs,
):
    """Open the dataset and aggregate the data for each variable"""
    with Client(n_workers=4, threads_per_worker=6) as client:
        era5_ds = open_dataset(fps, drop_vars)
        era5_ds.load()
    logging.info("Dataset opened and read into memory.")

    for agg_var in agg_vars:
        agg_ds = resample(era5_ds, agg_var)
        logging.info(f"Dataset for {agg_var} resampled.")
        regrid_ds = regrid(agg_ds, agg_var, grid_kwargs)
        logging.info(f"Dataset regridded {agg_var} writing.")
        out_fp = write_data(regrid_ds, output_dir, agg_var, year)
        logging.info(year, agg_var, f"done, written to {out_fp}")

    del era5_ds


def process_era5(
    era5_dir,
    output_dir,
    fn_str,
    year,
    drop_vars,
    agg_vars,
    no_clobber,
    grid_kwargs,
):
    """Resample year of data"""

    if not check_no_clobber(no_clobber, year, agg_vars, output_dir, fn_str):
        fps = get_year_filepaths(era5_dir, year, fn_str)
        open_resample_regrid(
            fps,
            drop_vars,
            agg_vars,
            output_dir,
            year,
            grid_kwargs,
        )


def get_grid_info(tmp_file, geo_file):
    ds = xr.open_dataset(tmp_file)
    geo_ds = xr.open_dataset(geo_file)

    # The proj4 string for the WRF projection is:
    # +proj=stere +units=m +a=6370000.0 +b=6370000.0 +lat_0=90.0 +lon_0=-152 +lat_ts=64 +nadgrids=@null
    # this was determined separately using the WRF-Python package
    # which has spotty availability / compatability
    #
    # here is the code for how that was done:
    # wrf_proj = PolarStereographic(
    #     **{"TRUELAT1": geo_ds.attrs["TRUELAT1"], "STAND_LON": geo_ds.attrs["STAND_LON"]}
    # ).proj4()
    wrf_proj = "+proj=stere +units=m +a=6370000.0 +b=6370000.0 +lat_0=90.0 +lon_0=-152 +lat_ts=64 +nadgrids=@null"

    # WGS84 projection
    wgs_proj = Proj(proj="latlong", datum="WGS84")
    wgs_to_wrf_transformer = Transformer.from_proj(wgs_proj, wrf_proj)

    # this is where we plug in the center longitude of the domain to get the center x, y in projected space
    e, n = wgs_to_wrf_transformer.transform(
        geo_ds.attrs["CEN_LON"], geo_ds.attrs["TRUELAT1"]
    )
    # now compute the rest of the grid based on x/y dimension lengths and grid spacing
    dx = dy = 4000
    nx = ds.XLONG.shape[1]
    ny = ds.XLONG.shape[0]
    x0 = -(nx - 1) / 2.0 * dx + e
    y0 = -(ny - 1) / 2.0 * dy + n
    # 2d grid coordinate values
    x = np.arange(nx) * dx + x0
    y = np.arange(ny) * dy + y0

    wrf_crs = CRS.from_proj4(wrf_proj)

    return {"x": x, "y": y, "wrf_crs": wrf_crs}


def regrid(ds, var_id, grid_kwargs):
    """Regrid ERA5 data to EPSG:3338"""
    x, y, wrf_crs = [grid_kwargs[k] for k in ["x", "y", "wrf_crs"]]

    ds_proj = (
        ds.rename({"south_north": "y", "west_east": "x"})
        .assign_coords({"y": ("y", y), "x": ("x", x)})
        .drop_vars(["XLONG", "XLAT"])
        .rio.set_spatial_dims("x", "y")
        .rio.write_crs(wrf_crs)
    )

    ds_3338 = ds_proj.rio.reproject("EPSG:3338")
    # make sure units is not lost here
    ds_3338[var_id].attrs["units"] = ds[var_id].attrs["units"]

    return ds_3338


def main(era5_dir, output_dir, geo_file, year, fn_str, no_clobber):
    # want these variables at daily resolution
    agg_vars = ["t2min", "t2", "t2max", "pr"]
    # make output dirs for these
    for var in agg_vars:
        output_dir.joinpath(var).mkdir(exist_ok=True)
    # get list of variables to exclude from the open_mfdataset() call using a single sample file
    tmp_file = era5_dir.joinpath(f"1995/{fn_str.format(date='1995-01-01')}")
    drop_vars = get_drop_vars(tmp_file)
    grid_kwargs = get_grid_info(tmp_file, geo_file)

    process_era5(
        era5_dir,
        output_dir,
        fn_str,
        year,
        drop_vars,
        agg_vars,
        no_clobber,
        grid_kwargs,
    )


if __name__ == "__main__":
    era5_dir, output_dir, geo_file, year, fn_str, no_clobber = parse_args()
    main(era5_dir, output_dir, geo_file, year, fn_str, no_clobber)
