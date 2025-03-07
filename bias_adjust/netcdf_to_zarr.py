"""Convert netCDF files to zarr format.

Supply a parent directory containing netCDF files, a string for fetching the files from that directory,
and a path to write the zarr store to.

example usage:
    python optimize_inputs.py --netcdf_dir /beegfs/CMIP6/kmredilla/daily_era5_4km_3338/ --year_str t2max/t2max_{year}_era5_4km_3338.nc --start_year 1965 --end_year 2014 --zarr_path /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/optimized_inputs/era5_t2max.zarr
"""

import argparse
import logging
import json
import shutil
from pathlib import Path
import xarray as xr
from dask.distributed import Client
from bias_adjust import drop_non_coord_vars


def validate_args(args):
    """Validate the supplied command line args."""
    args.netcdf_dir = Path(args.netcdf_dir)
    if not args.netcdf_dir.exists():
        raise FileNotFoundError(f"Directory {args.netcdf_dir} not found.")
    args.zarr_path = Path(args.zarr_path)
    if not args.zarr_path.parent.exists():
        raise FileNotFoundError(
            (
                f"Parent directory of requested zarr outputs directory, {args.zarr_path.parent},"
                " does not exist, and needs to for this script to run."
            )
        )

    if args.glob_str:
        if args.year_str or args.start_year or args.end_year:
            raise ValueError(
                "If glob_str is provided, year_str, start_year, and end_year must not be provided."
            )

    if args.start_year or args.end_year or args.year_str:
        if not (args.start_year and args.end_year):
            raise ValueError(
                "If start_year or end_year is provided, both must be provided."
            )
        if not args.year_str:
            raise ValueError(
                "If start_year and end_year are provided, year_str must be provided."
            )
        if args.glob_str:
            raise ValueError(
                "If start_year and end_year are provided, glob_str must not be provided."
            )

        try:
            args.start_year = int(args.start_year)
        except ValueError:
            raise ValueError(f"start_year must be an integer, got {args.start_year}")
        try:
            args.end_year = int(args.end_year)
        except ValueError:
            raise ValueError(f"end_year must be an integer, got {args.end_year}")
        if not (1950 <= args.start_year < args.end_year <= 2100):
            raise ValueError(
                (
                    f"start_year and end_year must be between 1950 and 2100, with start_year < end_year."
                    f" got {args.start_year} and {args.end_year}"
                )
            )

    if args.chunks_dict:
        if not isinstance(args.chunks_dict, dict):
            raise ValueError(
                f"chunks_dict must be a dictionary, got {args.chunks_dict}"
            )

    return args


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--netcdf_dir",
        type=str,
        help="Path to directory containing data files to optimize.",
        required=True,
    )
    parser.add_argument(
        "--glob_str",
        type=str,
        help="Glob string for getting data files in netcdf_dir. Required if files to optimize are not in the netcdf_dir root.",
        default=None,
    )
    parser.add_argument(
        "--year_str",
        type=str,
        help="String for getting data files in netcdf_dir based on start and end years. Requires both start_year and end_year to use.",
        default=None,
    )
    parser.add_argument(
        "--start_year",
        type=str,
        help="Starting year of data to optimize. Required if year_str is provided.",
        default=None,
    )
    parser.add_argument(
        "--end_year",
        type=str,
        help="Ending year of data to optimize. Required if year_str is provided.",
        default=None,
    )
    parser.add_argument(
        "--chunks_dict",  # this is just a template for now, in case we want to make this configurable
        type=json.loads,
        help="Dictionary of chunks to use for rechunking",
        default='{"time": -1, "x": 10, "y": 10}',
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        help="Path to write rechunked zarr store",
        required=True,
    )
    args = parser.parse_args()

    args = validate_args(args)

    return (
        args.netcdf_dir,
        args.glob_str,
        args.year_str,
        args.start_year,
        args.end_year,
        args.chunks_dict,
        args.zarr_path,
    )


def get_input_filepaths(
    netcdf_dir, glob_str=None, year_str=None, start_year=None, end_year=None
):
    """Get filepaths for all files in netcdf_dir matching glob_str or year_str + start_year + end_year.

    Default behaviors:
    - if nothing but netcdf_dir is provided, return all files in netcdf_dir (i.e. set glob_str to "*")
    - if glob_str is provided, return all files in netcdf_dir that match glob_str
    - if year_str is provided, return all files in netcdf_dir that match year_str formatted for all years in range(start_year, end_year + 1)

    example year_str: "GFDL-ESM4/historical/day/tasmax/tasmax_day_GFDL-ESM4_historical_regrid_{year}0101-{year}1231.nc"
    example glob_str: "GFDL-ESM4/historical/day/tasmax/tasmax_day_GFDL-ESM4_historical_regrid_*.nc"
    """
    if year_str is not None:
        fps = [
            netcdf_dir.joinpath(year_str.format(year=year))
            for year in range(start_year, end_year + 1)
        ]

        if not all([fp.exists() for fp in fps]):
            bad_files = "\n".join([str(fp) for fp in fps if not fp.exists()])
            raise FileNotFoundError(
                (
                    f"Files not found for all years in range {start_year} to {end_year} using year_str: {year_str}."
                    f"Expected files:\n {bad_files}"
                )
            )
    else:
        if glob_str is None:
            glob_str = "*"
        fps = list(netcdf_dir.glob(glob_str))

        if not fps:
            raise FileNotFoundError(
                f"No files found matching glob_str: {glob_str} in the data directory, {netcdf_dir}"
            )

    return fps


if __name__ == "__main__":
    (
        netcdf_dir,
        glob_str,
        year_str,
        start_year,
        end_year,
        chunks_dict,
        zarr_path,
    ) = parse_args()

    fps = get_input_filepaths(netcdf_dir, glob_str, year_str, start_year, end_year)

    with Client(n_workers=8) as client:
        with xr.open_mfdataset(fps, parallel=True, engine="h5netcdf") as ds:
            ds = ds.load()

    ds = drop_non_coord_vars(ds)
    var_id = list(ds.data_vars)[0]
    # hardcoding chunks stuff for now
    ds[var_id].encoding["preferred_chunks"] = chunks_dict
    ds[var_id].encoding["chunks"] = (ds.time.values.shape[0], 50, 50)
    ds[var_id].encoding["chunksizes"] = (ds.time.values.shape[0], 50, 50)

    logging.info(f"Optimizing {len(fps)} files in {netcdf_dir} to {zarr_path}")

    if zarr_path.exists():
        shutil.rmtree(zarr_path, ignore_errors=True)

    ds.to_zarr(zarr_path)
