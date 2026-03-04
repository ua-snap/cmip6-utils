"""Convert netCDF files to zarr format.

Supply a parent directory containing netCDF files, a string for fetching the files from that directory,
and a path to write the zarr store to.

example usage:
    python netcdf_to_zarr.py \
        --netcdf_dir /beegfs/CMIP6/kmredilla/daily_era5_4km_3338/ \
        --year_str t2max/t2max_{year}_era5_4km_3338.nc \
        --start_year 1965 --end_year 2014 \
        --zarr_path /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/optimized_inputs/era5_t2max.zarr
"""

import argparse
import logging
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import xarray as xr
from dask.distributed import Client

from zarr.sync import ThreadSynchronizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def validate_zarr_readback(zarr_path, expected_var_id, max_retries=120, retry_delay=60):
    """Validate that written zarr can be read back with actual data.

    This forces the writer node to verify data is accessible, which helps
    ensure it will be visible to other nodes in a distributed filesystem.
    Retries for up to 2 hours by default to handle slow filesystem propagation.

    Args:
        zarr_path: Path to zarr store
        expected_var_id: Variable to check
        max_retries: Number of read attempts (default: 120 = 2 hours with 60s delay)
        retry_delay: Seconds between retries (default: 60)

    Returns:
        True if successful

    Raises:
        ValueError: If data cannot be read after retries
    """
    import zarr
    import gc

    for attempt in range(1, max_retries + 1):
        try:
            elapsed_time = (attempt - 1) * retry_delay / 60  # minutes
            logging.info(
                f"Read-back validation attempt {attempt}/{max_retries} (elapsed: {elapsed_time:.1f} min)..."
            )

            # Close any open connections and force fresh read
            gc.collect()  # Force garbage collection to close file handles

            # Try system sync first
            try:
                os.sync()
            except:
                pass

            # Open fresh without any caching
            ds = xr.open_zarr(zarr_path, consolidated=False)

            if expected_var_id not in ds.data_vars:
                raise ValueError(f"Variable '{expected_var_id}' not found in dataset")

            arr = ds[expected_var_id]

            # Check actual data, not just metadata
            logging.info(f"Checking data validity by loading a sample...")
            sample = arr.isel(
                {dim: slice(0, min(50, arr.sizes[dim])) for dim in arr.dims}
            )
            sample_data = sample.compute()  # Force actual read from disk

            if sample_data.size == 0:
                raise ValueError("Sample is empty")

            if sample_data.isnull().all():
                raise ValueError("Sample is all NaN")

            # Check that we can access actual chunk files
            z = zarr.open_group(zarr_path, "r")
            if expected_var_id not in z:
                raise ValueError(f"Variable {expected_var_id} not in zarr group")

            var_array = z[expected_var_id]
            chunk_keys = [
                k for k in var_array.chunk_store.keys() if expected_var_id in str(k)
            ]
            chunk_count = len(chunk_keys)
            logging.info(f"Found {chunk_count} chunk files for {expected_var_id}")

            if chunk_count == 0:
                raise ValueError("No chunk files found!")

            # Success!
            logging.info(f"✓ Read-back validation PASSED on attempt {attempt}")
            logging.info(f"  - Sample shape: {sample_data.shape}")
            logging.info(f"  - Sample mean: {float(sample_data.mean()):.4f}")
            logging.info(
                f"  - Sample range: [{float(sample_data.min()):.4f}, {float(sample_data.max()):.4f}]"
            )
            logging.info(f"  - Chunk count: {chunk_count}")
            ds.close()
            return True

        except Exception as e:
            logging.warning(f"✗ Read-back validation attempt {attempt} failed: {e}")

            if attempt < max_retries:
                logging.info(f"Waiting {retry_delay}s before retry...")
                time.sleep(retry_delay)

                # Try to force filesystem visibility
                try:
                    os.sync()
                except:
                    pass

                # List the directory structure to force metadata refresh
                try:
                    subprocess.run(
                        ["find", str(zarr_path), "-type", "f", "-name", "*.*.*"],
                        capture_output=True,
                        check=False,
                        timeout=30,
                    )
                except:
                    pass
            else:
                raise ValueError(
                    f"Failed to validate zarr after {max_retries} attempts ({max_retries * retry_delay / 3600:.1f} hours). "
                    f"Last error: {e}"
                )

    return False


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
        default='{"time": -1, "x": 50, "y": 50}',
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

    # with Client(n_workers=12, threads_per_worker=2, memory_limit="3GB") as client:
    # with Client(n_workers=12, threads_per_worker=2, memory_limit="3GB") as client:
    # the data_vars="minimal" argument is a workaround for behavior in
    # xarray.open_mfdataset that will assign concat dimension to dimensionless
    # data variables (such as spatial_ref)
    with xr.open_mfdataset(
        fps, parallel=True, engine="h5netcdf", data_vars="minimal"
    ) as ds:
        ds = ds.load()

    var_id = list(ds.data_vars)[0]
    # hardcoding chunks stuff for now
    ds = ds.chunk(chunks_dict)

    logging.info(
        f"Converting {len(fps)} files in {netcdf_dir} to Zarr store at {zarr_path}"
    )

    if zarr_path.exists():
        shutil.rmtree(zarr_path, ignore_errors=True)

    synchronizer = ThreadSynchronizer()
    ds.to_zarr(zarr_path, synchronizer=synchronizer)
    logging.info(f"Initial write to {zarr_path} completed")

    # CRITICAL: Validate we can read it back
    logging.info("=" * 60)
    logging.info("Starting read-after-write validation (up to 2 hours)...")
    logging.info("=" * 60)

    try:
        validate_zarr_readback(zarr_path, var_id, max_retries=120, retry_delay=60)
        logging.info("=" * 60)
        logging.info("✓✓✓ Zarr store validated and confirmed readable ✓✓✓")
        logging.info("=" * 60)
    except Exception as e:
        logging.error("=" * 60)
        logging.error(f"✗✗✗ FATAL: Cannot read back written data: {e} ✗✗✗")
        logging.error("This data should NOT be used as input to other scripts!")
        logging.error("=" * 60)
        sys.exit(1)

    logging.info(f"Conversion and validation complete.")
