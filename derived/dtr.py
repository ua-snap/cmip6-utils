"""Script for making daily / diurnal temperature range (dtr) data from tmax and tmin data.
Here, tmax and tmin refer to the daily maximum and minimum temperature data,
but not necessarily "tas" or temperature at surface - other temperature variables should work as well.
This script is designed to work with any gridded daily tmax and tmin data, not just CMIP6. E.g., it can be used for ERA5 data.
The only requirement is that the gridded daily data is in a flat file structure in the input directories,
and can be opened with xarray.open_mfdataset.

Example usage:
    python dtr.py \
        --tmax_dir /import/beegfs/CMIP6/arctic-cmip6/regrid/GFDL-ESM4/historical/day/tasmax \
        --tmin_dir /import/beegfs/CMIP6/arctic-cmip6/regrid/GFDL-ESM4/historical/day/tasmin \
        --output_dir /import/beegfs/CMIP6/snapdata/dtr_processing/netcdf/GFDL-ESM4/historical/day/dtr \
        --dtr_tmp_fn dtr_GFDL-ESM4_historical_{start_date}_{end_date}.nc

    or ERA5 e.g.:

    python dtr.py \
        --tmax_dir /import/beegfs/CMIP6/arctic-cmip6/daily_era5_4km_3338/netcdf/t2max \
        --tmin_dir /import/beegfs/CMIP6/arctic-cmip6/daily_era5_4km_3338/netcdf/t2min \
        --output_dir /import/beegfs/CMIP6/snapdata/dtr_processing/era5_dtr \
        --dtr_tmp_fn dtr_{year}_4km_3338.nc


    python dtr.py \
        --tmax_dir /center1/CMIP6/kmredilla/daily_era5_4km_3338/netcdf/t2max \
        --tmin_dir /center1/CMIP6/kmredilla/daily_era5_4km_3338/netcdf/t2min \
        --output_dir /import/beegfs/CMIP6/kmredilla/dtr_processing/era5_dtr/dtr \
        --dtr_tmp_fn dtr_{year}_4km_3338.nc
"""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import xarray as xr
import string
from datetime import datetime
import os
import dask
from dask.distributed import Client, LocalCluster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def configure_dask_for_dtr(n_workers=4, threads_per_worker=4, memory_limit="28GB"):
    """Configure Dask LocalCluster optimized for DTR calculation on 128GB nodes.

    DTR calculation is I/O intensive (reading many files) and compute simple (subtraction).

    Args:
        n_workers: Number of worker processes (default: 4)
        threads_per_worker: Threads per worker (default: 4)
        memory_limit: Memory limit per worker (default: 28GB)

    Returns:
        client: Dask distributed client
    """
    # Close any existing clients
    try:
        client = Client.current()
        client.close()
    except ValueError:
        pass

    # Configure global dask settings
    dask.config.set(
        {
            # Memory management
            "distributed.worker.memory.target": 0.75,
            "distributed.worker.memory.spill": 0.85,
            "distributed.worker.memory.pause": 0.90,
            "distributed.worker.memory.terminate": 0.95,
            # I/O optimization for reading many files
            "distributed.comm.timeouts.tcp": "120s",
            "distributed.scheduler.bandwidth": 1e9,
            # Array settings
            "array.slicing.split_large_chunks": True,
            "array.chunk-size": "128 MiB",
        }
    )

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=True,
        dashboard_address=None,
    )

    client = Client(cluster)

    logging.info(f"Dask cluster configured for DTR calculation:")
    logging.info(f"  Workers: {n_workers}, Threads/worker: {threads_per_worker}")
    logging.info(f"  Memory per worker: {memory_limit}")
    logging.info(f"  Dashboard: {client.dashboard_link}")

    return client


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tmax_dir",
        type=str,
        help="ERA5: Directory containing daily maximum temperature data saved by year (and nothing else)",
        required=False,
    )
    parser.add_argument(
        "--tmin_dir",
        type=str,
        help="ERA5: Directory containing daily minimum temperature data saved by year (and nothing else)",
        required=False,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="CMIP6: Directory containing batch files of source CMIP6 filepaths",
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory for writing daily temperature range data",
        required=True,
    )
    parser.add_argument(
        "--dtr_tmp_fn",
        type=str,
        help="Template filename for the daily temperature range data",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    return (
        Path(args.tmax_dir) if args.tmax_dir is not None else None,
        Path(args.tmin_dir) if args.tmin_dir is not None else None,
        Path(args.input_dir) if args.input_dir is not None else None,
        Path(args.output_dir),
        args.dtr_tmp_fn,
        args.model,
        args.scenario,
    )


def get_tmax_tmin_fps_era5(tmax_dir, tmin_dir):
    """Helper function for getting tasmax and tasmin filepaths. Put in function for checking prior to slurming.
    Assumes that all files in the input directories are the target input files.
    """
    tmax_fps = list(tmax_dir.glob("*"))
    tmin_fps = list(tmin_dir.glob("*"))

    assert (
        len(tmax_fps) > 0
    ), f"No tasmax files found in the input directory, in {tmax_dir}"
    assert (
        len(tmin_fps) > 0
    ), f"No tasmin files found in the input directory, in {tmin_dir}"
    assert len(tmax_fps) == len(
        tmin_fps
    ), f"Number of tmax and tmin files must be the same. tmax: {len(tmax_fps)} files in {tmax_dir}, tmin: {len(tmin_fps)} files in {tmin_dir}"

    return tmax_fps, tmin_fps


def get_tmax_tmin_fps_cmip6(input_dir, model, scenario):
    """Helper function for getting tasmax and tasmin filepaths. Put in function for checking prior to slurming.
    Assumes that all files in the input directories are the target input files.
    """

    # Read all of the files in cmip6_files_dir and concatenate them into a single list
    cmip6_file_list = []
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                cmip6_file_list.extend([line.strip() for line in f if line.strip()])

    tmax_fps = [
        fp
        for fp in cmip6_file_list
        if "tasmax" in fp and model in fp and scenario in fp
    ]
    tmin_fps = [
        fp
        for fp in cmip6_file_list
        if "tasmin" in fp and model in fp and scenario in fp
    ]

    tmax_fps.sort()
    tmin_fps.sort()

    print("tmax_fps:", tmax_fps)
    print("tmin_fps:", tmin_fps)

    assert (
        len(tmax_fps) > 0
    ), f"No tasmax files found in the input directory, in {input_dir}"
    assert (
        len(tmin_fps) > 0
    ), f"No tasmin files found in the input directory, in {input_dir}"
    assert len(tmax_fps) == len(
        tmin_fps
    ), f"Number of tmax and tmin files must be the same. tmax: {len(tmax_fps)} files in {input_dir}, tmin: {len(tmin_fps)} files in {input_dir}"

    return tmax_fps, tmin_fps


def get_var_id(ds):
    """Get the variable id from the dataset attributes.
    This is a helper function for getting the variable id from the dataset attributes.
    """
    if "variable_id" in ds.attrs.keys():
        var_id = ds.attrs["variable_id"]
        assert var_id in ds.data_vars, f"{var_id} not in {ds.data_vars}"
    else:
        valid_vars = [var for var in ds.data_vars if set(ds[var].dims) == set(ds.dims)]
        assert (
            len(valid_vars) == 1
        ), f"Dataset must have exactly one variable indexed by all dimensions. Found: {valid_vars}"
        var_id = valid_vars[0]

    return var_id


def get_start_end_dates(ds):
    """Get the start and end dates from the dataset attributes."""
    start_date = ds.time.min().dt.strftime("%Y%m%d").values.item()
    end_date = ds.time.max().dt.strftime("%Y%m%d").values.item()
    return start_date, end_date


def extract_format_keys(template):
    """Extract keys from a string to be formatted. Returns a set of keys."""
    formatter = string.Formatter()
    return set([key for _, key, _, _ in formatter.parse(template) if key is not None])


def make_output_filepath(output_dir, dtr_tmp_fn, start_date, end_date):
    """Make the output file path from the template and start and end dates."""
    keys = extract_format_keys(dtr_tmp_fn)

    start_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")

    if keys == {"end_date", "start_date"}:
        output_fp = output_dir.joinpath(
            dtr_tmp_fn.format(
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
            )
        )
    elif keys == {"year"}:
        start_year = start_date.year
        end_year = end_date.year
        if start_year != end_year:
            raise ValueError(
                f"Start and end dates must be in the same year for template {dtr_tmp_fn}"
            )
        output_fp = output_dir.joinpath(dtr_tmp_fn.format(year=start_year))
    else:
        raise ValueError(
            f"Template DTR filename, {dtr_tmp_fn}, must have either start_date and end_date or year as keys. Found: {keys}"
        )

    return output_fp


def validate_output_file(output_fp, var_id="dtr"):
    """Validate that the output file exists and contains valid data.

    Checks multiple samples across the dataset to handle filesystem cache
    coherency issues on distributed systems like beegfs.

    Args:
        output_fp: Path to output file
        var_id: Variable identifier to check

    Raises:
        ValueError: If output is invalid or all samples are NaN
    """
    if not output_fp.exists():
        raise ValueError(f"Output file not created: {output_fp}")

    # Check file size
    size_mb = output_fp.stat().st_size / (1024 * 1024)
    if size_mb < 0.1:
        raise ValueError(
            f"Output file suspiciously small ({size_mb:.2f} MB): {output_fp}"
        )

    # Try to open and check variable
    try:
        with xr.open_dataset(output_fp) as ds:
            if var_id not in ds.data_vars:
                raise ValueError(
                    f"Variable '{var_id}' not found in output: {output_fp}"
                )

            arr = ds[var_id]
            if arr.size == 0:
                raise ValueError(f"Variable '{var_id}' is empty in: {output_fp}")

            # Check multiple samples to catch filesystem cache coherency issues
            # where some chunks may not be visible yet on different nodes
            samples_to_check = [
                ("start", {dim: slice(0, min(10, arr.sizes[dim])) for dim in arr.dims}),
                (
                    "middle",
                    {
                        dim: slice(arr.sizes[dim] // 2, arr.sizes[dim] // 2 + 10)
                        for dim in arr.dims
                    },
                ),
                (
                    "end",
                    {
                        dim: slice(max(0, arr.sizes[dim] - 10), arr.sizes[dim])
                        for dim in arr.dims
                    },
                ),
            ]

            all_nan_count = 0

            for location, selection in samples_to_check:
                try:
                    sample = arr.isel(selection)
                    sample_data = sample.compute()

                    if sample_data.isnull().all():
                        all_nan_count += 1
                        logging.warning(
                            f"  WARNING: {location} sample is all NaN in {output_fp.name}"
                        )
                    else:
                        logging.debug(
                            f"  {location} sample valid: "
                            f"min={float(sample_data.min()):.4f}, "
                            f"max={float(sample_data.max()):.4f}, "
                            f"mean={float(sample_data.mean()):.4f}"
                        )
                except Exception as e:
                    logging.error(
                        f"  ERROR reading {location} sample from {output_fp.name}: {e}"
                    )
                    raise

            # Only fail if ALL samples are NaN (suggests real problem)
            # Partial NaN samples may be filesystem cache issues that resolve
            if all_nan_count == len(samples_to_check):
                raise ValueError(
                    f"Variable '{var_id}' appears to be all NaN in: {output_fp}. "
                    f"Checked {len(samples_to_check)} locations, all returned NaN. "
                    f"This may indicate a filesystem cache coherency issue or failed computation."
                )

            if all_nan_count > 0:
                logging.warning(
                    f"  {all_nan_count}/{len(samples_to_check)} samples were all NaN, "
                    f"but validation passed (some data found)"
                )

    except Exception as e:
        raise ValueError(f"Cannot validate output {output_fp}: {e}")

    logging.info(f"✓ Output validated: {output_fp.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    tmax_dir, tmin_dir, input_dir, output_dir, dtr_tmp_fn, model, scenario = (
        parse_args()
    )

    success = False
    client = None

    try:
        # Configure Dask
        logging.info("Configuring Dask cluster...")
        client = configure_dask_for_dtr(
            n_workers=4, threads_per_worker=4, memory_limit="28GB"
        )

        # Get file paths
        logging.info("Collecting input file paths...")
        if input_dir:
            tmax_fps, tmin_fps = get_tmax_tmin_fps_cmip6(input_dir, model, scenario)
            logging.info(f"Processing CMIP6 data: {model} {scenario}")
        else:
            tmax_fps, tmin_fps = get_tmax_tmin_fps_era5(tmax_dir, tmin_dir)
            logging.info(f"Processing ERA5 data")

        logging.info(f"Found {len(tmax_fps)} tmax files and {len(tmin_fps)} tmin files")

        # Optimized chunking for DTR calculation
        # DTR is simple subtraction, so larger chunks are better for I/O efficiency
        chunks = {"time": 365, "x": 200, "y": 200}  # One year at a time

        logging.info(
            f"Using chunk strategy: time={chunks['time']}, x={chunks['x']}, y={chunks['y']}"
        )
        logging.info("Opening tmax dataset...")

        # Open with explicit chunking - don't use context managers since we need the data later
        tmax_ds = xr.open_mfdataset(
            tmax_fps, engine="h5netcdf", parallel=True, use_cftime=True, chunks=chunks
        )

        logging.info("Opening tmin dataset...")
        tmin_ds = xr.open_mfdataset(
            tmin_fps, engine="h5netcdf", parallel=True, use_cftime=True, chunks=chunks
        )

        # Get variable IDs
        tmax_var_id = get_var_id(tmax_ds)
        tmin_var_id = get_var_id(tmin_ds)
        logging.info(f"Processing variables: {tmax_var_id}, {tmin_var_id}")

        # Validate units match
        units = tmax_ds[tmax_var_id].attrs["units"]
        if units != tmin_ds[tmin_var_id].attrs["units"]:
            raise ValueError(
                f"Units mismatch: tmax has '{units}', tmin has '{tmin_ds[tmin_var_id].attrs['units']}'"
            )

        # Calculate DTR (lazy evaluation - no persist()!)
        logging.info("Calculating DTR (lazy evaluation)...")
        dtr = tmax_ds[tmax_var_id] - tmin_ds[tmin_var_id]

        dtr.name = "dtr"
        dtr.attrs = {
            "long_name": "Daily temperature range",
            "units": units,
        }

        # Replace negative values (tasmax - tasmin < 0) with 0.0000999
        # Using this number instead of zero helps identify tweaked spots
        logging.info("Applying negative value correction...")
        dtr = dtr.where((dtr.isnull() | (dtr >= 0)), 0.0000999)

        # Create dataset with matching dimension order
        dtr_ds = dtr.to_dataset().transpose(*list(tmax_ds[tmax_var_id].dims))
        dtr_ds.attrs = {k: v for k, v in tmax_ds.attrs.items() & tmin_ds.attrs.items()}
        dtr_ds.attrs["variable_id"] = "dtr"

        # Write output files
        output_dir.mkdir(parents=True, exist_ok=True)
        years = np.unique(dtr_ds.time.dt.year)
        total_years = len(years)
        logging.info(f"Writing {total_years} years of DTR data...")

        for idx, year in enumerate(years, 1):
            logging.info(f"Processing year {year} ({idx}/{total_years})...")
            year_ds = dtr_ds.sel(time=str(year))
            start_date, end_date = get_start_end_dates(year_ds)
            output_fp = make_output_filepath(
                output_dir, dtr_tmp_fn, start_date, end_date
            )

            # Check if output already exists
            if output_fp.exists():
                logging.info(f"Output already exists, overwriting: {output_fp.name}")
                output_fp.unlink()

            logging.info(
                f"Computing and writing {year_ds.dtr.name} for {start_date}-{end_date}..."
            )
            try:
                year_ds.to_netcdf(output_fp, compute=True)
            except Exception as e:
                logging.error(f"Failed to write {output_fp}: {e}")
                if output_fp.exists():
                    output_fp.unlink()
                raise

            # Validate output
            validate_output_file(output_fp, var_id="dtr")
            logging.info(f"✓ Completed year {year} ({idx}/{total_years})")

        logging.info(f"Successfully processed all {total_years} years")
        success = True

    except Exception as e:
        logging.error(f"FATAL ERROR during DTR processing: {e}")
        logging.error(f"DTR calculation FAILED")
        sys.exit(1)

    finally:
        # Cleanup Dask client
        if client is not None:
            logging.info("Closing Dask client...")
            client.close()

    if not success:
        logging.error("DTR processing did not complete successfully")
        sys.exit(1)

    logging.info("DTR processing completed successfully")
    sys.exit(0)
