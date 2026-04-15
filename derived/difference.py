"""Script for computing the difference between two variables and saving it as a new variable.
This will be used primarily for computing daily temperature range (dtr) from tmax and tmin data, 
and for computing tasmin from tmax and dtr data.
This script is designed to work with zarr stores as inputs. 

Example usage:
    python difference.py \
        --minuend_store /center1/CMIP6/kmredilla/cmip6_4km_downscaling/adjusted/tasmax_GFDL-ESM4_historical_adjusted.zarr \
        --subtrahend_store /center1/CMIP6/kmredilla/cmip6_4km_downscaling/adjusted/dtr_GFDL-ESM4_historical_adjusted.zarr \
        --output_store /center1/CMIP6/kmredilla/cmip6_4km_downscaling/derived/tasmin/tasmin_GFDL-ESM4_historical_adjusted.zarr \
        --new_var_id tasmin
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import xarray as xr
import string
from datetime import datetime
import dask
from dask.distributed import Client, LocalCluster

from zarr.sync import ThreadSynchronizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def configure_dask_for_difference(
    n_workers=4, threads_per_worker=4, memory_limit="28GB"
):
    """Configure Dask LocalCluster optimized for difference calculations on 128GB nodes.

    Difference calculation is simple (subtraction) but needs careful memory management
    for large zarr stores.

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
            # I/O optimization
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

    logging.info(f"Dask cluster configured for difference calculation:")
    logging.info(f"  Workers: {n_workers}, Threads/worker: {threads_per_worker}")
    logging.info(f"  Memory per worker: {memory_limit}")

    return client


def force_filesystem_cache_refresh(zarr_path, max_attempts=10, delay=10):
    """Aggressively force beegfs to refresh its cache for a zarr store.

    This is critical for multi-node jobs where the writer and reader are on different nodes.

    Args:
        zarr_path: Path to zarr store
        max_attempts: Number of attempts to verify chunks are visible
        delay: Seconds between attempts
    """
    logging.info(f"Forcing filesystem cache refresh for {zarr_path}...")

    for attempt in range(1, max_attempts + 1):
        try:
            # Force kernel to flush anything pending
            try:
                os.sync()
            except:
                pass

            # List all chunk files to force metadata + inode cache refresh
            # Zarr chunks are named like "0.0.0", "1.2.3", etc.
            result = subprocess.run(
                ["find", str(zarr_path), "-type", "f", "-name", "*.*.*"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            chunk_files = [
                line
                for line in result.stdout.strip().split("\n")
                if line
                and not line.endswith(".zarray")
                and not line.endswith(".zattrs")
            ]
            logging.info(f"  Attempt {attempt}: Found {len(chunk_files)} chunk files")

            if len(chunk_files) > 0:
                # Try to actually read a few bytes from chunk files
                import random

                sample_files = random.sample(chunk_files, min(5, len(chunk_files)))
                for chunk_file in sample_files:
                    try:
                        with open(chunk_file, "rb") as f:
                            data = f.read(1024)  # Read first 1KB
                            if len(data) > 0:
                                logging.info(
                                    f"    Read {len(data)} bytes from {os.path.basename(chunk_file)}"
                                )
                    except Exception as e:
                        logging.warning(f"    Failed to read {chunk_file}: {e}")

                logging.info(f"  ✓ Cache refresh successful - chunks are visible")
                return True

            if attempt < max_attempts:
                logging.info(f"  No chunks found yet, waiting {delay}s...")
                time.sleep(delay)

        except subprocess.TimeoutExpired:
            logging.warning(
                f"  Timeout finding chunk files, attempt {attempt}/{max_attempts}"
            )
            if attempt < max_attempts:
                time.sleep(delay)
        except Exception as e:
            logging.warning(f"  Cache refresh attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                time.sleep(delay)

    raise ValueError(
        f"Could not verify zarr chunks are visible after {max_attempts} attempts"
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
            # Check multiple samples to handle NaNs at domain edges
            logging.info(f"Checking data validity by loading samples...")

            samples_to_check = [
                ("start", {dim: slice(0, min(50, arr.sizes[dim])) for dim in arr.dims}),
                (
                    "middle",
                    {
                        dim: slice(arr.sizes[dim] // 2, arr.sizes[dim] // 2 + 50)
                        for dim in arr.dims
                    },
                ),
                (
                    "end",
                    {
                        dim: slice(max(0, arr.sizes[dim] - 50), arr.sizes[dim])
                        for dim in arr.dims
                    },
                ),
            ]

            valid_sample_found = False
            sample_data = None

            for location, selection in samples_to_check:
                sample = arr.isel(selection)
                sample_data = sample.compute()  # Force actual read from disk

                if sample_data.size == 0:
                    continue

                if not sample_data.isnull().all():
                    valid_sample_found = True
                    logging.info(f"  ✓ Found valid data in {location} sample")
                    break
                else:
                    logging.info(
                        f"  ~ {location} sample is all NaN (may be edge of domain)"
                    )

            if not valid_sample_found:
                raise ValueError("All samples (start, middle, end) are NaN or empty")

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


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minuend_store",
        type=str,
        help="Directory containing 'minued' data (that from which subtrahend is subtracted)",
        required=True,
    )
    parser.add_argument(
        "--subtrahend_store",
        type=str,
        help="Directory containing 'subtrahend' data (that which is subtracted from minuend)",
        required=True,
    )
    parser.add_argument(
        "--output_store",
        type=str,
        help="Directory for writing difference data.",
        required=True,
    )
    parser.add_argument(
        "--new_var_id",
        type=str,
        help="New variable id for the resulting difference data",
    )
    args = parser.parse_args()

    return (
        Path(args.minuend_store),
        Path(args.subtrahend_store),
        Path(args.output_store),
        args.new_var_id,
    )


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


def validate_output_zarr(output_store, var_id, min_size_mb=5):
    """Validate that the written zarr store is valid and has reasonable size.

    Args:
        output_store: Path to the zarr store
        var_id: Variable identifier
        min_size_mb: Minimum expected size in MB

    Raises:
        ValueError: If output is invalid
    """
    import os

    logging.info(f"Validating written zarr store at {output_store}...")

    # Check that the path exists
    if not output_store.exists():
        raise ValueError(f"Output zarr store was not created at {output_store}")

    # Check directory size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(output_store):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)

    size_mb = total_size / (1024 * 1024)
    logging.info(f"Output zarr store size: {size_mb:.2f} MB")

    if size_mb < min_size_mb:
        raise ValueError(
            f"Output zarr store is suspiciously small ({size_mb:.2f} MB < {min_size_mb} MB). "
            "Computation may have failed."
        )

    # Try to open and validate the zarr store
    try:
        out_ds_check = xr.open_dataset(output_store, engine="zarr")
    except Exception as e:
        raise ValueError(f"Cannot open output zarr store: {e}")

    if var_id not in out_ds_check.data_vars:
        raise ValueError(f"Output variable '{var_id}' not found in {output_store}")

    arr = out_ds_check[var_id]
    if arr.size == 0:
        raise ValueError(f"Output for {output_store} is empty")

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
                    f"  WARNING: {location} sample is all NaN in {output_store}"
                )
            else:
                logging.debug(
                    f"  {location} sample valid: "
                    f"min={float(sample_data.min()):.4f}, "
                    f"max={float(sample_data.max()):.4f}, "
                    f"mean={float(sample_data.mean()):.4f}"
                )
        except Exception as e:
            logging.error(f"  ERROR reading {location} sample from {output_store}: {e}")
            raise

    # Only fail if ALL samples are NaN (suggests real problem)
    # Partial NaN samples may be filesystem cache issues that resolve
    if all_nan_count == len(samples_to_check):
        raise ValueError(
            f"Output for {output_store} appears to be all NaN. "
            f"Checked {len(samples_to_check)} locations, all returned NaN. "
            f"This may indicate a filesystem cache coherency issue or failed computation."
        )

    if all_nan_count > 0:
        logging.warning(
            f"  {all_nan_count}/{len(samples_to_check)} samples were all NaN, "
            f"but validation passed (some data found)"
        )

    logging.info(f"Written zarr validation passed for {output_store}")
    return True


if __name__ == "__main__":
    minuend_store, subtrahend_store, output_store, new_var_id = parse_args()

    success = False
    client = None

    try:
        # Configure Dask
        logging.info("Configuring Dask cluster...")
        client = configure_dask_for_difference(
            n_workers=4, threads_per_worker=4, memory_limit="28GB"
        )

        logging.info(
            f"Computing difference: {minuend_store.name} - {subtrahend_store.name}"
        )

        # Optimized chunking for difference calculation
        # Simple subtraction, so larger chunks are better
        chunk_dict = {"time": 365, "x": 150, "y": 150}  # One year at a time

        logging.info(
            f"Using chunk strategy: time={chunk_dict['time']}, x={chunk_dict['x']}, y={chunk_dict['y']}"
        )

        # CRITICAL: Force filesystem cache refresh before opening zarr files
        logging.info("=" * 60)
        logging.info("FORCING FILESYSTEM CACHE REFRESH FOR INPUT FILES")
        logging.info("=" * 60)

        try:
            force_filesystem_cache_refresh(minuend_store, max_attempts=10, delay=10)
        except Exception as e:
            logging.error(f"Failed to refresh cache for {minuend_store}: {e}")
            raise

        try:
            force_filesystem_cache_refresh(subtrahend_store, max_attempts=10, delay=10)
        except Exception as e:
            logging.error(f"Failed to refresh cache for {subtrahend_store}: {e}")
            raise

        logging.info("=" * 60)
        logging.info("Cache refresh complete, opening zarr files...")
        logging.info("=" * 60)

        # Open datasets with explicit chunking
        logging.info("Opening minuend dataset...")
        minu_ds = xr.open_dataset(minuend_store, engine="zarr", chunks=chunk_dict)

        logging.info("Opening subtrahend dataset...")
        subtr_ds = xr.open_dataset(subtrahend_store, engine="zarr", chunks=chunk_dict)

        minu_var_id = get_var_id(minu_ds)
        subtr_var_id = get_var_id(subtr_ds)

        logging.info(f"Computing: {minu_var_id} - {subtr_var_id} = {new_var_id}")

        minuend = minu_ds[minu_var_id]
        diff = minuend - subtr_ds[subtr_var_id]

        # get units and make sure they are the same between both inputs
        units = minuend.attrs["units"]
        if units != subtr_ds[subtr_var_id].attrs["units"]:
            raise ValueError(
                f"Units mismatch: minuend has '{units}', subtrahend has '{subtr_ds[subtr_var_id].attrs['units']}'"
            )

        diff.name = new_var_id
        diff.attrs = {
            "units": units,
        }
        diff.encoding = minuend.encoding

        # the list here at the end is just making sure we have a matching dim order
        diff_ds = diff.to_dataset().transpose(*list(minuend.dims))
        diff_ds.attrs = {
            k: v for k, v in minu_ds.attrs.items() & subtr_ds.attrs.items()
        }
        # give this a variable_id attribute for consistency (helps with e.g. regridding with regrid.py)
        diff_ds.attrs["variable_id"] = new_var_id

        variable_id = diff_ds.attrs["variable_id"]

        # Apply variable-specific thresholding
        min_tasmin = 203.15
        if variable_id == "tasmin":
            logging.info("Squeezing tasmin values below limit...")
            count_below_threshold = (
                (diff_ds[variable_id] < min_tasmin).sum().compute().item()
            )
            logging.info(
                f"Count of values below {min_tasmin} K: {count_below_threshold}"
            )
            diff_ds[variable_id] = diff_ds[variable_id].where(
                (diff_ds[variable_id] >= min_tasmin) | diff_ds[variable_id].isnull(),
                min_tasmin,
            )

        # Remove existing output
        if output_store.exists():
            logging.info(f"Removing existing output at {output_store}")
            try:
                shutil.rmtree(output_store)
            except Exception as e:
                logging.error(f"Failed to remove existing output: {e}")
                raise

        # Write output
        logging.info(f"Writing {variable_id} to {output_store}")
        synchronizer = ThreadSynchronizer()
        try:
            diff_ds.to_zarr(output_store, synchronizer=synchronizer, compute=True)
            logging.info(f"Initial write to {output_store} completed")
        except Exception as e:
            logging.error(f"Failed to write zarr store: {e}")
            # Clean up partial write
            if output_store.exists():
                shutil.rmtree(output_store, ignore_errors=True)
            raise

        # CRITICAL: Validate we can read it back
        logging.info("=" * 60)
        logging.info("Starting read-after-write validation (up to 2 hours)...")
        logging.info("=" * 60)

        try:
            validate_zarr_readback(
                output_store, variable_id, max_retries=120, retry_delay=60
            )
            logging.info("=" * 60)
            logging.info("✓✓✓ Difference data validated and confirmed readable ✓✓✓")
            logging.info("=" * 60)
        except Exception as e:
            logging.error("=" * 60)
            logging.error(f"✗✗✗ FATAL: Cannot read back written data: {e} ✗✗✗")
            logging.error("This data should NOT be used as input to other scripts!")
            logging.error("=" * 60)
            # Clean up unreadable output
            if output_store.exists():
                shutil.rmtree(output_store, ignore_errors=True)
            raise

        logging.info(f"Difference calculation completed successfully for {variable_id}")
        success = True

    except Exception as e:
        logging.error(f"FATAL ERROR during difference calculation: {e}")
        logging.error(f"Difference calculation FAILED")
        # Clean up any partial output
        if output_store.exists():
            logging.info(f"Cleaning up failed output at {output_store}")
            shutil.rmtree(output_store, ignore_errors=True)
        sys.exit(1)

    finally:
        # Always cleanup Dask client
        if client is not None:
            logging.info("Closing Dask client...")
            client.close()

    if not success:
        logging.error("Difference calculation did not complete successfully")
        sys.exit(1)

    logging.info("Exiting with success")
    sys.exit(0)
