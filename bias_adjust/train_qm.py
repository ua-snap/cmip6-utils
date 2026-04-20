"""Script to train a quantile mapping adjustment for a given model. Uses fixed historical reference years for training.

Notes:
- Assumes time series of sim and ref are the same!

Example usage:
    python train_qm.py \
        --sim_path /beegfs/CMIP6/kmredilla/zarr_bias_adjust_inputs/zarr/pr_GFDL-ESM4_historical.zarr \
        --ref_path /beegfs/CMIP6/kmredilla/zarr_bias_adjust_inputs/zarr/pr_era5.zarr \
        --train_path /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted/trained/qdm_trained_pr_GFDL-ESM4.zarr \
        --tmp_path /center1/CMIP6/kmredilla/tmp
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# import icclim
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
import xarray as xr
from zarr.sync import ThreadSynchronizer
import numcodecs
from xclim import sdba
from luts import sim_ref_var_lu, varid_adj_kind_lu, jitter_under_lu, adapt_freq_thresh_lu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def configure_dask_for_training(n_workers=4, threads_per_worker=4, memory_limit="30GB", local_directory=None):
    """Configure Dask LocalCluster optimized for QDM training on 128GB nodes.

    Args:
        n_workers: Number of worker processes (default: 4)
        threads_per_worker: Threads per worker (default: 4)
        memory_limit: Memory limit per worker (default: 30GB = 120GB total for 4 workers)

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
            # Memory management - more aggressive for large climate data
            "distributed.worker.memory.target": 0.70,  # Start spilling at 70%
            "distributed.worker.memory.spill": 0.80,  # Agressively spill at 80%
            "distributed.worker.memory.pause": 0.85,  # Pause at 85%
            "distributed.worker.memory.terminate": 0.95,  # Kill worker at 95%
            # Optimize for I/O with zarr
            "distributed.comm.timeouts.tcp": "120s",
            "distributed.scheduler.bandwidth": 1e9,  # Assume 1 Gbps network
            # Array optimization
            "array.slicing.split_large_chunks": True,
            "array.chunk-size": "128 MiB",  # Target chunk size
            # Disable work stealing for deterministic results
            "distributed.scheduler.work-stealing": False,
        }
    )

    # Create LocalCluster with explicit resource limits
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=True,  # Use processes not threads for GIL-bound work
        dashboard_address=None,  # Disable dashboard on compute nodes
        local_directory=str(local_directory) if local_directory else None,
    )

    client = Client(cluster)

    logging.info(f"Dask cluster configured:")
    logging.info(f"  Workers: {n_workers}")
    logging.info(f"  Threads per worker: {threads_per_worker}")
    logging.info(f"  Memory per worker: {memory_limit}")
    logging.info(f"  Total memory: {n_workers * 30} GB (out of ~128 GB available)")
    logging.info(f"  Dashboard: {client.dashboard_link}")

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
            logging.info(f"  Starting attempt {attempt}/{max_attempts}...")

            # Force kernel to flush anything pending
            try:
                os.sync()
                logging.info(f"    os.sync() completed")
            except Exception as e:
                logging.warning(f"    os.sync() failed: {e}")

            # Find chunk files - zarr chunks are named like "0.0.0", "1.2.3", etc.
            # They're in subdirectories for each variable
            logging.info(f"    Running find command on {zarr_path}...")
            result = subprocess.run(
                ["find", str(zarr_path), "-type", "f", "-name", "*.*.*"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            logging.info(
                f"    find command completed with return code {result.returncode}"
            )
            logging.info(f"    stdout length: {len(result.stdout)} bytes")

            if result.stderr:
                logging.warning(f"    find stderr: {result.stderr[:500]}")

            chunk_files = [
                line
                for line in result.stdout.strip().split("\n")
                if line
                and not line.endswith(".zarray")
                and not line.endswith(".zattrs")
            ]
            logging.info(f"  Attempt {attempt}: Found {len(chunk_files)} chunk files")

            if len(chunk_files) > 0:
                # Try to actually read a few bytes from chunk files to force data into cache
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
        expected_var_id: Variable to check (for logging only - trained QDM has different vars)
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

            # Trained QDM datasets don't contain the original variable
            # They contain xclim parameters like hist_q, ref_q, af, etc.
            if len(ds.data_vars) == 0:
                raise ValueError("Dataset has no data variables")

            logging.info(f"Dataset contains variables: {list(ds.data_vars.keys())}")

            # Pick first variable to validate (they should all be present or all absent)
            var_name = list(ds.data_vars.keys())[0]
            arr = ds[var_name]

            # Check actual data, not just metadata
            # Check multiple samples to handle NaNs at domain edges
            logging.info(
                f"Checking data validity by loading samples from '{var_name}'..."
            )

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
            if var_name not in z:
                raise ValueError(f"Variable {var_name} not in zarr group")

            var_array = z[var_name]
            chunk_keys = [k for k in var_array.chunk_store.keys() if var_name in str(k)]
            chunk_count = len(chunk_keys)
            logging.info(f"Found {chunk_count} chunk files for {var_name}")

            if chunk_count == 0:
                raise ValueError("No chunk files found!")

            # Success!
            logging.info(f"✓ Read-back validation PASSED on attempt {attempt}")
            logging.info(f"  - Validated variable: {var_name}")
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

    args.sim_path = Path(args.sim_path)
    args.ref_path = Path(args.ref_path)
    args.train_path = Path(args.train_path)
    args.tmp_path = Path(args.tmp_path)
    if not args.sim_path.exists():
        raise FileNotFoundError(f"Zarr store {args.sim_path} not found.")
    if not args.ref_path.exists():
        raise FileNotFoundError(f"Zarr store {args.ref_path} not found.")
    if not args.train_path.parent.exists():
        raise FileNotFoundError(
            f"Parent directory of requested training outputs directory, {args.train_path.parent},"
            " does not exist, and needs to for this script to run."
        )

    return args


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim_path",
        type=str,
        help="path to zarr store of historical simulation data",
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        help="path to zarr store of reference data",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        help="Path to write trained QM object to",
    )
    parser.add_argument(
        "--tmp_path",
        type=str,
        help="path to temporary scratch space for dask",
    )
    args = parser.parse_args()

    args = validate_args(args)

    return (
        args.sim_path,
        args.ref_path,
        args.train_path,
        args.tmp_path,
    )


# --- Utility functions for diagnostics and jitter ---
def check_data_validity(ds, var_id, label):
    """Check if the dataset variable is all NaN or empty and raise an error if invalid.

    Enhanced version that checks multiple samples across the dataset to catch
    filesystem cache coherency issues on distributed systems.
    """
    if var_id not in ds.data_vars:
        error_msg = f"{label} variable '{var_id}' not found in dataset."
        logging.error(error_msg)
        raise ValueError(error_msg)

    arr = ds[var_id]
    if arr.size == 0:
        error_msg = f"{label} data for variable '{var_id}' is empty!"
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Log array info for debugging
    logging.info(f"Checking {label} data for '{var_id}':")
    logging.info(f"  Shape: {arr.shape}")
    logging.info(f"  Chunks: {arr.chunks}")
    logging.info(f"  Dtype: {arr.dtype}")

    # Check multiple samples to be thorough - distributed filesystems can have
    # inconsistent cache states where some chunks are visible and others aren't
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

            logging.info(
                f"  {location} sample: shape={sample_data.shape}, "
                f"min={float(sample_data.min()):.4f}, "
                f"max={float(sample_data.max()):.4f}, "
                f"mean={float(sample_data.mean()):.4f}"
            )

            if sample_data.isnull().all():
                all_nan_count += 1
                logging.warning(f"  WARNING: {location} sample is all NaN!")
        except Exception as e:
            logging.error(f"  ERROR reading {location} sample: {e}")
            raise

    if all_nan_count == len(samples_to_check):
        error_msg = (
            f"{label} data for variable '{var_id}' appears to be all NaN! "
            f"Checked {len(samples_to_check)} locations, all returned NaN. "
            f"This suggests a filesystem cache coherency issue."
        )
        logging.error(error_msg)
        logging.error(f"  Dataset path: {ds.encoding.get('source', 'unknown')}")
        raise ValueError(error_msg)

    if all_nan_count > 0:
        logging.warning(
            f"  {all_nan_count}/{len(samples_to_check)} samples were all NaN, but some had data"
        )

    logging.info(f"{label} data validation passed for variable '{var_id}'")
    return True


def apply_jitter(da):
    """Apply jitter to the data if the variable is in the jitter under lookup table."""
    var_id = da.name
    jitter_under_thresh = jitter_under_lu[var_id]
    da = sdba.processing.jitter_under_thresh(da, thresh=jitter_under_thresh)
    logging.info(f"Jitter under {jitter_under_thresh} applied to data")
    return da


def get_var_id(ds):
    """Get the variable ID from the dataset.

    Filters out metadata variables (mask, spatial_ref, etc.) by selecting
    only variables with 3 dimensions (time, x, y).
    """
    # Filter to only 3D variables (time, x, y) - excludes mask (2D), spatial_ref (0D), etc.
    climate_vars = [v for v in ds.data_vars if len(ds[v].dims) == 3]

    if len(climate_vars) == 0:
        raise ValueError(f"No 3D climate variable found in dataset. Available variables: {list(ds.data_vars)}")
    if len(climate_vars) > 1:
        raise ValueError(f"Multiple 3D variables found in dataset: {climate_vars}")

    var_id = climate_vars[0]
    return var_id


def ensure_matching_time_coords(hist_ds, ref_ds):
    """Ensure that the time coordinates of two datasets match."""
    if all(ref_ds.time.values == hist_ds.time.values) is False:
        # if the first dates match, and the hours don't, we can just fix that by using the hour used in the historical data
        if (
            ref_ds.time.values[0].strftime("%Y-%m-%d")
            == hist_ds.time.values[0].strftime("%Y-%m-%d")
        ) and (ref_ds.time.size == hist_ds.time.size):
            if ref_ds.time.values.min().hour != hist_ds.time.values.min().hour:
                # just make the hour the same for one of them
                start_time = ref_ds.time.values.min()
                end_time = ref_ds.time.values.max()
                use_hour = hist_ds.time.values.min().hour
                new_ref_times = xr.cftime_range(
                    f"{start_time.year}-{str(start_time.month).zfill(2)}-{str(start_time.day).zfill(2)} {use_hour}:00:00",
                    f"{end_time.year}-{str(end_time.month).zfill(2)}-{str(end_time.day).zfill(2)} {use_hour}:00:00",
                    freq="D",
                    calendar="noleap",
                )
                ref_ds = ref_ds.assign(time=new_ref_times)
                assert all(
                    ref_ds.time.values == hist_ds.time.values
                ), "Hist and ref time values do not match after adjusting hours"

    return hist_ds, ref_ds


def ensure_correct_ref_precip_units(ref):
    """Just make sure that the reference precipitation units are correct for the QM training."""
    # need to set the correct compatible precipitation units for ERA5 if precip
    if ref.attrs["units"] == "mm":
        ref.attrs["units"] = "mm d-1"
    elif ref.attrs["units"] == "m":
        ref.attrs["units"] = "m d-1"
    else:
        raise ValueError(
            f"Reference precipitation units are not compatible with QM training. Found {ref.attrs['units']}"
        )
    return ref


def keep_attrs(train_ds, hist_ds, hist_path):
    """Keep the attributes of the historical dataset for the trained QM object.

    Params:
    -------
    train_ds: xarray.Dataset
        The trained QM object dataset.
    hist_ds: xarray.Dataset
        The historical dataset.

    Returns:
    --------
    train_ds : xarray.Dataset
        The trained QM object dataset with the attributes of the historical dataset.
    """
    check_attrs = [
        "activity_id",
        "experiment_id",
        "source_id",
        "variabl_id",
        "table_id",
    ]

    for attr in check_attrs:
        if attr in hist_ds.attrs:
            train_ds.attrs[attr] = hist_ds.attrs[attr]

    train_ds.attrs["parent_path"] = str(hist_path)

    return train_ds


def validate_training_output(qm_train, var_id):
    """Validate that the training produced valid quantile mappings.

    Args:
        qm_train: The trained QDM object
        var_id: Variable identifier (for logging only)

    Raises:
        ValueError: If training output is invalid
    """
    logging.info(f"Validating training output for {var_id}...")

    # Check that the dataset exists
    if not hasattr(qm_train, "ds"):
        raise ValueError("Trained QM object has no 'ds' attribute")

    # Check that the dataset has data variables (xclim stores QDM parameters, not the original variable)
    if len(qm_train.ds.data_vars) == 0:
        raise ValueError("Trained QM dataset has no data variables")

    logging.info(f"  Trained QM dataset contains: {list(qm_train.ds.data_vars.keys())}")

    # Check for 'quantiles' dimension (expected in QDM output)
    if "quantiles" not in qm_train.ds.dims:
        raise ValueError(
            "No 'quantiles' dimension found in QDM output - training may have failed"
        )

    # Check that at least one quantile variable is not all NaN
    # Common xclim QDM variables: hist_q, ref_q, af (adjustment factors)
    # Check multiple spatial samples to handle NaNs at domain edges
    has_valid_data = False
    for var_name in qm_train.ds.data_vars:
        arr = qm_train.ds[var_name]
        if arr.size > 0:
            # Check start, middle, and end samples (like check_data_validity)
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

            for location, selection in samples_to_check:
                sample = arr.isel(selection)
                if not sample.isnull().all().compute():
                    has_valid_data = True
                    logging.info(
                        f"  ✓ Variable '{var_name}' contains valid data ({location} sample)"
                    )
                    break

            if has_valid_data:
                break

    if not has_valid_data:
        raise ValueError(
            "All variables in trained QM dataset are NaN or empty (checked start, middle, and end samples)"
        )

    logging.info(f"Training output validation passed for {var_id}")
    return True


def validate_written_zarr(train_path, var_id, min_size_mb=1):
    """Validate that the written zarr store is valid and has reasonable size.

    Args:
        train_path: Path to the zarr store
        var_id: Variable identifier
        min_size_mb: Minimum expected size in MB

    Raises:
        ValueError: If output is invalid
    """
    import os

    logging.info(f"Validating written zarr store at {train_path}...")

    # Check that the path exists
    if not train_path.exists():
        raise ValueError(f"Output zarr store was not created at {train_path}")

    # Check directory size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(train_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)

    size_mb = total_size / (1024 * 1024)
    logging.info(f"Output zarr store size: {size_mb:.2f} MB")

    if size_mb < min_size_mb:
        raise ValueError(
            f"Output zarr store is suspiciously small ({size_mb:.2f} MB < {min_size_mb} MB). "
            "Training may have failed."
        )

    # Try to open and validate the zarr store
    try:
        out_ds_check = xr.open_zarr(train_path, consolidated=True)
    except Exception as e:
        raise ValueError(f"Cannot open output zarr store: {e}")

    if var_id not in out_ds_check.data_vars:
        raise ValueError(f"Output variable '{var_id}' not found in {train_path}")

    arr = out_ds_check[var_id]
    if arr.size == 0:
        raise ValueError(f"Output for {train_path} is empty")

    # Sample check (avoid loading entire array)
    sample = arr.isel({dim: slice(0, min(10, arr.sizes[dim])) for dim in arr.dims})
    if sample.isnull().all().compute():
        raise ValueError(f"Output for {train_path} is all NaN")

    logging.info(f"Written zarr validation passed for {train_path}")
    return True


if __name__ == "__main__":
    # Version marker for debugging
    logging.info("=" * 70)
    logging.info("TRAIN_QM.PY - VERSION 2026-03-04 - Cache refresh diagnostics enabled")
    logging.info("=" * 70)

    (sim_path, ref_path, train_path, tmp_path) = parse_args()

    # Track success/failure for proper exit code
    success = False
    client = None

    try:
        # Configure Dask with explicit cluster
        logging.info("Configuring Dask cluster...")
        worker_dir = tmp_path / f"train-{sim_path.stem}-{os.getpid()}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        client = configure_dask_for_training(
            n_workers=4,
            threads_per_worker=4,
            memory_limit="28GB",  # 4 workers × 28GB = 112GB, leaving 16GB for system
            local_directory=worker_dir,
        )

        logging.info(f"Starting QM training for {sim_path.name}")
        logging.info(f"Opening input datasets...")

        # Optimized chunking strategy:
        # - Time: chunk into ~2 year blocks (730 days) for QDM dayofyear grouping
        # - Spatial: larger chunks (100x100) to reduce overhead
        # This balances memory usage with computational efficiency
        time_chunk = 730  # ~2 years
        spatial_chunk = 100

        chunk_dict = {"time": time_chunk, "x": spatial_chunk, "y": spatial_chunk}

        logging.info(
            f"Using chunk strategy: time={time_chunk}, x={spatial_chunk}, y={spatial_chunk}"
        )

        # CRITICAL: Force filesystem cache refresh before opening zarr files
        # This is essential for multi-node jobs where writer != reader node
        logging.info("=" * 60)
        logging.info("FORCING FILESYSTEM CACHE REFRESH - VERSION 2026-03-04")
        logging.info("=" * 60)

        try:
            logging.info(f"  Refreshing cache for sim_path: {sim_path}")
            force_filesystem_cache_refresh(sim_path, max_attempts=10, delay=10)
            logging.info(f"  ✓ sim_path cache refresh completed")
        except Exception as e:
            logging.error(f"  ✗ Failed to refresh cache for {sim_path}: {e}")
            logging.exception("Full traceback:")
            raise

        try:
            logging.info(f"  Refreshing cache for ref_path: {ref_path}")
            force_filesystem_cache_refresh(ref_path, max_attempts=10, delay=10)
            logging.info(f"  ✓ ref_path cache refresh completed")
        except Exception as e:
            logging.error(f"  ✗ Failed to refresh cache for {ref_path}: {e}")
            logging.exception("Full traceback:")
            raise

        logging.info("=" * 60)
        logging.info("✓ Cache refresh complete, opening zarr files...")
        logging.info("=" * 60)

        # Open with optimized chunking - use consolidated=False to avoid stale metadata
        hist_ds = xr.open_zarr(sim_path, chunks=chunk_dict, consolidated=False)

        # Convert calendar and rechunk - this is expensive, so log it
        logging.info("Converting reference calendar to noleap...")
        ref_ds = xr.open_zarr(ref_path, consolidated=False).convert_calendar(
            "noleap", align_on="date"
        )
        ref_ds = ref_ds.chunk(chunk_dict)
        hist_ds, ref_ds = ensure_matching_time_coords(hist_ds, ref_ds)

        var_id = get_var_id(hist_ds)
        logging.info(f"Processing variable: {var_id}")

        if var_id not in ref_ds.data_vars:
            ref_var_id = sim_ref_var_lu[var_id]
            logging.info(f"Renaming reference variable {ref_var_id} to {var_id}")
            ref_ds = ref_ds.rename({ref_var_id: var_id})

        # Input data validation - these now raise errors instead of returning False
        logging.info("Validating input data...")
        check_data_validity(hist_ds, var_id, "Historical")
        check_data_validity(ref_ds, var_id, "Reference")

        # get dataarrays
        ref = ref_ds[var_id]
        hist = hist_ds[var_id]

        if var_id == "pr":
            logging.info("Ensuring correct precipitation units for reference data")
            ref = ensure_correct_ref_precip_units(ref)

        # ensure data does not have zeros, depending on variable
        if var_id in jitter_under_lu.keys():
            logging.info(f"Applying jitter to {var_id}")
            hist = apply_jitter(hist)
            ref = apply_jitter(ref)

        # CRITICAL: Rechunk time dimension for training
        # xclim's QDM training requires time to be in a single chunk
        # Use smaller spatial chunks to compensate for large time chunk
        # Memory per chunk: 18,250 × 50 × 50 × 4 bytes = ~1.8GB (×2 for hist+ref = 3.6GB)
        logging.info("Rechunking data for training (time=-1 required by xclim)...")
        training_chunks = {"time": -1, "x": 30, "y": 30}
        hist = hist.chunk(training_chunks)
        ref = ref.chunk(training_chunks)
        logging.info(f"  Training chunks: time=-1, x=30, y=30")
        logging.info(f"  Memory per spatial chunk: ~1.3GB (both arrays)")
        logging.info(f"  hist chunks: {hist.chunks}")
        logging.info(f"  ref chunks: {ref.chunks}")

        logging.info(f"Starting QDM training for {var_id}...")
        train_kwargs = dict(
            ref=ref,
            hist=hist,
            nquantiles=100,
            group="time.dayofyear",
            window=31,
            kind=varid_adj_kind_lu[var_id],
        )
        if var_id in adapt_freq_thresh_lu:
            thresh = adapt_freq_thresh_lu[var_id]
            logging.info(f"Using adapt_freq_thresh={thresh} for {var_id}")
            train_kwargs.update(adapt_freq_thresh=thresh)

        qm_train = sdba.QuantileDeltaMapping.train(**train_kwargs)
        logging.info(f"QDM training completed for {var_id}")

        # Validate training output before writing
        validate_training_output(qm_train, var_id)

        qm_train.ds = keep_attrs(qm_train.ds, hist_ds, sim_path)

        # Remove existing output if present
        if train_path.exists():
            logging.info(f"Removing existing output at {train_path}")
            try:
                shutil.rmtree(train_path)
            except Exception as e:
                logging.error(f"Failed to remove existing output: {e}")
                raise

        # Write output with optimized zarr settings
        logging.info(f"Writing QDM object to {train_path}")
        synchronizer = ThreadSynchronizer()

        # Configure compression for trained data (quantiles compress well)
        encoding = {}
        for var in qm_train.ds.data_vars:
            encoding[var] = {
                "compressor": numcodecs.Blosc(
                    cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE
                )
            }

        try:
            qm_train.ds.to_zarr(
                train_path,
                encoding=encoding,
                synchronizer=synchronizer,
                consolidated=True,  # Faster subsequent reads
                compute=True,  # Force synchronous write completion
            )
            logging.info(f"Initial write to {train_path} completed")
        except Exception as e:
            logging.error(f"Failed to write zarr store: {e}")
            # Clean up partial write
            if train_path.exists():
                shutil.rmtree(train_path, ignore_errors=True)
            raise

        # CRITICAL: Validate we can read it back
        logging.info("=" * 60)
        logging.info("Starting read-after-write validation (up to 2 hours)...")
        logging.info("=" * 60)

        try:
            validate_zarr_readback(train_path, var_id, max_retries=120, retry_delay=60)
            logging.info("=" * 60)
            logging.info("✓✓✓ Trained QDM object validated and confirmed readable ✓✓✓")
            logging.info("=" * 60)
        except Exception as e:
            logging.error("=" * 60)
            logging.error(f"✗✗✗ FATAL: Cannot read back written data: {e} ✗✗✗")
            logging.error("This data should NOT be used as input to other scripts!")
            logging.error("=" * 60)
            # Clean up unreadable output
            if train_path.exists():
                shutil.rmtree(train_path, ignore_errors=True)
            raise

        logging.info(f"QM training pipeline completed successfully for {var_id}")
        success = True

    except Exception as e:
        logging.error(f"FATAL ERROR during training or writing: {e}")
        logging.error(f"Training FAILED for {sim_path.name}")
        # Clean up any partial output
        if train_path.exists():
            logging.info(f"Cleaning up failed output at {train_path}")
            shutil.rmtree(train_path, ignore_errors=True)
        # Exit with error code
        sys.exit(1)

    finally:
        # Always cleanup Dask client
        if client is not None:
            logging.info("Closing Dask client...")
            client.close()

    if not success:
        logging.error("Training did not complete successfully")
        sys.exit(1)

    logging.info("Exiting with success")
    sys.exit(0)
