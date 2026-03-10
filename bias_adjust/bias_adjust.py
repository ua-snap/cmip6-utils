"""Script for bias adjusting a given model and scenario. Uses a pre-trained quantile mapping adjustment object.

Example usage:
    python bias_adjust.py --train_path /center1/CMIP6/kmredilla/bias_adjustment_testing/trained_qdm_pr_GFDL-ESM4.zarr --sim_path /center1/CMIP6/kmredilla/zarr_bias_adjust_inputs/pr_GFDL-ESM4_historical.zarr --adj_path /center1/CMIP6/kmredilla/cmip6_4km_3338_downscaled/pr_GFDL-ESM4_historical_adj.zarr --tmp_path /center1/CMIP6/kmredilla/tmp
"""

import argparse
import datetime
import logging
import os
import shutil
import subprocess
import sys
import time

# import multiprocessing as mp
from itertools import product
from pathlib import Path
import dask
from dask.distributed import Client, LocalCluster
import xarray as xr
from xclim import sdba
import numcodecs

from zarr.sync import ThreadSynchronizer

# from luts import jitter_under_lu
from train_qm import get_var_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def configure_dask_for_adjustment(
    n_workers=4, threads_per_worker=4, memory_limit="30GB"
):
    """Configure Dask LocalCluster optimized for bias adjustment on 128GB nodes.

    Adjustment is more I/O intensive than training, so optimize accordingly.

    Args:
        n_workers: Number of worker processes (default: 4)
        threads_per_worker: Threads per worker (default: 4)
        memory_limit: Memory limit per worker (default: 30GB)

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
            # Memory management - more conservative thresholds to avoid OOM
            "distributed.worker.memory.target": 0.70,  # Start managing at 70%
            "distributed.worker.memory.spill": 0.80,  # Spill to disk at 80%
            "distributed.worker.memory.pause": 0.85,  # Pause at 85%
            "distributed.worker.memory.terminate": 0.95,
            # I/O optimization
            "distributed.comm.timeouts.tcp": "120s",
            "distributed.scheduler.bandwidth": 1e9,
            # Array settings
            "array.slicing.split_large_chunks": True,
            "array.chunk-size": "128 MiB",
        }
    )

    # Explicitly set worker space directory
    worker_space = Path.home() / "dask-worker-space"
    worker_space.mkdir(exist_ok=True)

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=True,
        dashboard_address=None,
        local_directory=str(worker_space),  # Ensure consistent worker space
    )

    client = Client(cluster)

    logging.info(f"Dask cluster configured for adjustment:")
    logging.info(f"  Workers: {n_workers}, Threads/worker: {threads_per_worker}")
    logging.info(f"  Memory per worker: {memory_limit}")
    logging.info(f"  Worker space: {worker_space}")
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

            # Check multiple samples to catch filesystem cache coherency issues
            # where some chunks may not be visible yet on different nodes
            logging.info(f"Checking data validity by loading multiple samples...")
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

            all_nan_count = 0
            sample_stats = []

            for location, selection in samples_to_check:
                sample = arr.isel(selection)
                sample_data = sample.compute()  # Force actual read from disk

                if sample_data.size == 0:
                    raise ValueError(f"{location} sample is empty")

                if sample_data.isnull().all():
                    all_nan_count += 1
                    logging.warning(f"  WARNING: {location} sample is all NaN")
                else:
                    sample_stats.append(
                        (
                            location,
                            float(sample_data.mean()),
                            float(sample_data.min()),
                            float(sample_data.max()),
                        )
                    )

            # Only fail if ALL samples are NaN (suggests real problem)
            if all_nan_count == len(samples_to_check):
                raise ValueError(
                    f"All {len(samples_to_check)} samples are NaN! "
                    f"This suggests a filesystem cache coherency issue or failed computation."
                )

            if all_nan_count > 0:
                logging.warning(
                    f"  {all_nan_count}/{len(samples_to_check)} samples were all NaN, "
                    f"but validation passed (some data found)"
                )

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
            if sample_stats:
                logging.info(
                    f"  - Valid samples: {len(sample_stats)}/{len(samples_to_check)}"
                )
                for location, mean, min_val, max_val in sample_stats:
                    logging.info(
                        f"    {location}: mean={mean:.4f}, range=[{min_val:.4f}, {max_val:.4f}]"
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


def add_global_attrs(adj_ds, src_ds):
    """Add global attributes to a new adjusted dataset

    Args:
        adj_ds (xarray.Dataset): dataset of a adjusted data
        src_ds (xarray.Dataset): dataset of source data

    Returns:
        xarray.Dataset with updated global attributes
    """
    # create new global attributes
    new_attrs = {
        "history": "File was processed by Scenarios Network for Alaska and Arctic Planning (SNAP) using xclim",
        "contact": "uaf-snap-data-tools@alaska.edu",
        "Conventions": "CF-1.7",
        "creation_date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        # experimental idea - store attributes from source file in an attribute for reference
        # use a string as netcdf cannot be serialized if this is a dict
        # (it can be reconstructed with eval() if desired later)
        "parent_attributes": str(src_ds.attrs),
    }
    for attr in [
        "variable_id",
        "source_id",
        "institution_id",
        "mip_era",
        "activity_id",
        "experiment_id",
        "table_id",
    ]:
        if attr in src_ds.attrs:
            new_attrs[attr] = src_ds.attrs[attr]

    adj_ds.attrs = new_attrs

    return adj_ds


def drop_non_coord_vars(ds, keep_spatial_ref=True, keep_latlon=True):
    """Function to drop all coordinates from xarray dataset which are not coordinate variables, i.e. which are not solely indexed by a dimension of the same name

    Args:
        ds (xarray.Dataset): dataset to drop non-coordinate-variables from
        keep_spatial_ref (bool): whether to keep the spatial_ref variable, which is not a coordinate variable but is useful for some applications
        keep_latlon (bool): whether to keep the lat and lon coordinates if present, which are not coordinate variables but are useful for some applications

    Returns:
        ds (xarray.Dataset): dataset with only dimension coordinates
    """
    coords_to_drop = [coord for coord in ds.coords if ds[coord].dims != (coord,)]
    if keep_spatial_ref:
        coords_to_drop.remove("spatial_ref")
    if keep_latlon:
        coords_to_drop.remove("lat")
        coords_to_drop.remove("lon")

    vars_to_drop = [var for var in ds.data_vars if len(ds[var].dims) < 3]
    ds = ds.drop_vars(coords_to_drop + vars_to_drop)

    return ds


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_path",
        type=str,
        help="Path to trained quantile mapping adjustment netcdf file",
        required=True,
    )
    parser.add_argument(
        "--sim_path",
        type=str,
        help="Path to model data to be adjusted",
        required=True,
    )
    parser.add_argument(
        "--adj_path",
        type=str,
        help="Path to write adjusted data",
        required=True,
    )
    parser.add_argument(
        "--tmp_path",
        type=str,
        help="Path to directory where dask temporary files will be written.",
        required=True,
    )

    args = parser.parse_args()

    return (
        Path(args.train_path),
        Path(args.sim_path),
        Path(args.adj_path),
        Path(args.tmp_path),
    )


def validate_sim_source(train_ds, sim_ds):
    """Validate that training and sim datasets have matching source_id.

    Raises:
        ValueError: If source_ids don't match
    """
    logging.info("Validating source_id")
    if "source_id" not in train_ds.attrs:
        raise ValueError("Training dataset missing 'source_id' attribute")
    if "source_id" not in sim_ds.attrs:
        raise ValueError("Simulation dataset missing 'source_id' attribute")

    if train_ds.attrs["source_id"] != sim_ds.attrs["source_id"]:
        raise ValueError(
            f"Source ID mismatch: training has '{train_ds.attrs['source_id']}' "
            f"but simulation has '{sim_ds.attrs['source_id']}'"
        )

    logging.info(
        f"Simulated data source (model) validated: {train_ds.attrs['source_id']}"
    )


def validate_input_data(ds, var_id, label):
    """Validate that input data is not empty or all NaN.

    Checks multiple samples across the dataset to handle filesystem cache
    coherency issues on distributed systems like beegfs.

    Args:
        ds: xarray Dataset
        var_id: variable identifier
        label: descriptive label for error messages

    Raises:
        ValueError: If data is invalid or all samples are NaN
    """
    if var_id not in ds.data_vars:
        raise ValueError(f"{label} variable '{var_id}' not found in dataset")

    arr = ds[var_id]
    if arr.size == 0:
        raise ValueError(f"{label} for {var_id} is empty")

    # Check multiple samples to catch filesystem cache coherency issues
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
                logging.warning(f"  WARNING: {location} sample is all NaN for {label}")
        except Exception as e:
            logging.error(f"  ERROR reading {location} sample for {label}: {e}")
            raise

    # Only fail if ALL samples are NaN (suggests real problem)
    if all_nan_count == len(samples_to_check):
        raise ValueError(
            f"{label} for {var_id} appears to be all NaN! "
            f"Checked {len(samples_to_check)} locations, all returned NaN. "
            f"This may indicate a filesystem cache coherency issue."
        )

    if all_nan_count > 0:
        logging.warning(
            f"  {all_nan_count}/{len(samples_to_check)} samples were all NaN for {label}, "
            f"but validation passed (some data found)"
        )

    logging.info(f"{label} validation passed for {var_id}")


def validate_output_zarr(adj_path, var_id, min_size_mb=10):
    """Validate written zarr output.

    Args:
        adj_path: Path to zarr store
        var_id: variable identifier
        min_size_mb: minimum expected size in MB

    Raises:
        ValueError: If output is invalid
    """
    import os

    logging.info(f"Validating output at {adj_path}...")

    if not adj_path.exists():
        raise ValueError(f"Output zarr store not created at {adj_path}")

    # Check size
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(adj_path)
        for filename in filenames
        if os.path.exists(os.path.join(dirpath, filename))
    )

    size_mb = total_size / (1024 * 1024)
    logging.info(f"Output zarr store size: {size_mb:.2f} MB")

    if size_mb < min_size_mb:
        raise ValueError(
            f"Output suspiciously small ({size_mb:.2f} MB < {min_size_mb} MB)"
        )

    # Validate content
    try:
        adj_ds_check = xr.open_zarr(adj_path, consolidated=True)
    except Exception as e:
        raise ValueError(f"Cannot open output zarr: {e}")

    validate_input_data(adj_ds_check, var_id, "Output")
    logging.info("Output validation passed")


if __name__ == "__main__":
    train_path, sim_path, adj_path, tmp_path = parse_args()

    success = False
    client = None

    try:
        # Configure Dask
        logging.info("Configuring Dask cluster...")
        client = configure_dask_for_adjustment(
            n_workers=4, threads_per_worker=4, memory_limit="28GB"
        )

        logging.info(f"Starting bias adjustment for {sim_path.name}")

        # Initial chunking for loading - will be rechunked before adjustment
        # xclim requires time=-1, but we load with larger chunks first for efficiency
        chunk_dict = {"time": 365, "x": 150, "y": 150}

        logging.info(
            f"Using initial chunk strategy: time={chunk_dict['time']}, x={chunk_dict['x']}, y={chunk_dict['y']}"
        )

        # open connection to trained QM dataset
        logging.info(f"Loading trained QM dataset from {train_path}")
        train_ds = xr.open_zarr(
            train_path, consolidated=True
        )  # Training data is small, no need to chunk
        qm = sdba.QuantileDeltaMapping.from_dataset(train_ds)

        # Load simulation data with optimized chunks
        logging.info(f"Loading simulation dataset from {sim_path}")
        sim_ds = xr.open_zarr(sim_path, chunks=chunk_dict, consolidated=True)

        # Validate source matching
        validate_sim_source(train_ds, sim_ds)

        var_id = get_var_id(sim_ds)
        logging.info(f"Processing variable: {var_id}")

        # Validate input data
        validate_input_data(sim_ds, var_id, "Input simulation data")

        # CRITICAL: Rechunk time dimension for adjustment
        # xclim's QDM adjust() requires time to be in a single chunk (same as training)
        # Use smaller spatial chunks to compensate for large time chunk
        # Memory per chunk: ~36,500 days × 50 × 50 × 4 bytes = ~3.6GB (worst case for long scenarios)
        logging.info("Rechunking data for adjustment (time=-1 required by xclim)...")
        sim_var = sim_ds[var_id]
        adjustment_chunks = {"time": -1, "x": 50, "y": 50}
        sim_var_rechunked = sim_var.chunk(adjustment_chunks)
        logging.info(f"  Adjustment chunks: time=-1, x=50, y=50")
        logging.info(f"  Memory per spatial chunk: ~3.6GB (worst case)")
        logging.info(f"  Rechunked shape: {sim_var_rechunked.shape}")
        logging.info(f"  Rechunked chunks: {sim_var_rechunked.chunks}")

        logging.info(f"Applying bias adjustment for {var_id}...")
        scen = qm.adjust(
            sim_var_rechunked,
            extrapolation="constant",
            interp="nearest",
        )
        scen_ds = scen.to_dataset(name=var_id)
        scen_ds = drop_non_coord_vars(scen_ds)
        scen_ds = add_global_attrs(scen_ds, sim_ds)
        logging.info(f"Bias adjustment completed for {var_id}")

        # CRITICAL: Rechunk after adjustment to avoid memory issues during squeeze operations
        # The adjustment step requires time=-1, but squeeze operations need smaller chunks
        # Use moderate time chunks to balance memory and task overhead
        logging.info("Rechunking adjusted data for squeeze operations...")
        squeeze_chunks = {"time": 365, "x": 100, "y": 100}
        scen_ds = scen_ds.chunk(squeeze_chunks)
        logging.info(f"  Rechunked to: time=365, x=100, y=100 for memory efficiency")

        # Remove existing output
        if adj_path.exists():
            logging.info(f"Removing existing adjusted data store at {adj_path}")
            try:
                shutil.rmtree(adj_path)
            except Exception as e:
                logging.error(f"Failed to remove existing output: {e}")
                raise

        # Apply variable-specific thresholding
        if var_id == "dtr":
            logging.info("##### START SQUEEZING DTR #####")
            rechunked = scen_ds[var_id].chunk(dict(y=-1, x=-1))
            max_value = rechunked.max().values
            min_value = rechunked.min().values
            lower_thresh = rechunked.quantile(0.0000002).values
            upper_thresh = rechunked.quantile(0.9999998).values

            # Count the number of pixels above and below the thresholds
            num_below = (
                scen_ds[var_id]
                .where(scen_ds[var_id] < lower_thresh)
                .count()
                .compute()
                .item()
            )
            num_above = (
                scen_ds[var_id]
                .where(scen_ds[var_id] > upper_thresh)
                .count()
                .compute()
                .item()
            )
            logging.info(f"Number of pixels below lower threshold: {num_below}")
            logging.info(f"Number of pixels above upper threshold: {num_above}")

            scen_ds[var_id] = scen_ds[var_id].where(
                (scen_ds[var_id] >= lower_thresh) | scen_ds[var_id].isnull(),
                other=lower_thresh,
            )
            scen_ds[var_id] = scen_ds[var_id].where(
                (scen_ds[var_id] <= upper_thresh) | scen_ds[var_id].isnull(),
                other=upper_thresh,
            )

            logging.info(f"Max DTR value: {max_value}")
            logging.info(f"Min DTR value: {min_value}")
            logging.info(f"Set values below {lower_thresh} to {lower_thresh}")
            logging.info(f"Set values above {upper_thresh} to {upper_thresh}")
            logging.info("##### FINISH SQUEEZING DTR #####")

        max_tasmax = 333.15
        max_pr = 1650
        min_pr = 0

        # tasmin is squeezed separately in the derived/difference.py script
        var_ds = scen_ds[var_id]
        if var_id == "tasmax":
            logging.info("Squeezing tasmax values above limit")
            count_above_threshold = (var_ds > max_tasmax).sum().compute().item()
            logging.info(
                f"Count of values above {max_tasmax} K: {count_above_threshold}"
            )
            var_ds = var_ds.where((var_ds <= max_tasmax) | var_ds.isnull(), max_tasmax)
        elif var_id == "pr":
            logging.info("Squeezing pr values above and below limits")
            count_above_threshold = (var_ds > max_pr).sum().compute().item()
            count_below_zero = (var_ds < min_pr).sum().compute().item()
            logging.info(
                f"Count of values above {max_pr} mm/day: {count_above_threshold}"
            )
            logging.info(f"Count of values below {min_pr} mm/day: {count_below_zero}")
            var_ds = var_ds.where((var_ds <= max_pr) | var_ds.isnull(), max_pr)
            var_ds = var_ds.where((var_ds >= min_pr) | var_ds.isnull(), min_pr)
        scen_ds[var_id] = var_ds

        logging.info(f"Writing adjusted data to {adj_path}")
        synchronizer = ThreadSynchronizer()

        # Configure compression optimized for climate data
        # Output chunks optimized for downstream analysis (year-by-year reads)
        encoding = {
            var_id: {
                "compressor": numcodecs.Blosc(
                    cname="zstd",
                    clevel=3,  # Lower compression for faster writes
                    shuffle=numcodecs.Blosc.BITSHUFFLE,
                ),
                "chunks": (365, 150, 150),  # Year-at-a-time for downstream reads
            }
        }

        try:
            # Use compute with progress tracking
            logging.info("Computing and writing results (this may take a while)...")
            scen_ds.to_zarr(
                adj_path,
                encoding=encoding,
                synchronizer=synchronizer,
                consolidated=True,
                compute=True,
            )
            logging.info(f"Initial write to {adj_path} completed")
        except Exception as e:
            logging.error(f"Failed to write zarr store: {e}")
            if adj_path.exists():
                shutil.rmtree(adj_path, ignore_errors=True)
            raise

        # CRITICAL: Validate we can read it back
        logging.info("=" * 60)
        logging.info("Starting read-after-write validation (up to 2 hours)...")
        logging.info("=" * 60)

        try:
            validate_zarr_readback(adj_path, var_id, max_retries=120, retry_delay=60)
            logging.info("=" * 60)
            logging.info("✓✓✓ Adjusted data validated and confirmed readable ✓✓✓")
            logging.info("=" * 60)
        except Exception as e:
            logging.error("=" * 60)
            logging.error(f"✗✗✗ FATAL: Cannot read back written data: {e} ✗✗✗")
            logging.error("This data should NOT be used as input to other scripts!")
            logging.error("=" * 60)
            # Clean up unreadable output
            if adj_path.exists():
                shutil.rmtree(adj_path, ignore_errors=True)
            raise

        logging.info(f"Bias adjustment pipeline completed successfully for {var_id}")
        success = True

    except Exception as e:
        logging.error(f"FATAL ERROR during processing or writing: {e}")
        logging.error(f"Bias adjustment FAILED for {sim_path.name}")
        # Clean up any partial output
        if adj_path.exists():
            logging.info(f"Cleaning up failed output at {adj_path}")
            shutil.rmtree(adj_path, ignore_errors=True)
        sys.exit(1)

    finally:
        # Always cleanup Dask client
        if client is not None:
            logging.info("Closing Dask client...")
            client.close()

    if not success:
        logging.error("Bias adjustment did not complete successfully")
        sys.exit(1)

    logging.info("Exiting with success")
    sys.exit(0)
