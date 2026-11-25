"""Script for bias adjusting a given model and scenario. Uses a pre-trained quantile mapping adjustment object.

Example usage:
    python bias_adjust.py \
        --train_path /beegfs/CMIP6/.../trained_datasets/trained_qdm_pr_GFDL-ESM4.zarr \
        --sim_path /beegfs/CMIP6/.../cmip6_zarr/pr_GFDL-ESM4_historical.zarr \
        --adj_path /beegfs/CMIP6/.../adjusted/pr_GFDL-ESM4_historical_adjusted.zarr \
        --tmp_path /beegfs/CMIP6/.../tmp

Processing outline (see patch_31d_review/README.md, Appendix A3):
  1. Apply the QDM ranking-window patch (config.QDM_RANK_WINDOWED / --rank_main_only).
  2. Open the sim store with time=-1 spatial chunks (config.ADJUST_CHUNK_SIZE per resolution)
     and the trained QDM store rechunked to the same spatial chunks.
  3. qm.adjust(...) -> transpose to (time, y, x) -> fillna(sim) -> cast float32
     -> ONE distributed compute that gathers the whole result to the client.
  4. Close the dask cluster, then squeeze (clip) in memory per config.SQUEEZE_LIMITS
     (dtr: histogram-quantile limits), recording counts in the global attrs.
  5. Synchronous write to "<adj_path>.writing" -> rename -> read-back validation.
"""

import argparse
import datetime
import json
import logging
import os
import resource
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import dask
import numcodecs
import numpy as np
import psutil
import xarray as xr
import xclim
from dask.distributed import Client, LocalCluster
from distributed.diagnostics import MemorySampler
from xclim import sdba
from zarr.sync import ThreadSynchronizer

import qdm_window_patch
from config import (
    ADJUST_CHUNK_SIZE,
    DASK_ADJUST,
    DTR_HIST_BINS,
    DTR_SQUEEZE_QUANTILES,
    GRID_SHAPE_TO_RESOLUTION,
    OUTPUT_CHUNKS,
    OUTPUT_COMPRESSOR,
    OUTPUT_DTYPE,
    QDM_NQUANTILES,
    QDM_RANK_WINDOWED,
    QDM_WINDOW,
    SQUEEZE_LIMITS,
)
from train_qm import get_var_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

BIAS_ADJUST_VERSION = "2026-11-16 patch_31d"


def configure_dask_for_adjustment(
    n_workers=4,
    threads_per_worker=4,
    memory_limit="30GB",
    worker_dir=None,
    spill_to_disk=True,
):
    """Configure Dask LocalCluster optimized for bias adjustment.

    Args:
        n_workers: Number of worker processes
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker (string or bytes)
        worker_dir: Directory for dask worker files (default: None, uses dask default)

    Returns:
        (client, cluster)
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
            # (target/spill are disabled when spill_to_disk is False)
            "distributed.worker.memory.target": 0.70 if spill_to_disk else False,
            "distributed.worker.memory.spill": 0.80 if spill_to_disk else False,
            "distributed.worker.memory.pause": 0.85,  # Pause at 85%
            "distributed.worker.memory.terminate": 0.95,
            # I/O optimization
            "distributed.comm.timeouts.tcp": "120s",
            "distributed.scheduler.bandwidth": 1e9,
            # Array settings
            "array.slicing.split_large_chunks": True,
            "array.chunk-size": "128 MiB",
            # Determinism - disable work stealing for reproducible results
            "distributed.scheduler.work-stealing": False,
        }
    )

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=True,
        dashboard_address=None,
        local_directory=str(worker_dir) if worker_dir else None,
    )

    client = Client(cluster)

    logging.info(f"Dask cluster configured for adjustment:")
    logging.info(f"  Workers: {n_workers}, Threads/worker: {threads_per_worker}")
    logging.info(f"  Memory per worker: {memory_limit}")
    logging.info(f"  Spill to disk: {spill_to_disk}")
    if worker_dir:
        logging.info(f"  Worker directory: {worker_dir}")
    logging.info(f"  Dashboard: {client.dashboard_link}")

    return client, cluster


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


def add_global_attrs(adj_ds, src_ds, extra_attrs=None):
    """Add global attributes to a new adjusted dataset

    Args:
        adj_ds (xarray.Dataset): dataset of a adjusted data
        src_ds (xarray.Dataset): dataset of source data
        extra_attrs (dict): provenance attributes to add (patch/run settings)

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

    if extra_attrs:
        new_attrs.update(extra_attrs)

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
    if keep_spatial_ref and "spatial_ref" in coords_to_drop:
        coords_to_drop.remove("spatial_ref")
    if keep_latlon:
        if "lat" in coords_to_drop:
            coords_to_drop.remove("lat")
        if "lon" in coords_to_drop:
            coords_to_drop.remove("lon")

    vars_to_drop = [var for var in ds.data_vars if len(ds[var].dims) < 3]
    ds = ds.drop_vars(coords_to_drop + vars_to_drop)

    return ds


def str2bool(value):
    """argparse helper for explicit true/false flags."""
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("true", "t", "yes", "y", "1"):
        return True
    if v in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_isel(spec):
    """Parse an --isel spec like "y=0:50,x=100:150" into {dim: slice}."""
    if not spec:
        return None
    out = {}
    for part in spec.split(","):
        dim, rng = part.split("=")
        start, stop = rng.split(":")
        out[dim.strip()] = slice(int(start), int(stop))
    return out


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_path",
        type=str,
        help="Path to trained quantile mapping adjustment zarr store",
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
    # --- optional overrides (defaults come from config.py) ---
    parser.add_argument(
        "--rank_main_only",
        type=str2bool,
        default=None,
        help="true: legacy exact-day-of-year ranking; false: rank over the training window. "
        f"Default: {'false' if QDM_RANK_WINDOWED else 'true'} (config.QDM_RANK_WINDOWED).",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=sorted(ADJUST_CHUNK_SIZE.keys()),
        default=None,
        help="Resolution label; inferred from the sim grid shape if omitted.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Spatial chunk edge for the time=-1 adjustment chunks (overrides config).",
    )
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--threads_per_worker", type=int, default=None)
    parser.add_argument(
        "--memory_limit",
        type=str,
        default=None,
        help="Per-worker dask memory limit, e.g. 30GB (overrides the derived value).",
    )
    parser.add_argument(
        "--isel",
        type=str,
        default=None,
        help='Spatial subset for tests, e.g. "y=0:50,x=100:150" (applied to sim and trained data).',
    )
    parser.add_argument(
        "--worker_dir_local",
        type=str2bool,
        default=DASK_ADJUST["worker_dir_local"],
        help="true: dask worker files under $SLURM_TMPDIR (or /tmp); false: under --tmp_path.",
    )
    parser.add_argument(
        "--spill_to_disk",
        type=str2bool,
        default=DASK_ADJUST["spill_to_disk"],
        help="Allow dask workers to spill to the worker dir (pointless on tmpfs).",
    )
    parser.add_argument(
        "--run_label",
        type=str,
        default="",
        help="Free-text label recorded in the output attrs (e.g. test id).",
    )

    args = parser.parse_args()
    args.train_path = Path(args.train_path)
    args.sim_path = Path(args.sim_path)
    args.adj_path = Path(args.adj_path)
    args.tmp_path = Path(args.tmp_path)
    args.isel = parse_isel(args.isel)
    if args.rank_main_only is None:
        args.rank_main_only = not QDM_RANK_WINDOWED
    return args


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


def infer_resolution(sim_var, override=None):
    """Resolution label from the (y, x) grid shape, unless overridden."""
    if override:
        return override
    shape = (sim_var.sizes["y"], sim_var.sizes["x"])
    if shape in GRID_SHAPE_TO_RESOLUTION:
        return GRID_SHAPE_TO_RESOLUTION[shape]
    raise ValueError(
        f"Cannot infer resolution from grid shape {shape}; pass --resolution or --chunk_size."
    )


def resolve_memory_limit(args, n_workers, result_gb):
    """Per-worker dask memory limit in bytes (see config.DASK_ADJUST for the rationale)."""
    if args.memory_limit:
        return args.memory_limit, "cli"
    mem_env = os.environ.get("SLURM_MEM_PER_NODE")
    if mem_env:
        node_gb = float(mem_env) / 1024.0
        source = "SLURM_MEM_PER_NODE"
    else:
        node_gb = psutil.virtual_memory().total / 2**30
        source = "psutil.virtual_memory"
    reserve = max(
        DASK_ADJUST["client_reserve_min_gb"],
        DASK_ADJUST["client_reserve_factor"] * result_gb,
    )
    per_worker = (node_gb - reserve) * DASK_ADJUST["memory_fraction"] / n_workers
    per_worker = min(
        DASK_ADJUST["worker_memory_max_gb"],
        max(DASK_ADJUST["worker_memory_min_gb"], per_worker),
    )
    logging.info(
        f"Memory sizing: node={node_gb:.1f}GB ({source}), result={result_gb:.1f}GB, "
        f"client reserve={reserve:.1f}GB -> {per_worker:.1f}GB per worker"
    )
    return int(per_worker * 2**30), source


def worker_peak_memory_gb(client):
    """Per-worker peak RSS (GB) from the workers' SystemMonitor history (last ~1 h)."""
    try:
        peaks = client.run(
            lambda dask_worker: max(dask_worker.monitor.quantities["memory"])
        )
        return {addr: round(v / 2**30, 2) for addr, v in peaks.items()}
    except Exception as e:  # diagnostics only, never fatal
        logging.warning(f"Could not collect per-worker peak memory: {e}")
        return {}


def count_chunk_files(zarr_path, var_id):
    """Number of chunk files present on disk for var_id (metadata-only, cheap)."""
    try:
        d = Path(zarr_path) / var_id
        return sum(1 for p in d.iterdir() if not p.name.startswith("."))
    except Exception:
        return -1


def squeeze_in_memory(arr, var_id):
    """Clip an in-memory numpy array per config (NaN-preserving); return (arr, attrs).

    Reproduces the previous lazy `.where((v >= lo) | isnull, lo)` / `.where((v <= hi) |
    isnull, hi)` logic exactly: np.clip / np.minimum / np.maximum propagate NaN.
    """
    attrs = {}
    lo = hi = None
    if var_id == "dtr":
        logging.info("##### START SQUEEZING DTR #####")
        arr_min = float(np.nanmin(arr))
        arr_max = float(np.nanmax(arr))
        logging.info(f"DTR range: [{arr_min:.2f}, {arr_max:.2f}] K")
        logging.info(f"Computing histogram with {DTR_HIST_BINS} bins...")
        hist, bin_edges = np.histogram(
            arr, bins=DTR_HIST_BINS, range=(arr_min, arr_max)
        )
        cumsum = np.cumsum(hist)
        cdf = cumsum / cumsum[-1]
        lower_quantile, upper_quantile = DTR_SQUEEZE_QUANTILES
        lower_idx = np.searchsorted(cdf, lower_quantile)
        upper_idx = np.searchsorted(cdf, upper_quantile)
        lo = float(bin_edges[lower_idx])
        hi = float(bin_edges[upper_idx + 1])  # Use upper edge of bin
        logging.info(f"Computed quantile-based thresholds: [{lo:.2f}, {hi:.2f}] K")
        attrs.update(
            {
                "squeeze_raw_min": arr_min,
                "squeeze_raw_max": arr_max,
                "squeeze_quantiles": list(DTR_SQUEEZE_QUANTILES),
            }
        )
    elif var_id in SQUEEZE_LIMITS:
        lo, hi = SQUEEZE_LIMITS[var_id]
    else:
        logging.info(f"No squeeze limits configured for {var_id}")
        return arr, attrs

    n_below = int(np.count_nonzero(arr < lo)) if lo is not None else 0
    n_above = int(np.count_nonzero(arr > hi)) if hi is not None else 0
    logging.info(f"Squeezing {var_id} to [{lo}, {hi}]")
    logging.info(f"Count of values below {lo}: {n_below}")
    logging.info(f"Count of values above {hi}: {n_above}")
    if lo is not None and hi is not None:
        np.clip(arr, lo, hi, out=arr)
    elif hi is not None:
        np.minimum(arr, hi, out=arr)
    elif lo is not None:
        np.maximum(arr, lo, out=arr)
    attrs.update(
        {
            "squeeze_lower_limit": "none" if lo is None else lo,
            "squeeze_upper_limit": "none" if hi is None else hi,
            "squeeze_count_below": n_below,
            "squeeze_count_above": n_above,
        }
    )
    if var_id == "dtr":
        logging.info("##### FINISH SQUEEZING DTR #####")
    return arr, attrs


def git_commit_of(path):
    """Code version for provenance.

    A frozen code snapshot (run directory copy) carries a CODE_VERSION file next to
    the script; use it verbatim. Otherwise the short git hash of the containing repo,
    with "-dirty" appended when the work tree has uncommitted changes; 'unknown' if
    neither is available.
    """
    here = Path(path).resolve().parent
    version_file = here / "CODE_VERSION"
    if version_file.exists():
        return version_file.read_text().strip()

    def _git(*cmd):
        out = subprocess.run(
            ["git", "-C", str(here), *cmd],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return out.stdout.strip()

    try:
        commit = _git("rev-parse", "--short", "HEAD")
        if not commit:
            return "unknown"
        if _git("status", "--porcelain", "--untracked-files=no"):
            commit += "-dirty"
        return commit
    except Exception:
        return "unknown"


if __name__ == "__main__":
    args = parse_args()
    train_path, sim_path, adj_path, tmp_path = (
        args.train_path,
        args.sim_path,
        args.adj_path,
        args.tmp_path,
    )

    success = False
    client = None
    cluster = None
    t_start = time.time()
    phase = {}
    summary = {
        "sim": sim_path.name,
        "host": socket.gethostname(),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "cpus_on_node": os.cpu_count(),
        "bias_adjust_version": BIAS_ADJUST_VERSION,
        "run_label": args.run_label,
    }

    try:
        logging.info("=" * 70)
        logging.info(f"BIAS_ADJUST.PY - {BIAS_ADJUST_VERSION}")
        logging.info("=" * 70)

        # --- QDM ranking-window patch (the point of patch_31d) ---
        qdm_window_patch.apply(main_only=args.rank_main_only)
        summary["rank_main_only"] = args.rank_main_only

        # --- Open sim metadata first: resolution, chunks and memory sizing depend on it ---
        logging.info(f"Loading simulation dataset from {sim_path}")
        sim_ds = xr.open_zarr(sim_path, consolidated=True)
        var_id = get_var_id(sim_ds)
        logging.info(f"Processing variable: {var_id}")
        if args.isel:
            logging.info(f"Applying test subset isel={args.isel}")
            sim_ds = sim_ds.isel(args.isel)
        resolution = infer_resolution(sim_ds[var_id], args.resolution)
        chunk_size = args.chunk_size or ADJUST_CHUNK_SIZE[resolution]
        adjustment_chunks = {"time": -1, "x": chunk_size, "y": chunk_size}
        sim_ds = sim_ds.chunk(adjustment_chunks)
        sim_var = sim_ds[var_id]
        result_gb = sim_var.size * np.dtype(OUTPUT_DTYPE).itemsize / 2**30
        summary.update(
            {
                "var_id": var_id,
                "resolution": resolution,
                "shape": list(sim_var.shape),
                "chunk_size": chunk_size,
                "result_gb": round(result_gb, 2),
            }
        )
        logging.info(
            f"Resolution {resolution}; adjustment chunks {adjustment_chunks}; "
            f"result size {result_gb:.2f} GB ({OUTPUT_DTYPE})"
        )

        # --- Dask cluster ---
        n_workers = args.n_workers or DASK_ADJUST["n_workers"][resolution]
        threads = (
            args.threads_per_worker or DASK_ADJUST["threads_per_worker"][resolution]
        )
        memory_limit, mem_source = resolve_memory_limit(args, n_workers, result_gb)
        if args.worker_dir_local:
            local_root = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / os.environ.get(
                "USER", "dask"
            )
        else:
            local_root = tmp_path
        worker_base_dir = local_root / f"bias-adjust-{sim_path.stem}-{os.getpid()}"
        logging.info(f"Creating worker base directory: {worker_base_dir}")
        try:
            worker_base_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass  # Race condition in parallel jobs - safe to ignore
        logging.info("Configuring Dask cluster...")
        client, cluster = configure_dask_for_adjustment(
            n_workers=n_workers,
            threads_per_worker=threads,
            memory_limit=memory_limit,
            worker_dir=worker_base_dir,
            spill_to_disk=args.spill_to_disk,
        )
        summary.update(
            {
                "n_workers": n_workers,
                "threads_per_worker": threads,
                "memory_limit_gb": round(
                    (
                        memory_limit
                        if isinstance(memory_limit, (int, float))
                        else dask.utils.parse_bytes(memory_limit)
                    )
                    / 2**30,
                    1,
                ),
                "memory_limit_source": mem_source,
                "worker_dir": str(worker_base_dir),
                "spill_to_disk": args.spill_to_disk,
            }
        )

        logging.info(f"Starting bias adjustment for {sim_path.name}")

        # --- Trained QM dataset ---
        logging.info(f"Loading trained QM dataset from {train_path}")
        train_ds = xr.open_zarr(train_path, consolidated=True)

        # Some training datasets have dimension order (x, y) instead of (y, x); this
        # caused non-deterministic NaN creation during bias adjustment. Transpose to match.
        for var in train_ds.data_vars:
            if train_ds[var].dims[:2] == ("x", "y"):
                logging.warning(
                    f"Training data has dimension order {train_ds[var].dims}"
                )
                logging.warning(
                    f"Transposing {var} from ('x', 'y', ...) to ('y', 'x', ...)"
                )
                train_ds[var] = train_ds[var].transpose("y", "x", ...)
                logging.info(f"After transpose: {var} dims = {train_ds[var].dims}")
        if args.isel:
            train_ds = train_ds.isel(args.isel)
        # Align the trained-store chunks with the adjustment chunks explicitly (xclim's
        # map_blocks would otherwise unify_chunks into irregular pieces).
        train_chunks = {"y": chunk_size, "x": chunk_size}
        for d in ("dayofyear", "quantiles"):
            if d in train_ds.dims:
                train_chunks[d] = -1
        train_ds = train_ds.chunk(train_chunks)
        logging.info(f"Trained af chunks: {train_ds['af'].chunks}")

        validate_sim_source(train_ds, sim_ds)
        validate_input_data(sim_ds, var_id, "Input simulation data")

        qm = sdba.QuantileDeltaMapping.from_dataset(train_ds)

        logging.info(f"Applying bias adjustment for {var_id}...")
        scen = qm.adjust(
            sim_var,
            extrapolation="constant",
            interp="nearest",
        )
        logging.info(f"After QDM adjust - dims={scen.dims}, shape={scen.shape}")

        # QDM may return dimensions in a different order - ensure (time, y, x)
        expected_dims = ("time", "y", "x")
        if scen.dims != expected_dims:
            logging.info(f"Transposing QDM output from {scen.dims} to {expected_dims}")
            scen = scen.transpose(*expected_dims)

        # xclim QDM returns NaN when sim=0 and hist_q_min > 0 (a snow-specific failure
        # mode where warming drives cells to zero that historically had non-zero snow).
        # Fill QDM-produced NaN with the unadjusted sim value: sim=0 fills as 0 (correct),
        # ocean/no-data NaN fills as NaN (preserved).
        # BUT only inside cells that have a trained QDM. Cells where the reference (ERA5/WRF)
        # is NaN but the GCM is not get an all-NaN `af`; they must stay NaN, otherwise the
        # fillna writes raw, unadjusted GCM values there (3,439 such cells at 12 km).
        # Result: output NaN == (sim NaN | untrained cell), matching the published products.
        logging.info(
            "Applying fillna fix: replacing QDM-produced NaN with unadjusted sim values "
            "(trained cells only)"
        )
        untrained = train_ds["af"].isnull().all(["dayofyear", "quantiles"])
        scen = scen.fillna(sim_var).where(~untrained).astype(OUTPUT_DTYPE)
        scen.name = var_id

        # --- ONE distributed compute: gather the whole adjusted array to the client ---
        logging.info("Computing adjusted data (single pass, gathered to client)...")
        t0 = time.time()
        sampler = MemorySampler()
        with sampler.sample("adjust", client=client, measure="process", interval=5):
            scen, untrained = dask.compute(scen, untrained)
        phase["adjust_compute_s"] = round(time.time() - t0, 1)
        n_untrained = int(untrained.sum())
        n_untrained_sim_finite = int(
            (untrained & scen.isel(time=0).isnull() & sim_var.isel(time=0).notnull())
            .sum()
            .compute()
        )
        summary["n_untrained_cells"] = n_untrained
        logging.info(
            f"Cells without a trained QDM: {n_untrained} "
            f"({n_untrained_sim_finite} of them have finite sim values -> left NaN)"
        )
        try:
            ms_df = sampler.to_pandas()
            summary["cluster_peak_gb"] = round(
                float(ms_df["adjust"].max()) / 2**30, 2
            )
        except Exception as e:
            logging.warning(f"MemorySampler summary failed: {e}")
        summary["worker_peak_gb"] = worker_peak_memory_gb(client)
        logging.info(
            f"Adjustment computed in {phase['adjust_compute_s']} s; "
            f"cluster peak {summary.get('cluster_peak_gb')} GB; "
            f"worker peaks {summary['worker_peak_gb']}"
        )

        # Workers are no longer needed: free their memory before the in-memory squeeze/write
        logging.info("Closing Dask cluster (all remaining work is in-process)...")
        client.close()
        cluster.close()
        client = cluster = None

        scen_ds = scen.to_dataset(name=var_id)
        scen_ds = drop_non_coord_vars(scen_ds)
        if scen_ds[var_id].dims != expected_dims:
            raise RuntimeError(f"Unexpected dims after compute: {scen_ds[var_id].dims}")
        logging.info(f"Bias adjustment completed for {var_id}")

        # --- Squeeze in memory (identical limits/logic to the previous lazy version) ---
        t0 = time.time()
        arr = scen_ds[var_id].values
        if arr.dtype != np.dtype(OUTPUT_DTYPE):
            raise RuntimeError(f"Expected {OUTPUT_DTYPE} data, got {arr.dtype}")
        arr, squeeze_attrs = squeeze_in_memory(
            arr, var_id
        )  # in place on the numpy buffer
        if arr is not scen_ds[var_id].data:  # defensive: only copy if a copy was made
            scen_ds[var_id].data[...] = arr
        phase["squeeze_s"] = round(time.time() - t0, 1)

        provenance = {
            "qdm_rank_main_only": str(args.rank_main_only),
            "qdm_rank_window_days": 1 if args.rank_main_only else QDM_WINDOW,
            "qdm_train_window_days": QDM_WINDOW,
            "qdm_nquantiles": QDM_NQUANTILES,
            "qdm_adjust_chunk_size": chunk_size,
            "n_cells_without_trained_qdm": n_untrained,
            "nan_policy": "NaN where sim is NaN or the cell has no trained QDM (all-NaN af); "
            "other QDM-produced NaN filled with the unadjusted sim value",
            "xclim_version": xclim.__version__,
            "cmip6_utils_commit": git_commit_of(__file__),
            "bias_adjust_version": BIAS_ADJUST_VERSION,
            "bias_adjust_run_label": args.run_label,
            "bias_adjust_job_id": str(os.environ.get("SLURM_JOB_ID", "")),
            "trained_qdm_path": str(train_path),
            "sim_path": str(sim_path),
        }
        provenance.update(squeeze_attrs)
        scen_ds = add_global_attrs(scen_ds, sim_ds, extra_attrs=provenance)

        # --- Write: .writing sibling -> rename -> read-back validation ---
        logging.info(f"Writing adjusted data to {adj_path}")
        synchronizer = ThreadSynchronizer()
        shuffle = {
            "bitshuffle": numcodecs.Blosc.BITSHUFFLE,
            "shuffle": numcodecs.Blosc.SHUFFLE,
            "noshuffle": numcodecs.Blosc.NOSHUFFLE,
        }[OUTPUT_COMPRESSOR["shuffle"]]
        encoding = {
            var_id: {
                "compressor": numcodecs.Blosc(
                    cname=OUTPUT_COMPRESSOR["cname"],
                    clevel=OUTPUT_COMPRESSOR["clevel"],
                    shuffle=shuffle,
                ),
                "chunks": OUTPUT_CHUNKS,
                "dtype": OUTPUT_DTYPE,
            }
        }

        if adj_path.exists():
            logging.info(f"Removing existing adjusted data store at {adj_path}")
            shutil.rmtree(adj_path)

        # Write to a .writing sibling first, then atomically rename to adj_path.
        # This ensures a failed or partial run never corrupts the live output.
        # The data is fully in memory (numpy) here: nothing is recomputed during the write.
        tmp_adj_path = adj_path.parent / (adj_path.name + ".writing")
        try:
            if tmp_adj_path.exists():
                shutil.rmtree(tmp_adj_path)

            t0 = time.time()
            logging.info(f"Writing to temp path {tmp_adj_path.name}...")
            scen_ds.to_zarr(
                tmp_adj_path,
                encoding=encoding,
                synchronizer=synchronizer,
                consolidated=True,
            )
            phase["write_s"] = round(time.time() - t0, 1)
            n_files_after_write = count_chunk_files(tmp_adj_path, var_id)
            logging.info(
                f"Write finished in {phase['write_s']} s; chunk files visible: {n_files_after_write}"
            )

            tmp_adj_path.rename(adj_path)
            n_files_after_rename = count_chunk_files(adj_path, var_id)
            summary["chunk_files_after_write"] = n_files_after_write
            summary["chunk_files_after_rename"] = n_files_after_rename
            logging.info(
                f"✓ Write succeeded and renamed to final path (chunk files: {n_files_after_rename})"
            )
        except Exception as e:
            logging.error(f"Write failed: {e}")
            if tmp_adj_path.exists():
                shutil.rmtree(tmp_adj_path, ignore_errors=True)
            raise RuntimeError(f"Failed to write {adj_path}: {e}")

        logging.info(f"Initial write to {adj_path} completed successfully")

        # CRITICAL: Validate we can read it back
        logging.info("=" * 60)
        logging.info("Starting read-after-write validation (up to 2 hours)...")
        logging.info("=" * 60)

        try:
            t0 = time.time()
            validate_zarr_readback(adj_path, var_id, max_retries=120, retry_delay=60)
            phase["readback_s"] = round(time.time() - t0, 1)
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
        logging.exception("Traceback:")
        logging.error(f"Bias adjustment FAILED for {sim_path.name}")
        for path_to_clean in [adj_path, adj_path.parent / (adj_path.name + ".writing")]:
            if path_to_clean.exists():
                logging.info(f"Cleaning up {path_to_clean}")
                shutil.rmtree(path_to_clean, ignore_errors=True)

    finally:
        # Cleanup Dask client (but leave worker directory - Dask may still be using it)
        if client is not None:
            logging.info("Closing Dask client...")
            try:
                client.close()
                cluster.close()
            except Exception:
                pass
        phase["total_s"] = round(time.time() - t_start, 1)
        summary["phase_s"] = phase
        summary["client_maxrss_gb"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2
        )
        summary["success"] = success
        logging.info("RUN_SUMMARY " + json.dumps(summary, default=str))

    if not success:
        logging.error("Bias adjustment did not complete successfully")
        sys.exit(1)

    logging.info("Exiting with success")
    sys.exit(0)
