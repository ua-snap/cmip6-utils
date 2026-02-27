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
import shutil
import sys
from pathlib import Path

# import icclim
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
import xarray as xr
from zarr.sync import ThreadSynchronizer
import numcodecs
from xclim import sdba
from luts import sim_ref_var_lu, varid_adj_kind_lu, jitter_under_lu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def configure_dask_for_training(n_workers=4, threads_per_worker=4, memory_limit='30GB'):
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
    dask.config.set({
        # Memory management - more aggressive for large climate data
        'distributed.worker.memory.target': 0.70,  # Start spilling at 70%
        'distributed.worker.memory.spill': 0.80,   # Agressively spill at 80%
        'distributed.worker.memory.pause': 0.85,   # Pause at 85%
        'distributed.worker.memory.terminate': 0.95,  # Kill worker at 95%
        
        # Optimize for I/O with zarr
        'distributed.comm.timeouts.tcp': '120s',
        'distributed.scheduler.bandwidth': 1e9,  # Assume 1 Gbps network
        
        # Array optimization
        'array.slicing.split_large_chunks': True,
        'array.chunk-size': '128 MiB',  # Target chunk size
        
        # Disable work stealing for deterministic results
        'distributed.scheduler.work-stealing': False,
    })
    
    # Create LocalCluster with explicit resource limits
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=True,  # Use processes not threads for GIL-bound work
        dashboard_address=None,  # Disable dashboard on compute nodes
    )
    
    client = Client(cluster)
    
    logging.info(f"Dask cluster configured:")
    logging.info(f"  Workers: {n_workers}")
    logging.info(f"  Threads per worker: {threads_per_worker}")
    logging.info(f"  Memory per worker: {memory_limit}")
    logging.info(f"  Total memory: {n_workers * 30} GB (out of ~128 GB available)")
    logging.info(f"  Dashboard: {client.dashboard_link}")
    
    return client




def validate_args(args):
    """Validate the supplied command line args."""

    args.sim_path = Path(args.sim_path)
    args.ref_path = Path(args.ref_path)
    args.train_path = Path(args.train_path)
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
    """Check if the dataset variable is all NaN or empty and raise an error if invalid."""
    if var_id not in ds.data_vars:
        error_msg = f"{label} variable '{var_id}' not found in dataset."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    arr = ds[var_id]
    if arr.size == 0:
        error_msg = f"{label} data for variable '{var_id}' is empty!"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Compute a sample to check for all NaN (avoid loading entire array)
    sample = arr.isel({dim: slice(0, min(10, arr.sizes[dim])) for dim in arr.dims})
    if sample.isnull().all().compute():
        error_msg = f"{label} data for variable '{var_id}' appears to be all NaN!"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
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
    """Get the variable ID from the dataset."""
    if len(ds.data_vars) > 1:
        raise ValueError("More than one variable found in dataset.")
    var_id = [v for v in ds.data_vars][0]
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
        var_id: Variable identifier
        
    Raises:
        ValueError: If training output is invalid
    """
    logging.info(f"Validating training output for {var_id}...")
    
    # Check that the dataset exists and has the variable
    if not hasattr(qm_train, 'ds'):
        raise ValueError("Trained QM object has no 'ds' attribute")
    
    if var_id not in qm_train.ds.data_vars:
        raise ValueError(f"Variable '{var_id}' not found in trained QM dataset")
    
    # Check that quantiles exist and are not all NaN
    arr = qm_train.ds[var_id]
    if arr.size == 0:
        raise ValueError(f"Trained QM data for '{var_id}' is empty")
    
    # Sample check for NaN values
    sample = arr.isel({dim: slice(0, min(10, arr.sizes[dim])) for dim in arr.dims})
    if sample.isnull().all().compute():
        raise ValueError(f"Trained QM data for '{var_id}' is all NaN")
    
    # Check for 'quantiles' dimension (expected in QDM output)
    if 'quantiles' not in qm_train.ds.dims:
        logging.warning("No 'quantiles' dimension found in QDM output - this may indicate a problem")
    
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
        out_ds_check = xr.open_zarr(train_path)
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
    (sim_path, ref_path, train_path, tmp_path) = parse_args()
    
    # Track success/failure for proper exit code
    success = False
    client = None

    try:
        # Configure Dask with explicit cluster
        logging.info("Configuring Dask cluster...")
        client = configure_dask_for_training(
            n_workers=4,
            threads_per_worker=4,
            memory_limit='28GB'  # 4 workers × 28GB = 112GB, leaving 16GB for system
        )
        
        logging.info(f"Starting QM training for {sim_path.name}")
        logging.info(f"Opening input datasets...")
        
        # Optimized chunking strategy:
        # - Time: chunk into ~2 year blocks (730 days) for QDM dayofyear grouping
        # - Spatial: larger chunks (100x100) to reduce overhead
        # This balances memory usage with computational efficiency
        time_chunk = 730  # ~2 years
        spatial_chunk = 100
        
        chunk_dict = {
            "time": time_chunk,
            "x": spatial_chunk,
            "y": spatial_chunk
        }
        
        logging.info(f"Using chunk strategy: time={time_chunk}, x={spatial_chunk}, y={spatial_chunk}")
        
        # Open with optimized chunking
        hist_ds = xr.open_zarr(sim_path, chunks=chunk_dict)
        
        # Convert calendar and rechunk - this is expensive, so log it
        logging.info("Converting reference calendar to noleap...")
        ref_ds = xr.open_zarr(ref_path).convert_calendar("noleap", align_on="date")
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

        logging.info(f"Starting QDM training for {var_id}...")
        train_kwargs = dict(
            ref=ref,
            hist=hist,
            nquantiles=100,
            group="time.dayofyear",
            window=31,
            kind=varid_adj_kind_lu[var_id],
        )
        if var_id == "pr":
            # do the adapt frequency thingy for precipitation data
            logging.info("Using adapt_freq_thresh for precipitation")
            train_kwargs.update(adapt_freq_thresh="0.254 mm d-1")

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
                'compressor': numcodecs.Blosc(
                    cname='zstd',
                    clevel=5,
                    shuffle=numcodecs.Blosc.SHUFFLE
                )
            }
        
        try:
            qm_train.ds.to_zarr(
                train_path,
                encoding=encoding,
                synchronizer=synchronizer,
                consolidated=True,  # Faster subsequent reads
                compute=True  # Force synchronous write completion
            )
            logging.info(f"Successfully wrote QDM object to {train_path}")
        except Exception as e:
            logging.error(f"Failed to write zarr store: {e}")
            # Clean up partial write
            if train_path.exists():
                shutil.rmtree(train_path, ignore_errors=True)
            raise

        # Output validation - this now raises errors instead of warnings
        validate_written_zarr(train_path, var_id, min_size_mb=1)
        
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
