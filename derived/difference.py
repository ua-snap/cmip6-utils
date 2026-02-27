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


def configure_dask_for_difference(n_workers=4, threads_per_worker=4, memory_limit='28GB'):
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
    dask.config.set({
        # Memory management
        'distributed.worker.memory.target': 0.75,
        'distributed.worker.memory.spill': 0.85,
        'distributed.worker.memory.pause': 0.90,
        'distributed.worker.memory.terminate': 0.95,
        
        # I/O optimization
        'distributed.comm.timeouts.tcp': '120s',
        'distributed.scheduler.bandwidth': 1e9,
        
        # Array settings
        'array.slicing.split_large_chunks': True,
        'array.chunk-size': '128 MiB',
    })
    
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
    
    # Sample check (avoid loading entire array)
    sample = arr.isel({dim: slice(0, min(10, arr.sizes[dim])) for dim in arr.dims})
    if sample.isnull().all().compute():
        raise ValueError(f"Output for {output_store} is all NaN")
    
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
            n_workers=4,
            threads_per_worker=4,
            memory_limit='28GB'
        )
        
        logging.info(f"Computing difference: {minuend_store.name} - {subtrahend_store.name}")
        
        # Optimized chunking for difference calculation
        # Simple subtraction, so larger chunks are better
        chunk_dict = {
            'time': 365,  # One year at a time
            'x': 150,
            'y': 150
        }
        
        logging.info(f"Using chunk strategy: time={chunk_dict['time']}, x={chunk_dict['x']}, y={chunk_dict['y']}")
        
        # Force beegfs cache refresh for minuend data
        logging.info(f"Forcing cache refresh for {minuend_store}...")
        subprocess.run(['ls', '-lR', str(minuend_store)], capture_output=True, check=False)
        time.sleep(5)
        
        # Open datasets with explicit chunking
        logging.info("Opening minuend dataset...")
        minu_ds = xr.open_dataset(minuend_store, engine="zarr", chunks=chunk_dict)
        
        # Force beegfs cache refresh for subtrahend data
        logging.info(f"Forcing cache refresh for {subtrahend_store}...")
        subprocess.run(['ls', '-lR', str(subtrahend_store)], capture_output=True, check=False)
        time.sleep(5)
        
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
        diff_ds.attrs = {k: v for k, v in minu_ds.attrs.items() & subtr_ds.attrs.items()}
        # give this a variable_id attribute for consistency (helps with e.g. regridding with regrid.py)
        diff_ds.attrs["variable_id"] = new_var_id

        variable_id = diff_ds.attrs["variable_id"]
        
        # Apply variable-specific thresholding
        min_tasmin = 203.15
        if variable_id == "tasmin":
            logging.info("Squeezing tasmin values below limit...")
            count_below_threshold = (diff_ds[variable_id] < min_tasmin).sum().compute().item()
            logging.info(f"Count of values below {min_tasmin} K: {count_below_threshold}")
            diff_ds[variable_id] = diff_ds[variable_id].where(
                (diff_ds[variable_id] >= min_tasmin) | diff_ds[variable_id].isnull(),
                min_tasmin
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
            logging.info(f"Successfully wrote difference data to {output_store}")
            
            # Force filesystem sync for beegfs cache coherency
            logging.info("Forcing filesystem sync...")
            subprocess.run(['sync'], check=True)
            time.sleep(10)
            logging.info("Filesystem sync complete")
        except Exception as e:
            logging.error(f"Failed to write zarr store: {e}")
            # Clean up partial write
            if output_store.exists():
                shutil.rmtree(output_store, ignore_errors=True)
            raise
        
        # Validate output
        validate_output_zarr(output_store, variable_id, min_size_mb=5)
        
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
