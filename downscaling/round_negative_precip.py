"""Modify existing Zarr stores containing precipitation data in place on disk.
This script is intended to be used to post-process downscaled CMIP6 data

Example usage:
    python round_negative_precip.py --zarr-dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/adjusted

"""

import os
import zarr
import numpy as np
import argparse
from pathlib import Path
from typing import Union, List, Tuple
import dask.array as da
from dask.distributed import Client, as_completed
from dask import delayed
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_zarr_stores(zarr_dir: str, n_workers: int = 4) -> None:
    """
    Process all zarr stores in a directory, rounding negative values. Uses Dask for parallelization.
    This assumes that negative precipitation errors are close to 0!
    Reports the percentage of pixels changed.

    Parameters
    ----------
    zarr_dir : str
        Path to directory containing zarr stores
    n_workers : int, optional
        Number of Dask workers to use for parallel processing, by default 4

    Returns
    -------
    None
    """
    zarr_dir = Path(zarr_dir)

    if not zarr_dir.exists():
        logger.error(f"Directory '{zarr_dir}' does not exist.")
        return

    # Find all zarr stores in the directory using the global VAR_ID which should be set to "pr"
    if zarr_dir.name.endswith(".zarr"):
        # If a specific zarr store is provided, then this is the only one to process
        zarr_stores = [zarr_dir]
    else:
        # otherwise, find all zarr stores in the directory
        zarr_stores = list(zarr_dir.glob(f"*{VAR_ID}*.zarr"))

    logger.info("The following zarr stores will be examined for negative values: ")
    for store in zarr_stores:
        logger.info(f"  - {store.name}")

    if not zarr_stores:
        logger.info(f"No zarr stores found in '{zarr_dir}'")
        return

    logger.info(f"Found {len(zarr_stores)} zarr store(s) in '{zarr_dir}'")
    print("-" * 60)

    # Set up Dask client for parallel processing
    with Client(
        n_workers=n_workers,
        memory_limit="96GB",  # probably sane default for t2small nodes
    ) as client:
        logger.info(f"Dask client started with {n_workers} workers")
        logger.info(f"Dashboard available at: {client.dashboard_link}")

        # Create delayed tasks for each zarr store
        tasks = []
        for store_path in zarr_stores:
            task = delayed(process_single_store)(store_path)
            tasks.append(task)

        # Execute tasks and collect results
        futures = client.compute(tasks)

        # Process results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    print_array_results(result)
            except Exception as e:
                logger.error(f"Error processing store: {str(e)}")


def process_single_store(store_path: Path) -> Union[dict, None]:
    """
    Process a single zarr store and return results.

    Parameters
    ----------
    store_path : Path
        Path to the zarr store

    Returns
    -------
    dict or None
        Dictionary containing processing results or None if error occurred
    """

    try:
        # Open the zarr store
        z = zarr.open(str(store_path), mode="r+")
        pr = z.get(VAR_ID)
        assert isinstance(z.get("pr"), zarr.Array), "Expected 'pr' to be a zarr.Array"

        return process_pr(pr, store_path.name)

    except Exception as e:
        logger.error(f"Error processing '{store_path.name}': {str(e)}")
        return None


def _round_negative_values_chunked(arr: zarr.Array) -> None:
    """
    Round negative values in a zarr array using chunked processing.
    Assumes negative values are close to zero and will thus be rounded to zero

    This function processes the array in chunks to handle large arrays efficiently
    and avoid memory issues while working within zarr's indexing limitations.

    Parameters
    ----------
    arr : zarr.Array
        The zarr array to modify

    Returns
    -------
    None
    """
    # Get the chunk shape from the zarr array
    chunks = arr.chunks

    # Iterate through all chunks
    for chunk_coords in np.ndindex(
        *[len(range(0, s, c)) for s, c in zip(arr.shape, chunks)]
    ):
        # Calculate the slice for this chunk
        slices = []
        for i, (coord, chunk_size, dim_size) in enumerate(
            zip(chunk_coords, chunks, arr.shape)
        ):
            start = coord * chunk_size
            end = min(start + chunk_size, dim_size)
            slices.append(slice(start, end))

        chunk_slice = tuple(slices)

        # Read the chunk data
        chunk_data = arr[chunk_slice]

        # Find and modify negative values
        negative_mask = chunk_data < 0
        if np.any(negative_mask):
            chunk_data[negative_mask] = chunk_data[negative_mask].round()
            # Write the modified chunk back
            arr[chunk_slice] = chunk_data


def process_pr(arr: zarr.Array, name: str) -> dict:
    """
    Process a single zarr array using Dask for chunked operations.

    Parameters
    ----------
    arr : zarr.Array
        The zarr array to process
    name : str
        Name identifier for the array

    Returns
    -------
    dict
        Dictionary containing processing results with keys:
        - 'name': str, name of the array
        - 'shape': tuple, shape of the array
        - 'dtype': numpy.dtype, data type of the array
        - 'total_pixels': int, total number of pixels
        - 'negative_count': int, number of negative pixels found
        - 'percentage_changed': float, percentage of pixels changed
    """
    # Convert zarr array to dask array for efficient chunked processing
    dask_arr = da.from_zarr(arr)

    # Count total pixels
    total_pixels = dask_arr.size

    # Create mask for negative values
    negative_mask = dask_arr < 0

    # Count negative values using Dask
    negative_count = da.sum(negative_mask).compute()

    results = {
        "name": name,
        "shape": arr.shape,
        "dtype": arr.dtype,
        "total_pixels": total_pixels,
        "negative_count": int(negative_count),
        "changed_count": 0,
        "percentage_changed": 0.0,
    }

    if negative_count > 0:
        # Set negative values to zero using chunked operations
        # We need to work with the original zarr array for writing
        _round_negative_values_chunked(arr)

        # re-do negative mask after modification
        rounded_negative_mask = dask_arr < 0
        rounded_negative_count = da.sum(rounded_negative_mask).compute()
        if rounded_negative_count > 0:
            logger.warning(
                f"After rounding, {rounded_negative_count} negative values remain in {name}"
            )
        else:
            logger.info(f"All negative values in {name} have been rounded to zero")

        changed_count = int(negative_count - rounded_negative_count)
        results["changed_count"] = changed_count
        percentage_changed = (changed_count / total_pixels) * 100
        results["percentage_changed"] = percentage_changed

    return results


# def print_results(results: dict) -> None:
#     """
#     Print processing results in a formatted way.

#     Parameters
#     ----------
#     results : dict
#         Dictionary containing processing results
#     """
#     if results.get("type") == "group":
#         print(f"Processing group: {results['name']}")
#         for array_result in results["arrays"]:
#             print_array_results(array_result, indent="  ")
#     else:
#         print_array_results(results)

#     print()


def print_array_results(results: dict, indent: str = "") -> None:
    """
    Print results for a single array.

    Parameters
    ----------
    results : dict
        Dictionary containing array processing results
    indent : str, optional
        Indentation string for formatting, by default ""
    """
    print(f"{indent}Processing array: {results['name']}")
    print(f"{indent}  Shape: {results['shape']}")
    print(f"{indent}  Data type: {results['dtype']}")

    changed_count = results["negative_count"]

    if changed_count > 0:
        percentage_changed = results["percentage_changed"]
        print(
            f"{indent}  Changed {changed_count:,} pixels ({percentage_changed:.2f}%) from negative to zero"
        )
    else:
        print(f"{indent}  No negative pixels found")


def main() -> None:
    """
    Main function to parse command line arguments and initiate processing.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Process zarr stores in a directory, setting negative values to zero and reporting changes."
    )
    parser.add_argument(
        "--zarr-dir",
        help="EITHER path to directory containing zarr stores OR path to a specific zarr store",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of Dask workers to use for parallel processing (default: 4)",
    )

    args = parser.parse_args()

    process_zarr_stores(args.zarr_dir, args.workers)


if __name__ == "__main__":
    VAR_ID = "pr"
    main()
