"""Drop a variable from Zarr stores in a directory.
Currently, it only supports dropping a scalar variable, i.e. a variable with zero dimensions.
This is useful for removing variables that are not needed or that may cause issues in downstream processing.


"""

from pathlib import Path
import zarr
import argparse


def find_zarr_with_variable(zarr_dir, var_name):
    """
    Returns a list of Zarr store names in the given directory that contain the supplied variable name.
    """
    zarr_stores = list(
        zarr_dir.glob("*.zarr")
    )  # Ensure we only consider directories that are Zarr stores

    print(f"Found {len(zarr_stores)} Zarr stores in {zarr_dir}")
    stores_with_variable = []
    for store_path in zarr_stores:
        try:
            z = zarr.open(store_path, mode="r")
            if var_name in z:
                stores_with_variable.append(store_path)
        except Exception:
            continue  # Skip stores that can't be read as Zarr
    return stores_with_variable


def drop_variable_from_zarr(store_path, var_name):
    """
    Drops the specified variable from the Zarr store and overwrites the existing store.
    """
    store = zarr.open(store_path, mode="a")

    if var_name in store:
        assert (
            store[var_name].ndim == 0
        ), f"variable '{var_name}' is not a scalar, which is unexpected."
        del store[var_name]  # Drop the variable - this removes it on disk!

    else:
        print(f"variable '{var_name}' not found in {store_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drop a variable from Zarr stores in a directory."
    )
    parser.add_argument(
        "--zarr_dir", type=str, help="Path to the directory containing Zarr stores."
    )
    parser.add_argument("--var_name", type=str, help="Name of the variable to drop.")
    args = parser.parse_args()

    args.zarr_dir = Path(args.zarr_dir)

    return args


def main(zarr_dir, var_name):
    zarr_stores_with_variable = find_zarr_with_variable(zarr_dir, var_name)
    print(f"Zarr stores with '{var_name}' variable: ")
    for store_path in zarr_stores_with_variable:
        print(store_path)

    for store_path in zarr_stores_with_variable:
        drop_variable_from_zarr(store_path, var_name)
        print(f"Dropped '{var_name}' from {store_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args.zarr_dir, args.var_name)
