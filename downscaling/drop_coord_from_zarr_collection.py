from pathlib import Path
import zarr
import argparse


def find_zarr_with_height_variable(zarr_dir):
    """
    Returns a list of Zarr store names in the given directory that contain 'height' as a variable.
    """
    zarr_stores = list(
        zarr_dir.glob("*.zarr")
    )  # Ensure we only consider directories that are Zarr stores

    print(f"Found {len(zarr_stores)} Zarr stores in {zarr_dir}")
    stores_with_height = []
    for store_path in zarr_stores:
        try:
            z = zarr.open(store_path, mode="r")
            if "height" in z:
                stores_with_height.append(store_path)
        except Exception:
            continue  # Skip stores that can't be read as Zarr
    return stores_with_height


def drop_coordinate_from_zarr(store_path, coord_name):
    """
    Drops the specified coordinate from the Zarr store and overwrites the existing store.
    """
    store = zarr.open(store_path, mode="a")

    if coord_name in store:
        assert (
            store[coord_name].ndim == 0
        ), f"Coordinate '{coord_name}' is not a scalar, which is unexpected."
        del store[coord_name]  # Drop the coordinate - this removes it on disk!

    else:
        print(f"Coordinate '{coord_name}' not found in {store_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drop a coordinate from Zarr stores in a directory."
    )
    parser.add_argument(
        "--zarr_dir", type=str, help="Path to the directory containing Zarr stores."
    )
    parser.add_argument(
        "--coord_name", type=str, help="Name of the coordinate to drop."
    )
    args = parser.parse_args()

    args.zarr_dir = Path(args.zarr_dir)

    return args


def main(zarr_dir, coord_name):
    zarr_stores_with_height = find_zarr_with_height_variable(zarr_dir)
    print(f"Zarr stores with '{coord_name}' variable: ")
    for store_path in zarr_stores_with_height:
        print(store_path)

    for store_path in zarr_stores_with_height:
        drop_coordinate_from_zarr(store_path, coord_name)
        print(f"Dropped '{coord_name}' from {store_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args.zarr_dir, args.coord_name)
