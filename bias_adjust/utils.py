import warnings


def validate_path_arg(arg_path, arg_name):
    """Validate that a path supplied as a argparse argument exists."""
    msg = f"Path supplied for {arg_name}, {arg_path}, does not exist. Aborting."
    if not arg_path.exists():
        raise FileNotFoundError(msg.format(arg_name=arg_name, arg_path=arg_path))

    return


def check_for_input_data(expected_stores):
    """Check if any of the expected stores exist."""
    found_stores = [store for store in expected_stores if store.exists()]
    if not any(found_stores):
        raise ValueError(
            f"No zarr stores in the input directory ({expected_stores[0].parent}) match the models / scenarios / variables supplied. Aborting."
        )
    else:
        missing_stores = set(expected_stores) - set(found_stores)
        if not len(missing_stores) == 0:
            missing_stores_str = "\n".join(
                [f"- {str(store)}" for store in list(missing_stores)]
            )
            warnings.warn(
                f"Some model / scenario / variable combinations were not found in the input directory and will be skipped: \n{missing_stores_str}\n",
                UserWarning,
            )

    return
