import operator
import re

import pytest
import xarray as xr
import numpy as np
from distributed import Client
from dask_jobqueue import SLURMCluster
from pathlib import Path
from xclim.core.units import convert_units_to
from collections import defaultdict


def pull_dims_from_source(ds):
    """Pull dimensions from the source attribute of the dataset.
    If dataset variable id does not match the filename variable id, rename it.
    (This allows for datasets with generic variable names like "data" to be used,
    as long as their filepath starts with the variable id.)
    """

    var_id = list(ds.data_vars)[0]  # assume first var is the one we want
    src_fp = Path(ds.encoding["source"])
    fn = src_fp.name
    fp_var_id, fp_model, fp_scenario = fn.split("_")[
        :3
    ]  # assumes filename begins with the var id
    assert var_id == fp_var_id

    # some stores don't have source_id and experiment_id attributes yet
    try:
        model = ds.attrs["source_id"]
        assert model == fp_model, f"Model mismatch: {model} != {fp_model}"
    except KeyError:
        model = fp_model

    try:
        scenario = ds.attrs["experiment_id"]
        assert (
            scenario == fp_scenario
        ), f"Scenario mismatch: {scenario} != {fp_scenario}"
    except KeyError:
        scenario = fp_scenario

    # add model and scenario to dataset as dimensions using an array with one value each
    ds = ds.expand_dims({"model": [model], "scenario": [scenario]})

    return ds


def percentage_condition(da, condition):
    """Calculate the percentage of pixels over time and x/y axes that meet a given condition."""

    # Parse the condition string
    match = re.match(r"([<>!=]=?|==)\s*(.*)", condition.strip())
    if not match:
        raise ValueError("Condition must be a comparison string like '> 24'")

    op_str, value_str = match.groups()
    ops = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    if op_str not in ops:
        raise ValueError(f"Unsupported operator: {op_str}")

    value = float(value_str)
    condition_mask = ops[op_str](da, value)
    condition_count = condition_mask.sum(dim=["y", "x", "time"])
    # Total number of pixels for each model and scenario
    total_count = (
        condition_mask.sizes["y"]
        * condition_mask.sizes["x"]
        * condition_mask.sizes["time"]
    )

    # Compute percentage for each model and scenario
    # Compute total count excluding NaNs for each model and scenario
    percentage_condition = condition_count / total_count * 100

    # Create a mask indicating where all values are NaN for each model and scenario.
    # This is used to mask the final percentages, since the comparison oeprators return False for NaN values.
    valid_count = (~da.isnull()).sum(dim=da.dims)
    nan_mask = valid_count == 0
    nan_mask = da.isnull().all(dim=["y", "x", "time"])

    # mask off invalid combinations
    percentage_condition = percentage_condition.where(~nan_mask)

    return percentage_condition


class SanityChecker:
    """Test suite for adjusted climate model data stored in zarr format."""

    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.zarr_files = self._discover_zarr_files()

        self.cluster = SLURMCluster(
            cores=28,
            processes=14,
            # n_workers=14,
            memory="128GB",
            # queue="debug",
            queue="t2small",
            # walltime="01:00:00",
            walltime="12:00:00",
            log_directory="/beegfs/CMIP6/kmredilla/tmp/dask_jobqueue_logs",
            account="cmip6",
            interface="ib0",
        )
        self.client = Client(self.cluster)

        # client = Client(n_workers=4, threads_per_worker=6)
        print(f"Using Dask client: {self.client}")

    def close_cluster(self):
        """Close the Dask cluster."""
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()

    def _discover_zarr_files(self):
        """Discover all zarr files in the data folder."""
        zarr_pattern = "*_*_*.zarr"
        files = list(self.data_folder.glob(zarr_pattern))

        zarr_info = {}
        for file_path in files:
            filename = file_path.name
            # Parse filename: <variable>_<model>_<scenario>.zarr
            parts = filename.replace(".zarr", "").split("_")
            assert (
                len(parts) >= 3
            ), f"Filename '{filename}' does not match expected pattern '<variable>_<model>_<scenario>.zarr'"
            variable = parts[0]
            model = parts[1]
            scenario = parts[2]

            if variable not in zarr_info:
                zarr_info[variable] = []
            zarr_info[variable].append(
                {
                    "path": file_path,
                    "model": model,
                    "scenario": scenario,
                    "filename": filename,
                }
            )

        return zarr_info

    def validate_attrs(self, ds, model, scenario, var_id):
        """Validate dataset attributes."""
        if "source_id" not in ds.attrs or ds.attrs["source_id"] != model:
            raise ValueError(
                f"Model attribute mismatch: expected '{model}', found '{ds.attrs.get('source_id', 'missing')}'"
            )
        if "experiment_id" not in ds.attrs or ds.attrs["experiment_id"] != scenario:
            raise ValueError(
                f"Scenario attribute mismatch: expected '{scenario}', found '{ds.attrs.get('experiment_id', 'missing')}'"
            )
        if "variable_id" not in ds.attrs or ds.attrs["variable_id"] != var_id:
            raise ValueError(
                f"Variable attribute mismatch: expected '{var_id}', found '{ds.attrs.get('variable_id', 'missing')}'"
            )

    def load_dataset(self, file_info, var_id=None):
        """Load a zarr dataset with error handling."""
        file_path = file_info["path"]
        try:
            ds = xr.open_zarr(file_path)
            ds = ds.load()  # Load the dataset into memory
            print(f"Loaded dataset from {file_info['filename']}")
        except Exception as e:
            pytest.fail(f"Failed to load zarr file {file_path}: {str(e)}")

        if var_id is not None:
            self.validate_attrs(ds, file_info["model"], file_info["scenario"], var_id)

        return ds

    def open_mfdataset(self, files):
        """Open multiple zarr files as a single dataset."""
        paths = [file_info["path"] for file_info in files]

        try:
            ds = xr.open_mfdataset(
                paths, combine="by_coords", preprocess=pull_dims_from_source
            )
        except Exception as e:
            pytest.fail(
                f"Failed on open_mfdataset with files {files[:2]}...{files[-2:]}: {str(e)}"
            )

        return ds

    def print_zarr_stores_table(self, variable_files):

        models = set()
        scenarios = set()
        presence = defaultdict(dict)

        for file_info in variable_files:
            model = file_info["model"]
            scenario = file_info["scenario"]
            models.add(model)
            scenarios.add(scenario)
            presence[model][scenario] = "✓"

        models = sorted(models)
        scenarios = sorted(scenarios)

        header = ["Model"] + scenarios
        rows = []
        for model in models:
            row = [model]
            for scenario in scenarios:
                row.append(presence[model].get(scenario, ""))
            rows.append(row)

        # Print as table
        col_widths = [
            max(len(str(cell)) for cell in col) for col in zip(*([header] + rows))
        ]
        fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
        print(fmt.format(*header))
        print("-+-".join("-" * w for w in col_widths))
        for row in rows:
            print(fmt.format(*row))

    def test_pr_bounds(self):
        """Test that precipitation data is within valid bounds (0 to 1650 mm/d)."""
        if "pr" not in self.zarr_files:
            pytest.skip("No precipitation (pr) files found")

        var_id = "pr"  # Precipitation variable ID
        # for file_info in self.zarr_files[var_id]
        pr_files = self.zarr_files[var_id]

        # Print table of presence/absence of zarr stores (model x scenario)
        self.print_zarr_stores_table(pr_files)

        self.cluster.scale(n=140)
        ds = self.open_mfdataset(pr_files)

        # Expect exactly 'pr' variable
        if var_id not in ds.data_vars:
            raise ValueError(
                f"Expected variable 'pr' not found. "
                f"Available variables: {list(ds.data_vars.keys())}"
            )

        pr_da = ds[var_id]
        pr_da = convert_units_to(pr_da, "mm/d")

        # Check for values below 0
        negative_pr_perc = percentage_condition(pr_da, "< 0")

        if any(negative_pr_perc) > 0:
            pytest.fail(
                f"Precipitation below 0 mm/d found: " f"{negative_pr_perc.to_pandas()}"
            )

        extreme_pr_perc = percentage_condition(pr_da, "> 1650")

        # Check for values above 1650 mm/d
        if any(extreme_pr_perc) > 0:
            pytest.fail(
                f"Precipitation above 1650 mm/d found: "
                f"{extreme_pr_perc.to_pandas()}"
            )

        self.cluster.scale(n=0)  # Scale down the cluster after use

        print(f"✓ PR bounds check passed")

    def test_tasmax_bounds(self):
        """Test that maximum temperature data is below 40°C."""
        if "tasmax" not in self.zarr_files:
            pytest.skip("No maximum temperature (tasmax) files found")

        var_id = "tasmax"  # Maximum temperature variable ID
        for file_info in self.zarr_files[var_id]:
            ds = self.load_dataset(file_info, var_id)

            # Expect exactly 'tasmax' variable
            if var_id not in ds.data_vars:
                raise ValueError(
                    f"Expected variable 'tasmax' not found in {file_info['filename']}. "
                    f"Available variables: {list(ds.data_vars.keys())}"
                )

            tasmax_da = ds[var_id]
            tasmax_da = convert_units_to(tasmax_da, "degC")

            max_val = tasmax_da.max().values
            if max_val > 40:
                pytest.fail(
                    f"Maximum temperature above 40°C found in {file_info['filename']}: "
                    f"maximum value = {max_val:.2f} °C"
                )

            print(
                f"✓ TASMAX bounds check passed for {file_info['filename']} "
                f"(max: {max_val:.2f} °C)"
            )

    def test_tasmin_bounds(self):
        """Test that minimum temperature data is above -70°C."""
        if "tasmin" not in self.zarr_files:
            pytest.skip("No minimum temperature (tasmin) files found")

        var_id = "tasmin"  # Minimum temperature variable ID
        for file_info in self.zarr_files[var_id]:
            ds = self.load_dataset(file_info, var_id)

            # Expect exactly 'tasmin' variable
            if var_id not in ds.data_vars:
                raise ValueError(
                    f"Expected variable 'tasmin' not found in {file_info['filename']}. "
                    f"Available variables: {list(ds.data_vars.keys())}"
                )

            tasmin_da = ds[var_id]
            tasmin_da = convert_units_to(tasmin_da, "degC")

            min_val = tasmin_da.min().values
            if min_val < -70:
                pytest.fail(
                    f"Minimum temperature below -70°C found in {file_info['filename']}: "
                    f"minimum value = {min_val:.2f} °C"
                )

            print(
                f"✓ TASMIN bounds check passed for {file_info['filename']} "
                f"(min: {min_val:.2f} °C)"
            )

    def test_tasmin_tasmax_consistency(self):
        """Test that tasmin <= tasmax for corresponding model/scenario pairs."""
        if "tasmin" not in self.zarr_files or "tasmax" not in self.zarr_files:
            pytest.skip("Both tasmin and tasmax files required for consistency check")

        # Create lookup for tasmin files
        tasmin_lookup = {}
        for file_info in self.zarr_files["tasmin"]:
            key = (file_info["model"], file_info["scenario"])
            tasmin_lookup[key] = file_info

        # Check each tasmax file against corresponding tasmin
        for tasmax_info in self.zarr_files["tasmax"]:
            key = (tasmax_info["model"], tasmax_info["scenario"])

            if key not in tasmin_lookup:
                print(f"⚠ No corresponding tasmin file for {tasmax_info['filename']}")
                continue

            tasmin_info = tasmin_lookup[key]

            # Load both datasets
            ds_min = self.load_dataset(tasmin_info)
            ds_max = self.load_dataset(tasmax_info)

            # Expect exactly 'tasmin' and 'tasmax' variables
            if "tasmin" not in ds_min.data_vars:
                raise ValueError(
                    f"Expected variable 'tasmin' not found in {tasmin_info['filename']}. "
                    f"Available variables: {list(ds_min.data_vars.keys())}"
                )

            if "tasmax" not in ds_max.data_vars:
                raise ValueError(
                    f"Expected variable 'tasmax' not found in {tasmax_info['filename']}. "
                    f"Available variables: {list(ds_max.data_vars.keys())}"
                )

            tasmin_data = ds_min["tasmin"]
            tasmax_data = ds_max["tasmax"]

            # Ensure both datasets have the same dimensions and coordinates
            try:
                # Align the datasets (in case of slight coordinate differences)
                tasmin_aligned, tasmax_aligned = xr.align(
                    tasmin_data, tasmax_data, join="inner"
                )

                # Check for tasmin > tasmax
                violation_mask = tasmin_aligned > tasmax_aligned

                if violation_mask.any():
                    n_violations = violation_mask.sum().values
                    total_points = violation_mask.size

                    # Get some example violations
                    violation_indices = np.where(violation_mask.values)
                    if len(violation_indices[0]) > 0:
                        # Get first violation for reporting
                        idx = tuple(
                            violation_indices[i][0]
                            for i in range(len(violation_indices))
                        )
                        tasmin_val = float(tasmin_aligned.values[idx])
                        tasmax_val = float(tasmax_aligned.values[idx])

                        pytest.fail(
                            f"Found {n_violations} points where tasmin > tasmax in "
                            f"{tasmin_info['model']}_{tasmin_info['scenario']} "
                            f"({n_violations}/{total_points} = "
                            f"{100*n_violations/total_points:.3f}% of points). "
                            f"Example: tasmin={tasmin_val:.2f}, tasmax={tasmax_val:.2f}"
                        )

                print(
                    f"✓ Temperature consistency check passed for "
                    f"{tasmin_info['model']}_{tasmin_info['scenario']}"
                )

            except Exception as e:
                pytest.fail(
                    f"Error comparing tasmin/tasmax for "
                    f"{tasmin_info['model']}_{tasmin_info['scenario']}: {str(e)}"
                )


def test_precipitation_bounds(sanity_checker):
    """Test precipitation data bounds."""
    sanity_checker.test_pr_bounds()
    sanity_checker.close_cluster()


# def test_maximum_temperature_bounds(sanity_checker):
#     """Test maximum temperature bounds."""
#     sanity_checker.test_tasmax_bounds()


# def test_minimum_temperature_bounds(sanity_checker):
#     """Test minimum temperature bounds."""
#     sanity_checker.test_tasmin_bounds()


# def test_temperature_consistency(sanity_checker):
#     """Test temperature consistency between tasmin and tasmax."""
#     sanity_checker.test_tasmin_tasmax_consistency()


if __name__ == "__main__":
    # Example usage for running directly
    import sys

    if len(sys.argv) > 1:
        data_folder = sys.argv[1]

    print(f"Testing climate data in: {data_folder}")

    # Create Dask client
    print("Starting Dask client...")

    print(f"Testing climate data in: {data_folder}")
    tester = SanityChecker(data_folder)

    print(f"Found zarr files: {list(tester.zarr_files.keys())}")
    for var, files in tester.zarr_files.items():
        print(f"  {var}: {len(files)} files")

    # Run tests
    try:
        print("\n" + "=" * 50)
        print("Running precipitation bounds test...")
        tester.test_pr_bounds()

        # print("\n" + "=" * 50)
        # print("Running maximum temperature bounds test...")
        # tester.test_tasmax_bounds()

        # print("\n" + "=" * 50)
        # print("Running minimum temperature bounds test...")
        # tester.test_tasmin_bounds()

        # print("\n" + "=" * 50)
        # print("Running temperature consistency test...")
        # tester.test_tasmin_tasmax_consistency()

        print("\n" + "=" * 50)
        print("All tests completed successfully! ✓")

    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        sys.exit(1)
