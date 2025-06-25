"""This test script performs sanity checks on downscaled data stored in Zarr format.
It is designed to be run on the whole corpus of downscaled data
It checks for valid bounds on precipitation, maximum temperature, and minimum temperature,
as well as consistency between tasmin and tasmax variables.
I'm not sure this is the best use case for pytest, just experimenting with it, and it works.
It uses dask via dask-jobqueue to create a cluster from multiple slurm nodes.
This is based on the "health checks" of Lavoie et al 2024 https://doi.org/10.1038/s41597-023-02855-z

Example usage:
    pytest -s sanity_checks.py --data-folder /center1/CMIP6/kmredilla/cmip6_4km_downscaling/adjusted
    # -s flag prints output in real time

Notes:
    As should be noted elsewhere, the location of the data should probably be on
    /center or some other filesystem for consistent dask processing.
"""

import operator
import re
import pytest
import xarray as xr
from distributed import Client
from dask_jobqueue import SLURMCluster
from pathlib import Path
from xclim.core.units import convert_units_to
from collections import defaultdict


def pull_dims_from_source(ds: xr.Dataset) -> xr.Dataset:
    """Pull dimensions from the source attribute of the dataset.

    If dataset variable id does not match the filename variable id, rename it.
    This allows for datasets with generic variable names like "data" to be used,
    as long as their filepath starts with the variable id.

    Args:
        ds (xr.Dataset): The input xarray Dataset.

    Returns:
        xr.Dataset: The dataset with added 'model' and 'scenario' dimensions.
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


def percentage_condition(da: xr.DataArray, condition: str) -> xr.DataArray:
    """Calculate the percentage of pixels over time and x/y axes that meet a given condition.

    Args:
        da (xr.DataArray): The data array to check.
        condition (str): The condition as a string (e.g., '> 24', '<= 0').

    Returns:
        xr.DataArray: The percentage of pixels meeting the condition for each model and scenario.
    """
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
    """Test suite for downscaled CMIP6 data stored in zarr format."""

    def __init__(self, data_folder: str) -> None:
        """Initialize the SanityChecker.

        Args:
            data_folder (str): Path to the folder containing zarr files.
        """
        self.data_folder = Path(data_folder)
        self.zarr_stores = self._discover_zarr_stores()

        self.cluster = SLURMCluster(
            cores=28,
            processes=14,
            memory="128GB",
            queue="t2small",
            walltime="02:00:00",
            log_directory="/beegfs/CMIP6/kmredilla/tmp/dask_jobqueue_logs",
            account="cmip6",
            interface="ib0",
        )
        self.client = Client(self.cluster)
        print(f"Using Dask client: {self.client}")

    def close_cluster(self) -> None:
        """Close the Dask cluster."""
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()

    def _discover_zarr_stores(self) -> dict:
        """Discover all zarr files in the data folder.

        Returns:
            dict: Dictionary mapping variable names to lists of file info dicts.
        """
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

    def _any_skipna(self, da: xr.DataArray) -> bool:
        """Check if any value in the DataArray is True, ignoring NaNs.
        Because the skipna=True is not available in xarray.DataArray.any(),
        this is simply a wrapper to set NaNs to False before calling .any().

        Args:
            da (xr.DataArray): The DataArray to check.

        Returns:
            bool: True if any value is not NaN, False otherwise.
        """
        return da.astype(bool).where(da.notnull(), False).any().values

    def validate_attrs(
        self, ds: xr.Dataset, model: str, scenario: str, var_id: str
    ) -> None:
        """Validate dataset attributes.

        Args:
            ds (xr.Dataset): The dataset to validate.
            model (str): Expected model name.
            scenario (str): Expected scenario name.
            var_id (str): Expected variable id.

        Raises:
            ValueError: If any attribute does not match the expected value.
        """
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

    def load_dataset(self, file_info: dict, var_id: str = None) -> xr.Dataset:
        """Load a zarr dataset with error handling.

        Args:
            file_info (dict): Dictionary with file information.
            var_id (str, optional): Variable id to validate. Defaults to None.

        Returns:
            xr.Dataset: Loaded xarray dataset.

        Raises:
            pytest.fail: If loading fails.
        """
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

    def open_mfdataset(self, files: list) -> xr.Dataset:
        """Open multiple zarr files as a single dataset.

        Args:
            files (list): List of file info dictionaries.

        Returns:
            xr.Dataset: Combined xarray dataset.

        Raises:
            pytest.fail: If opening fails.
        """
        paths = [file_info["path"] for file_info in files]

        try:
            ds = xr.open_mfdataset(
                paths,
                combine="by_coords",
                preprocess=pull_dims_from_source,
                coords="minimal",
            )
        except Exception as e:
            pytest.fail(
                f"Failed on open_mfdataset with files {files[:2]}...{files[-2:]}: {str(e)}"
            )

        return ds

    def print_zarr_stores_table(self, variable_files: list) -> None:
        """Print a table showing presence/absence of zarr stores by model and scenario.

        Args:
            variable_files (list): List of file info dictionaries for a variable.
        """
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

    def test_pr_bounds(self) -> None:
        """Test that precipitation data is within valid bounds (0 to 1650 mm/d).

        Raises:
            pytest.skip: If no precipitation files found.
            pytest.fail: If data is out of bounds.
        """
        if "pr" not in self.zarr_stores:
            pytest.skip("No precipitation (pr) files found")

        var_id = "pr"  # Precipitation variable ID
        pr_files = self.zarr_stores[var_id]

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

        if self._any_skipna(negative_pr_perc):
            pytest.fail(
                f"Precipitation below 0 mm/d found: \n"
                f"{negative_pr_perc.to_pandas()}"
            )

        extreme_pr_perc = percentage_condition(pr_da, "> 1650")

        # Check for values above 1650 mm/d
        if self._any_skipna(extreme_pr_perc):
            pytest.fail(
                f"Precipitation above 1650 mm/d found: \n"
                f"{extreme_pr_perc.to_pandas()}"
            )

        self.cluster.scale(n=0)  # Scale down the cluster after use

        print(f"✓ PR bounds check passed")

    def test_tasmax_bounds(self) -> None:
        """Test that maximum temperature data is below 40°C.

        Raises:
            pytest.skip: If no tasmax files found.
            pytest.fail: If data is out of bounds.
        """
        if "tasmax" not in self.zarr_stores:
            pytest.skip("No maximum temperature (tasmax) files found")

        var_id = "tasmax"  # Maximum temperature variable ID
        tasmax_files = self.zarr_stores[var_id]
        self.print_zarr_stores_table(tasmax_files)

        self.cluster.scale(n=140)
        ds = self.open_mfdataset(tasmax_files)

        # Expect exactly 'tasmax' variable
        if var_id not in ds.data_vars:
            raise ValueError(
                f"Expected variable 'tasmax' not found. "
                f"Available variables: {list(ds.data_vars.keys())}"
            )

        tasmax_da = ds[var_id]
        tasmax_da = convert_units_to(tasmax_da, "degC")

        # for reference, maximum temperature of WRF downscaled 4km ERA5 is 34.4 °C
        high_tasmax_perc = percentage_condition(tasmax_da, "> 40")
        if self._any_skipna(high_tasmax_perc):
            pytest.fail(
                f"Maximum temperature above 40°C found: \n"
                f"{high_tasmax_perc.to_pandas()}"
            )

        self.cluster.scale(n=0)

        print(f"✓ TASMAX bounds check passed. ")

    def test_tasmin_bounds(self) -> None:
        """Test that minimum temperature data is above -70°C.

        Raises:
            pytest.skip: If no tasmin files found.
            pytest.fail: If data is out of bounds.
        """
        if "tasmin" not in self.zarr_stores:
            pytest.skip("No minimum temperature (tasmin) files found")

        var_id = "tasmin"  # Minimum temperature variable ID
        tasmin_files = self.zarr_stores[var_id]
        self.print_zarr_stores_table(tasmin_files)

        self.cluster.scale(n=140)
        ds = self.open_mfdataset(tasmin_files)

        # Expect exactly 'tasmin' variable
        if var_id not in ds.data_vars:
            raise ValueError(
                f"Expected variable 'tasmin' not found. "
                f"Available variables: {list(ds.data_vars.keys())}"
            )

        tasmin_da = ds[var_id]
        tasmin_da = convert_units_to(tasmin_da, "degC")

        low_tasmin_perc = percentage_condition(tasmin_da, "< -70")
        if self._any_skipna(low_tasmin_perc):
            pytest.fail(
                f"Minimum temperature below -70°C found: \n"
                f"{low_tasmin_perc.to_pandas()}"
            )

        self.cluster.scale(n=0)

        print(f"✓ TASMIN bounds check passed.")

    def test_tasmin_tasmax_consistency(self) -> None:
        """Test that tasmin <= tasmax for corresponding model/scenario pairs.

        Raises:
            pytest.skip: If required files are missing.
            pytest.fail: If tasmin > tasmax anywhere.
        """
        if "tasmin" not in self.zarr_stores or "tasmax" not in self.zarr_stores:
            pytest.skip("Both tasmin and tasmax files required for consistency check")

        tasmin_stores = self.zarr_stores["tasmin"]

        # Create lookup for tasmin files
        tasmin_lookup = {}
        for file_info in tasmin_stores:
            key = (file_info["model"], file_info["scenario"])
            tasmin_lookup[key] = file_info

        # Check each tasmax file against corresponding tasmin
        tasmax_stores = self.zarr_stores["tasmax"]
        tasmax_to_open = []
        tasmin_to_open = []
        for tasmax_store_info in tasmax_stores:
            key = (tasmax_store_info["model"], tasmax_store_info["scenario"])

            if key not in tasmin_lookup:
                print(
                    f"⚠ No corresponding tasmin file for {tasmax_store_info['filename']}"
                )
                continue

            tasmin_store_info = tasmin_lookup[key]
            tasmax_to_open.append(tasmax_store_info)
            tasmin_to_open.append(tasmin_store_info)

        # Load both datasets
        self.cluster.scale(n=140)
        ds_min = self.open_mfdataset(tasmin_to_open)
        ds_max = self.open_mfdataset(tasmax_to_open)

        # Expect exactly 'tasmin' and 'tasmax' variables
        if "tasmin" not in ds_min.data_vars:
            raise ValueError(
                f"Expected variable 'tasmin' not found. "
                f"Available variables: {list(ds_min.data_vars.keys())}"
            )

        if "tasmax" not in ds_max.data_vars:
            raise ValueError(
                f"Expected variable 'tasmax' not found. "
                f"Available variables: {list(ds_max.data_vars.keys())}"
            )

        tasmin_data = ds_min["tasmin"]
        tasmax_data = ds_max["tasmax"]

        # Align the datasets (in case of slight coordinate differences)
        tasmin_aligned, tasmax_aligned = xr.align(
            tasmin_data, tasmax_data, join="inner"
        )

        # Check for tasmin > tasmax
        violation_mask = tasmin_aligned > tasmax_aligned

        violation_perc = percentage_condition(violation_mask.astype(float), "== 1")
        if self._any_skipna(violation_perc):
            pytest.fail(
                f"Instances where tasmin > tasmax found: \n"
                f"{violation_perc.to_pandas()}"
            )

        self.cluster.scale(n=0)

        print(f"✓ Temperature consistency check passed.")


def test_precipitation_bounds(sanity_checker):
    """Test precipitation data bounds."""
    sanity_checker.test_pr_bounds()


def test_maximum_temperature_bounds(sanity_checker):
    """Test maximum temperature bounds."""
    sanity_checker.test_tasmax_bounds()


def test_minimum_temperature_bounds(sanity_checker):
    """Test minimum temperature bounds."""
    sanity_checker.test_tasmin_bounds()


def test_temperature_consistency(sanity_checker: SanityChecker) -> None:
    """Test temperature consistency between tasmin and tasmax.

    Args:
        sanity_checker (SanityChecker): The SanityChecker instance.
    """
    sanity_checker.test_tasmin_tasmax_consistency()
