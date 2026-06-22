"""Assemble all per-(output_var, model, scenario) fragments into the single
master Dataset (model x scenario x era x period x aggregation x y x x for
each of the 10 data variables), compute the CMIP6-Ensemble multi-model
mean, attach all attrs, and write zarr + netCDF via write_outputs.py.

Fragments are discovered by globbing <output_root>/intermediate/fragments/
(not by re-reading job_list.json), so a partial/rerun set of fragments is
handled gracefully -- whatever's missing just stays NaN.

Usage:
    python combine.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG_PATH, Config, load_config
import metadata
from write_outputs import write_outputs

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# Mirrors the spec's "Data vars" list order exactly.
MASTER_VARS = ["tmin", "tmax", "tmean", "dtr", "pr", "pr_tot", "hurs", "hursmin", "sfcwind", "snw"]


def discover_fragments(config: Config) -> dict:
    """{output_var: [{"model":..., "scenario":..., "path":...}, ...]}"""
    by_var = {}
    for path in sorted(config.fragments_dir.glob("*.zarr")):
        parts = path.name[: -len(".zarr")].split("__")
        if len(parts) != 3:
            continue
        output_var, model, scenario = parts
        by_var.setdefault(output_var, []).append({"model": model, "scenario": scenario, "path": path})
    return by_var


def load_grid_reference(config: Config) -> dict:
    ds = xr.open_zarr(config.grid_reference_zarr_path, consolidated=False)
    extras = {}
    for name in ("lat", "lon", "spatial_ref"):
        if name in ds.variables:
            extras[name] = ds[name].compute()
    return ds["y"].values, ds["x"].values, extras


def combine_variable(
    output_var: str,
    fragments: list,
    config: Config,
    model_dim_values: list,
    y: np.ndarray,
    x: np.ndarray,
) -> xr.DataArray:
    era_names = [e.name for e in config.eras]
    period_names = [p.name for p in config.periods]
    agg_names = config.aggregations

    model_index = {name: i for i, name in enumerate(model_dim_values)}
    scenario_index = {name: i for i, name in enumerate(config.scenarios)}

    shape = (
        len(model_dim_values),
        len(config.scenarios),
        len(era_names),
        len(period_names),
        len(agg_names),
        len(y),
        len(x),
    )
    arr = np.full(shape, np.nan, dtype=np.float32)

    for frag in fragments:
        if frag["model"] not in model_index:
            raise ValueError(f"Fragment model {frag['model']!r} not in configured models for {output_var}")
        if frag["scenario"] not in scenario_index:
            raise ValueError(f"Fragment scenario {frag['scenario']!r} not in configured scenarios for {output_var}")
        mi = model_index[frag["model"]]
        si = scenario_index[frag["scenario"]]
        da = xr.open_zarr(frag["path"], consolidated=True)[output_var]
        da = da.reindex(era=era_names, period=period_names, aggregation=agg_names)
        arr[mi, si] = da.values

    ensemble_idx = model_index[config.ensemble_name]
    member_idx = [model_index[m] for m in config.ensemble_members]
    with np.errstate(invalid="ignore"):
        arr[ensemble_idx] = np.nanmean(arr[member_idx], axis=0)

    out = xr.DataArray(
        arr,
        dims=("model", "scenario", "era", "period", "aggregation", "y", "x"),
        coords={
            "model": model_dim_values,
            "scenario": config.scenarios,
            "era": era_names,
            "period": period_names,
            "aggregation": agg_names,
            "y": y,
            "x": x,
        },
        name=output_var,
        attrs=metadata.variable_attrs(output_var, config),
    )
    return out


def build_master_dataset(config: Config) -> xr.Dataset:
    fragments_by_var = discover_fragments(config)
    model_dim_values = config.all_model_dim_values
    y, x, grid_extras = load_grid_reference(config)

    data_vars = {}
    for output_var in MASTER_VARS:
        fragments = fragments_by_var.get(output_var, [])
        if not fragments:
            print(f"WARNING: no fragments found for {output_var} -- it will be entirely NaN", file=sys.stderr)
        print(f"combining {output_var}: {len(fragments)} fragments")
        data_vars[output_var] = combine_variable(output_var, fragments, config, model_dim_values, y, x)

    ds = xr.Dataset(data_vars, coords=grid_extras)

    ds["era"].attrs.update(metadata.era_attrs(config))
    ds["model"].attrs.update(metadata.model_attrs(config))
    ds["period"].attrs.update(metadata.period_attrs(config))
    ds["scenario"].attrs.update(metadata.scenario_attrs(config))
    ds["aggregation"].attrs.update(metadata.aggregation_attrs(config))
    ds.attrs.update(metadata.global_attrs(config))

    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config = load_config(args.config)
    ds = build_master_dataset(config)
    write_outputs(ds, config)


if __name__ == "__main__":
    main()
