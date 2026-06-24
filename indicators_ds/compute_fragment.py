"""Compute one job's indicator fragment from job_list.json.

Usage (one task, e.g. for local testing):
    python compute_fragment.py --job-index 0 --config config_12km.yaml

Usage (from within a SLURM array task):
    python compute_fragment.py --job-index $SLURM_ARRAY_TASK_ID --config config_12km.yaml

Reads the source zarr(s) for the job's indicator (tasmax and/or tasmin),
loads the full daily series into memory once, and computes
temporal_min/mean/max for every (era, period) using a two-stage method:
count qualifying days per calendar year (the indicator's criterion --
see CRITERIA below), then min/mean/max those per-year counts across the
years within each era. This generalizes climatologies'
compute_fragment.py's `pr_tot_aggregate` (sum-per-year, then
min/mean/max-across-years) from summing values to counting days.

Writes the result fragment to <output_root>/intermediate/fragments/.

A fragment is a small zarr store with dims (era, period, aggregation, y, x)
holding just one (indicator, model, scenario) combination -- combine.py
later assembles all fragments into the full master array.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG_PATH, Config, Era, Period, load_config
from periods import in_period_and_label_year
import units

# Ocean/out-of-domain grid cells are NaN throughout in every source
# variable (confirmed during climatologies' data inventory). nanmin/
# nanmean/nanmax on those cells warn on every single (era, period)
# reduction -- benign, but would otherwise flood SLURM logs across every
# job.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# Day-level qualifying criteria -- this pipeline's own authoritative
# definition of each indicator (see config_12km.yaml's indicators: block
# for the prose description of each). Inputs arrive already converted to
# degC (see run_job). Note dw and ftc's tasmin operators are not the same
# (dw is strict "<", ftc is "<=") -- this is intentional, not a typo.
CRITERIA = {
    "su": lambda tasmax: tasmax > 25.0,
    "dw": lambda tasmin: tasmin < -30.0,
    "ftc": lambda tasmax, tasmin: (tasmax > 0.0) & (tasmin <= 0.0),
}


def load_var(path: str, varname: str) -> xr.DataArray:
    ds = xr.open_zarr(path, consolidated=False)
    da = ds[varname]
    da = da.transpose("time", "y", "x")
    return da.load()


def count_aggregate(
    values_by_name: dict[str, np.ndarray],
    time: xr.DataArray,
    criterion,
    eras: list[Era],
    periods: list[Period],
) -> np.ndarray:
    """Per period: count qualifying days per calendar year (excluding
    incomplete season-years at the series' edges); per era: temporal_min/
    mean/max across those per-year counts."""
    ny, nx = next(iter(values_by_name.values())).shape[1:]
    out = np.full((len(eras), len(periods), 3, ny, nx), np.nan, dtype=np.float32)

    for pi, period in enumerate(periods):
        in_period, label_year = in_period_and_label_year(time, period.months)
        sel_values = {name: arr[in_period] for name, arr in values_by_name.items()}
        sel_label_years = label_year[in_period]
        sel_months = time.dt.month.values[in_period]
        required_months = set(period.months)

        year_counts = {}
        for yr in np.unique(sel_label_years):
            ymask = sel_label_years == yr
            months_present = set(np.unique(sel_months[ymask]).tolist())
            if months_present != required_months:
                continue  # incomplete season-year at a time-series edge; exclude

            year_slices = {name: arr[ymask] for name, arr in sel_values.items()}
            qualifies = criterion(**year_slices)
            with np.errstate(invalid="ignore"):
                year_count = np.nansum(qualifies, axis=0).astype(np.float32)

            # A day only counts as "qualifying" if every required variable
            # is present that day (NaN comparisons are already False, so
            # qualifies handles that correctly on its own) -- but nansum
            # of an all-missing-day pixel returns 0, not NaN. Restore NaN
            # at pixels where *every* day this year is missing at least
            # one required variable (e.g. the ocean mask), or every
            # out-of-domain cell would silently become a fake 0.
            isnan_stack = np.stack([np.isnan(arr) for arr in year_slices.values()], axis=0)
            day_invalid = np.any(isnan_stack, axis=0)
            no_valid_days = np.all(day_invalid, axis=0)
            year_count[no_valid_days] = np.nan
            year_counts[int(yr)] = year_count

        for ei, era in enumerate(eras):
            stack = [v for yr, v in year_counts.items() if era.start_year <= yr <= era.end_year]
            if not stack:
                continue
            stack = np.stack(stack, axis=0)
            with np.errstate(invalid="ignore"):
                out[ei, pi, 0] = np.nanmin(stack, axis=0)
                out[ei, pi, 1] = np.nanmean(stack, axis=0)
                out[ei, pi, 2] = np.nanmax(stack, axis=0)
    return out


def make_fragment_dataset(
    values: np.ndarray,
    output_var: str,
    config: Config,
    y: xr.DataArray,
    x: xr.DataArray,
) -> xr.Dataset:
    return xr.Dataset(
        {output_var: (("era", "period", "aggregation", "y", "x"), values)},
        coords={
            "era": [e.name for e in config.eras],
            "period": [p.name for p in config.periods],
            "aggregation": config.aggregations,
            "y": y.values,
            "x": x.values,
        },
    )


def write_fragment(ds: xr.Dataset, output_var: str, model: str, scenario: str, config: Config) -> Path:
    config.fragments_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.fragments_dir / f"{output_var}__{model}__{scenario}.zarr"
    ds.to_zarr(out_path, mode="w", consolidated=True)
    return out_path


def run_job(job: dict, config: Config) -> Path:
    indicator = config.indicators[job["indicator"]]

    das = {}
    for name in indicator.requires:
        da = load_var(job["source_paths"][name], job["source_vars"][name])
        das[name] = units.convert_value_if_needed(name, da, config)

    time = next(iter(das.values()))["time"]
    for name, da in das.items():
        if not da["time"].equals(time):
            raise ValueError(
                f"time index mismatch for {job['model']}/{job['scenario']} "
                f"indicator={job['indicator']!r} ({name} has {da.sizes['time']} steps)"
            )

    values_by_name = {name: da.values for name, da in das.items()}
    criterion = CRITERIA[job["indicator"]]
    values = count_aggregate(values_by_name, time, criterion, config.eras, config.periods)

    any_da = next(iter(das.values()))
    ds = make_fragment_dataset(values, indicator.output_var, config, any_da["y"], any_da["x"])
    return write_fragment(ds, indicator.output_var, job["model"], job["scenario"], config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--job-list", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config = load_config(args.config)
    job_list_path = args.job_list or config.job_list_path
    with open(job_list_path) as f:
        jobs = json.load(f)

    job = jobs[args.job_index]
    print(f"[{args.job_index}] indicator={job['indicator']} model={job['model']} scenario={job['scenario']}")
    written = run_job(job, config)
    print(f"  wrote {written}")


if __name__ == "__main__":
    main()
