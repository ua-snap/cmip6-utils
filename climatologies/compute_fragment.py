"""Compute one job's climatology fragment(s) from job_list.json.

Usage (one task, e.g. for local testing):
    python compute_fragment.py --job-index 0

Usage (from within a SLURM array task):
    python compute_fragment.py --job-index $SLURM_ARRAY_TASK_ID

Reads the source zarr(s) for the job, loads the full daily series into
memory once, computes temporal_min/mean/max for every (era, period) per
PLAN.md S5.3 (direct method, used by "direct" and "tmean" jobs) or S5.4
(two-stage method, used for the Pr_tot half of "pr" jobs), and writes the
result fragment(s) to <output_root>/intermediate/fragments/.

A fragment is a small zarr store with dims (era, period, aggregation, y, x)
holding just one (output_var, model, scenario) combination -- combine.py
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

# Ocean/out-of-domain grid cells are NaN throughout in every source
# variable (confirmed during data inventory). nanmin/nanmean/nanmax on
# those cells warn on every single (era, period) reduction -- benign, but
# would otherwise flood SLURM logs across ~427 jobs.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")


def load_var(path: str, varname: str) -> xr.DataArray:
    ds = xr.open_zarr(path, consolidated=False)
    da = ds[varname]
    da = da.transpose("time", "y", "x")
    return da.load()


def direct_aggregate(da: xr.DataArray, eras: list[Era], periods: list[Period]) -> np.ndarray:
    """PLAN.md S5.3: temporal_min/mean/max over every day in period+era, directly."""
    ny, nx = da.sizes["y"], da.sizes["x"]
    out = np.full((len(eras), len(periods), 3, ny, nx), np.nan, dtype=np.float32)
    values = da.values
    time = da["time"]

    for pi, period in enumerate(periods):
        in_period, label_year = in_period_and_label_year(time, period.months)
        for ei, era in enumerate(eras):
            mask = in_period & (label_year >= era.start_year) & (label_year <= era.end_year)
            if not mask.any():
                continue
            sub = values[mask]
            with np.errstate(invalid="ignore"):
                out[ei, pi, 0] = np.nanmin(sub, axis=0)
                out[ei, pi, 1] = np.nanmean(sub, axis=0)
                out[ei, pi, 2] = np.nanmax(sub, axis=0)
    return out


def pr_tot_aggregate(da: xr.DataArray, eras: list[Era], periods: list[Period]) -> np.ndarray:
    """PLAN.md S5.4: per-year period-sum, then temporal_min/mean/max across years."""
    ny, nx = da.sizes["y"], da.sizes["x"]
    out = np.full((len(eras), len(periods), 3, ny, nx), np.nan, dtype=np.float32)
    values = da.values
    time = da["time"]

    for pi, period in enumerate(periods):
        in_period, label_year = in_period_and_label_year(time, period.months)
        sel_values = values[in_period]
        sel_label_years = label_year[in_period]
        sel_months = time.dt.month.values[in_period]
        required_months = set(period.months)

        year_sums = {}
        for yr in np.unique(sel_label_years):
            ymask = sel_label_years == yr
            months_present = set(np.unique(sel_months[ymask]).tolist())
            if months_present != required_months:
                continue  # incomplete season-year at a time-series edge; exclude
            with np.errstate(invalid="ignore"):
                year_sums[int(yr)] = np.nansum(sel_values[ymask], axis=0)

        for ei, era in enumerate(eras):
            stack = [v for yr, v in year_sums.items() if era.start_year <= yr <= era.end_year]
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


def run_job(job: dict, config: Config) -> list[Path]:
    written = []

    if job["kind"] == "direct":
        da = load_var(job["source_paths"]["main"], job["source_vars"]["main"])
        values = direct_aggregate(da, config.eras, config.periods)
        ds = make_fragment_dataset(values, job["output_var"], config, da["y"], da["x"])
        written.append(write_fragment(ds, job["output_var"], job["model"], job["scenario"], config))

    elif job["kind"] == "tmean":
        da_max = load_var(job["source_paths"]["tmax"], job["source_vars"]["tmax"])
        da_min = load_var(job["source_paths"]["tmin"], job["source_vars"]["tmin"])
        if not da_max["time"].equals(da_min["time"]):
            raise ValueError(
                f"tmax/tmin time index mismatch for {job['model']}/{job['scenario']} "
                f"({da_max.sizes['time']} vs {da_min.sizes['time']} steps)"
            )
        da_mean = (da_max + da_min) / 2.0
        values = direct_aggregate(da_mean, config.eras, config.periods)
        ds = make_fragment_dataset(values, job["output_var"], config, da_max["y"], da_max["x"])
        written.append(write_fragment(ds, job["output_var"], job["model"], job["scenario"], config))

    elif job["kind"] == "pr":
        da = load_var(job["source_paths"]["main"], job["source_vars"]["main"])

        pr_values = direct_aggregate(da, config.eras, config.periods)
        pr_ds = make_fragment_dataset(pr_values, "Pr", config, da["y"], da["x"])
        written.append(write_fragment(pr_ds, "Pr", job["model"], job["scenario"], config))

        pr_tot_values = pr_tot_aggregate(da, config.eras, config.periods)
        pr_tot_ds = make_fragment_dataset(pr_tot_values, "Pr_tot", config, da["y"], da["x"])
        written.append(write_fragment(pr_tot_ds, "Pr_tot", job["model"], job["scenario"], config))

    else:
        raise ValueError(f"Unknown job kind: {job['kind']!r}")

    return written


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
    print(f"[{args.job_index}] {job['family']}/{job['kind']} model={job['model']} scenario={job['scenario']}")
    written = run_job(job, config)
    for p in written:
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
