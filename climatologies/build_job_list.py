"""Scan disk for what's actually available and write intermediate/job_list.json.

Each entry describes one unit of work for compute_fragment.py: a single
(output_var or "tmean"/"pr", model, scenario) combination, with the exact
source path(s) and source variable name(s) needed to read it. The SLURM
array index maps directly to a position in this list.

Run manually (not via SLURM -- this is a fast disk scan):
    python build_job_list.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG_PATH, load_config
from source_catalog import discover_model_scenario_paths


def build_direct_jobs(config) -> dict:
    """Returns {family_name: [job, ...]} for every source family, including one
    WRF-ERA5/historical job per family if its era5_zarr file exists."""
    jobs_by_family = {}
    for name, family in config.source_families.items():
        jobs = []
        for hit in discover_model_scenario_paths(family.adjusted_glob):
            jobs.append(
                {
                    "family": name,
                    "output_var": family.output_var,
                    "kind": "direct",
                    "model": hit["model"],
                    "scenario": hit["scenario"],
                    "source_paths": {"main": hit["path"]},
                    "source_vars": {"main": family.cmip6_var},
                }
            )
        if os.path.exists(family.era5_zarr):
            jobs.append(
                {
                    "family": name,
                    "output_var": family.output_var,
                    "kind": "direct",
                    "model": config.reference_model,
                    "scenario": "historical",
                    "source_paths": {"main": family.era5_zarr},
                    "source_vars": {"main": family.era5_var},
                }
            )
        jobs_by_family[name] = jobs
    return jobs_by_family


def build_tmean_jobs(config, jobs_by_family: dict) -> list:
    tmax_lookup = {(j["model"], j["scenario"]): j for j in jobs_by_family["tasmax"]}
    tmin_lookup = {(j["model"], j["scenario"]): j for j in jobs_by_family["tasmin"]}
    common = sorted(set(tmax_lookup) & set(tmin_lookup))
    jobs = []
    for model, scenario in common:
        tmax_job = tmax_lookup[(model, scenario)]
        tmin_job = tmin_lookup[(model, scenario)]
        jobs.append(
            {
                "family": "tmean",
                "output_var": config.derived["tmean"].output_var,
                "kind": "tmean",
                "model": model,
                "scenario": scenario,
                "source_paths": {
                    "tmax": tmax_job["source_paths"]["main"],
                    "tmin": tmin_job["source_paths"]["main"],
                },
                "source_vars": {
                    "tmax": tmax_job["source_vars"]["main"],
                    "tmin": tmin_job["source_vars"]["main"],
                },
            }
        )
    return jobs


def build_job_list(config) -> list:
    jobs_by_family = build_direct_jobs(config)
    all_jobs = []
    for name, jobs in jobs_by_family.items():
        if name == "pr":
            # pr jobs additionally produce Pr_tot from the same source read.
            for j in jobs:
                j = dict(j)
                j["kind"] = "pr"
                all_jobs.append(j)
        else:
            all_jobs.extend(jobs)
    all_jobs.extend(build_tmean_jobs(config, jobs_by_family))
    return all_jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config = load_config(args.config)
    jobs = build_job_list(config)

    config.job_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.job_list_path, "w") as f:
        json.dump(jobs, f, indent=2)

    by_family = {}
    for j in jobs:
        by_family.setdefault(j["family"], 0)
        by_family[j["family"]] += 1
    print(f"Wrote {len(jobs)} jobs to {config.job_list_path}")
    for fam, n in sorted(by_family.items()):
        print(f"  {fam}: {n}")


if __name__ == "__main__":
    main()
