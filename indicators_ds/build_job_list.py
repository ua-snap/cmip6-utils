"""Scan disk for what's actually available and write intermediate/job_list.json.

Each entry describes one unit of work for compute_fragment.py: a single
(indicator, model, scenario) combination, with the exact source path(s)
and source variable name(s) needed to read it. The SLURM array index maps
directly to a position in this list.

Run manually (not via SLURM -- this is a fast disk scan):
    python build_job_list.py --config config_12km.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG_PATH, Config, IndicatorDef, SourceFamily, load_config
from source_catalog import discover_model_scenario_paths


def discover_family_hits(family: SourceFamily, config: Config) -> dict:
    """{(model, scenario): {"path": ..., "var": ...}} for everything on disk
    for this source family, including a WRF-ERA5/historical hit if its
    era5_zarr file exists."""
    hits = {}
    for hit in discover_model_scenario_paths(family.adjusted_glob):
        hits[(hit["model"], hit["scenario"])] = {"path": hit["path"], "var": family.cmip6_var}
    if os.path.exists(family.era5_zarr):
        hits[(config.reference_model, "historical")] = {"path": family.era5_zarr, "var": family.era5_var}
    return hits


def build_indicator_jobs(name: str, indicator: IndicatorDef, config: Config, hits_by_family: dict) -> list:
    """One job per (model, scenario) that has hits for *every* source
    family this indicator requires (an intersection -- e.g. ftc needs both
    tasmax and tasmin, the same pattern climatologies uses for tmean)."""
    required_hits = [hits_by_family[fam] for fam in indicator.requires]
    common_keys = set(required_hits[0])
    for hits in required_hits[1:]:
        common_keys &= set(hits)

    jobs = []
    for model, scenario in sorted(common_keys):
        source_paths = {fam: hits_by_family[fam][(model, scenario)]["path"] for fam in indicator.requires}
        source_vars = {fam: hits_by_family[fam][(model, scenario)]["var"] for fam in indicator.requires}
        jobs.append(
            {
                "indicator": name,
                "model": model,
                "scenario": scenario,
                "source_paths": source_paths,
                "source_vars": source_vars,
            }
        )
    return jobs


def build_job_list(config: Config) -> list:
    hits_by_family = {
        name: discover_family_hits(family, config) for name, family in config.source_families.items()
    }
    all_jobs = []
    for name, indicator in config.indicators.items():
        all_jobs.extend(build_indicator_jobs(name, indicator, config, hits_by_family))
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

    by_indicator = {}
    for j in jobs:
        by_indicator.setdefault(j["indicator"], 0)
        by_indicator[j["indicator"]] += 1
    print(f"Wrote {len(jobs)} jobs to {config.job_list_path}")
    for name, n in sorted(by_indicator.items()):
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
