"""Loader for config.yaml -- the single source of truth for this pipeline.

Every other script imports `load_config` from here rather than reading
config.yaml directly, so there is exactly one place that knows the file's
schema.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Era:
    name: str
    start_year: int
    end_year: int


@dataclass(frozen=True)
class Period:
    name: str
    months: list  # chronological order, e.g. [12, 1, 2] for DJF
    wrap: bool = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "wrap", self.months[0] > self.months[-1])


@dataclass(frozen=True)
class SourceFamily:
    name: str
    output_var: str
    cmip6_var: str
    era5_var: str
    units: str
    long_name: str
    adjusted_glob: str
    era5_zarr: str


@dataclass(frozen=True)
class DerivedVar:
    name: str
    output_var: str
    requires: list
    units: str
    long_name: str
    description: str


@dataclass(frozen=True)
class Config:
    paths: dict
    source_families: dict  # name -> SourceFamily
    derived: dict  # name -> DerivedVar
    models: list
    reference_model: str
    ensemble_name: str
    ensemble_members: list
    scenarios: list
    eras: list  # list[Era]
    periods: list  # list[Period]
    aggregations: list
    grid_reference_zarr_path: str
    slurm: dict
    output: dict
    metadata: dict

    @property
    def output_root(self) -> Path:
        return Path(self.paths["output_root"])

    @property
    def fragments_dir(self) -> Path:
        return self.output_root / "intermediate" / "fragments"

    @property
    def job_list_path(self) -> Path:
        return self.output_root / "intermediate" / "job_list.json"

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs" / "slurm"

    @property
    def final_output_dir(self) -> Path:
        return self.output_root / "output"

    @property
    def all_model_dim_values(self) -> list:
        """Exact Model coordinate order: named models, then ensemble, then reference."""
        return list(self.models) + [self.ensemble_name, self.reference_model]


def load_config(path: str | os.PathLike) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    source_families = {
        name: SourceFamily(name=name, **spec)
        for name, spec in raw["source_families"].items()
    }
    derived = {
        name: DerivedVar(name=name, **spec) for name, spec in raw["derived"].items()
    }
    eras = [Era(**e) for e in raw["eras"]]
    periods = [Period(name=p["name"], months=list(p["months"])) for p in raw["periods"]]

    return Config(
        paths=raw["paths"],
        source_families=source_families,
        derived=derived,
        models=list(raw["models"]),
        reference_model=raw["reference_model"],
        ensemble_name=raw["ensemble"]["name"],
        ensemble_members=list(raw["ensemble"]["members"]),
        scenarios=list(raw["scenarios"]),
        eras=eras,
        periods=periods,
        aggregations=list(raw["aggregations"]),
        grid_reference_zarr_path=raw["grid_reference"]["zarr_path"],
        slurm=raw["slurm"],
        output=raw["output"],
        metadata=raw["metadata"],
    )


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
