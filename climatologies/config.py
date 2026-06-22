"""Loader for the pipeline's config YAML (e.g. config_12km.yaml,
config_4km.yaml) -- the single source of truth for a given pipeline run.

Every other script imports `load_config` from here rather than reading
the config YAML directly, so there is exactly one place that knows the
file's schema.
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
    output_variable_order: list  # exact order of the 10 master output_var names
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
    qc: dict
    units: dict

    @property
    def output_root(self) -> Path:
        return Path(self.paths["output_root"])

    @property
    def qc_dir(self) -> Path:
        return self.output_root / "qc"

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
        """Exact model coordinate order: named models, then ensemble, then reference."""
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

    output_variable_order = list(raw["output_variable_order"])
    derivable_vars = {f.output_var for f in source_families.values()} | {d.output_var for d in derived.values()}
    declared_vars = set(output_variable_order)
    if declared_vars != derivable_vars:
        missing = derivable_vars - declared_vars
        extra = declared_vars - derivable_vars
        raise ValueError(
            f"{path}'s output_variable_order doesn't match the output_vars "
            f"produced by source_families+derived. Missing from order: {sorted(missing)}. "
            f"In order but not produced by any family/derived entry: {sorted(extra)}."
        )

    return Config(
        paths=raw["paths"],
        source_families=source_families,
        derived=derived,
        output_variable_order=output_variable_order,
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
        qc=raw["qc"],
        units=raw["units"],
    )


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config_12km.yaml"
