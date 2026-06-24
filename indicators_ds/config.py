"""Loader for the pipeline's config YAML (e.g. config_12km.yaml) -- the
single source of truth for a given pipeline run.

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
    months: list  # chronological order, e.g. [1..12] for Annual
    wrap: bool = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "wrap", self.months[0] > self.months[-1])


@dataclass(frozen=True)
class SourceFamily:
    name: str
    cmip6_var: str
    era5_var: str
    units: str
    adjusted_glob: str
    era5_zarr: str


@dataclass(frozen=True)
class IndicatorDef:
    name: str
    output_var: str
    requires: list  # source_families keys needed to compute this indicator
    units: str
    long_name: str
    description: str
    plausible_min: float  # user-set plausibility bounds for qc.py's range check
    plausible_max: float


@dataclass(frozen=True)
class Config:
    paths: dict
    source_families: dict  # name -> SourceFamily
    indicators: dict  # name -> IndicatorDef
    output_variable_order: list  # exact order of the master output_var names
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
    units: dict
    qc: dict

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
        name: SourceFamily(name=name, **spec) for name, spec in raw["source_families"].items()
    }
    indicators = {
        name: IndicatorDef(name=name, **spec) for name, spec in raw["indicators"].items()
    }
    eras = [Era(**e) for e in raw["eras"]]
    periods = [Period(name=p["name"], months=list(p["months"])) for p in raw["periods"]]

    output_variable_order = list(raw["output_variable_order"])
    declared_vars = set(output_variable_order)
    produced_vars = {i.output_var for i in indicators.values()}
    if declared_vars != produced_vars:
        missing = produced_vars - declared_vars
        extra = declared_vars - produced_vars
        raise ValueError(
            f"{path}'s output_variable_order doesn't match the output_vars "
            f"produced by indicators. Missing from order: {sorted(missing)}. "
            f"In order but not produced by any indicator: {sorted(extra)}."
        )

    for name, indicator in indicators.items():
        unknown = set(indicator.requires) - set(source_families)
        if unknown:
            raise ValueError(f"indicator {name!r} requires unknown source_families: {sorted(unknown)}")

    return Config(
        paths=raw["paths"],
        source_families=source_families,
        indicators=indicators,
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
        units=raw["units"],
        qc=raw["qc"],
    )


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config_12km.yaml"
