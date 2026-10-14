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

# Daily climatologies use a noleap calendar; Feb 29 is dropped from leap-year sources.
NOLEAP_DAYS = 365


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
    grid_reference_overrides: dict  # output_var -> zarr path; empty when unused
    slurm: dict
    output: dict
    metadata: dict
    qc: dict
    units: dict
    daily: bool = False

    @property
    def output_root(self) -> Path:
        root = Path(self.paths["output_root"])
        return root / "daily" if self.daily else root

    @property
    def period_dim(self) -> str:
        """Name of the temporal-bin dimension in fragments and final outputs."""
        return "dayofyear" if self.daily else "period"

    @property
    def period_labels(self) -> list:
        """Coordinate values of `period_dim`: period names, or 1..365 (noleap) day-of-year."""
        if self.daily:
            return list(range(1, NOLEAP_DAYS + 1))
        return [p.name for p in self.periods]

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

    def output_zarr_path(self, output_var: str) -> Path:
        return self.final_output_dir / self.output["zarr_name"].format(
            var=output_var + self._output_name_suffix
        )

    def output_netcdf_path(self, output_var: str) -> Path:
        return self.final_output_dir / self.output["netcdf_name"].format(
            var=output_var + self._output_name_suffix
        )

    @property
    def _output_name_suffix(self) -> str:
        return "_daily" if self.daily else ""

    def grid_reference_path_for(self, output_var: str) -> str:
        """Zarr used for y/x/lat/lon/spatial_ref when combining this variable."""
        return self.grid_reference_overrides.get(
            output_var, self.grid_reference_zarr_path
        )

    @property
    def all_model_dim_values(self) -> list:
        """Exact model coordinate order: named models, then ensemble, then reference."""
        return list(self.models) + [self.ensemble_name, self.reference_model]


def load_config(path: str | os.PathLike, daily: bool = False) -> Config:
    """Load a config YAML. `daily=True` switches to per-day-of-year climatologies
    (written under <output_root>/daily/) instead of per-period ones."""
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
    derivable_vars = {f.output_var for f in source_families.values()} | {
        d.output_var for d in derived.values()
    }
    declared_vars = set(output_variable_order)
    if declared_vars != derivable_vars:
        missing = derivable_vars - declared_vars
        extra = declared_vars - derivable_vars
        raise ValueError(
            f"{path}'s output_variable_order doesn't match the output_vars "
            f"produced by source_families+derived. Missing from order: {sorted(missing)}. "
            f"In order but not produced by any family/derived entry: {sorted(extra)}."
        )

    for key in ("zarr_name", "netcdf_name"):
        template = raw["output"][key]
        if "{var}" not in template:
            raise ValueError(
                f"{path}'s output.{key} must contain {{var}} placeholder "
                f"(e.g. cmip6_wrf_climatologies_{{var}}.zarr), got {template!r}."
            )

    grid_reference_overrides = dict(raw["grid_reference"].get("overrides") or {})
    unknown_overrides = sorted(set(grid_reference_overrides) - declared_vars)
    if unknown_overrides:
        raise ValueError(
            f"{path}'s grid_reference.overrides keys must be output_var names "
            f"from output_variable_order. Unknown: {unknown_overrides}."
        )

    if daily and "pr_tot" in derived:
        # pr_tot's per-year period-sum is just pr itself for a one-day period.
        pr_tot_var = derived["pr_tot"].output_var
        output_variable_order = [v for v in output_variable_order if v != pr_tot_var]

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
        grid_reference_overrides=grid_reference_overrides,
        slurm=raw["slurm"],
        output=raw["output"],
        metadata=raw["metadata"],
        qc=raw["qc"],
        units=raw["units"],
        daily=daily,
    )


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config_12km.yaml"
