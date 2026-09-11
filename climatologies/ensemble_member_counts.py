"""Render and tabulate effective CMIP6 ensemble-member counts for QC.

For every configured output variable and period, this script counts the
configured CMIP6 ensemble members with a finite value in each grid cell. It
uses the same future scenario, era, and aggregation selection as ``qc.py``'s
delta maps, so the two products can be reviewed side by side.

Run after the per-variable combined Zarr outputs have been written:

    python ensemble_member_counts.py --config config_12km.yaml

By default, the summary CSV is written to
``<output_root>/qc/ensemble_member_counts.csv`` and the maps are written to
``<output_root>/qc/ensemble_member_counts/<aggregation>/<variable>/``.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from config import DEFAULT_CONFIG_PATH, Config, load_config

FIGURE_WIDTH_PER_ERA = 4
FIGURE_HEIGHT_PER_SCENARIO = 3.2
MAP_DPI = 110
ZERO_MEMBER_COLOR = "#bdbdbd"
LOGGER = logging.getLogger(__name__)
SUMMARY_FIELDNAMES = [
    "variable",
    "scenario",
    "era",
    "period",
    "aggregation",
    "n_models_min",
    "n_models_max",
    "n_cells_with_data",
    "zero_count_fraction",
]


def finite_member_count(values: xr.DataArray, members: Sequence[str]) -> xr.DataArray:
    """Count configured members with finite values at every non-model cell.

    Args:
        values: DataArray with a ``model`` dimension and one or more remaining
            dimensions.
        members: Configured CMIP6 ensemble-member model names.

    Returns:
        Integer DataArray over every dimension except ``model``.

    Raises:
        ValueError: If the model dimension or a configured member is absent.
    """
    if "model" not in values.dims:
        raise ValueError("Member-count input must include a 'model' dimension.")
    if not members:
        raise ValueError("At least one configured ensemble member is required.")

    available_members = {str(model) for model in values["model"].values}
    missing_members = sorted(set(members) - available_members)
    if missing_members:
        raise ValueError(
            "Configured ensemble members are absent from the dataset model "
            f"coordinate: {', '.join(missing_members)}"
        )

    selected = values.sel(model=list(members))
    return np.isfinite(selected).sum(dim="model").astype(np.int16)


def member_count_colormap(
    member_total: int,
) -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    """Create a discrete, colorblind-safe member-count scale.

    Count zero is gray to explicitly mean "no finite ensemble-member value";
    counts one through ``member_total`` use a sequential viridis scale.
    """
    if member_total < 1:
        raise ValueError("The configured ensemble must contain at least one member.")

    viridis = plt.get_cmap("viridis", member_total)
    colors = [ZERO_MEMBER_COLOR, *[viridis(index) for index in range(member_total)]]
    cmap = mcolors.ListedColormap(colors, name="effective_member_count")
    boundaries = np.arange(-0.5, member_total + 1.5, 1)
    return cmap, mcolors.BoundaryNorm(boundaries, cmap.N)


def count_summary(counts: np.ndarray) -> tuple[int, int, float]:
    """Return the minimum, maximum, and zero-count share for one map panel."""
    if counts.ndim != 2:
        raise ValueError(
            f"Member-count panel must be two-dimensional, got {counts.ndim} dimensions."
        )
    if counts.size == 0:
        raise ValueError("Member-count panel is empty.")

    return int(counts.min()), int(counts.max()), float(np.mean(counts == 0))


def member_count_range(
    counts: np.ndarray, *, mask: np.ndarray | None = None
) -> tuple[int, int] | None:
    """Return the minimum and maximum contributor count in selected cells.

    Args:
        counts: Integer member-count grid.
        mask: Optional Boolean grid identifying cells to summarize.

    Returns:
        The inclusive count range, or ``None`` when no cells are selected.

    Raises:
        ValueError: If the mask shape does not match the count grid.
    """
    count_values = np.asarray(counts)
    if mask is None:
        selected = count_values.ravel()
    else:
        selection = np.asarray(mask, dtype=bool)
        if selection.shape != count_values.shape:
            raise ValueError(
                "Member-count mask shape must match the count grid: "
                f"{selection.shape} != {count_values.shape}."
            )
        selected = count_values[selection]

    if selected.size == 0:
        return None
    return int(selected.min()), int(selected.max())


def member_count_label(
    counts: np.ndarray, *, mask: np.ndarray | None = None
) -> str:
    """Format a precise N label for one contributor-count grid."""
    count_range = member_count_range(counts, mask=mask)
    if count_range is None:
        return "N=none"
    count_min, count_max = count_range
    if count_min == count_max:
        return f"N={count_min}"
    return f"N={count_min}\N{EN DASH}{count_max}"


def validate_coordinate_values(
    ds: xr.Dataset, coordinate: str, requested_values: Sequence[str]
) -> None:
    """Raise a clear error if a requested config value is not in a coordinate."""
    if coordinate not in ds.coords:
        raise ValueError(f"Master output has no '{coordinate}' coordinate.")

    available_values = {str(value) for value in ds[coordinate].values}
    missing_values = sorted(set(requested_values) - available_values)
    if missing_values:
        raise ValueError(
            f"Requested {coordinate} values are absent from the master output: "
            f"{', '.join(missing_values)}"
        )


def render_member_count_map(
    data: xr.DataArray,
    output_var: str,
    period: str,
    aggregation: str,
    scenarios: Sequence[str],
    eras: Sequence[str],
    members: Sequence[str],
    output_path: Path,
) -> None:
    """Render one scenario-by-era grid of effective member-count maps.

    The color scale is fixed to zero through the configured member count across
    every output variable and period, making all resulting panels comparable.
    """
    cmap, norm = member_count_colormap(len(members))
    fig, axes = plt.subplots(
        len(scenarios),
        len(eras),
        figsize=(
            FIGURE_WIDTH_PER_ERA * len(eras),
            FIGURE_HEIGHT_PER_SCENARIO * len(scenarios),
        ),
        squeeze=False,
    )
    image = None

    for scenario_index, scenario in enumerate(scenarios):
        for era_index, era in enumerate(eras):
            ax = axes[scenario_index, era_index]
            panel = data.sel(
                scenario=scenario,
                era=era,
                period=period,
                aggregation=aggregation,
            )
            counts = finite_member_count(panel, members).values
            _, _, zero_share = count_summary(counts)
            contributor_label = member_count_label(counts, mask=counts > 0)
            image = ax.imshow(counts, cmap=cmap, norm=norm, interpolation="none")
            ax.set_title(
                f"{scenario} / {era}\n"
                f"{contributor_label}; no data {zero_share:.1%}",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"{output_var}: effective CMIP6 ensemble-member count\n"
        f"Period={period}, Aggregation={aggregation}; gray=0 finite contributors",
        fontsize=11,
    )
    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            ticks=np.arange(len(members) + 1),
            shrink=0.7,
            label="finite CMIP6 ensemble members",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)


def configured_delta_map_selection(config: Config) -> tuple[list[str], list[str], str]:
    """Return the scenario, era, and aggregation selection used for delta maps."""
    try:
        delta_cfg = config.qc["delta_maps"]
        scenarios = [str(value) for value in delta_cfg["scenarios"]]
        eras = [str(value) for value in delta_cfg["future_eras"]]
        aggregation = str(delta_cfg["aggregation"])
    except KeyError as exc:
        raise ValueError(
            "Config QC settings must define qc.delta_maps selection fields."
        ) from exc

    return scenarios, eras, aggregation


def write_member_count_summary(config: Config, output_path: Path | None = None) -> Path:
    """Write per-climatology CMIP6 contributor-count ranges to CSV.

    The range excludes cells with zero available members, which are reported
    separately through ``zero_count_fraction``. This avoids treating the ocean
    mask as a contributing-model count of zero.
    """
    summary_path = output_path or config.qc_dir / "ensemble_member_counts.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for output_var in config.output_variable_order:
            ds = xr.open_zarr(config.output_zarr_path(output_var), consolidated=True)
            try:
                if output_var not in ds.data_vars:
                    raise ValueError(
                        f"Configured output variable is absent from {output_var}'s Zarr."
                    )
                counts = finite_member_count(ds[output_var], config.ensemble_members)
                for scenario in counts["scenario"].values:
                    for era in counts["era"].values:
                        for period in counts["period"].values:
                            for aggregation in counts["aggregation"].values:
                                count_values = counts.sel(
                                    scenario=scenario,
                                    era=era,
                                    period=period,
                                    aggregation=aggregation,
                                ).values
                                _, _, zero_fraction = count_summary(count_values)
                                populated_range = member_count_range(
                                    count_values, mask=count_values > 0
                                )
                                if populated_range is None:
                                    populated_min = populated_max = 0
                                else:
                                    populated_min, populated_max = populated_range
                                writer.writerow(
                                    {
                                        "variable": output_var,
                                        "scenario": str(scenario),
                                        "era": str(era),
                                        "period": str(period),
                                        "aggregation": str(aggregation),
                                        "n_models_min": populated_min,
                                        "n_models_max": populated_max,
                                        "n_cells_with_data": int(
                                            np.count_nonzero(count_values)
                                        ),
                                        "zero_count_fraction": f"{zero_fraction:.6f}",
                                    }
                                )
            finally:
                ds.close()

    return summary_path


def generate_member_count_maps(
    config: Config,
    *,
    aggregation: str | None = None,
    periods: Sequence[str] | None = None,
    output_root: Path | None = None,
) -> list[Path]:
    """Render count-map grids aligned with the configured delta-map panels."""
    scenarios, eras, default_aggregation = configured_delta_map_selection(config)
    selected_aggregation = aggregation or default_aggregation
    maps_root = output_root or config.qc_dir / "ensemble_member_counts"
    generated_paths = []

    for output_var in config.output_variable_order:
        ds = xr.open_zarr(config.output_zarr_path(output_var), consolidated=True)
        try:
            if output_var not in ds.data_vars:
                raise ValueError(
                    f"Configured output variable is absent from {output_var}'s Zarr."
                )
            validate_coordinate_values(ds, "scenario", scenarios)
            validate_coordinate_values(ds, "era", eras)
            validate_coordinate_values(ds, "aggregation", [selected_aggregation])
            selected_periods = periods or [str(period) for period in ds["period"].values]
            validate_coordinate_values(ds, "period", selected_periods)

            for period in selected_periods:
                output_path = (
                    maps_root
                    / selected_aggregation
                    / output_var
                    / f"{output_var}__{period}.png"
                )
                render_member_count_map(
                    data=ds[output_var],
                    output_var=output_var,
                    period=period,
                    aggregation=selected_aggregation,
                    scenarios=scenarios,
                    eras=eras,
                    members=config.ensemble_members,
                    output_path=output_path,
                )
                generated_paths.append(output_path)
        finally:
            ds.close()

    return generated_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Pipeline configuration YAML path.",
    )
    parser.add_argument(
        "--aggregation",
        help="Aggregation to map; defaults to qc.delta_maps.aggregation in the selected config.",
    )
    parser.add_argument(
        "--period",
        action="append",
        dest="periods",
        help="Period to map; repeat to limit output. Defaults to every period in the master output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output root; defaults to <output_root>/qc/ensemble_member_counts.",
    )
    return parser.parse_args()


def main() -> None:
    """Render effective ensemble-member count maps from final Zarr outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    config = load_config(args.config)
    summary_path = write_member_count_summary(config)
    LOGGER.info("Wrote %s", summary_path)
    generated_paths = generate_member_count_maps(
        config,
        aggregation=args.aggregation,
        periods=args.periods,
        output_root=args.output_dir,
    )
    for output_path in generated_paths:
        LOGGER.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
