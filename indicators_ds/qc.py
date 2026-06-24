"""QC for the indicators_ds *pipeline*, not the source data.

The source data has already been through its own QC process (see
climatologies/qc.py, which checks the same tasmax/tasmin inputs this
pipeline reads) -- this script does not re-check whether the inputs are
physically plausible. Every check here compares quantities this pipeline
itself computed against a mathematical relationship that must hold if the
computation is correct (e.g. temporal_min/temporal_max are literal
per-year counts, so they must be exact integers; a multi-model ensemble
mean must equal nanmean of its configured members), plus one plausibility
check against the per-indicator bounds set in config.yaml's
indicators:.../plausible_min/plausible_max. A violation in the calc
checks means *this pipeline* has a bug; a violation in the plausible-range
check is worth a look but isn't necessarily a bug.

Run manually after a pipeline run finishes:
    python qc.py --config config_12km.yaml

Reads only the small combined master Zarr output (not the heavy daily
source data), except for the fragment-presence cross-check, which just
lists fragment filenames (no data read).

Writes to <paths.output_root>/qc/:
    nan_checks.log            -- fragment-presence / output-NaN coverage-gap cross-check
    calc_checks.log           -- min<=mean<=max, integer min/max, non-negativity,
                                  plausible range, ensemble re-derivation
    delta_maps/<var>/<var>__<period>.png  -- one PNG per indicator x every
        configured period, each a scenario x era grid of
        (CMIP6-Ensemble projection) - (WRF-ERA5 historical baseline)

This reads (but does not modify) intermediate/fragments/ for the
coverage-gap cross-check -- run cleanup_intermediate.py (once it exists --
deferred for this pipeline's first pass) only after this script has been
run and its results reviewed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG_PATH, Config, load_config


class Log:
    """Accumulates report lines and echoes them to stdout as it goes."""

    def __init__(self, title: str):
        self.lines = [title, "=" * len(title), ""]
        print(title)

    def write(self, line: str = ""):
        self.lines.append(line)
        print(line)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n")


# --------------------------------------------------------------------------
# NaN pattern checks
# --------------------------------------------------------------------------


def check_coverage_gaps(ds: xr.Dataset, config: Config, log: Log) -> int:
    log.write("## Coverage-gap cross-check")
    log.write("For every (variable, model, scenario): if a fragment exists on disk, the")
    log.write("output must have *some* non-NaN data; if no fragment exists, the output")
    log.write("must be entirely NaN. A violation means combine.py dropped or misplaced a")
    log.write("fragment, or a fragment job silently produced empty output.")
    log.write("")
    violations = 0
    for output_var in config.output_variable_order:
        existing = set()
        for p in config.fragments_dir.glob(f"{output_var}__*__*.zarr"):
            parts = p.name[: -len(".zarr")].split("__")
            if len(parts) == 3:
                existing.add((parts[1], parts[2]))
        da = ds[output_var]
        for model in ds["model"].values:
            if model == config.ensemble_name:
                continue  # not fragment-backed, derived in combine.py
            for scenario in ds["scenario"].values:
                all_nan = bool(np.isnan(da.sel(model=model, scenario=scenario).values).all())
                has_fragment = (model, scenario) in existing
                if has_fragment and all_nan:
                    violations += 1
                    log.write(f"  VIOLATION: {output_var}/{model}/{scenario} has a fragment but output is entirely NaN")
                if not has_fragment and not all_nan:
                    violations += 1
                    log.write(f"  VIOLATION: {output_var}/{model}/{scenario} has no fragment but output has data")
    log.write(f"Total violations: {violations}")
    log.write("")
    return violations


# --------------------------------------------------------------------------
# Calculation-correctness checks
# --------------------------------------------------------------------------


def check_min_mean_max(ds: xr.Dataset, config: Config, log: Log) -> dict:
    log.write("## min <= mean <= max")
    results = {}
    for var in config.output_variable_order:
        da = ds[var]
        vmin = da.sel(aggregation="temporal_min").values
        vmean = da.sel(aggregation="temporal_mean").values
        vmax = da.sel(aggregation="temporal_max").values
        tol = 1e-3
        bad = (vmin > vmean + tol) | (vmean > vmax + tol)
        n_bad = int(np.nansum(bad))
        results[var] = n_bad
        log.write(f"  {var}: {n_bad} violations")
    log.write("")
    return results


def check_min_max_are_integers(ds: xr.Dataset, config: Config, log: Log) -> dict:
    log.write("## temporal_min/temporal_max are exact per-year counts")
    log.write("Unlike temporal_mean, min/max are never averaged across years -- they're")
    log.write("literally one of the per-year day-counts pulled into the era's value pool")
    log.write("(see compute_fragment.py's count_aggregate), so they must be exact integers")
    log.write("(within float32 round-off). A non-integer value here means the aggregation")
    log.write("or criterion logic has a bug.")
    log.write("")
    log.write(f"{config.ensemble_name} is excluded here -- it's a nanmean *across models*")
    log.write("of each model's own (integer) min/max, which is itself generally fractional;")
    log.write("that's expected, not a bug.")
    log.write("")
    models = [m for m in ds["model"].values if m != config.ensemble_name]
    results = {}
    for var in config.output_variable_order:
        da = ds[var].sel(model=models)
        n_checked = 0
        n_bad = 0
        for agg in ("temporal_min", "temporal_max"):
            values = da.sel(aggregation=agg).values
            valid = ~np.isnan(values)
            n_checked += int(valid.sum())
            non_integer = np.abs(values[valid] - np.round(values[valid])) > 1e-3
            n_bad += int(non_integer.sum())
        results[var] = {"n_checked": n_checked, "n_violations": n_bad}
        log.write(f"  {var}: checked {n_checked}, violations {n_bad}")
    log.write("")
    return results


def check_nonnegative(ds: xr.Dataset, config: Config, log: Log) -> dict:
    log.write("## All values are non-negative")
    log.write("Every indicator here is a day-count, which can never be negative.")
    log.write("")
    results = {}
    for var in config.output_variable_order:
        values = ds[var].values
        n_bad = int(np.nansum(values < -1e-3))
        results[var] = n_bad
        log.write(f"  {var}: {n_bad} violations")
    log.write("")
    return results


def check_plausible_range(ds: xr.Dataset, config: Config, log: Log) -> dict:
    log.write("## Plausible value range (config.yaml's plausible_min/plausible_max)")
    log.write("Bounds on a single year's indicator count for this domain; since")
    log.write("temporal_min/mean/max are all pooled from a set of per-year counts, every")
    log.write("one of them must fall within the same bounds as any individual year would.")
    log.write("A violation is worth a closer look but isn't necessarily a pipeline bug --")
    log.write("it could be a genuinely surprising value in the source data.")
    log.write("")
    results = {}
    for indicator in config.indicators.values():
        da = ds[indicator.output_var]
        values = da.values
        valid = ~np.isnan(values)
        n_checked = int(valid.sum())
        out_of_range = (values[valid] < indicator.plausible_min - 1e-3) | (values[valid] > indicator.plausible_max + 1e-3)
        n_bad = int(out_of_range.sum())
        results[indicator.output_var] = {"n_checked": n_checked, "n_violations": n_bad}
        log.write(
            f"  {indicator.output_var} (expected [{indicator.plausible_min}, {indicator.plausible_max}]): "
            f"checked {n_checked}, violations {n_bad}"
        )
    log.write("")
    return results


def check_ensemble_derivation(ds: xr.Dataset, config: Config, log: Log) -> dict:
    log.write("## CMIP6-Ensemble re-derivation")
    log.write("Recomputes nanmean of the configured ensemble members directly from the")
    log.write("final array and compares to the stored ensemble value -- catches drift")
    log.write("between the config's member list and what's actually in the output.")
    log.write("")
    results = {}
    for var in config.output_variable_order:
        da = ds[var]
        members = da.sel(model=config.ensemble_members)
        with np.errstate(invalid="ignore"):
            recomputed = members.mean(dim="model", skipna=True).values
        stored = da.sel(model=config.ensemble_name).values
        valid = ~np.isnan(recomputed) & ~np.isnan(stored)
        nan_mismatch = int((np.isnan(recomputed) != np.isnan(stored)).sum())
        if valid.any():
            close = np.isclose(recomputed[valid], stored[valid], rtol=1e-5, atol=1e-3)
            n_bad = int((~close).sum())
        else:
            n_bad = 0
        results[var] = {"n_checked": int(valid.sum()), "n_violations": n_bad, "nan_pattern_mismatches": nan_mismatch}
        log.write(f"  {var}: checked {results[var]['n_checked']}, violations {n_bad}, NaN-pattern mismatches {nan_mismatch}")
    log.write("")
    return results


# --------------------------------------------------------------------------
# Delta maps
# --------------------------------------------------------------------------


def render_delta_map(ds: xr.Dataset, config: Config, output_var: str, period: str, out_path: Path):
    cfg = config.qc["delta_maps"]
    baseline = cfg["baseline"]
    agg = cfg["aggregation"]
    units = ds[output_var].attrs.get("units", "")

    baseline_da = ds[output_var].sel(
        model=baseline["model"], scenario=baseline["scenario"], era=baseline["era"], period=period, aggregation=agg
    ).values

    scenarios = cfg["scenarios"]
    future_eras = cfg["future_eras"]
    deltas = np.full((len(scenarios), len(future_eras), *baseline_da.shape), np.nan, dtype=np.float32)
    for i, scenario in enumerate(scenarios):
        for j, era in enumerate(future_eras):
            proj_da = ds[output_var].sel(
                model=cfg["projection_model"], scenario=scenario, era=era, period=period, aggregation=agg
            ).values
            deltas[i, j] = proj_da - baseline_da

    finite = deltas[np.isfinite(deltas)]
    vmax = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(len(scenarios), len(future_eras), figsize=(4 * len(future_eras), 3.2 * len(scenarios)), squeeze=False)
    im = None
    for i, scenario in enumerate(scenarios):
        for j, era in enumerate(future_eras):
            ax = axes[i, j]
            im = ax.imshow(deltas[i, j], cmap="RdBu_r", norm=norm)
            mean_d = np.nanmean(deltas[i, j])
            ax.set_title(f"{scenario} / {era}\nmean {mean_d:+.2f} {units}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"{output_var} ({units}): {cfg['projection_model']} projection minus "
        f"{baseline['model']} {baseline['scenario']} {baseline['era']} baseline\n"
        f"Period={period}, Aggregation={agg}",
        fontsize=11,
    )
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label=f"delta ({units})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def generate_delta_maps(ds: xr.Dataset, config: Config):
    # Every configured period, not just a representative subset -- just
    # "Annual" for this first pass, but the loop generalizes for free if
    # sub-annual periods are added later.
    for output_var in config.output_variable_order:
        for period in ds["period"].values:
            period = str(period)
            out_path = config.qc_dir / "delta_maps" / output_var / f"{output_var}__{period}.png"
            render_delta_map(ds, config, output_var, period, out_path)
            print(f"  wrote {out_path}")


# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    config = load_config(args.config)

    zarr_path = config.final_output_dir / config.output["zarr_name"]
    ds = xr.open_zarr(zarr_path, consolidated=True)

    nan_log = Log("QC: NaN pattern checks")
    nan_log.write(f"Master output: {zarr_path}")
    nan_log.write("")
    check_coverage_gaps(ds, config, nan_log)
    nan_log.save(config.qc_dir / "nan_checks.log")

    calc_log = Log("QC: calculation-correctness checks")
    calc_log.write(f"Master output: {zarr_path}")
    calc_log.write("")
    min_mean_max = check_min_mean_max(ds, config, calc_log)
    integer_check = check_min_max_are_integers(ds, config, calc_log)
    nonneg_check = check_nonnegative(ds, config, calc_log)
    range_check = check_plausible_range(ds, config, calc_log)
    ensemble_check = check_ensemble_derivation(ds, config, calc_log)

    total_violations = (
        sum(min_mean_max.values())
        + sum(r["n_violations"] for r in integer_check.values())
        + sum(nonneg_check.values())
        + sum(r["n_violations"] for r in range_check.values())
        + sum(r["n_violations"] for r in ensemble_check.values())
    )
    calc_log.write("## Summary")
    calc_log.write("ALL CHECKS PASSED" if total_violations == 0 else f"{total_violations} TOTAL VIOLATIONS -- see above for detail")
    calc_log.save(config.qc_dir / "calc_checks.log")

    print("generating delta maps...")
    generate_delta_maps(ds, config)

    print()
    print(f"QC complete. Output under {config.qc_dir}")


if __name__ == "__main__":
    main()
