"""QC for the climatology *pipeline*, not the source data.

The source data has already been through its own QC process -- this
script does not re-check whether values are physically plausible. Every
check here compares quantities this pipeline itself computed against a
mathematical relationship that must hold if the computation is correct
(e.g. pr_tot must equal pr's mean times the period's day-count; a
multi-model ensemble mean must equal nanmean of its configured members).
A violation means *this pipeline* has a bug; it says nothing about
whether the underlying climate data is "right".

Run manually after a pipeline run finishes:
    python qc.py --config config_12km.yaml

Reads only the small combined master Zarr output (not the heavy daily
source data), except for the fragment-presence cross-check, which just
lists fragment filenames (no data read).

Writes to <paths.output_root>/qc/:
    nan_checks.log            -- pr/pr_tot NaN-mask equality, coverage-gap cross-check
    calc_checks.log           -- min<=mean<=max, pr_tot identity, tmean bounds, ensemble re-derivation
    delta_maps/<var>/<var>__<period>.png  -- one PNG per variable x every
        configured period, each a scenario x era grid of
        (CMIP6-Ensemble projection) - (WRF-ERA5 historical baseline)

This reads (but does not modify) intermediate/fragments/ for the
coverage-gap cross-check -- run cleanup_intermediate.py only after this
script has been run and its results reviewed.
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

NOLEAP_DAYS_IN_MONTH = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def period_day_count(months: list) -> int:
    return sum(NOLEAP_DAYS_IN_MONTH[m] for m in months)


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
# NaN pattern checks -- only checks that compare quantities this pipeline
# computed against each other or against its own fragment manifest. Do NOT
# add checks here that compare different models'/eras'/variables' masks
# against each other -- the source data legitimately has different masks
# per variable (e.g. snw is land-only) and per model, and that is not this
# pipeline's concern.
# --------------------------------------------------------------------------


def check_pr_pr_tot_nan_match(ds: xr.Dataset, log: Log) -> int:
    log.write("## pr vs pr_tot NaN-mask equality")
    log.write("pr_tot is derived entirely from pr inside this pipeline, so any cell")
    log.write("that's NaN in one must be NaN in the other. A mismatch is a pipeline bug.")
    log.write("")
    pr_nan = np.isnan(ds["pr"].values)
    prtot_nan = np.isnan(ds["pr_tot"].values)
    mismatch = pr_nan != prtot_nan
    n_mismatch = int(mismatch.sum())
    log.write(f"Total cells compared: {pr_nan.size}")
    log.write(f"Mismatches: {n_mismatch}")
    if n_mismatch:
        idx = np.argwhere(mismatch)[:20]
        log.write("Sample mismatching indices (model,scenario,era,period,agg,y,x):")
        for row in idx:
            log.write(f"  {tuple(row.tolist())}")
    log.write("")
    return n_mismatch


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


def check_pr_tot_identity(ds: xr.Dataset, config: Config, log: Log) -> dict:
    log.write("## pr_tot identity: pr_tot.temporal_mean == pr.temporal_mean * days_in_period")
    log.write("Exact for any model on a fixed-length (noleap) calendar -- pr_tot's mean is")
    log.write("literally the same sum as pr's mean, just normalized by years vs. by days.")
    log.write("WRF-ERA5 excluded due to real gregorian calendar (variable Feb length).")
    log.write("")
    log.write("Wrapping periods (DJF, ONDJFM) get a looser, era-length-scaled tolerance:")
    log.write("pr_tot's per-year completeness exclusion (see compute_fragment.py's")
    log.write("pr_tot_aggregate) legitimately drops a boundary year that pr's direct")
    log.write("pooling still includes, whenever an era's start year lands exactly on the")
    log.write("first day of a model's available record (e.g. era 1965-2014 needs Dec 1964,")
    log.write("which doesn't exist). That's at most ~1 year out of the era's length, not a bug.")
    log.write("")
    excluded = set(config.qc["calc_checks"]["fixed_calendar_models_excluded"])
    rtol = config.qc["calc_checks"]["rtol"]
    period_lookup = {p.name: p for p in config.periods}
    era_lengths = {e.name: e.end_year - e.start_year + 1 for e in config.eras}
    models = [m for m in ds["model"].values if m not in excluded]

    pr_mean = ds["pr"].sel(aggregation="temporal_mean", model=models)
    prtot_mean = ds["pr_tot"].sel(aggregation="temporal_mean", model=models)

    results = {}
    for period_name in ds["period"].values:
        period_name = str(period_name)
        period_obj = period_lookup[period_name]
        ndays = period_day_count(period_obj.months)
        n_checked_total = 0
        n_bad_total = 0
        for era in ds["era"].values:
            era_rtol = max(rtol, 3.0 / era_lengths[str(era)]) if period_obj.wrap else rtol
            expected = (pr_mean.sel(period=period_name, era=era) * ndays).values
            actual = prtot_mean.sel(period=period_name, era=era).values
            valid = ~np.isnan(expected) & ~np.isnan(actual)
            if not valid.any():
                continue
            close = np.isclose(actual[valid], expected[valid], rtol=era_rtol, atol=1e-3)
            n_checked_total += int(valid.sum())
            n_bad_total += int((~close).sum())
        if n_checked_total == 0:
            continue
        results[period_name] = {"n_checked": n_checked_total, "n_violations": n_bad_total}
        log.write(f"  {period_name} (days={ndays}): checked {n_checked_total}, violations {n_bad_total}")
    log.write("")
    return results


def check_tmean_bounds(ds: xr.Dataset, config: Config, log: Log) -> dict:
    log.write("## tmean vs tmax/tmin")
    log.write("tmean = (tmax+tmin)/2 daily, before aggregation, so:")
    log.write("  temporal_mean: tmean.mean == (tmax.mean + tmin.mean)/2  [exact, mean is linear]")
    log.write("  temporal_min:  tmean.min  >= (tmax.min  + tmin.min )/2 [bound only]")
    log.write("  temporal_max:  tmean.max  <= (tmax.max  + tmin.max )/2 [bound only]")
    log.write("")
    rtol = config.qc["calc_checks"]["rtol"]
    tmax, tmin, tmean = ds["tmax"], ds["tmin"], ds["tmean"]
    results = {}

    expected_mean = ((tmax.sel(aggregation="temporal_mean") + tmin.sel(aggregation="temporal_mean")) / 2).values
    actual_mean = tmean.sel(aggregation="temporal_mean").values
    valid = ~np.isnan(expected_mean) & ~np.isnan(actual_mean)
    close = np.isclose(actual_mean[valid], expected_mean[valid], rtol=rtol, atol=1e-3)
    results["mean_equality"] = {"n_checked": int(valid.sum()), "n_violations": int((~close).sum())}

    lower = ((tmax.sel(aggregation="temporal_min") + tmin.sel(aggregation="temporal_min")) / 2).values
    actual_min = tmean.sel(aggregation="temporal_min").values
    valid2 = ~np.isnan(lower) & ~np.isnan(actual_min)
    bad_min = actual_min[valid2] < (lower[valid2] - 1e-3)
    results["min_bound"] = {"n_checked": int(valid2.sum()), "n_violations": int(bad_min.sum())}

    upper = ((tmax.sel(aggregation="temporal_max") + tmin.sel(aggregation="temporal_max")) / 2).values
    actual_max = tmean.sel(aggregation="temporal_max").values
    valid3 = ~np.isnan(upper) & ~np.isnan(actual_max)
    bad_max = actual_max[valid3] > (upper[valid3] + 1e-3)
    results["max_bound"] = {"n_checked": int(valid3.sum()), "n_violations": int(bad_max.sum())}

    for name, r in results.items():
        log.write(f"  {name}: checked {r['n_checked']}, violations {r['n_violations']}")
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
    # Every configured period, not just a representative subset -- fragments/the
    # combined output already hold every period, so this is free.
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
    check_pr_pr_tot_nan_match(ds, nan_log)
    check_coverage_gaps(ds, config, nan_log)
    nan_log.save(config.qc_dir / "nan_checks.log")

    calc_log = Log("QC: calculation-correctness checks")
    calc_log.write(f"Master output: {zarr_path}")
    calc_log.write("")
    min_mean_max = check_min_mean_max(ds, config, calc_log)
    pr_tot_identity = check_pr_tot_identity(ds, config, calc_log)
    tmean_bounds = check_tmean_bounds(ds, config, calc_log)
    ensemble_check = check_ensemble_derivation(ds, config, calc_log)

    total_violations = (
        sum(min_mean_max.values())
        + sum(r["n_violations"] for r in pr_tot_identity.values())
        + sum(r["n_violations"] for r in tmean_bounds.values())
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
