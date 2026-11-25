"""Diagnostics for the QDM ranking-window patch (issue #125).

Two sub-commands:

pixel   Prove, on a tiny spatial subset, what each ranking mode does: run QDM adjust in
        legacy (main_only=True) and windowed (main_only=False) mode with xclim's extra
        output enabled, and compare the `sim_q` xclim assigns to chosen dates against an
        independent hand computation of the percentile rank (scipy rankdata, average
        ties) within (a) the same day-of-year population and (b) the +/-15-day window
        population across all years. Writes a markdown note.

    python compare_rank_modes.py pixel \
        --sim_path .../cmip6_zarr/pr_CESM2_ssp585.zarr \
        --train_path .../trained_datasets/trained_qdm_pr_CESM2.zarr \
        --isel "y=100:105,x=140:145" --dates 2050-05-06 2015-01-03 2099-12-25 \
        --out_md .../tests/diagnostics/T0_pixel_proof_pr_CESM2_ssp585.md

summary Compare two adjusted stores (legacy vs windowed ranking, same inputs): cellwise
        q50/q90/q99/q99.9 over time, time-mean, time-max, wet-day frequency (pr), with
        difference maps (NetCDF), domain statistics (CSV) and a quick-look PNG.

    python compare_rank_modes.py summary \
        --legacy .../tests/adjusted_legacy/pr_CESM2_ssp585_adjusted.zarr \
        --windowed .../tests/adjusted_windowed_c25/pr_CESM2_ssp585_adjusted.zarr \
        --out_dir .../tests/diagnostics/T4_pr_CESM2_ssp585
"""

import argparse
import logging
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr
import xclim
from scipy.stats import rankdata
from xclim import sdba

import qdm_window_patch
from config import QDM_WINDOW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

QUANTILES = [0.5, 0.9, 0.99, 0.999]
WET_DAY_THRESH = (
    0.254  # mm d-1, trace precipitation (matches luts.adapt_freq_thresh_lu)
)


def parse_isel(spec):
    out = {}
    for part in spec.split(","):
        dim, rng = part.split("=")
        a, b = rng.split(":")
        out[dim.strip()] = slice(int(a), int(b))
    return out


def climate_var(ds):
    vs = [v for v in ds.data_vars if ds[v].ndim == 3]
    assert len(vs) == 1, vs
    return vs[0]


def xclim_pct_rank(values):
    """Replicate xclim.sdba.utils.rank(pct=True) on a 1-D population (NaN ignored)."""
    vals = np.asarray(values, dtype="float64")
    ok = ~np.isnan(vals)
    pct = np.full(vals.shape, np.nan)
    r = rankdata(vals[ok], method="average") / ok.sum()  # xarray: rank / count
    mn, mx = r.min(), r.max()
    pct[ok] = mx * (r - mn) / (mx - mn)  # xclim rescales [1/N, 1] -> [0, 1]
    return pct


def run_pixel(args):
    isel = parse_isel(args.isel)
    sim_ds = xr.open_zarr(args.sim_path, consolidated=True).isel(isel)
    var_id = climate_var(sim_ds)
    sim = sim_ds[var_id].load()
    train_ds = xr.open_zarr(args.train_path, consolidated=True)
    for v in train_ds.data_vars:
        if train_ds[v].dims[:2] == ("x", "y"):
            train_ds[v] = train_ds[v].transpose("y", "x", ...)
    train_ds = train_ds.isel(isel).load()
    logging.info(
        f"{var_id}: sim subset {dict(sim.sizes)}, kind={train_ds.attrs.get('adj_params')}"
    )

    results = {}
    for main_only in (True, False):
        qdm_window_patch.apply(main_only=main_only)
        qm = sdba.QuantileDeltaMapping.from_dataset(train_ds)
        with xclim.set_options(sdba_extra_output=True):
            out = qm.adjust(sim, extrapolation="constant", interp="nearest")
        if isinstance(out, xr.DataArray):
            raise RuntimeError("Expected a Dataset with sim_q (sdba_extra_output)")
        results[main_only] = out.transpose("time", "y", "x").load()

    # pick the centre pixel of the subset (first non-NaN pixel if the centre is NaN)
    ny, nx = sim.sizes["y"], sim.sizes["x"]
    cy, cx = ny // 2, nx // 2
    if np.isnan(sim[:, cy, cx]).all():
        cand = np.argwhere(~np.isnan(sim.values).all(axis=0))
        if len(cand) == 0:
            raise RuntimeError("Subset is all NaN; choose another --isel")
        cy, cx = cand[0]
    series = sim[:, cy, cx].values.astype("float64")
    times = sim.time.values
    doy = sim.time.dt.dayofyear.values
    N = len(times)
    half = QDM_WINDOW // 2

    lines = [
        f"# T0 pixel-level proof: {var_id} {Path(args.sim_path).name}",
        "",
        f"* sim subset: `{args.isel}`, pixel (y={int(sim.y[cy])}, x={int(sim.x[cx])}) "
        f"[subset index y={cy}, x={cx}]",
        f"* xclim {xclim.__version__}; training window {QDM_WINDOW} d; "
        f"N timesteps {N}; calendar {type(times[0]).__name__}",
        "",
        "`sim_q` is the percentile rank xclim assigns to the simulated value before looking up",
        "the adjustment factor. Hand values use scipy `rankdata` (average ties) over the stated",
        "population, rescaled exactly as `xclim.sdba.utils.rank(pct=True)` does.",
        "",
        "| date | sim value | population | N pop | hand sim_q | xclim sim_q (legacy) | xclim sim_q (windowed) | match | scen legacy | scen windowed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    all_ok = True
    for date in args.dates:
        y_, m_, d_ = (int(x) for x in date.split("-"))
        idx = [
            i for i, t in enumerate(times) if (t.year, t.month, t.day) == (y_, m_, d_)
        ]
        if not idx:
            lines.append(f"| {date} | not in series | | | | | | | | |")
            continue
        t = idx[0]
        v = series[t]
        same_doy = np.where(doy == doy[t])[0]
        # legacy population: exact day-of-year across all years
        pct_legacy = xclim_pct_rank(series[same_doy])
        hand_legacy = pct_legacy[np.where(same_doy == t)[0][0]]
        # windowed population: +/-half days around every same-DOY date (clipped at the ends,
        # which is what xclim's NaN-padded rolling window amounts to)
        win_idx = np.unique(
            np.concatenate(
                [np.arange(max(0, i - half), min(N, i + half + 1)) for i in same_doy]
            )
        )
        pct_win = xclim_pct_rank(series[win_idx])
        hand_win = pct_win[np.where(win_idx == t)[0][0]]
        xq_legacy = float(results[True].sim_q[t, cy, cx])
        xq_win = float(results[False].sim_q[t, cy, cx])
        ok_l = np.isclose(hand_legacy, xq_legacy, atol=1e-9)
        ok_w = np.isclose(hand_win, xq_win, atol=1e-9)
        all_ok &= bool(ok_l and ok_w)
        lines.append(
            f"| {date} | {v:.4g} | same DOY | {len(same_doy)} | {hand_legacy:.6f} | "
            f"{xq_legacy:.6f} | — | {'OK' if ok_l else 'MISMATCH'} | "
            f"{float(results[True].scen[t, cy, cx]):.4g} | "
            f"{float(results[False].scen[t, cy, cx]):.4g} |"
        )
        lines.append(
            f"| {date} | {v:.4g} | ±{half} d window | {len(win_idx)} | {hand_win:.6f} | "
            f"— | {xq_win:.6f} | {'OK' if ok_w else 'MISMATCH'} | | |"
        )

    # whole-subset statistics of the two modes
    d = results[False].scen - results[True].scen
    lines += [
        "",
        f"**Result: {'all hand-computed ranks match xclim' if all_ok else 'MISMATCH FOUND'}**",
        "",
        "## Whole-subset difference (windowed − legacy) of adjusted values",
        "",
        f"* mean {float(d.mean()):.4g}, std {float(d.std()):.4g}, min {float(d.min()):.4g}, "
        f"max {float(d.max()):.4g}",
        f"* |diff| 99th percentile {float(np.nanpercentile(np.abs(d.values), 99)):.4g}; "
        f"fraction of values changed > 1e-6: {float((np.abs(d) > 1e-6).mean()):.4f}",
        f"* legacy scen NaN count {int(results[True].scen.isnull().sum())}, "
        f"windowed {int(results[False].scen.isnull().sum())}, sim {int(sim.isnull().sum())}",
    ]
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    logging.info("\n".join(lines))
    logging.info(f"Wrote {out}")
    if not all_ok:
        raise SystemExit("T0 MISMATCH")


def run_summary(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    leg = xr.open_zarr(args.legacy, consolidated=True)
    win = xr.open_zarr(args.windowed, consolidated=True)
    v = climate_var(leg)
    chunks = {"time": -1, "y": 50, "x": 50}
    a = leg[v].transpose("time", "y", "x").chunk(chunks)
    b = win[v].transpose("time", "y", "x").chunk(chunks)
    logging.info(
        f"{v}: comparing {Path(args.legacy).name} vs {Path(args.windowed).name}"
    )

    maps = {}
    qa = a.quantile(QUANTILES, dim="time", skipna=True)
    qb = b.quantile(QUANTILES, dim="time", skipna=True)
    maps["mean_legacy"], maps["mean_windowed"] = a.mean("time"), b.mean("time")
    maps["max_legacy"], maps["max_windowed"] = a.max("time"), b.max("time")
    if v == "pr":
        maps["wetfreq_legacy"] = (a >= WET_DAY_THRESH).sum("time") / a.count("time")
        maps["wetfreq_windowed"] = (b >= WET_DAY_THRESH).sum("time") / b.count("time")
    maps["absdiff_mean_over_time"] = abs(b - a).mean("time")
    computed = xr.Dataset(maps)
    qa, qb, computed = dask.compute(qa, qb, computed)
    for i, q in enumerate(QUANTILES):
        computed[f"q{q}_legacy"] = qa.isel(quantile=i, drop=True)
        computed[f"q{q}_windowed"] = qb.isel(quantile=i, drop=True)
        computed[f"q{q}_diff"] = computed[f"q{q}_windowed"] - computed[f"q{q}_legacy"]
    computed["mean_diff"] = computed["mean_windowed"] - computed["mean_legacy"]
    computed["max_diff"] = computed["max_windowed"] - computed["max_legacy"]
    if v == "pr":
        computed["wetfreq_diff"] = (
            computed["wetfreq_windowed"] - computed["wetfreq_legacy"]
        )
    computed.attrs = {
        "legacy": str(args.legacy),
        "windowed": str(args.windowed),
        "variable": v,
    }
    computed.to_netcdf(out_dir / "rank_mode_comparison_maps.nc")

    rows = []
    for name in [k for k in computed.data_vars if k.endswith("_diff")]:
        d = computed[name].values
        d = d[np.isfinite(d)]
        rows.append(
            {
                "stat": name,
                "mean": d.mean(),
                "median": np.median(d),
                "p05": np.percentile(d, 5),
                "p95": np.percentile(d, 95),
                "min": d.min(),
                "max": d.max(),
                "mean_abs": np.abs(d).mean(),
                "frac_positive": (d > 0).mean(),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "rank_mode_comparison_stats.csv", index=False)
    logging.info("\n" + df.to_string(index=False))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    for ax, name in zip(
        axs.flat[:3], ["q0.999_legacy", "q0.999_windowed", "q0.999_diff"]
    ):
        computed[name].plot(ax=ax, robust=True)
        ax.set_title(name)
    d = computed["q0.999_diff"].values
    axs.flat[3].hist(d[np.isfinite(d)], bins=100)
    axs.flat[3].set_title("q0.999 diff (windowed − legacy)")
    fig.suptitle(f"{v}: {Path(args.legacy).name}")
    fig.tight_layout()
    fig.savefig(out_dir / "rank_mode_comparison_q999.png", dpi=110)
    logging.info(f"Wrote outputs to {out_dir}")


def run_regress(args):
    """Compare two stores that should be numerically identical (T1 regression)."""
    a_ds = xr.open_zarr(args.a, consolidated=True)
    b_ds = xr.open_zarr(args.b, consolidated=True)
    va, vb = climate_var(a_ds), climate_var(b_ds)
    a = a_ds[va].transpose("time", "y", "x").chunk({"time": 365, "y": 100, "x": 100})
    b = b_ds[vb].transpose("time", "y", "x").chunk({"time": 365, "y": 100, "x": 100})
    logging.info(
        f"A: {Path(args.a).name} dims={a_ds[va].dims} dtype={a.dtype} | "
        f"B: {Path(args.b).name} dims={b_ds[vb].dims} dtype={b.dtype}"
    )
    if a.shape != b.shape:
        raise SystemExit(f"REGRESS FAIL: shape {a.shape} vs {b.shape}")
    if not np.array_equal(a.time.values, b.time.values):
        raise SystemExit("REGRESS FAIL: time coordinates differ")
    a64, b64 = a.astype("float64"), b.astype("float64")
    diff = abs(a64 - b64)
    nan_mismatch = (a.isnull() != b.isnull()).sum()
    n_diff = (diff > args.atol).sum()
    max_diff = diff.max()
    mean_diff = diff.mean()
    n_total = a.count()
    nan_mismatch, n_diff, max_diff, mean_diff, n_total = dask.compute(
        nan_mismatch, n_diff, max_diff, mean_diff, n_total
    )
    res = {
        "a": str(args.a),
        "b": str(args.b),
        "n_values": int(n_total),
        "nan_mask_mismatches": int(nan_mismatch),
        f"n_abs_diff_gt_{args.atol}": int(n_diff),
        "max_abs_diff": float(max_diff),
        "mean_abs_diff": float(mean_diff),
    }
    ok = int(nan_mismatch) == 0 and int(n_diff) == 0
    res["status"] = "IDENTICAL" if ok else "DIFFERENT"
    logging.info("REGRESS_RESULT " + str(res))
    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([res]).to_csv(out, mode="a", header=not out.exists(), index=False)
    if not ok and args.strict:
        raise SystemExit("REGRESS FAIL")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    pr_ = sub.add_parser("regress", help="A vs B numerical identity check")
    pr_.add_argument("--a", required=True)
    pr_.add_argument("--b", required=True)
    pr_.add_argument("--atol", type=float, default=1e-5)
    pr_.add_argument("--out_csv", default=None)
    pr_.add_argument("--strict", action="store_true")
    pp = sub.add_parser("pixel")
    pp.add_argument("--sim_path", required=True)
    pp.add_argument("--train_path", required=True)
    pp.add_argument("--isel", required=True, help='e.g. "y=100:105,x=140:145"')
    pp.add_argument("--dates", nargs="+", required=True, help="YYYY-MM-DD ...")
    pp.add_argument("--out_md", required=True)
    ps = sub.add_parser("summary")
    ps.add_argument("--legacy", required=True)
    ps.add_argument("--windowed", required=True)
    ps.add_argument("--out_dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "pixel":
        run_pixel(args)
    elif args.cmd == "regress":
        run_regress(args)
    else:
        run_summary(args)
