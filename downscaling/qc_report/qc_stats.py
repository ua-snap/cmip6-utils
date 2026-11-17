"""Compute every statistic the QC report needs for one (resolution, variable, model).

Each zarr store (new output, previous output, un-adjusted CMIP6, ERA5) is read ONCE, in
(all time x block x block) spatial windows processed by a thread pool. Everything the report
draws is derived from that single pass, plus a few cheap single-pixel reads for the
data-dependent "top" pixels. Results go to <output_root>/<res>/data/ as one NetCDF + one
JSON per scenario (and one pair per ERA5 variable, shared by all models).

usage: python qc_stats.py --res 12km --var pr --model MIROC6 [--threads 8]
"""

import argparse
import json
import os
import resource
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from qc_common import (
    Store,
    exceed_mask,
    group_for,
    load_config,
    nearest_index,
    out_dir,
    paths_for,
    same_grid,
    same_time,
    scenarios_for,
    windows,
)

F32 = np.float32
# temp-file suffix: PIDs are not unique across Slurm jobs (each job has its own PID namespace)
UNIQUE = (
    f"{os.environ.get('SLURM_JOB_ID', 'nojob')}_{os.uname().nodename}_{os.getpid()}"
)


def log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f"[{datetime.now():%H:%M:%S} peak {rss:.1f} GB] {msg}", flush=True)


def sorted_quantiles(s, n, levels):
    """Linear-interpolated quantiles of rows of s (C, T) sorted ascending with NaN last."""
    C = s.shape[0]
    out = np.full((len(levels), C), np.nan, F32)
    ok = n > 0
    rows = np.arange(C)
    nm1 = np.maximum(n - 1, 0)
    for k, p in enumerate(levels):
        pos = p * nm1
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, nm1)
        f = (pos - lo).astype(F32)
        vlo = s[rows, lo]
        vhi = s[rows, hi]
        out[k] = np.where(ok, vlo + (vhi - vlo) * f, np.nan)
    return out


def sort_cells(a):
    """(T, wy, wx) -> rows sorted per cell (C, T), valid count (C,), min, max."""
    T = a.shape[0]
    s = np.ascontiguousarray(a.reshape(T, -1).T)
    s.sort(axis=1)  # NaN sorts last
    n = (~np.isnan(s)).sum(1)
    rows = np.arange(s.shape[0])
    mn = np.where(n > 0, s[:, 0], np.nan).astype(F32)
    mx = np.where(n > 0, s[rows, np.maximum(n - 1, 0)], np.nan).astype(F32)
    return s, n, mn, mx


def masked_mean(a):
    v = ~np.isnan(a)
    n = v.sum(0)
    s = np.nansum(a, axis=0, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / n).astype(F32), n


# ---------------------------------------------------------------------------------- ERA5


def era5_pass(cfg, era5, check, fixed, threads):
    ny, nx = era5.y.size, era5.x.size
    maps = {k: np.full((ny, nx), np.nan, F32) for k in ("mean", "min", "max")}
    series = np.full((len(fixed), era5.ntime), np.nan, F32)
    total = {"exceed": 0, "valid": 0}

    def work(ys, xs):
        a = era5.block(ys, xs)
        r = {"ys": ys, "xs": xs}
        r["mean"], n = masked_mean(a)
        r["min"], r["max"] = np.fmin.reduce(a, axis=0), np.fmax.reduce(
            a, axis=0
        )  # NaN-skipping, no copies
        r["exceed"] = int(exceed_mask(a, check).sum())
        r["valid"] = int(n.sum())
        r["pts"] = {
            i: a[:, iy - ys.start, ix - xs.start].copy()
            for i, (_, iy, ix) in enumerate(fixed)
            if ys.start <= iy < ys.stop and xs.start <= ix < xs.stop
        }
        return r

    with ThreadPoolExecutor(threads) as ex:
        for fut in as_completed(
            [ex.submit(work, ys, xs) for ys, xs in windows(ny, nx, cfg["block_size"])]
        ):
            r = fut.result()
            for k in maps:
                maps[k][r["ys"], r["xs"]] = r[k]
            total["exceed"] += r["exceed"]
            total["valid"] += r["valid"]
            for i, v in r["pts"].items():
                series[i] = v
    return maps, series, total


def get_era5(cfg, res, var, fixed, fixed_names, threads, grid_ref):
    """ERA5 stats are shared by all models; compute once and cache."""
    vcfg = cfg["variables"][var]
    g = group_for(cfg, res, var)
    path = Path(g["era5_zarr"]) / f"{vcfg['era5_var']}_era5.zarr"
    data = out_dir(cfg, res) / "data"
    nc = data / f"era5_{var}.nc"
    if nc.exists():
        ds = xr.open_dataset(nc).load()
        if list(ds["fixed_name"].values) == fixed_names:
            return ds
    log(f"ERA5 pass: {path}")
    era5 = Store(path, cfg, vcfg["units"])
    same_grid(era5, grid_ref, f"ERA5 vs new ({var})")
    maps, series, total = era5_pass(cfg, era5, vcfg["check"], fixed, threads)
    ds = xr.Dataset(
        {
            **{f"era5_{k}": (("y", "x"), v) for k, v in maps.items()},
            "era5_fixed": (("fixed", "time_era5"), series),
        },
        coords={
            "y": era5.y,
            "x": era5.x,
            "fixed_name": ("fixed", fixed_names),
            "doy_era5": ("time_era5", era5.doy),
            "month_era5": ("time_era5", era5.month),
            "ymd_era5": ("time_era5", era5.ymd),
        },
        attrs={
            "path": str(path),
            "exceed_total": total["exceed"],
            "valid_total": total["valid"],
        },
    )
    tmp = nc.with_suffix(f".tmp{UNIQUE}.nc")
    ds.to_netcdf(tmp)
    os.replace(tmp, nc)
    return ds


# ------------------------------------------------------------------------- scenario pass


def scenario_pass(cfg, new, prev, sim, vcfg, fixed, threads):
    check = vcfg["check"]
    low = vcfg["extreme"] == "min"
    k_days = vcfg.get("top_point_days", 0)
    levels = np.asarray(cfg["comparison_quantiles"])
    ny, nx, T = new.y.size, new.x.size, new.ntime
    L = len(levels)
    nan2 = lambda: np.full((ny, nx), np.nan, F32)  # noqa: E731
    M = {
        k: nan2()
        for k in (
            "mean_new",
            "min_new",
            "max_new",
            "mean_sim",
            "min_sim",
            "max_sim",
            "mean_prev",
            "mad",
            "maxad",
            "frac_equal",
        )
    }
    M["exceed_count"] = np.zeros((ny, nx), np.int32)
    M["nan_mismatch"] = np.zeros((ny, nx), np.int32)
    Q = {
        "q_new": np.full((L, ny, nx), np.nan, F32),
        "q_prev": np.full((L, ny, nx), np.nan, F32),
    }
    fixed_new = np.full((len(fixed), T), np.nan, F32)
    fixed_sim = np.full((len(fixed), T), np.nan, F32)
    cands, topdays, hist_vals = [], [], []
    tot = {
        "valid_new": 0,
        "exceed_new": 0,
        "exceed_sim": 0,
        "valid_sim": 0,
        "pairs": 0,
        "equal": 0,
        "sum_abs_diff": 0.0,
        "max_abs_diff": 0.0,
        "nan_mismatch": 0,
    }

    def work(ys, xs):
        r = {"ys": ys, "xs": xs}
        local = [
            (i, iy - ys.start, ix - xs.start)
            for i, (_, iy, ix) in enumerate(fixed)
            if ys.start <= iy < ys.stop and xs.start <= ix < xs.stop
        ]
        # ---- new output
        N = new.block(ys, xs)
        wy, wx = N.shape[1:]
        E = exceed_mask(N, check)
        cnt = E.sum(0).astype(np.int32)
        r["exceed_count"] = cnt
        r["hist_vals"] = N[E]
        r["mean_new"], nvalid = masked_mean(N)
        r["fixed_new"] = {i: N[:, a, b].copy() for i, a, b in local}
        flat = cnt.ravel()
        cand_idx = [int(j) for j in np.argsort(-flat, kind="stable")[:6] if flat[j] > 0]
        r["cands"] = [
            (
                int(flat[j]),
                ys.start + j // wx,
                xs.start + j % wx,
                N[:, j // wx, j % wx].copy(),
            )
            for j in cand_idx
        ]
        if k_days:
            t, a, b = np.nonzero(E)
            v = N[t, a, b]
            order = np.argsort(v if low else -v, kind="stable")[:k_days]
            r["topdays"] = [
                (float(v[o]), int(t[o]), ys.start + int(a[o]), xs.start + int(b[o]))
                for o in order
            ]
        # ---- un-adjusted CMIP6 (same grid)
        if sim is not None:
            S = sim.block(ys, xs)
            r["exceed_sim"] = int(exceed_mask(S, check).sum())
            r["mean_sim"], ns = masked_mean(S)
            r["valid_sim"] = int(ns.sum())
            r["min_sim"], r["max_sim"] = np.fmin.reduce(S, axis=0), np.fmax.reduce(
                S, axis=0
            )
            r["fixed_sim"] = {i: S[:, a, b].copy() for i, a, b in local}
            r["cand_sim"] = [
                S[:, iy - ys.start, ix - xs.start].copy() for _, iy, ix, _ in r["cands"]
            ]
            del S
        # ---- previous output: daily differences, then its sorted distribution
        if prev is not None:
            O = prev.block(ys, xs)
            nN, nO = np.isnan(N), np.isnan(O)
            both = ~nN & ~nO
            d = np.subtract(N, O)
            np.abs(d, out=d)
            d[~both] = 0
            nb = both.sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                r["mad"] = np.where(
                    nb > 0, d.sum(0, dtype=np.float64) / nb, np.nan
                ).astype(F32)
            r["maxad"] = np.where(nb > 0, d.max(0), np.nan).astype(F32)
            eq = (N == O) | (nN & nO)
            anyv = (~nN).any(0) | (~nO).any(0)
            r["frac_equal"] = np.where(anyv, eq.mean(0), np.nan).astype(F32)
            r["nan_mismatch"] = (nN != nO).sum(0).astype(np.int32)
            r["pairs"], r["equal"] = int(both.sum()), int((eq & both).sum())
            r["sum_abs_diff"], r["max_abs_diff"] = float(
                d.sum(dtype=np.float64)
            ), float(d.max())
            del d, eq, both, nN, nO
            r["mean_prev"], _ = masked_mean(O)
            s, n, _, _ = sort_cells(O)
            del O
            r["q_prev"] = sorted_quantiles(s, n, levels).reshape(L, wy, wx)
            del s
        # ---- sorted distribution of the new output
        s, n, mn, mx = sort_cells(N)
        del N
        r["q_new"] = sorted_quantiles(s, n, levels).reshape(L, wy, wx)
        r["min_new"], r["max_new"] = mn.reshape(wy, wx), mx.reshape(wy, wx)
        r["valid_new"] = int(nvalid.sum())
        return r

    with ThreadPoolExecutor(threads) as ex:
        futs = [
            ex.submit(work, ys, xs) for ys, xs in windows(ny, nx, cfg["block_size"])
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            ys, xs = r["ys"], r["xs"]
            for k in M:
                if k in r:
                    M[k][ys, xs] = r[k]
            for k in Q:
                if k in r:
                    Q[k][:, ys, xs] = r[k]
            for k in tot:
                if k in r and k != "nan_mismatch":
                    tot[k] = max(tot[k], r[k]) if k == "max_abs_diff" else tot[k] + r[k]
            tot["exceed_new"] += int(r["exceed_count"].sum())
            for j, v in r["fixed_new"].items():
                fixed_new[j] = v
            for j, v in r.get("fixed_sim", {}).items():
                fixed_sim[j] = v
            for c, cs in zip(r["cands"], r.get("cand_sim", [None] * len(r["cands"]))):
                cands.append((*c, cs))
            topdays += r.get("topdays", [])
            hist_vals.append(r["hist_vals"])
            if i % 5 == 0 or i == len(futs):
                log(f"  window {i}/{len(futs)}")
    tot["nan_mismatch"] = int(M["nan_mismatch"].sum())
    hv = np.concatenate(hist_vals) if hist_vals else np.zeros(0, F32)
    tot["hist_values_total"] = int(hv.size)
    cap = cfg["max_hist_values"]
    if hv.size > cap:
        hv = np.random.default_rng(cfg["random_seed"]).choice(hv, cap, replace=False)
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    top = cands[:6] if len(cands) >= 6 else []
    topdays.sort(key=lambda t: t[0] if low else -t[0])
    return M, Q, fixed_new, fixed_sim, top, topdays[:k_days], hv, tot


# ------------------------------------------------------------------------- dtr upstream


def dtr_upstream(cfg, res, var, model, scenario):
    g = group_for(cfg, res, var)
    up = g.get("dtr_upstream")
    if var != "dtr" or not up:
        return None
    data = {}
    for stage, base in up.items():
        files = sorted(Path(base, model, scenario, "day", var).glob("*.nc"))
        if not files:
            continue
        acc = None
        for f in files:
            with xr.open_dataset(f) as ds:
                a = ds[var].transpose("time", "lat", "lon").values.astype(np.float64)
                lat, lon = ds["lat"].values, ds["lon"].values
            v = ~np.isnan(a)
            cur = {
                "min": np.where(v, a, np.inf).min(0),
                "max": np.where(v, a, -np.inf).max(0),
                "sum": np.where(v, a, 0).sum(0),
                "n": v.sum(0),
            }
            if acc is None:
                acc = cur
            else:
                acc["min"] = np.minimum(acc["min"], cur["min"])
                acc["max"] = np.maximum(acc["max"], cur["max"])
                acc["sum"] += cur["sum"]
                acc["n"] += cur["n"]
        ok = acc["n"] > 0
        for k in ("min", "max"):
            data[f"{stage}_{k}"] = (
                (f"{stage}_lat", f"{stage}_lon"),
                np.where(ok, acc[k], np.nan).astype(F32),
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            data[f"{stage}_mean"] = (
                (f"{stage}_lat", f"{stage}_lon"),
                np.where(ok, acc["sum"] / acc["n"], np.nan).astype(F32),
            )
        data[f"{stage}_lat"] = (f"{stage}_lat", lat)
        data[f"{stage}_lon"] = (f"{stage}_lon", lon)
    return xr.Dataset(data) if data else None


# ------------------------------------------------------------------------------- driver


def fixed_points(cfg, ref):
    names = list(dict.fromkeys(n for s in cfg["place_sets"] for n in s))
    pts = [
        (
            n,
            nearest_index(ref.y, cfg["places"][n]["y"]),
            nearest_index(ref.x, cfg["places"][n]["x"]),
        )
        for n in names
    ]
    # random locations: same in every report of a resolution (seeded, from the valid-cell mask)
    valid = ~np.isnan(ref._read({"time": 0}))
    cells = np.argwhere(valid)
    rng = np.random.default_rng(cfg["random_seed"])
    pick = cells[rng.choice(len(cells), cfg["n_random_points"], replace=False)]
    pts += [(f"random_{i + 1}", int(a), int(b)) for i, (a, b) in enumerate(pick)]
    return pts


def run(cfg, res, var, model, threads, only_scenarios=None):
    vcfg = cfg["variables"][var]
    units = vcfg["units"]
    data = out_dir(cfg, res) / "data"
    data.mkdir(parents=True, exist_ok=True)
    scens = scenarios_for(cfg, res, var, model)
    if only_scenarios:
        scens = [s for s in scens if s in only_scenarios]
    if not scens:
        raise SystemExit(f"no new outputs for {var} {model} at {res}")
    ref = Store(paths_for(cfg, res, var, model, scens[0])["new"], cfg, units)
    fixed = fixed_points(cfg, ref)
    fixed_names = [f[0] for f in fixed]
    era5 = get_era5(cfg, res, var, fixed, fixed_names, threads, ref)
    hist_stores = {}
    for scen in scens:
        t0 = time.time()
        P = paths_for(cfg, res, var, model, scen)
        log(f"{var} {model} {scen}: new={P['new']}")
        new = Store(P["new"], cfg, units)
        same_grid(new, ref, "new vs reference grid")
        prev = sim = None
        if P["previous"] is not None:
            prev = Store(P["previous"], cfg, units)
            same_grid(prev, new, f"previous vs new ({scen})")
            same_time(prev, new, f"previous vs new ({scen})")
        else:
            log("  no previous output: comparison skipped")
        if P["sim"] is not None:
            sim = Store(P["sim"], cfg, units)
            same_grid(sim, new, f"un-adjusted vs new ({scen})")
            same_time(sim, new, f"un-adjusted vs new ({scen})")
        M, Q, fx_new, fx_sim, top, topdays, hv, tot = scenario_pass(
            cfg, new, prev, sim, vcfg, fixed, threads
        )
        if scen == "historical":
            hist_stores = {"new": new, "sim": sim}
        log(f"  pass done in {time.time() - t0:.0f} s")

        dv = {k: (("y", "x"), v) for k, v in M.items()}
        dv.update({k: (("level", "y", "x"), v) for k, v in Q.items()})
        dv["fixed_new"] = (("fixed", "time"), fx_new)
        dv["fixed_sim"] = (("fixed", "time"), fx_sim)
        dv["hist_values"] = (("hv",), hv)
        coords = {
            "y": new.y,
            "x": new.x,
            "level": cfg["comparison_quantiles"],
            "fixed_name": ("fixed", fixed_names),
            "fixed_iy": ("fixed", [f[1] for f in fixed]),
            "fixed_ix": ("fixed", [f[2] for f in fixed]),
            "ymd": ("time", new.ymd),
            "doy": ("time", new.doy),
            "month": ("time", new.month),
        }
        summary = {
            "res": res,
            "var": var,
            "model": model,
            "scenario": scen,
            "units": units,
            "check": vcfg["check"],
            "paths": {k: str(v) if v else None for k, v in P.items()},
            "new_attrs": {
                k: v
                for k, v in new.attrs.items()
                if isinstance(v, (str, int, float))
                and any(s in k for s in ("rank", "window", "code", "version", "commit"))
            },
            "totals": tot,
            "ntime": new.ntime,
            "era5_exceed_total": int(era5.attrs["exceed_total"]),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        # top-6 pixels (most days beyond the check threshold) + their series in every source
        if top:
            dv["top_count"] = (("top",), [t[0] for t in top])
            coords["top_iy"] = ("top", [t[1] for t in top])
            coords["top_ix"] = ("top", [t[2] for t in top])
            dv["top_new"] = (("top", "time"), np.stack([t[3] for t in top]))
            if sim is not None:
                dv["top_sim"] = (("top", "time"), np.stack([t[4] for t in top]))
            era5_store = Store(era5.attrs["path"], cfg, units)
            dv["top_era5"] = (
                ("top", "time_era5"),
                np.stack([era5_store.point(t[1], t[2]) for t in top]),
            )
            if scen != "historical" and hist_stores:
                for k in ("new", "sim"):
                    if hist_stores.get(k) is not None:
                        dv[f"top_hist_{k}"] = (
                            ("top", "time_hist"),
                            np.stack([hist_stores[k].point(t[1], t[2]) for t in top]),
                        )
                h = hist_stores["new"]
                coords.update(
                    {
                        "doy_hist": ("time_hist", h.doy),
                        "month_hist": ("time_hist", h.month),
                    }
                )
        # dtr: the most extreme pixel-days, with tasmax/tasmin on those days
        if topdays:
            g = group_for(cfg, res, var)
            k = len(topdays)
            dv["td_value"] = (("td",), [t[0] for t in topdays])
            coords.update(
                {
                    "td_it": ("td", [t[1] for t in topdays]),
                    "td_iy": ("td", [t[2] for t in topdays]),
                    "td_ix": ("td", [t[3] for t in topdays]),
                }
            )
            for other in ("tasmax", "tasmin"):
                p = Path(g["new_adjusted"]) / f"{other}_{model}_{scen}_adjusted.zarr"
                vals = [np.nan] * k
                if p.exists():
                    st = Store(p, cfg, "K")
                    vals = [st.value_at(t[1], t[2], t[3]) for t in topdays]
                dv[f"td_{other}"] = (("td",), vals)
            era5_store = Store(era5.attrs["path"], cfg, units)
            dv["td_new"] = (
                ("td", "time"),
                np.stack([new.point(t[2], t[3]) for t in topdays]),
            )
            if sim is not None:
                dv["td_sim"] = (
                    ("td", "time"),
                    np.stack([sim.point(t[2], t[3]) for t in topdays]),
                )
            dv["td_era5"] = (
                ("td", "time_era5"),
                np.stack([era5_store.point(t[2], t[3]) for t in topdays]),
            )
        if "time_era5" in {d for v in dv.values() for d in v[0]}:
            coords.update(
                {
                    "doy_era5": ("time_era5", era5["doy_era5"].values),
                    "month_era5": ("time_era5", era5["month_era5"].values),
                }
            )
        ds = xr.Dataset(dv, coords=coords)
        up = dtr_upstream(cfg, res, var, model, scen)
        if up is not None:
            ds = ds.merge(up)
        stem = data / f"{var}_{model}_{scen}"
        tmp = stem.with_suffix(f".tmp{UNIQUE}.nc")
        enc = {k: {"zlib": True, "complevel": 1} for k in ds.data_vars}
        ds.to_netcdf(tmp, encoding=enc)
        os.replace(tmp, stem.with_suffix(".nc"))
        summary["seconds"] = round(time.time() - t0, 1)
        stem.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str))
        log(f"  wrote {stem}.nc ({summary['seconds']} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--res", required=True)
    ap.add_argument("--var", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--scenarios", nargs="*")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    run(
        cfg,
        a.res,
        a.var,
        a.model,
        a.threads or cfg["resolutions"][a.res]["threads"],
        a.scenarios,
    )
