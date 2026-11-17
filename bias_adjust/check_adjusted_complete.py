"""Check that bias-adjusted zarr stores are complete and consistent with their inputs.

Two levels:

cheap (metadata only, seconds per store)
  - store exists, no "<store>.writing" sibling left behind
  - consolidated metadata opens; the climate variable is present with dims (time, y, x)
  - dtype == config.OUTPUT_DTYPE, shape == sim store shape
  - every zarr chunk of the variable exists on disk (nchunks_initialized == nchunks)
  - time coordinate identical to the sim store's
  - provenance attrs present; qdm_rank_main_only == --expect_main_only (if given)

full (reads both stores once each)
  - NaN count equal to the sim's, AND per-cell NaN-count map (sum over time) equal,
    AND per-timestep NaN-count series (sum over y, x) equal  -> NaN mask equality
    (the fillna fix guarantees output NaN == input NaN; this is the strongest cheap
    proxy that needs no chunk alignment between the two stores)
  - finite nanmin / nanmax, count outside config.SQUEEZE_LIMITS (should be 0)
  - optional --hash: sha1 over the md5 of every chunk file, for bitwise run-to-run
    comparisons (T5 determinism test)

Example usage:
    python check_adjusted_complete.py \
        --adjusted_dir /beegfs/CMIP6/jdpaul3/cmip6_patch_31d_reruns/12km/adjusted \
        --sim_dirs /beegfs/.../cmip6_12km_downscaling/cmip6_zarr /beegfs/.../sfcWind_hurs_hursmin/cmip6_zarr \
        --mode full --expect_main_only false \
        --out_csv /beegfs/CMIP6/jdpaul3/cmip6_patch_31d_reruns/12km/qc/completeness.csv

    --stores 'pr_*'      glob on store names
    --shard 2/8          process every 8th store starting at index 2 (slurm array use)

Exit status 1 if any checked store is not OK.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from config import OUTPUT_DTYPE, SQUEEZE_LIMITS, cmip6_zarr_tmp_fn, trained_qm_tmp_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

EXPECTED_DIMS = ("time", "y", "x")
PROVENANCE_ATTRS = (
    "qdm_rank_main_only",
    "qdm_rank_window_days",
    "xclim_version",
    "cmip6_utils_commit",
    "bias_adjust_version",
)


def parse_store_name(adj_path):
    """'{var}_{model}_{scenario}_adjusted.zarr' -> (var, model, scenario)."""
    stem = Path(adj_path).name.replace("_adjusted.zarr", "")
    var_id, model, scenario = stem.split("_", 2)
    return var_id, model, scenario


def _grid_shape(path):
    """(ny, nx) of a zarr store from its metadata (no data read), or None."""
    try:
        with xr.open_zarr(path, consolidated=True) as ds:
            return ds.sizes["y"], ds.sizes["x"]
    except Exception:
        return None


def _pick_by_grid(adj_path, candidates):
    """Among existing candidate stores, return the one on the same (y, x) grid as
    adj_path. The same file name exists in both the 12 km and 4 km input trees, so
    the first match by name alone can be the wrong resolution."""
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None
    target = _grid_shape(adj_path)
    if target is not None:
        for p in candidates:
            if _grid_shape(p) == target:
                return p
        logging.warning(f"no candidate on grid {target} for {Path(adj_path).name}")
        return None
    return candidates[0]


def find_sim_path(adj_path, sim_dirs):
    """Locate the sim store that produced adj_path among sim_dirs (same grid)."""
    var_id, model, scenario = parse_store_name(adj_path)
    name = cmip6_zarr_tmp_fn.format(var_id=var_id, model=model, scenario=scenario)
    return _pick_by_grid(adj_path, [Path(d) / name for d in sim_dirs])


def find_train_path(adj_path, train_dirs):
    """Locate the trained QDM store for adj_path among train_dirs (same grid)."""
    var_id, model, _ = parse_store_name(adj_path)
    name = trained_qm_tmp_fn.format(var_id=var_id, model=model)
    return _pick_by_grid(adj_path, [Path(d) / name for d in train_dirs or []])


def untrained_mask(train_path):
    """(y, x) bool DataArray: True where the trained af is all-NaN."""
    af = xr.open_zarr(train_path, consolidated=True)["af"]
    if af.dims[:2] == ("x", "y"):
        af = af.transpose("y", "x", ...)
    return af.isnull().all(["dayofyear", "quantiles"])


def climate_var(ds):
    vs = [v for v in ds.data_vars if ds[v].ndim == 3]
    if len(vs) != 1:
        raise ValueError(f"Expected one 3-D variable, found {vs}")
    return vs[0]


def cheap_check(adj_path, sim_path, expect_main_only=None):
    """Metadata-only completeness check. Returns a dict with 'status' and 'problems'."""
    adj_path = Path(adj_path)
    res = {"store": adj_path.name, "sim_path": str(sim_path), "mode": "cheap"}
    problems = []
    try:
        var_id, model, scenario = parse_store_name(adj_path)
        res.update({"var_id": var_id, "model": model, "scenario": scenario})
        if not adj_path.exists():
            problems.append("missing")
            res["status"] = "FAIL"
            res["problems"] = ";".join(problems)
            return res
        if (adj_path.parent / (adj_path.name + ".writing")).exists():
            problems.append("writing_sibling_present")
        if not (adj_path / ".zmetadata").exists():
            problems.append("no_zmetadata")

        ds = xr.open_zarr(adj_path, consolidated=True)
        v = climate_var(ds)
        if v != var_id:
            problems.append(f"var_name_{v}")
        da = ds[v]
        if tuple(da.dims) != EXPECTED_DIMS:
            problems.append(f"dims_{'_'.join(da.dims)}")
        if str(da.dtype) != OUTPUT_DTYPE:
            problems.append(f"dtype_{da.dtype}")
        res["shape"] = "x".join(str(s) for s in da.shape)

        z = zarr.open_group(str(adj_path), mode="r")
        zarr_var = z[v]
        res["nchunks"] = int(zarr_var.nchunks)
        res["nchunks_initialized"] = int(zarr_var.nchunks_initialized)
        if zarr_var.nchunks_initialized != zarr_var.nchunks:
            problems.append("missing_chunks")
        res["zarr_chunks"] = "x".join(str(c) for c in zarr_var.chunks)

        for a in PROVENANCE_ATTRS:
            if a not in ds.attrs:
                problems.append(f"no_attr_{a}")
        res["qdm_rank_main_only"] = ds.attrs.get("qdm_rank_main_only")
        if expect_main_only is not None and "qdm_rank_main_only" in ds.attrs:
            got = str(ds.attrs["qdm_rank_main_only"]).lower() == "true"
            if got != expect_main_only:
                problems.append(f"rank_main_only_is_{got}")
        res["cmip6_utils_commit"] = ds.attrs.get("cmip6_utils_commit")
        res["creation_date"] = ds.attrs.get("creation_date")
        for a in (
            "squeeze_count_below",
            "squeeze_count_above",
            "squeeze_lower_limit",
            "squeeze_upper_limit",
        ):
            res[a] = ds.attrs.get(a)

        if sim_path is not None and Path(sim_path).exists():
            sim = xr.open_zarr(sim_path, consolidated=True)
            sv = climate_var(sim)
            if tuple(sim[sv].transpose(*EXPECTED_DIMS).shape) != tuple(da.shape):
                problems.append("shape_mismatch_vs_sim")
            if sim.time.size == ds.time.size:
                if not np.array_equal(sim.time.values, ds.time.values):
                    problems.append("time_coord_mismatch")
            else:
                problems.append("time_length_mismatch")
            sim.close()
        else:
            problems.append("sim_not_found")
        ds.close()
    except Exception as e:  # any error is a failed check, never a crash
        problems.append(f"error:{type(e).__name__}:{str(e)[:120]}")

    res["problems"] = ";".join(problems)
    res["status"] = "OK" if not problems else "FAIL"
    return res


def nan_signature(da):
    """(total NaN count, per-cell NaN map, per-timestep NaN series) computed lazily."""
    isnull = da.isnull()
    per_cell = isnull.sum("time")
    per_time = isnull.sum(("y", "x"))
    return per_cell, per_time


def hash_store(adj_path, var_id):
    """sha1 over the sorted (name, md5 of DECODED bytes) of every chunk of var_id.

    Hashes the decompressed chunk contents, not the files: multithreaded Blosc
    may lay out its compressed blocks in a different order on every write, so
    identical data can produce different chunk files (seen in T5, 2026-11-16).
    """
    d = Path(adj_path) / var_id
    compressor = zarr.open_array(str(d), mode="r").compressor
    items = []
    for p in sorted(d.iterdir()):
        if p.name.startswith("."):
            continue
        raw = p.read_bytes()
        if compressor is not None:
            raw = compressor.decode(raw)
        items.append(f"{p.name}:{hashlib.md5(raw).hexdigest()}")
    return hashlib.sha1("\n".join(items).encode()).hexdigest()


def full_check(
    adj_path, sim_path, expect_main_only=None, do_hash=False, train_path=None
):
    """cheap_check + NaN mask == (sim NaN | untrained cell) + value range + optional hash."""
    res = cheap_check(adj_path, sim_path, expect_main_only)
    res["mode"] = "full"
    problems = [p for p in res["problems"].split(";") if p]
    if "missing" in problems or sim_path is None:
        return res
    t0 = time.time()
    try:
        ds = xr.open_zarr(adj_path, consolidated=True)
        v = climate_var(ds)
        da = ds[v]
        sim = xr.open_zarr(sim_path, consolidated=True)
        sv = climate_var(sim)
        sda = sim[sv].transpose(*EXPECTED_DIMS)

        if train_path is not None:
            res["train_path"] = str(train_path)
            sda = sda.where(~untrained_mask(train_path))
        else:
            problems.append("train_not_found_mask_is_sim_only")
        adj_cell, adj_time = nan_signature(da)
        sim_cell, sim_time = nan_signature(sda)
        adj_cell, adj_time, sim_cell, sim_time, vmin, vmax = dask.compute(
            adj_cell, adj_time, sim_cell, sim_time, da.min(), da.max()
        )
        res["nan_count"] = int(adj_cell.sum())
        res["expected_nan_count"] = int(sim_cell.sum())
        res["nanmin"] = float(vmin)
        res["nanmax"] = float(vmax)
        if res["nan_count"] != res["expected_nan_count"]:
            problems.append("nan_count_mismatch")
        if not np.array_equal(adj_cell.values, sim_cell.values):
            problems.append("nan_map_mismatch")
        if not np.array_equal(adj_time.values, sim_time.values):
            problems.append("nan_timeseries_mismatch")
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            problems.append("non_finite_minmax")

        lo, hi = SQUEEZE_LIMITS.get(v, (None, None))
        if lo is not None and float(vmin) < lo:
            problems.append(f"below_limit_{lo}")
        if hi is not None and float(vmax) > hi:
            problems.append(f"above_limit_{hi}")
        if v == "dtr" and float(vmin) < 0:
            problems.append("negative_dtr")

        if do_hash:
            res["chunk_hash"] = hash_store(adj_path, v)
        ds.close()
        sim.close()
    except Exception as e:
        problems.append(f"error:{type(e).__name__}:{str(e)[:120]}")
    res["full_check_s"] = round(time.time() - t0, 1)
    res["problems"] = ";".join(problems)
    res["status"] = "OK" if not problems else "FAIL"
    return res


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--adjusted_dir", type=str, required=True)
    p.add_argument("--sim_dirs", type=str, nargs="+", required=True)
    p.add_argument(
        "--train_dirs",
        type=str,
        nargs="+",
        default=None,
        help="trained_datasets dirs; full mode expects NaN = sim NaN | untrained cell",
    )
    p.add_argument(
        "--stores", type=str, default="*_adjusted.zarr", help="glob on store names"
    )
    p.add_argument("--mode", choices=["cheap", "full"], default="cheap")
    p.add_argument("--expect_main_only", type=str, default=None, help="true/false")
    p.add_argument("--hash", action="store_true", help="hash chunk files (full mode)")
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument(
        "--shard", type=str, default=None, help="i/n: every n-th store from i"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    expect = None
    if args.expect_main_only is not None:
        expect = args.expect_main_only.strip().lower() in ("true", "t", "1", "yes")
    stores = sorted(Path(args.adjusted_dir).glob(args.stores))
    stores = [s for s in stores if s.name.endswith(".zarr")]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        stores = stores[i::n]
    logging.info(
        f"Checking {len(stores)} store(s) in {args.adjusted_dir} ({args.mode})"
    )

    rows = []
    for s in stores:
        sim_path = find_sim_path(s, args.sim_dirs)
        if args.mode == "full":
            r = full_check(
                s,
                sim_path,
                expect,
                do_hash=args.hash,
                train_path=find_train_path(s, args.train_dirs),
            )
        else:
            r = cheap_check(s, sim_path, expect)
        rows.append(r)
        logging.info(f"{r['status']:4} {s.name} {r['problems']}")

    df = pd.DataFrame(rows)
    if args.out_csv:
        out = Path(args.out_csv)
        if args.shard:
            out = out.with_name(
                out.stem + f".shard{args.shard.replace('/', 'of')}" + out.suffix
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logging.info(f"Wrote {out}")
    n_fail = int((df["status"] != "OK").sum()) if len(df) else 0
    logging.info(f"SUMMARY: {len(df) - n_fail} OK, {n_fail} FAIL")
    sys.exit(1 if n_fail else 0)
