"""Land-only domain-wide Annual climatology deltas vs WRF-ERA5.

Writes ${output_root}/qc/annual_land_deltas/{var}.csv -- one file per
output variable. Each row is a named GCM x SSP x future era. The spatial
mean uses cells where the WRF LANDMASK (nearest-neighbor warped onto that
variable's grid) is 1 and WRF-ERA5 is finite.

Usage:
    python annual_land_deltas.py --config config_12km.yaml

qc.py also calls write_annual_land_deltas() at the end of a QC run.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG_PATH, Config, load_config

PERIOD = "Annual"


def analysis_mask(landmask: np.ndarray, era5: np.ndarray) -> np.ndarray:
    """True where LANDMASK==1 and WRF-ERA5 is finite."""
    return (np.asarray(landmask) == 1) & np.isfinite(era5)


def land_mean_delta(
    future: np.ndarray, era5: np.ndarray, mask: np.ndarray
) -> tuple[float, int] | None:
    """Mean of (future - era5) over mask; None if no finite cells."""
    delta = np.asarray(future, dtype=np.float64) - np.asarray(era5, dtype=np.float64)
    vals = delta[mask]
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return None
    return float(np.mean(finite)), int(finite.size)


def _lonlat_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(lat, dtype=np.float64).ravel())
    lon = np.deg2rad(np.asarray(lon, dtype=np.float64).ravel())
    return np.column_stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
    )


def target_latlon(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """2D lat/lon on the climatology grid.

    Uses stored lat/lon when present (most variables). 4km snw has only
    projected y/x, so those are converted from EPSG:3338.
    """
    if "lat" in ds and "lon" in ds:
        return ds["lat"].values, ds["lon"].values
    xx, yy = np.meshgrid(ds["x"].values, ds["y"].values)
    transformer = Transformer.from_crs("EPSG:3338", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    return np.asarray(lat), np.asarray(lon)


def regrid_landmask_nearest(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    src_mask: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor warp of a binary land mask onto dst lat/lon."""
    tree = cKDTree(_lonlat_to_xyz(src_lat, src_lon))
    _, idx = tree.query(_lonlat_to_xyz(dst_lat, dst_lon), k=1)
    dst = np.asarray(dst_lat)
    return np.asarray(src_mask).ravel()[idx].reshape(dst.shape).astype(np.int8)


def load_geo_em_landmask(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lat, lon, LANDMASK) from a WRF geo_em file, Time-squeezed.

    Uses mass-point XLAT_M/XLONG_M (same stagger as LANDMASK), not U/V/corners.
    """
    ds = xr.open_dataset(path)
    lat = ds["XLAT_M"].isel(Time=0).values
    lon = ds["XLONG_M"].isel(Time=0).values
    mask = ds["LANDMASK"].isel(Time=0).values.astype(np.int8)
    ds.close()
    return lat, lon, mask


def write_variable_csv(
    da: xr.DataArray,
    landmask: np.ndarray,
    config: Config,
    output_var: str,
    out_path: Path,
) -> None:
    cfg = config.qc["delta_maps"]
    baseline = cfg["baseline"]
    agg = cfg["aggregation"]
    era5 = da.sel(
        model=baseline["model"],
        scenario=baseline["scenario"],
        era=baseline["era"],
        period=PERIOD,
        aggregation=agg,
    ).values
    mask = analysis_mask(landmask, era5)
    units = da.attrs.get("units", "")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        geo_em = config.qc["land_mask"]["geo_em"]
        f.write(f"# variable: {output_var}\n")
        f.write(f"# geo_em: {geo_em}\n")
        f.write(
            f"# baseline: {baseline['model']} {baseline['scenario']} {baseline['era']}\n"
        )
        f.write(f"# period: {PERIOD}\n")
        f.write(f"# aggregation: {agg}\n")
        f.write("# mask: LANDMASK==1 and finite WRF-ERA5\n")
        writer = csv.DictWriter(
            f, fieldnames=["model", "scenario", "era", "delta", "units", "n_cells"]
        )
        writer.writeheader()
        for model in config.models:
            for scenario in cfg["scenarios"]:
                for era in cfg["future_eras"]:
                    future = da.sel(
                        model=model,
                        scenario=scenario,
                        era=era,
                        period=PERIOD,
                        aggregation=agg,
                    ).values
                    result = land_mean_delta(future, era5, mask)
                    if result is None:
                        continue
                    delta, n_cells = result
                    writer.writerow(
                        {
                            "model": model,
                            "scenario": scenario,
                            "era": era,
                            "delta": f"{float(delta):.2f}",
                            "units": units,
                            "n_cells": n_cells,
                        }
                    )


def write_annual_land_deltas(config: Config) -> None:
    src_lat, src_lon, src_mask = load_geo_em_landmask(config.qc["land_mask"]["geo_em"])
    warped = {}
    out_dir = config.qc_dir / "annual_land_deltas"
    for output_var in config.output_variable_order:
        ds = xr.open_zarr(config.output_zarr_path(output_var), consolidated=True)
        key = (int(ds.sizes["y"]), int(ds.sizes["x"]))
        if key not in warped:
            dst_lat, dst_lon = target_latlon(ds)
            warped[key] = regrid_landmask_nearest(
                src_lat, src_lon, src_mask, dst_lat, dst_lon
            )
        out_path = out_dir / f"{output_var}.csv"
        write_variable_csv(ds[output_var], warped[key], config, output_var, out_path)
        print(f"wrote {out_path}")
        ds.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    write_annual_land_deltas(load_config(args.config))


if __name__ == "__main__":
    main()
