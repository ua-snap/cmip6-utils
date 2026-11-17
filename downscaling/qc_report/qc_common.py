"""Shared helpers for the QC report tool: config loading, store discovery and reading.

No settings live here; everything editable is in config.yaml.
"""

import re
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
import zarr

HERE = Path(__file__).resolve().parent
SCENARIO_ORDER = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]


def load_config(path=None):
    with open(path or HERE / "config.yaml") as f:
        return yaml.safe_load(f)


def group_for(cfg, res, var):
    for name, g in cfg["resolutions"][res]["groups"].items():
        if var in g["variables"]:
            return g
    raise KeyError(f"{var} is not configured for {res}")


def out_dir(cfg, res):
    return Path(cfg["output_root"]) / res


def scenarios_for(cfg, res, var, model):
    """Scenarios with a new adjusted store, historical first."""
    g = group_for(cfg, res, var)
    found = []
    for p in Path(g["new_adjusted"]).glob(f"{var}_{model}_*_adjusted.zarr"):
        m = re.fullmatch(
            rf"{re.escape(var)}_{re.escape(model)}_(\w+?)_adjusted\.zarr", p.name
        )
        if m:
            found.append(m.group(1))
    return sorted(
        found, key=lambda s: (SCENARIO_ORDER.index(s) if s in SCENARIO_ORDER else 99, s)
    )


def jobs_for(cfg, res, variables=None, models=None):
    """All (var, model) pairs that have new outputs."""
    jobs = []
    for g in cfg["resolutions"][res]["groups"].values():
        for var in g["variables"]:
            if variables and var not in variables:
                continue
            ms = set()
            for p in Path(g["new_adjusted"]).glob(f"{var}_*_adjusted.zarr"):
                ms.add(p.name[len(var) + 1 :].rsplit("_", 2)[0])
            for m in sorted(ms):
                if not models or m in models:
                    jobs.append((var, m))
    return jobs


def paths_for(cfg, res, var, model, scenario):
    g = group_for(cfg, res, var)
    era5_var = cfg["variables"][var]["era5_var"]
    prev = g.get("previous_adjusted")
    p = {
        "new": Path(g["new_adjusted"]) / f"{var}_{model}_{scenario}_adjusted.zarr",
        "previous": Path(prev)
        / cfg["variables"][var]
        .get("previous_filename", cfg["previous_filename"])
        .format(var=var, model=model, scenario=scenario)
        if prev
        else None,
        "sim": Path(g["cmip6_zarr"]) / f"{var}_{model}_{scenario}.zarr",
        "era5": Path(g["era5_zarr"]) / f"{era5_var}_era5.zarr",
    }
    return {k: (v if v is not None and v.exists() else None) for k, v in p.items()}


def unit_transform(cfg, from_units, to_units):
    if from_units == to_units or from_units is None:
        return 1.0, 0.0
    for c in cfg["unit_conversions"]:
        if c["from"] == from_units and c["to"] == to_units:
            return float(c["scale"]), float(c["offset"])
    raise ValueError(
        f"No unit conversion {from_units!r} -> {to_units!r} in config.yaml"
    )


class Store:
    """A 3-D variable in a zarr store, read as (time, y, x) float32 in display units.

    Reads go straight through zarr in the store's native dimension order and are transposed
    in numpy: xarray's lazy indexing falls back to point-wise (vindex) selection for some
    dimension orders, which needs several GB of index arrays per block.
    """

    def __init__(self, path, cfg, to_units):
        self.path = Path(path)
        ds = xr.open_zarr(self.path, chunks=None)
        names = [n for n in ds.data_vars if ds[n].ndim == 3]
        if len(names) != 1:
            raise ValueError(f"{path}: expected one 3-D variable, found {names}")
        self.name = names[0]
        self.da = ds[self.name]
        if set(self.da.dims) != {"time", "y", "x"}:
            raise ValueError(f"{path}: unexpected dims {self.da.dims}")
        self.z = zarr.open_group(str(self.path), mode="r")[self.name]
        self.zdims = list(self.z.attrs["_ARRAY_DIMENSIONS"])
        fill = self.z.fill_value
        self.fill = (
            None
            if fill is None or (isinstance(fill, float) and np.isnan(fill))
            else fill
        )
        for k in ("scale_factor", "add_offset"):
            if k in self.z.attrs:
                raise NotImplementedError(f"{path}: packed data ({k}) is not supported")
        self.units = self.da.attrs.get("units")
        self.scale, self.offset = unit_transform(cfg, self.units, to_units)
        self.x = ds["x"].values
        self.y = ds["y"].values
        t = ds["time"]
        self.ymd = (t.dt.year * 10000 + t.dt.month * 100 + t.dt.day).values.astype(
            np.int64
        )
        self.doy = t.dt.dayofyear.values.astype(np.int16)
        self.month = t.dt.month.values.astype(np.int8)
        self.ntime = t.size
        self.attrs = dict(ds.attrs)

    def _read(self, sel):
        """sel: {dim: slice | int}; returns the remaining dims in (time, y, x) order."""
        key = tuple(sel.get(d, slice(None)) for d in self.zdims)
        a = self.z[key]
        keep = [d for d in self.zdims if not isinstance(sel.get(d), (int, np.integer))]
        a = np.transpose(a, [keep.index(d) for d in ("time", "y", "x") if d in keep])
        a = np.ascontiguousarray(a, dtype=np.float32)
        if self.fill is not None:
            a[a == self.fill] = np.nan
        return a

    def _convert(self, a):
        a = np.asarray(a)
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        if self.scale != 1.0:
            np.multiply(a, np.float32(self.scale), out=a)
        if self.offset != 0.0:
            np.add(a, np.float32(self.offset), out=a)
        return a

    def block(self, ys, xs):
        return self._convert(self._read({"y": ys, "x": xs}))

    def point(self, iy, ix):
        return self._convert(self._read({"y": int(iy), "x": int(ix)}))

    def value_at(self, it, iy, ix):
        return float(
            self._convert(
                self._read({"time": int(it), "y": int(iy), "x": int(ix)}).reshape(1)
            )[0]
        )


def same_grid(a, b, what):
    if a.x.shape != b.x.shape or a.y.shape != b.y.shape:
        raise ValueError(
            f"{what}: grid shapes differ {a.y.size}x{a.x.size} vs {b.y.size}x{b.x.size}"
        )
    if not (np.allclose(a.x, b.x) and np.allclose(a.y, b.y)):
        raise ValueError(f"{what}: x/y coordinates differ")


def same_time(a, b, what):
    if a.ntime != b.ntime or not np.array_equal(a.ymd, b.ymd):
        raise ValueError(f"{what}: time axes differ ({a.ntime} vs {b.ntime} steps)")


def windows(ny, nx, size):
    for y0 in range(0, ny, size):
        for x0 in range(0, nx, size):
            yield slice(y0, min(y0 + size, ny)), slice(x0, min(x0 + size, nx))


def nearest_index(coords, value):
    return int(np.abs(np.asarray(coords) - value).argmin())


def exceed_mask(a, check):
    k = check["kind"]
    if k == "above":
        return a > check["value"]
    if k == "below":
        return a < check["value"]
    if k == "outside":
        return (a < check["low"]) | (a > check["high"])
    raise ValueError(k)


def check_label(check, units):
    k = check["kind"]
    if k == "above":
        return f"> {check['value']:g} {units}"
    if k == "below":
        return f"< {check['value']:g} {units}"
    return f"outside [{check['low']:g}, {check['high']:g}] {units}"
