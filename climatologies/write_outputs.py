"""Write the combined master Dataset to both Zarr and NetCDF (identical
content, two formats), per the spec.
"""

from __future__ import annotations

import xarray as xr

from config import Config


def _strip_stale_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Source fragments/grid-reference files carry their own on-disk chunk
    encoding; leaving it in place can make to_zarr/to_netcdf raise on a
    chunk-shape mismatch against the newly-combined array shape."""
    for name in list(ds.data_vars) + list(ds.coords):
        ds[name].encoding.pop("chunks", None)
        ds[name].encoding.pop("preferred_chunks", None)
    return ds


def write_outputs(ds: xr.Dataset, config: Config) -> None:
    config.final_output_dir.mkdir(parents=True, exist_ok=True)
    ds = _strip_stale_encoding(ds)

    zarr_path = config.final_output_dir / config.output["zarr_name"]
    ds.to_zarr(zarr_path, mode="w", consolidated=True)
    print(f"wrote {zarr_path}")

    nc_path = config.final_output_dir / config.output["netcdf_name"]
    complevel = config.output["netcdf_compression_level"]
    encoding = {var: {"zlib": True, "complevel": complevel} for var in ds.data_vars}
    ds.to_netcdf(nc_path, engine="netcdf4", encoding=encoding)
    print(f"wrote {nc_path}")
