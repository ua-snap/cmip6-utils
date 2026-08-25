# Derived variables

Computes variables that aren't climate indicators but are derived from other variables — currently daily temperature range (DTR), and generic differencing used to derive things like `tasmin` from `tasmax` and `dtr`.

## Components

- `config.py` — shared constants
- `dtr.py` / `run_cmip6_dtr.py` / `run_wrf_era5_dtr.py` — compute DTR from tmax/tmin. Works on any gridded daily data openable with `xarray.open_mfdataset` in a flat directory structure, not just CMIP6 — the `wrf_era5` variant runs it on WRF-downscaled ERA5 data instead.
- `difference.py` / `run_cmip6_difference.py` / `run_wrf_era5_difference.py` — compute the difference between two zarr datasets (e.g. `tasmin = tasmax - dtr`)
- `dtr_qc.ipynb` — QC notebook for DTR output

The `run_*.py` scripts build and submit Slurm jobs over a suite of models/scenarios — see each script's docstring for a full example invocation.
