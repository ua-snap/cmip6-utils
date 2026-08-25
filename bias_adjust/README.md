# Bias Adjustment

Bias-adjusts regridded CMIP6 data against ERA5 using quantile mapping. This is one half of the statistical downscaling process (alongside `downscaling/`, which handles ERA5 prep and target grids). Inputs are converted to zarr, an adjustment is trained per model/variable on historical data, applied across all scenarios, then the adjusted output is post-processed and QC'd.

## Components

- `config.py`, `luts.py`, `utils.py` — shared constants, lookup tables, and path/validation helpers
- `netcdf_to_zarr.py` / `run_cmip6_netcdf_to_zarr.py` / `run_era5_netcdf_to_zarr.py` — convert CMIP6/ERA5 NetCDF inputs to zarr
- `train_qm.py` / `run_train_qm.py` — train a quantile-mapping adjustment per model/variable, using only "historical" CMIP6 data against ERA5 (assumes matching time series lengths)
- `bias_adjust.py` / `run_bias_adjust.py` — apply a trained adjustment to a given model/scenario/variable
- `generate_doy_summaries.py` — build day-of-year summary stats for QC of adjusted output
- `squeeze_hurs.py` — clip adjusted `hurs`/`hursmin` output to the valid `[0, 100]` range (while preserving NaNs), writing squeezed copies alongside a summary CSV
- `run_detrend_profiling.sh` — ad hoc comparison of detrending configs across regions, used during method selection

## Running

1. Convert inputs to zarr: `run_cmip6_netcdf_to_zarr.py`, `run_era5_netcdf_to_zarr.py`.
2. Train adjustments on historical data: `run_train_qm.py`.
3. Apply adjustments across all scenarios: `run_bias_adjust.py`.
4. QC: `generate_doy_summaries.py`, then `squeeze_hurs.py` if `hurs`/`hursmin` were among the adjusted variables.

Each `run_*.py` script builds and submits Slurm jobs over a suite of models/scenarios/variables — see each script's docstring for a full example invocation.
