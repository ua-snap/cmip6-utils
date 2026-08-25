# Downscaling

Scripts and notebooks supporting the downscaling of CMIP6 data onto a higher-resolution ERA5 grid. Regridding (`regridding/`) and bias-adjustment (`bias_adjust/`) do the heavy lifting; this folder covers everything else — ERA5 prep, cascade-regridding target grids, and post-processing.

## Which CMIP6 data?

Multiple variables (see https://github.com/ua-snap/prefect/tree/main/downscaling#supported-variables), regridded to a 4km or 12km grid in EPSG:3338, for the 13-model SNAP CMIP6 ensemble — selected for skill in Alaska and the Arctic using the [GCMEval tool](https://gcmeval.met.no/) ([paper](https://doi.org/10.1016/j.cliser.2020.100167)).

## `notebooks/`

Exploratory analysis behind the bias-adjustment method: how CMIP6 data is regridded and bias-adjusted with `xclim`, and how the final parameters were chosen. We used [Lavoie et al. 2024](https://doi.org/10.1038/s41597-023-02855-z) as a template — detrended quantile mapping for daily max temperature, daily temperature range, and precipitation — but found quantile delta mapping performed comparably and better for things like seasonal precipitation cycles (see `discrete_comparison_qm_method.ipynb`).

Depends on a local `baeda.py` module (bias-adjustment exploratory data analysis); run a Jupyter server from a compute node inside `notebooks/`. Many paths are hardcoded — these notebooks are now mainly a record of past decisions, not a runnable pipeline.

## Other utilities

- `prep_era5_variables.py` — rename ERA5 variable/file/directory names to what the downscaling flow expects; `--celsius-to-kelvin` converts temperature units
- `make_intermediate_target_grid_file.py` / `make_final_target_grid_file.py` — build the intermediate and final target grids used in cascade regridding (see `default_target_grid_files/README.md` for the grids currently in production use)
- `regrid_sftlf_to_target.py` — regrid a land-fraction (`sftlf`) file onto an intermediate/target grid, for cascade-regridding land masks
- `drop_variable_from_zarr_collection.py` — drop a scalar variable from a directory of zarr stores in place
- `round_negative_precip.py` — post-process downscaled precip, rounding small negative values (< -0.5mm) up to zero (warns if larger negatives remain, which would signal a real issue)
- `tests/` — sanity checks against the full downscaled data corpus (bounds checks, tasmin/tasmax consistency); run individually or together, ~10 min each on a Chinook compute node
