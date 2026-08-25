# SNAP CMIP6 Utilities

This is SNAP's repo for working with CMIP6 climate model data. It covers the full pipeline from mirroring raw data off of ESGF through producing regridded, downscaled, bias-adjusted, and derived climate products and indicators ready for release.

Pipeline stages run on UAF's Chinook HPC, often via Prefect (see https://github.com/ua-snap/prefect) and expect the data paths described in each subdirectory's README. 

Note that stage directories may include stale code that is unused in the production pipeline. For a stripped down version of the codebase utilities, see https://github.com/ua-snap/cmip6-downscaling and/or the official code release here: https://zenodo.org/records/22086730

## Setup

Create the conda environment from `environment.yml`:

```sh
conda env create -f environment.yml
conda activate cmip6-utils
```

(Python 3.9. Some subdirectories also have their own `conda_init.sh` for use in Slurm jobs — see individual READMEs.)

## Pipelines


### Getting the data

- [`transfers/`](transfers/README.md) — Mirror raw CMIP6 output from ESGF (LLNL node) to SNAP's Arctic Climate Data Node (ACDN) via Globus. Note that some elements of this stage will likely break with the introduction of ESGF next-generation architecture (ESGF-NG) in late 2026.

### Coarse Regridding and Computing Indicators

- [`regridding/`](regridding/README.md) — Regrid mirrored CMIP6 data to a common grid and crop to a pan-Arctic domain (50N-90N). Handles land-only variables by using the land area fraction variable (`sftlf`), if available.
- [`indicators/`](indicators/README.md) — Compute climate indicators (via `xclim`) from the regridded CMIP6 data.

### Statistical Downscaling

- [`downscaling/`](downscaling/README.md) and [`bias_adjust/`](bias_adjust/README.md) — Downscale CMIP6 data to a higher-resolution ERA5 target grid using quantile-mapping bias adjustment. Includes cascade regridding setup, which uses utilities from [`regridding/`](regridding/).
- [`derived/`](derived/README.md) — Compute derived (non-indicator) variables from downscaled data, such as daily temperature range.
- [`reformat/`](reformat/README.md) — Reformat final outputs (variable order, ensemble dimension, attributes) for public data release.


## Other

- [`archive/`](/archive) contains preliminary, superseded work from an earlier CDS/ECMWF-based approach to acquiring CMIP6 data, predating the ESGF/Globus pipeline above. It is not part of the active pipeline and is kept for reference only.