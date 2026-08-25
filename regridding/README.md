# Regridding

Regrids mirrored CMIP6 data (`transfers/`) onto a common grid and crops it to a pan-Arctic domain (50N-90N). Historically the common grid has been NCAR's CESM2 native grid, also shared by a few other chosen models (TaiESM1, NorESM2-MM) — see `explore_grids.ipynb` for the selection rationale.

Land-only variables are regridded alongside the source model's land fraction (`sftlf`) file where one is available, so masking stays consistent downstream; per-model `sftlf` paths are hardcoded in `config.py` (not all models have one).

This pipeline is orchestrated via Slurm jobs launched from a [Prefect](https://github.com/ua-snap/prefect) flow (separate repo) rather than run directly: `config.py` duplicates `transfers/config.py` for Prefect's use, and `slurm.py` prints the submitted job ID to stdout for the flow to parse.

## Components

- `config.py` — grid/model constants, including per-model `sftlf` paths
- `generate_batch_files.py` / `run_generate_batch_files.py` — group files needing regridding into batches by common grid, frequency, model, scenario, and variable
- `regrid.py` — regrids a batch of files to the target grid (crops to 50N+ first)
- `run_regrid_again.py` — builds batch files for a second regridding pass over data that's already been regridded once (e.g. onto a downscaling target grid)
- `qc.py` / `qc.ipynb` / `run_qc.py` — QC checks on regridded output
- `explore_grids.ipynb`, `explore_regridding.ipynb`, `explore_regridding_nan.ipynb`, `explore_regrid.ipynb` — exploratory notebooks (grid selection, `xesmf` behavior, NaN handling)
- `slurm.py`, `conda_init.sh` — Slurm job helpers

## Running

1. Copy `conda_init.sh` to your home directory (skip if already done for `transfers/`).
2. Set environment variables as full paths (Slurm doesn't expand `~/`): `SCRATCH_DIR` (scratch output location), `PROJECT_DIR` (path to this repo, used in place of `PYTHONPATH`), `CONDA_INIT`, `SLURM_EMAIL`.
3. Build batches: `python generate_batch_files.py` (reads grid info for every mirrored file, can take a while).
4. Regrid via the Prefect flow (see the `prefect` repo), which submits `regrid.py` as a Slurm job per batch.
5. Spot-check output: `qc.ipynb`.
6. `python run_qc.py ...` to build and submit a Slurm job running the full QC suite.
7. Copy regridded data off scratch to permanent storage, ideally from a `screen` session (iterate by model if `rsync` errors on the whole tree):
   ```sh
   rsync -av $SCRATCH_DIR/regrid /beegfs/CMIP6/arctic-cmip6/
   ```
8. Fix permissions (directories 755, files 644):
   ```sh
   find /beegfs/CMIP6/arctic-cmip6/regrid -type d -exec chmod 755 {} \;
   find /beegfs/CMIP6/arctic-cmip6/regrid -type f -exec chmod 644 {} \;
   ```
