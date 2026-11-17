# patch_31d: QDM ranking-window fix (issue #125)

## What was wrong

QDM is trained with `group="time.dayofyear", window=31` (`train_qm.py`), i.e. the
adjustment factors for a given day of year come from a ±15-day window across all training
years. In xclim 0.47.0, `QuantileDeltaMapping.adjust` ranks each simulated value against
the **exact day of year only** (`main_only=True`, hard-coded in
`xclim/sdba/_adjustment.py::qdm_adjust`), so the percentile used to look up the adjustment
factor comes from ~50–86 values instead of the ~1,550–2,666 the factors were built from.
The successor library xsdba exposes the consistent behaviour as `rank_window=True`.

## The fix

`qdm_window_patch.py` re-defines xclim's `qdm_adjust` (same `@map_blocks` decorator, same
code) with `main_only` taken from `config.QDM_RANK_WINDOWED` and rebinds
`xclim.sdba.adjustment.qdm_adjust`. The conda environment is untouched; the patch refuses
to run if the installed xclim is not the version it was written against. Training is
unchanged, so existing `trained_datasets/` are reused.

Switches:

* `config.QDM_RANK_WINDOWED = True` (default) / `False` (legacy behaviour).
* `bias_adjust.py --rank_main_only true|false` overrides config for A/B tests.

Every output store records which mode produced it: global attrs `qdm_rank_main_only`,
`qdm_rank_window_days`, `qdm_nquantiles`, `xclim_version`, `cmip6_utils_commit`,
`bias_adjust_version`, `bias_adjust_run_label`, plus the squeeze counts/limits applied.

## Other changes made at the same time (bias_adjust.py)

* Adjustment chunks are `time=-1` with a spatial edge from `config.ADJUST_CHUNK_SIZE`
  (15 at both resolutions, inferred from the grid shape) — windowed ranking needs ~60×
  more memory per block than exact-day ranking (~2.8 GB for a 15×15 SSP block); the
  trained store is rechunked to match. Outputs are bit-identical across chunk sizes and
  worker layouts (tested at 12 km and 4 km).
* Measured defaults (`config.DASK_ADJUST`, `config.SLURM_BIAS_ADJUST`): 12 km 4 workers ×
  4 threads on a 120G t2small job (~15 min/SSP under load); 4 km 6 × 4 on a 500G analysis
  job (~43 min/SSP). 4 km does not fit on 128 GB nodes.
* **Untrained cells stay NaN.** `main` (#112) filled every QDM NaN with the raw sim
  value, which also put unadjusted GCM values into cells whose trained `af` is all-NaN
  (no reference data; 3,439 such cells with finite sim at 12 km). Now NaN = sim NaN or
  untrained cell; other QDM-produced NaN are still filled from sim. The count is recorded
  in the `n_cells_without_trained_qdm` attr.
* **Compute once.** The adjusted array is gathered to the client in a single distributed
  compute; the dask cluster is then closed and the squeeze (clip) runs in memory. The
  previous code recomputed the whole QDM graph for every `.compute()` in the squeeze
  section (3× for pr, ~6× for dtr). The write path is unchanged: synchronous `to_zarr`
  from in-memory data to `<store>.writing`, rename, then read-back validation.
* Output standardised to dims `(time, y, x)`, `float32`, chunks `(365, 100, 100)`.
* hurs/hursmin are clipped to [0, 100] here (previously a separate `squeeze_hurs.py` step).
* Per-worker memory limit derived from `SLURM_MEM_PER_NODE` / node memory and the result
  size (`config.DASK_ADJUST`); overrides: `--n_workers --threads_per_worker --memory_limit`.
* Test helpers: `--isel "y=a:b,x=c:d"` spatial subset, `--run_label`, `--worker_dir_local`.
* The job log ends with a `RUN_SUMMARY {json}` line (timings per phase, cluster/worker
  peak memory, chunk-file counts after write and rename).
* `run_bias_adjust.py`: `--resolution`, `--time_limit`, `--cpus_per_task`, `--mem`,
  `--extra_args`, `--exclude`, `--skip_complete`, `--dry_run`; `#SBATCH` headers use full
  option names and set `--cpus-per-task` so shared partitions are not oversubscribed.

## Verification tools

* `compare_rank_modes.py pixel` — proves on a tiny subset that xclim's `sim_q` equals an
  independent hand-computed rank within the exact-DOY population (legacy) and within the
  ±15-day population (patched).
* `compare_rank_modes.py summary` — quantile/mean/max/wet-day difference maps and stats
  between a legacy and a windowed store.
* `check_adjusted_complete.py` — completeness (every chunk present, time coord, attrs) and
  NaN-mask equality with the expected mask (sim NaN ∪ untrained cells, `--train_dirs`);
  `--hash` for bitwise run-to-run comparison. The hash is over **decoded** chunk bytes:
  multithreaded Blosc writes identical data to differently-ordered compressed files.
* `summarize_runs.py` — collects the `RUN_SUMMARY` lines of a set of job logs into a CSV.

## derived/ (tasmin = tasmax − dtr)

* `difference.py` reads inputs in their native zarr chunks (the old fixed (365, 150, 150)
  straddled the new (365, 100, 100) chunks), keeps the minuend's encoding through the
  tasmin floor (`.where()` dropped it → zarr default lz4), merges global attrs by value
  (list-valued attrs broke the old set intersection) without per-store provenance, and
  records `minuend_store`, `subtrahend_store`, `squeeze_lower_limit`, `squeeze_count_below`.
* `run_cmip6_difference.py`: full-name `#SBATCH` options, `--cpus_per_task`, `--mem`,
  `--time_limit`, `--array_limit`, `--dry_run`, node-local numba/matplotlib caches, real
  exit status. Submit from a writable directory (or `sbatch --chdir=/tmp`): dask creates
  `dask-worker-space` in the working directory.
