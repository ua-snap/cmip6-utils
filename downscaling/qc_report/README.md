# Downscaled CMIP6 QC reports (PDF)

Script-based replacement for the production QC notebooks in `../notebooks/`
(`downscaled_{pr,tasmax,tasmin,dtr,hurs,hursmin,sfcWind}.ipynb`). It runs the same analysis
much faster, adds a new-vs-previous comparison, and writes PDFs that collaborators can open
without Jupyter or GitHub.

## What a report contains

There is one PDF per variable × model × resolution: `qc_<var>_<model>_<res>.pdf`.

**A. New vs previous outputs.** This section leads the document.
- A table per scenario with the domain-average |Δ| of:
  - the long-term mean;
  - the median;
  - the 99.9th percentile;
  - the daily values.
- The share of identical cell-days, and the number of cells whose NaN pattern differs.
- Quantile-difference profiles (new − previous, per grid cell, at quantile levels 0.0001 to
  0.9999). These show that the bulk of the distribution is essentially unchanged and that
  differences grow in the tails.
- Maps of the change in the long-term mean and in the 99.9th percentile.

**B. Standard QC of the new outputs**, per scenario. This is the same analysis as the notebooks:

| Variable | Checks |
|---|---|
| pr, tasmax, tasmin, sfcWind | <ul><li>Threshold check (pr > 400 mm d⁻¹, tasmax > 35 °C, tasmin < −55 °C, sfcWind > 31.3 m s⁻¹): a map of exceedance days, a histogram of the exceeding values, and counts for ERA5, un-adjusted and downscaled data.</li><li>The 6 pixels with the most exceedances: ECDFs in a 31-day window and day-of-year means.</li><li>ECDFs at 2 sets of hand-picked places and at 5 random locations.</li></ul> |
| hurs, hursmin | <ul><li>Out-of-range check [0, 100] %.</li><li>ECDFs for the fire season (May–Sep) and the full year at hand-picked places and random locations.</li><li>Day-of-year means.</li></ul> |
| dtr | <ul><li>Min, mean and max maps at each processing stage: source, first regrid, second regrid, final regrid, downscaled and ERA5.</li><li>The 10 most extreme pixel-days, with tasmax and tasmin on those days and ECDFs at those pixels.</li></ul> |

Each ECDF and day-of-year panel compares the same five sources as the notebooks:
- ERA5;
- historical, un-adjusted and downscaled;
- scenario, un-adjusted and downscaled.

`qc_summary_<res>.pdf` gathers the comparison metrics and threshold counts for every
model and scenario of a resolution. It replaces `counts_outside_thresholds.py`.

## Why it is fast

`qc_stats.py` reads each zarr store once, in (all time × 100 × 100) spatial blocks, using a
thread pool. Every map, quantile, count, histogram and point series is computed in that one
pass. The only other reads are single pixels: the data-dependent "top" pixels and ERA5
point series. ERA5 statistics are cached and shared by all models.

`qc_report.py` then reads only the small NetCDF/JSON files and draws the PDF with
matplotlib. No LaTeX, notebook or extra packages are needed beyond the `cmip6-utils` env.

At 12 km, one variable × model with all scenarios takes about 2–4 minutes for the
statistics and about 10 s for the PDF. The notebooks took hours.

## Usage

```sh
conda activate cmip6-utils
cd downscaling/qc_report
python run_qc.py --res 12km                     # all variables and models: Slurm array + summary job
python run_qc.py --res 12km --vars pr --models MIROC6
python run_qc.py --res 4km --skip_done          # only pairs without a PDF yet

# single pair, interactively on a compute node
python qc_stats.py --res 12km --var pr --model MIROC6
python qc_report.py model --res 12km --var pr --model MIROC6 [--png_dir preview/]
python qc_report.py summary --res 12km
```

Outputs go to `<output_root>/<res>/`:
- `pdf/`: the reports;
- `data/`: per-scenario statistics (`.nc` and `.json`);
- `slurm/`: job scripts and logs.

## Configuration

Everything editable is in [`config.yaml`](config.yaml):
- paths per resolution and variable group (new, previous, un-adjusted CMIP6, ERA5, dtr upstream stages);
- per-variable thresholds, units and ECDF mode;
- hand-picked places, the random-location seed and the quantile levels;
- Slurm resources;
- the report text.

To add the 4 km original variables, fill in the commented `original_vars` group under `4km`.

## Differences from the notebooks

- **Day-of-year windows.** ECDF windows are a true circular ±15-day window: 31 days, wrapping
  across the new year. The notebooks used 30 days without wrap-around.
- **Random locations.** These are seeded (`random_seed`), so they are the same in every
  report and every re-run.
- **Combined figures.** Day-of-year means for 6 pixels share one page, and the 5 random
  locations share one page with a locator map.
- **Counts.** Counts that the notebooks printed (exceedances, top-pixel coordinates, dtr
  extremes) are shown as tables. Top pixels also get lat/lon.
