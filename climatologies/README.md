# Downscaled CMIP6 Climatologies

Computes climatologies (temporal min/mean/max by month/season and by
configurable historical/future era) from WRF-downscaled, bias-adjusted
CMIP6 data and the WRF-downscaled ERA5 reference, for 13 CMIP6 models plus
a multi-model ensemble mean. Produces a single master output (identical
content in Zarr and NetCDF) with dimensions
`model, scenario, era, period, aggregation, y, x`.


## Quick start

Adjust paths in `config.yaml`, then run:

```sh
conda activate cmip6-utils
cd cmip6-utils/climatologies
bash slurm/run_pipeline.sh
```

That rebuilds the job list from whatever's currently on disk, regenerates
the SLURM scripts from `config.yaml`, and submits two jobs to the
`analysis` partition: a fragment-computation array job, then a combine job
that runs automatically once every fragment task succeeds. Check progress
with `squeue --me`. When it's done, the result is at:

```
${paths.output_root}/output/cmip6_wrf_climatologies.zarr
${paths.output_root}/output/cmip6_wrf_climatologies.nc
```

Then run the QC and  clean up the intermediate file:

```sh
sbatch slurm/submit_qc.sbatch
# check ${paths.output_root}/qc/{nan_checks,calc_checks}.log and
# ${paths.output_root}/qc/delta_maps/ before proceeding
python cleanup_intermediate.py --yes
```


## What it does

Input data lives in three separate directory trees (one each for
sfcWind/hurs/hursmin, the "original" temperature/precip/dtr variables, and
snw), each with an `era5_zarr/` subtree (the WRF-downscaled ERA5 reference)
and an `adjusted/` subtree (WRF-downscaled, bias-adjusted CMIP6, one zarr
store per model/scenario). All of it is daily data on the same 211x282
projected WRF grid.

The pipeline reduces that daily data down to 10 output variables, each
shaped `(Model, Scenario, Era, Period, Aggregation, y, x)`:

| Variable | Meaning |
|---|---|
| `tmin`, `tmax` | daily min/max near-surface air temperature (degC by default -- source data is Kelvin, see "Editing the config" below) |
| `tmean` | derived: daily `(tmax + tmin) / 2` (degC by default) |
| `dtr` | diurnal temperature range (its own bias-adjusted variable, not derived from tmax-tmin; degC by default, though its value is identical in degC and K since it's a difference) |
| `pr` | daily total precipitation |
| `pr_tot` | derived: total precipitation *summed over the period, per year* (see below) |
| `hurs`, `hursmin` | daily mean/min near-surface relative humidity |
| `sfcwind` | daily mean near-surface wind speed |
| `snw` | surface snow water equivalent |

All 10 data variable names are lowercase by convention. The 5 dimension
names (`Model`, `Scenario`, `Era`, `Period`, `Aggregation` in prose
throughout this doc) are likewise lowercase (`model`, `scenario`, `era`,
`period`, `aggregation`) in the actual output files -- this doc keeps the
capitalized form for readability since it matches the original spec, but
don't expect to see it that way if you open the Zarr/NetCDF yourself.

- **Model**: the 13 CMIP6 models, plus `CMIP6-Ensemble` (a multi-model
  mean) and `WRF-ERA5` (the reference; only populated under
  `Scenario=historical`).
- **Scenario**: `historical`, `ssp126`, `ssp245`, `ssp370`, `ssp585`.
- **Era**: configurable date ranges (e.g. `1981-2010`, `2040-2069`) — see
  below.
- **Period**: each calendar month, plus `DJF`/`MAM`/`JJA`/`SON` and the
  two custom seasons `AMJJAS`/`ONDJFM`.
- **Aggregation**: `temporal_min`, `temporal_mean`, `temporal_max`.

Not every model has data for every variable/scenario (e.g. `CESM2` has no
`tmax`/`tmin`/`dtr` data at all; `hursmin` only exists for 6 of 13 models).
Missing combinations are simply `NaN` throughout — this is expected, not a
bug.

## How it works

Four stages, each its own script:

1. **`build_job_list.py`** scans the input directories on disk (it never
   hardcodes which models/scenarios exist) and writes
   `intermediate/job_list.json` — one entry per (variable, model,
   scenario) combination that actually has data.
2. **`compute_fragment.py`** — run once per `job_list.json` entry (as a
   SLURM array task). Loads the *entire* daily time series for that one
   variable/model/scenario into memory, then computes every Era x Period
   x Aggregation value from it in one pass, and writes a small "fragment"
   zarr (`intermediate/fragments/{variable}__{model}__{scenario}.zarr`).
   Two methods are used (see "Design decisions" below): a direct
   day-level reduction for everything except `pr_tot`, and a per-year
   sum-then-reduce method for `pr_tot`.
3. **`combine.py`** assembles every fragment into the full master arrays,
   fills in `NaN` for any (model, scenario) combination with no fragment
   at all, computes the `CMIP6-Ensemble` mean, and attaches all
   attributes (via `metadata.py`).
4. **`write_outputs.py`** writes the combined dataset to both Zarr and
   NetCDF.

`slurm/run_pipeline.sh` runs stage 1, regenerates the SLURM scripts for
stages 2-4 from `config.yaml` (via `slurm/generate_sbatch.py`), and
submits the stage-2 array job followed by a stage-3+4 job chained with
`--dependency=afterok`.

## Editing the config

**`config.yaml` is the single place to change any parameter or any piece
of descriptive text that ends up in the output.** Nothing else in this
repo should need to change for the kinds of edits below — just edit the
YAML and re-run `slurm/run_pipeline.sh`.

- **Eras** — add, remove, or change the `eras:` list (each has a `name`,
  `start_year`, `end_year`).
- **Periods** — the `periods:` list (each has a `name` and `months:` in
  chronological order — order matters for detecting that a period like
  `DJF` crosses a year boundary).
- **Models / ensemble membership** — `models:` is the full model list;
  `ensemble.members:` is the (possibly smaller) subset averaged into
  `CMIP6-Ensemble`.
- **Input paths** — `source_families.*.adjusted_glob` / `.era5_zarr`. Each
  source family also carries its `units` and `long_name`.
- **SLURM resources** — `slurm:` (partition, mem, cpus, time, and the
  array concurrency throttle for the fragment stage).
- **Any text in the output's attributes** — the `metadata:` section
  (dataset title/institution/summary, the per-dimension description
  templates, the methodology note attached to most variables). See
  "Why dataclasses" below for why this is safe to edit freely.
- **Temperature units** — `units.temperature_unit`: `celsius` (default) or
  `kelvin`. Source data is always Kelvin; `celsius` applies a K -> degC
  shift to `tmax`/`tmin`/`tmean`. `dtr` is a temperature *difference*, so
  its stored values never change between unit systems (a 1 K difference
  is the same size as a 1 degC difference) -- only its `units` label
  follows the setting, for consistency with the other temperature
  variables. Changing this requires recomputing the `tmax`/`tmin`/`tmean`
  fragments (the value conversion happens in `compute_fragment.py`, not
  at write time) and re-running `combine.py`.

## Design decisions

A few of these were genuinely ambiguous from the original spec and were
resolved by explicit clarifying questions during development:

- **Aggregation method.** For every variable except `pr_tot`,
  `temporal_min`/`mean`/`max` are computed directly over every individual
  day in the given Period+Era (e.g. "the coldest single April day across
  1981-2010") — not by first collapsing each year to one value and then
  reducing across years. This was chosen because it's the simpler,
  single-stage reduction, and matches how the spec described `pr`
  specifically (separately from `pr_tot`, which does need the two-stage
  treatment).
- **`pr_tot`.** Per the spec, this needs the sum of `pr` over the period,
  computed *per year*, with `temporal_min`/`mean`/`max` then taken across
  those per-year sums (e.g. "the driest vs. wettest April on record").
- **Season-year convention for `DJF`/`ONDJFM`.** These cross a calendar
  year boundary. A season instance is labeled by the year of its *last*
  month (DJF's year = the Jan/Feb year), and it counts toward an era if
  that label-year falls in the era's range — so the boundary December is
  pulled in even though its own calendar year is technically one year
  before era start. This is the standard climatological convention.
- **`CMIP6-Ensemble` and missing data.** The ensemble mean is a `nanmean`
  across whichever configured member models actually have non-NaN data
  for a given slice — only `NaN` if *zero* members have data. Given that
  some variables (e.g. `hursmin`) only have 6 of 13 models, requiring all
  members to be present would make the ensemble almost always `NaN`.
- **Output location.** Intermediate fragments and the final master
  outputs live under `/beegfs/CMIP6/jdpaul3/climatologies/`, not in this
  repo and not under `/import/home` (a shared, mostly-full quota) —
  multi-GB pipeline output doesn't belong in either place.

## QC

`qc.py` validates the *pipeline*, not the source data -- the source data
has already been through its own QC process, and this script does not
re-check whether values are physically plausible. Every check compares
quantities this pipeline itself computed against a mathematical
relationship that must hold if the computation is correct (e.g. `pr_tot`
must equal `pr`'s mean times the period's day-count; the ensemble mean
must equal `nanmean` of its configured members). A violation means *this
pipeline* has a bug.

Run manually after a pipeline run, on a compute node (not the login node
-- it opens the full master output and a couple of variables at a time
can be a few GB):

```sh
sbatch slurm/submit_qc.sbatch
```

Writes to `${paths.output_root}/qc/`:
- `nan_checks.log` -- `pr`/`pr_tot` NaN-mask equality; a coverage-gap
  cross-check against `intermediate/fragments/`.
- `calc_checks.log` -- `min<=mean<=max`; the `pr_tot` exact-identity check
  (`pr_tot.temporal_mean == pr.temporal_mean x days_in_period`, exact on
  any fixed-calendar CMIP6 model, with an era-length-scaled tolerance for
  the wrapping periods `DJF`/`ONDJFM` -- see the file for why); `tmean` vs
  `tmax`/`tmin` bounds; ensemble re-derivation. Ends with a one-line
  `ALL CHECKS PASSED` / `N TOTAL VIOLATIONS` summary.
- `delta_maps/<var>/<var>__<period>.png` -- one PNG per variable x *every*
  configured Period (180 total for the default 10 vars x 18 periods),
  each an 8-panel scenario x era grid of
  `CMIP6-Ensemble[scenario,era] projection - WRF-ERA5[historical baseline]`.
  Doubles as a sanity figure (an obviously-wrong delta pattern usually
  means a sign/unit/scenario-label bug) and as a genuinely useful "does
  the projected change look physically sane" plot.

Once you've reviewed the QC output, run `python cleanup_intermediate.py
--yes` to delete `intermediate/fragments/` (defaults to a dry run without
`--yes`) -- do this *after* QC, not before, since the coverage-gap
cross-check reads fragment filenames.

All of the QC parameters (the baseline era, which scenarios/eras get
delta maps, the tolerance used by the identity checks) live under
`config.yaml`'s `qc:` section, same single-config-location rule as
everything else.

## Why dataclasses in `config.py`

`config.py` defines a handful of `@dataclass`-decorated classes (`Era`,
`Period`, `SourceFamily`, `DerivedVar`, `Config`) that `load_config()`
parses `config.yaml` into. They hold **zero hardcoded values** — every
field's value still comes from `config.yaml`; this is a thin loader, not
a second configuration surface.

The benefit: the same `Config` object gets passed into five different
scripts (`build_job_list.py`, `compute_fragment.py`, `combine.py`,
`metadata.py`, `slurm/generate_sbatch.py`). With a typed object, a typo
like `config.eras[0].start_yera` is caught by your editor before you even
run anything; with a plain dict, the equivalent
`config["eras"][0]["start_yera"]` only fails at runtime, possibly deep
inside a job that takes a while to get there. It also lets a field compute
itself once at load time instead of every caller re-deriving it — e.g.
`Period.wrap` (does this period cross a year boundary, like `DJF`?) is
computed once in `Period.__post_init__` and just read everywhere it's
needed afterward. Finally, `frozen=True` makes the loaded config
read-only, so nothing downstream can accidentally mutate shared
configuration mid-run.

## File reference

| File | Role |
|---|---|
| `config.yaml` | every tunable parameter and every piece of output-facing text |
| `config.py` | loads `config.yaml` into typed objects (no values of its own) |
| `periods.py` | period-membership and season-year ("label year") logic |
| `units.py` | K <-> degC conversion for temperature variables, driven by `config.yaml`'s `units:` section |
| `source_catalog.py` | glob/regex helpers for discovering what's on disk |
| `build_job_list.py` | stage 1 — writes `intermediate/job_list.json` |
| `compute_fragment.py` | stage 2 — one job -> one or two fragment zarr(s) |
| `metadata.py` | builds output attrs by filling `config.yaml` templates (no hardcoded text) |
| `combine.py` | stage 3 — fragments -> master Dataset, ensemble, NaN-fill |
| `write_outputs.py` | stage 4 — master Dataset -> Zarr + NetCDF |
| `qc.py` | validates the pipeline's own calculations + renders delta maps -- run manually, see "QC" above |
| `cleanup_intermediate.py` | deletes `intermediate/fragments/` -- run manually, after `qc.py`, see "QC" above |
| `slurm/generate_sbatch.py` | writes `slurm/submit_fragments.sbatch` / `submit_combine.sbatch` / `submit_qc.sbatch` from `config.yaml`'s `slurm:` section |
| `slurm/run_pipeline.sh` | runs stage 1, regenerates sbatch scripts, submits the SLURM jobs (fragments + combine only -- `qc.py`/`cleanup_intermediate.py` are run separately) |
