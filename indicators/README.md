# Indicators

Computes climate indicators (via [xclim](https://xclim.readthedocs.io/)) from regridded CMIP6 data (`regridding/`).

## Components

- `config.py` — paths and constants
- `luts.py` — lookup tables mapping variables to the indicators that depend on them (and vice versa), used to avoid opening the same dataset twice
- `indicators.py` — computes a set of indicators that share a source variable, for a given model and scenario
- `slurm.py` — builds and submits the indicator-computation Slurm jobs, and writes a `qc/qc.csv` to-do list of outputs for QC
- `qc.py` / `run_qc.py` — numeric QC checks driven by `qc/qc.csv`, with errors written to `qc/qc_error.txt`
- `visual_qc.ipynb` / `run_visual_qc.py` — visual QC notebook, run as a Slurm job
- `process_indicators.ipynb` — reference/orchestration notebook for running the pipeline across models and scenarios
- `shp/` — shapefiles used in indicator computation/QC

## Running

```sh
python indicators.py --indicators rx1day --model CESM2 --scenario ssp585 \
    --input_dir /beegfs/CMIP6/arctic-cmip6/regrid --out_dir <out_dir>
```

In practice this is run at scale via `slurm.py` across all models, scenarios, and indicators (see `process_indicators.ipynb`). Follow with `run_qc.py` and `run_visual_qc.py` for QC.
