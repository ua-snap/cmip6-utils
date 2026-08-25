# CMIP6 Transfers

Mirrors raw [CMIP6](https://www.wcrp-climate.org/wgcm-cmip) model output from LLNL's [ESGF](https://esgf.llnl.gov/) node to SNAP's Arctic Climate Data Node (ACDN) via [Globus](https://www.globus.org/), preserving the ESGF directory structure:

```
<activity>/<institution>/<model>/<scenario>/<variant>/<frequency>/<variable>/<grid type>/<version>/
```

under `/beegfs/CMIP6/arctic-cmip6/CMIP6/` on Chinook. Requires accounts on both Chinook and Globus.

**Note:** parts of this pipeline (the LLNL ESGF holdings audit) will likely need rework when ESGF-NG (the ESGF next-generation architecture) launches in late 2026.

## Target dataset

The models/scenarios/variables/frequencies to mirror are configured in `config.py`, based on collaborator input and subject to change. Only a single variant per model is transferred, so part of the pipeline audits current ESGF holdings to identify the variant with the best coverage across the requested combinations — the transferred data will be a subset of what's requested, since not everything requested exists in ESGF.

## Components

- `config.py` — requested model/scenario/variable/frequency combinations
- `esgf_holdings.py` / `esgf_holdings_e3sm.py` — audit ESGF holdings against `config.py`, writing `llnl_esgf_holdings*.csv` (add `--wrf` for the non-standard subdaily variables/frequencies used in WRF; not available for E3SM)
- `select_variants.ipynb` — explore per-model variant coverage to pick which variant to transfer
- `generate_manifest.py` — build the manifest of files to transfer (`llnl_manifest*.csv`)
- `generate_batch_files.py` — split the manifest into `<source> <destination>` batch files in `batch_files/`
- `batch_transfer.py` — submit Globus transfer jobs for all batch files
- `transfer.py` — older, non-SDK transfer script for ad hoc subsets
- `quick_ls.py` — run `ls` on a Globus path
- `remove_old_versions.py` — remove superseded ESGF dataset versions from the ACDN
- `holdings_summary.ipynb` / `holdings_summary_wrf.ipynb` — summary tables from the audit results
- `tests/` + `tests.slurm` — verify every manifest file is present on the ACDN and opens with `xarray`
- `utils.py`, `conda_init.sh` — shared helpers / Slurm shell init

## Running

1. Log into Globus and set `CLIENT_ID` — see Globus setup below.
2. Copy `conda_init.sh` to your home directory (assumes `miniconda3` is installed there).
3. Set environment variables as full paths (Slurm doesn't expand `~/`): `TEST_OUT_DIR` (test output location), `CONDA_INIT` (path to the copied script), `SLURM_EMAIL` (failure notifications).
4. Audit holdings:
   ```sh
   python esgf_holdings.py --node llnl --ncpus 24 | tee esgf_holdings_output.txt
   python esgf_holdings_e3sm.py --node llnl --ncpus 24 | tee esgf_holdings_e3sm_output.txt
   ```
5. Build the manifest: `python generate_manifest.py --node llnl`
6. Build batch files: `python generate_batch_files.py`
7. Run the transfer: `python batch_transfer.py --node llnl`
8. Verify: `sbatch tests.slurm` (requires `module load slurm` in `~/.bashrc` if not already loaded)

Add `--wrf` to steps 4-6 to additionally handle the WRF-only subdaily variables.

Expect Globus to prompt for additional consent/login the first time a step touches a new endpoint — follow the printed URL or `globus session consent` command.

## Globus setup

```sh
globus login
export CLIENT_ID=a316babe-5447-43b0-a82e-4fc86c91b71a
```

`CLIENT_ID` should be the same for all users. If you hit client errors, create your own app per the [Globus SDK tutorial](https://globus-sdk-python.readthedocs.io/en/stable/tutorial.html) and use its client ID instead.
