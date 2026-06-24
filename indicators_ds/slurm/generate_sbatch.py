"""Generate submit_fragments.sbatch, submit_combine.sbatch, and
submit_qc.sbatch from the given config -- so SLURM resource requests
(partition, mem, cpus, time, array concurrency) are tunable in one place
(the config YAML) without ever touching these scripts by hand.

Run this (or just run_pipeline.sh, which calls it for you, for the
fragments/combine scripts) after build_job_list.py and any time the
config's `slurm:` section changes:
    python slurm/generate_sbatch.py --config ../config_12km.yaml

submit_qc.sbatch is not part of run_pipeline.sh -- qc.py is run manually
after a pipeline run, via `sbatch slurm/submit_qc.sbatch`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INDICATORS_DS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(INDICATORS_DS_DIR))

from config import DEFAULT_CONFIG_PATH, load_config

CONDA_HOOK = (
    'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"\n'
    "conda activate {conda_env}\n"
)

FRAGMENTS_TEMPLATE = """#!/bin/sh
#SBATCH --job-name=indicators_ds_fragments
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --array=0-{max_index}%{concurrency}
#SBATCH --output={logs_dir}/fragment_%A_%a.out

set -e
echo Start slurm && date
{conda_hook}
python {indicators_ds_dir}/compute_fragment.py --job-index $SLURM_ARRAY_TASK_ID --config {config_path}
echo End slurm && date
"""

COMBINE_TEMPLATE = """#!/bin/sh
#SBATCH --job-name=indicators_ds_combine
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={logs_dir}/combine_%j.out

set -e
echo Start slurm && date
{conda_hook}
python {indicators_ds_dir}/combine.py --config {config_path}
echo End slurm && date
"""

QC_TEMPLATE = """#!/bin/sh
#SBATCH --job-name=indicators_ds_qc
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={logs_dir}/qc_%j.out

set -e
echo Start slurm && date
{conda_hook}
python {indicators_ds_dir}/qc.py --config {config_path}
echo End slurm && date
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    with open(config.job_list_path) as f:
        n_jobs = len(json.load(f))
    max_index = n_jobs - 1

    conda_hook = CONDA_HOOK.format(conda_env=config.slurm["conda_env"])
    frag_cfg = config.slurm["fragments"]
    comb_cfg = config.slurm["combine"]

    frag_script = FRAGMENTS_TEMPLATE.format(
        partition=config.slurm["partition"],
        cpus=frag_cfg["cpus_per_task"],
        mem=frag_cfg["mem"],
        time=frag_cfg["time"],
        max_index=max_index,
        concurrency=frag_cfg["array_concurrency"],
        logs_dir=config.logs_dir,
        conda_hook=conda_hook,
        indicators_ds_dir=INDICATORS_DS_DIR,
        config_path=config_path,
    )
    frag_path = Path(__file__).resolve().parent / "submit_fragments.sbatch"
    frag_path.write_text(frag_script)
    print(f"wrote {frag_path} (array=0-{max_index}%{frag_cfg['array_concurrency']}, {n_jobs} jobs)")

    comb_script = COMBINE_TEMPLATE.format(
        partition=config.slurm["partition"],
        cpus=comb_cfg["cpus_per_task"],
        mem=comb_cfg["mem"],
        time=comb_cfg["time"],
        logs_dir=config.logs_dir,
        conda_hook=conda_hook,
        indicators_ds_dir=INDICATORS_DS_DIR,
        config_path=config_path,
    )
    comb_path = Path(__file__).resolve().parent / "submit_combine.sbatch"
    comb_path.write_text(comb_script)
    print(f"wrote {comb_path}")

    qc_cfg = config.slurm["qc"]
    qc_script = QC_TEMPLATE.format(
        partition=config.slurm["partition"],
        cpus=qc_cfg["cpus_per_task"],
        mem=qc_cfg["mem"],
        time=qc_cfg["time"],
        logs_dir=config.logs_dir,
        conda_hook=conda_hook,
        indicators_ds_dir=INDICATORS_DS_DIR,
        config_path=config_path,
    )
    qc_path = Path(__file__).resolve().parent / "submit_qc.sbatch"
    qc_path.write_text(qc_script)
    print(f"wrote {qc_path}")


if __name__ == "__main__":
    main()
