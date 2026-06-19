"""Generate submit_fragments.sbatch and submit_combine.sbatch from
config.yaml -- so SLURM resource requests (partition, mem, cpus, time,
array concurrency) are tunable in one place (config.yaml) without ever
touching these scripts by hand.

Run this (or just run_pipeline.sh, which calls it for you) after
build_job_list.py and any time config.yaml's `slurm:` section changes:
    python slurm/generate_sbatch.py [--config ../config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CLIMATOLOGIES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CLIMATOLOGIES_DIR))

from config import DEFAULT_CONFIG_PATH, load_config

CONDA_HOOK = (
    'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"\n'
    "conda activate {conda_env}\n"
)

FRAGMENTS_TEMPLATE = """#!/bin/sh
#SBATCH --job-name=climatology_fragments
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --array=0-{max_index}%{concurrency}
#SBATCH --output={logs_dir}/fragment_%A_%a.out

echo Start slurm && date
{conda_hook}
python {climatologies_dir}/compute_fragment.py --job-index $SLURM_ARRAY_TASK_ID --config {config_path}
echo End slurm && date
"""

COMBINE_TEMPLATE = """#!/bin/sh
#SBATCH --job-name=climatology_combine
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={logs_dir}/combine_%j.out

echo Start slurm && date
{conda_hook}
python {climatologies_dir}/combine.py --config {config_path}
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
        climatologies_dir=CLIMATOLOGIES_DIR,
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
        climatologies_dir=CLIMATOLOGIES_DIR,
        config_path=config_path,
    )
    comb_path = Path(__file__).resolve().parent / "submit_combine.sbatch"
    comb_path.write_text(comb_script)
    print(f"wrote {comb_path}")


if __name__ == "__main__":
    main()
