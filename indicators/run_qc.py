"""Script to create a slurm job to run the qc.py script on indicator outputs.

python run_qc.py \
    --qc_script /beegfs/CMIP6/kmredilla/cmip6-utils/indicators/qc.py \
    --in_dir /beegfs/CMIP6/arctic-cmip6/CMIP6_common_regrid/ \
    --out_dir /beegfs/CMIP6/kmredilla/cmip6_indicators/netcdf/ \
    --slurm_dir /beegfs/CMIP6/kmredilla/cmip6_indicators/slurm
"""

import argparse
import logging
from pathlib import Path
from slurm import (
    make_sbatch_head,
    submit_sbatch,
)

tmp_qc_sbatch_fn = "indicators_qc.slurm"


def parse_args():
    """Parse some command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qc_script",
        type=str,
        help="Path to qc.py script",
        required=True,
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        help="Path to directory containing inputs for indicators",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to output directory for indicators",
        required=True,
    )
    parser.add_argument(
        "--slurm_dir",
        type=str,
        help="Path to directory where slurm scripts and logs will be written.",
        required=True,
    )
    args = parser.parse_args()

    return (
        args.qc_script,
        args.in_dir,
        args.out_dir,
        Path(args.slurm_dir),
    )


def write_sbatch_run_qc(
    in_dir,
    out_dir,
    slurm_dir,
    qc_script,
):
    """Write the sbatch file for the indicators QC."""
    # create the sbatch file
    sbatch_path = slurm_dir.joinpath(tmp_qc_sbatch_fn)
    sbatch_out_fp = slurm_dir.joinpath(sbatch_path.name.replace(".slurm", "_%j.out"))

    sbatch_head_kwargs = {
        "sbatch_out_fp": sbatch_out_fp,
        "partition": f"t2small",
    }
    sbatch_head = make_sbatch_head(**sbatch_head_kwargs)

    pycommands = "\n"
    pycommands += f"python {qc_script} --in_dir {in_dir} --out_dir {out_dir}\n"

    pycommands += "\n\n"

    pycommands += f"echo End indicators QC && date\n" "echo Job Completed"
    commands = sbatch_head + pycommands

    with open(sbatch_path, "w") as f:
        f.write(commands)
        logging.info(f"sbatch file written to {sbatch_path}")

    return sbatch_path


if __name__ == "__main__":

    (
        qc_script,
        in_dir,
        out_dir,
        slurm_dir,
    ) = parse_args()

    # clear preexisting slurm output files
    for file in slurm_dir.glob(tmp_qc_sbatch_fn.replace(".slurm", "*.out")):
        file.unlink()

    sbatch_kwargs = {
        "in_dir": in_dir,
        "out_dir": out_dir,
        "slurm_dir": slurm_dir,
        "qc_script": qc_script,
    }
    sbatch_path = write_sbatch_run_qc(**sbatch_kwargs)

    job_id = [submit_sbatch(sbatch_path)]
    print(job_id)
