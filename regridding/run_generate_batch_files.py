"""Script to build a slurm file that runs generate_batch_files.py."""

import argparse
from pathlib import Path
import subprocess


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conda_init_script",
        type=str,
        help="Path to script that initiates conda",
        required=True,
    )
    parser.add_argument(
        "--conda_env_name",
        type=str,
        help="Name of conda environment to activate",
        required=True,
    )
    parser.add_argument(
        "--generate_batch_files_script",
        type=str,
        help="Path to script that generates batch files",
        required=True,
    )
    parser.add_argument(
        "--cmip6_directory",
        type=str,
        help="Path to directory where CMIP6 files are stored",
        required=True,
    )
    parser.add_argument(
        "--regrid_batch_dir",
        type=str,
        help="Path to directory where batch files are written",
        required=True,
    )
    parser.add_argument(
        "--vars",
        type=str,
        help="list of variables used in generating batch files",
        required=True,
    )
    parser.add_argument(
        "--freqs",
        type=str,
        help="list of frequencies (mon or day) used in generating batch files",
        required=True,
    )
    parser.add_argument(
        "--models",
        type=str,
        help="list of models used in generating batch files",
        required=True,
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="list of scenarios used in generating batch files",
        required=True,
    )

    args = parser.parse_args()

    return (
        Path(args.conda_init_script),
        args.conda_env_name,
        Path(args.generate_batch_files_script),
        Path(args.cmip6_directory),
        Path(args.regrid_batch_dir),
        args.vars,
        args.freqs,
        args.models,
        args.scenarios,
    )


def submit_sbatch(sbatch_fp):
    """Submit a script to slurm via sbatch

    Args:
        sbatch_fp (pathlib.PosixPath): path to .slurm script to submit

    Returns:
        job id for submitted job
    """
    out = subprocess.check_output(["sbatch", str(sbatch_fp)])
    job_id = out.decode().replace("\n", "").split(" ")[-1]

    return job_id


if __name__ == "__main__":

    (
        conda_init_script,
        conda_env_name,
        generate_batch_files_script,
        cmip6_directory,
        regrid_batch_dir,
        vars,
        freqs,
        models,
        scenarios,
    ) = parse_args()

    regrid_batch_dir.mkdir(exist_ok=True, parents=True)
    slurm_dir = regrid_batch_dir.parent.joinpath("slurm")
    slurm_dir.mkdir(exist_ok=True)
    generate_batch_files_sbatch_fp = slurm_dir.joinpath("generate_batch_files.slurm")
    generate_batch_files_sbatch_out_fp = str(generate_batch_files_sbatch_fp).replace(
        ".slurm", "_%j.out"
    )

    sbatch_text = (
        "#!/bin/sh\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --cpus-per-task=24\n"
        f"#SBATCH -p t2small\n"
        f"#SBATCH --time=04:00:00\n"
        f"#SBATCH --output {generate_batch_files_sbatch_out_fp}\n"
        # print start time
        "echo Start slurm && date\n"
        # prepare shell for using activate
        f"source {conda_init_script}\n"
        f"conda activate {conda_env_name}\n"
        # run the generate batch files script
        f"python {generate_batch_files_script} --cmip6_directory '{cmip6_directory}' --regrid_batch_dir '{regrid_batch_dir}' --vars '{vars}' --freqs '{freqs}' --models '{models}' --scenarios '{scenarios}' \n"
    )

    # save the sbatch text as a new slurm file in the repo directory
    with open(generate_batch_files_sbatch_fp, "w") as f:
        f.write(sbatch_text)

    job_id = submit_sbatch(generate_batch_files_sbatch_fp)

    # print the job_id to stdout
    # there is no way to set an env var for the parent shell, so the only ways to
    # directly pass job ids are through stdout or a temp file
    print(job_id)
