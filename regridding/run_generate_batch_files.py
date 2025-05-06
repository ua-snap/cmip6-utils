"""Script to build a slurm file that runs generate_batch_files.py."""

import argparse
from pathlib import Path

# project
from slurm import submit_sbatch

batch_file_dir_name = "regrid_batch_files"


def parse_args():
    """Parse some command line arguments.

    Returns
    -------
    conda_env_name : str
        Name of conda environment to activate
    generate_batch_files_script : str
        Path to script that generates batch files
    cmip6_directory : str
        Path to directory where CMIP6 files are stored
    regrid_batch_dir : str
        Path to directory where batch files are written
    vars : str
        List of variables to generate batch files for
    freqs : str
        List of frequencies to use for generating batch files
    models : str
        List of models to use for generating batch files
    scenarios : str
        List of scenarios to use for generating batch files
    """
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--slurm_dir",
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
        args.conda_env_name,
        Path(args.generate_batch_files_script),
        Path(args.cmip6_directory),
        Path(args.slurm_dir),
        args.vars,
        args.freqs,
        args.models,
        args.scenarios,
    )


if __name__ == "__main__":

    (
        conda_env_name,
        generate_batch_files_script,
        cmip6_directory,
        slurm_dir,
        vars,
        freqs,
        models,
        scenarios,
    ) = parse_args()

    regrid_batch_dir = slurm_dir.joinpath(batch_file_dir_name)
    regrid_batch_dir.mkdir(exist_ok=True)
    generate_batch_files_sbatch_fp = slurm_dir.joinpath(
        "generate_regrid_batch_files.slurm"
    )
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
        # this should work to initialize conda without init script
        'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"\n'
        f"conda activate {conda_env_name}\n"
        # run the generate batch files script
        f"python {generate_batch_files_script} \
            --cmip6_directory '{cmip6_directory}' \
            --regrid_batch_dir '{regrid_batch_dir}' \
            --vars '{vars}' --freqs '{freqs}' --models '{models}' --scenarios '{scenarios}' \n"
    )

    # save the sbatch text as a new slurm file in the repo directory
    with open(generate_batch_files_sbatch_fp, "w") as f:
        f.write(sbatch_text)

    job_id = submit_sbatch(generate_batch_files_sbatch_fp)

    # print the job_id to stdout
    # there is no way to set an env var for the parent shell, so the only ways to
    # directly pass job ids are through stdout or a temp file
    print(job_id)
