"""Functions to assist with constructing slurm jobs."""

import logging
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def make_sbatch_head(
    partition, sbatch_out_path, conda_env_name, job_name, array_range=None
):
    """Make a string of SBATCH commands that can be written into a .slurm script

    Args:
        partition (str): name of the partition to use
        sbatch_out_path (path_like): path to where sbatch stdout should be written
        conda_env_name (str): name of the conda environment to activate
        job_name (str): name of the job to use in sbatch
        array_range (str): string to use in the SLURM array

    Returns:
        sbatch_head (str): string of SBATCH commands ready to be used as parameter in sbatch-writing functions.
    """
    sbatch_head = (
        "#!/bin/sh\n"
        f"#SBATCH --job-name={job_name}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH -p {partition}\n"
        f"#SBATCH --output {sbatch_out_path}\n"
    )
    if array_range is not None:
        sbatch_head += f"#SBATCH --array={array_range}%10\n"
    sbatch_head += (
        # print start time
        "echo Start slurm && date\n"
        # prepare shell for using activate - Chinook requirement
        # this seems to work to initialize conda without init script
        'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"\n'
        f"conda activate {conda_env_name}\n"
    )

    return sbatch_head


def submit_sbatch(sbatch_path):
    """Submit a script to slurm via sbatch.

    Parameters
    ----------
    sbatch_path : pathlib.PosixPath
        path to .slurm script to submit

    Returns
    -------
    str
        job id for submitted job
    """
    out = subprocess.check_output(["sbatch", str(sbatch_path)])
    job_id = out.decode().replace("\n", "").split(" ")[-1]

    return job_id
