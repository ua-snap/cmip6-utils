"""Script for constructing slurm jobs for computing daily temperature range for WRF-downscaled ERA5 data.

Example usage:
    python run_wrf_era5_dtr.py \
        --worker_script /import/beegfs/CMIP6/kmredilla/cmip6-utils/derived/dtr.py \
        --conda_env_name cmip6-utils \
        --era5_dir /center1/CMIP6/kmredilla/daily_era5_4km_3338 \
        --output_dir /import/beegfs/CMIP6/kmredilla/cmip6_4km_downscaling/era5_dtr \
        --slurm_dir /import/beegfs/CMIP6/kmredilla/cmip6_4km_downscaling/slurm \
        --partition t2small

Returns:
    Outputs are written directly to output_dir
"""

import argparse
import subprocess
import logging
from pathlib import Path
from itertools import product
from config import (
    era5_dtr_sbatch_fn,
    era5_dtr_tmp_fn,
)

# wonder if we should have a lut for this.
era5_tmax_var_id = "t2max"
era5_tmin_var_id = "t2min"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def validate_args(args):
    """Validate the arguments passed to the script."""
    args.worker_script = Path(args.worker_script)

    args.era5_dir = Path(args.era5_dir)
    if not args.era5_dir.exists():
        raise FileNotFoundError(
            f"Input directory, {args.era5_dir}, does not exist. Aborting."
        )
    args.output_dir = Path(args.output_dir)
    if not args.output_dir.parent.exists():
        raise FileNotFoundError(
            f"Parent of output directory, {args.output_dir.parent}, does not exist. Aborting."
        )
    args.slurm_dir = Path(args.slurm_dir)
    if not args.slurm_dir.parent.exists():
        raise FileNotFoundError(
            f"Parent of slurm directory, {args.slurm_dir.parent}, does not exist. Aborting."
        )

    return args


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker_script",
        type=str,
        help="Path to dtr processing script",
        required=True,
    )
    parser.add_argument(
        "--conda_env_name",
        type=str,
        help="Name of the conda environment to activate",
        required=True,
    )
    parser.add_argument(
        "--era5_dir",
        type=str,
        help="Path to directory of ERA5 data, with filepath structure <variable ID>/<files>",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory where DTR outputs will be written.",
        required=True,
    )
    parser.add_argument(
        "--slurm_dir",
        type=str,
        help="Path to directory where slurm stuff will be written.",
        required=True,
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="slurm partition",
        required=True,
    )
    parser.add_argument(
        "--clear_out_files",
        action="store_true",
        help="Remove output files in the slurm output files in slurm directory before running the job",
        default=True,
    )
    args = parser.parse_args()
    args = validate_args(args)

    return (
        args.worker_script,
        args.conda_env_name,
        args.era5_dir,
        args.output_dir,
        args.slurm_dir,
        args.partition,
        args.clear_out_files,
    )


def make_sbatch_head(partition, sbatch_out_fp, conda_env_name):
    """Make a string of SBATCH commands that can be written into a .slurm script

    Args:
        partition (str): name of the partition to use
        sbatch_out_fp (path_like): path to where sbatch stdout should be written
        conda_env_name (str): name of the conda environment to activate

    Returns:
        sbatch_head (str): string of SBATCH commands ready to be used as parameter in sbatch-writing functions.
        The following keys are left for filling with str.format:

            - output slurm filename
    """
    sbatch_head = (
        "#!/bin/sh\n"
        f"#SBATCH --job-name=era5_dtr\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH -p {partition}\n"
        f"#SBATCH --output {sbatch_out_fp}\n"
        # print start time
        "echo Start slurm && date\n"
        # prepare shell for using activate - Chinook requirement
        # this seems to work to initialize conda without init script
        'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"\n'
        f"conda activate {conda_env_name}\n"
    )

    return sbatch_head


def write_sbatch_dtr(
    sbatch_fp,
    sbatch_out_fp,
    worker_script,
    era5_dir,
    output_dir,
    sbatch_head,
):
    """Write an sbatch array script for executing the dtr processing for a suite of models and scenarios.

    Args:
        sbatch_fp (path_like): path to .slurm script to write sbatch commands to
        sbatch_out_fp (path_like): path to where sbatch stdout should be written
        worker_script (path_like): path to the script to be called to run the dtr processing
        era5_dir (path-like): path to directory of tasmax and tasmin files
        output_dir (path-like): directory to write the dtr data
        sbatch_head (dict): string for sbatch head script

    Returns:
        None, writes the commands to sbatch_fp
    """
    pycommands = "\n"
    pycommands += (
        # Extract the model and scenario to process for the current $SLURM_ARRAY_TASK_ID
        f"python {worker_script} "
        f"--tmax_dir {era5_dir.joinpath(era5_tmax_var_id)} "
        f"--tmin_dir {era5_dir.joinpath(era5_tmin_var_id)} "
        f"--output_dir {output_dir} "
        f"--dtr_tmp_fn {era5_dtr_tmp_fn}\n\n"
    )

    pycommands += f"echo End dtr processing && date\n\n"
    commands = sbatch_head.format(sbatch_out_fp=sbatch_out_fp) + pycommands

    with open(sbatch_fp, "w") as f:
        f.write(commands)

    logging.info(f"Wrote sbatch script to {sbatch_fp}")

    return


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
        worker_script,
        conda_env_name,
        era5_dir,
        output_dir,
        slurm_dir,
        partition,
        clear_out_files,
    ) = parse_args()

    output_dir.mkdir(exist_ok=True)
    # make the output directories

    slurm_dir.mkdir(exist_ok=True)
    if clear_out_files:
        for file in slurm_dir.glob(era5_dtr_sbatch_fn.replace(".slurm", "*.out")):
            file.unlink()

    # filepath for slurm script
    sbatch_fp = slurm_dir.joinpath(era5_dtr_sbatch_fn)
    # filepath for slurm stdout
    sbatch_out_fp = slurm_dir.joinpath(sbatch_fp.name.replace(".slurm", "_%j.out"))

    sbatch_head_kwargs = {
        "partition": partition,
        "sbatch_out_fp": sbatch_out_fp,
        "conda_env_name": conda_env_name,
    }
    sbatch_head = make_sbatch_head(**sbatch_head_kwargs)

    sbatch_dtr_kwargs = {
        "sbatch_fp": sbatch_fp,
        "sbatch_out_fp": sbatch_out_fp,
        "worker_script": worker_script,
        "era5_dir": era5_dir,
        "output_dir": output_dir,
        "sbatch_head": sbatch_head,
    }
    write_sbatch_dtr(**sbatch_dtr_kwargs)
    job_id = submit_sbatch(sbatch_fp)

    print(job_id)
