r"""Script for constructing slurm jobs for computing the difference between arbitrary datasets.
Datasets are assumed to be in a flat file structure in the input directory and in zarr format. 

Example usage:
    # example for derived tasmin. 
    python run_wrf_era5_difference.py \
        --worker_script /home/kmredilla/repos/cmip6-utils/derived/difference.py \
        --conda_env_name cmip6-utils \
        --minuend_store /center1/CMIP6/kmredilla/cmip6_4km_downscaling/era5_zarr/t2max_era5.zarr \
        --subtrahend_store /center1/CMIP6/kmredilla/cmip6_4km_downscaling/era5_zarr/dtr_era5.zarr \
        --output_store /center1/CMIP6/kmredilla/cmip6_4km_downscaling/era5_zarr/tasmin_era5.zarr \
        --new_var_id tasmin \
        --slurm_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/slurm \
        --partition t2small

Returns:
    Writes slurm job script for computing the difference between two datasets and calls sbatch to submit the job.
"""

import argparse
import subprocess
import logging
from pathlib import Path
from itertools import product
from config import era5_diff_sbatch_tmp_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def validate_args(args):
    """Validate the arguments passed to the script."""
    args.worker_script = Path(args.worker_script)
    minuend_store = Path(args.minuend_store)
    subtrahend_store = Path(args.subtrahend_store)
    output_store = Path(args.output_store)
    if not minuend_store.exists():
        raise FileNotFoundError(
            f"Minuend store {minuend_store} does not exist. Aborting."
        )
    if not subtrahend_store.exists():
        raise FileNotFoundError(
            f"Subtrahend store {subtrahend_store} does not exist. Aborting."
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
        "--minuend_store",
        type=str,
        help="Directory containing 'minued' data (that from which subtrahend is subtracted)",
        required=True,
    )
    parser.add_argument(
        "--subtrahend_store",
        type=str,
        help="Directory containing 'subtrahend' data (that which is subtracted from minuend)",
        required=True,
    )
    parser.add_argument(
        "--output_store",
        type=str,
        help="Directory for writing difference data.",
        required=True,
    )
    parser.add_argument(
        "--new_var_id",
        type=str,
        help="New variable id for the resulting difference data",
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
        args.minuend_store,
        args.subtrahend_store,
        args.output_store,
        args.new_var_id,
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
    """
    sbatch_head = (
        "#!/bin/sh\n"
        f"#SBATCH --job-name=era5_difference\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH -p {partition}\n"
        f"#SBATCH --output {sbatch_out_fp}\n"
        "echo Start slurm && date\n"
        'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"\n'
        f"conda activate {conda_env_name}\n"
    )

    return sbatch_head


def write_sbatch_diff(
    sbatch_fp,
    sbatch_out_fp,
    worker_script,
    minuend_store,
    subtrahend_store,
    output_store,
    new_var_id,
    sbatch_head,
):
    """Write an sbatch array script for executing the dtr processing for a suite of models and scenarios.

    Args:
        sbatch_fp (path_like): path to .slurm script to write sbatch commands to
        sbatch_out_fp (path_like): path to where sbatch stdout should be written
        worker_script (path_like): path to the script to be called to run the dtr processing
        minuend_store (str): path to the minuend data store
        subtrahend_store (str): path to the subtrahend data store
        output_store (str): path to the output data store
        new_var_id (str): variable id for the output data
        sbatch_head (dict): string for sbatch head script
    Returns:
        None, writes the commands to sbatch_fp
    """

    pycommands = "\n"
    # these are template filepaths with $model and $scenario placeholders for slurm task array config

    pycommands += (
        # Extract the model and scenario to process for the current $SLURM_ARRAY_TASK_ID
        f"python {worker_script} "
        f"--minuend_store {minuend_store} "
        f"--subtrahend_store {subtrahend_store} "
        f"--output_store {output_store} "
        f"--new_var_id {new_var_id}\n"
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
        minuend_store,
        subtrahend_store,
        output_store,
        new_var_id,
        slurm_dir,
        partition,
        clear_out_files,
    ) = parse_args()

    if clear_out_files:
        for file in slurm_dir.glob(era5_diff_sbatch_tmp_fn.replace(".slurm", "*.out")):
            file.unlink()

    # filepath for slurm script
    sbatch_fp = slurm_dir.joinpath(
        era5_diff_sbatch_tmp_fn.format(new_var_id=new_var_id)
    )
    # filepath for slurm stdout
    sbatch_out_fp = slurm_dir.joinpath(sbatch_fp.name.replace(".slurm", "_%j.out"))

    sbatch_head_kwargs = {
        "partition": partition,
        "sbatch_out_fp": sbatch_out_fp,
        "conda_env_name": conda_env_name,
    }
    sbatch_head = make_sbatch_head(**sbatch_head_kwargs)

    sbatch_diff_kwargs = {
        "sbatch_fp": sbatch_fp,
        "sbatch_out_fp": sbatch_out_fp,
        "worker_script": worker_script,
        "minuend_store": minuend_store,
        "subtrahend_store": subtrahend_store,
        "output_store": output_store,
        "new_var_id": new_var_id,
        "sbatch_head": sbatch_head,
    }
    write_sbatch_diff(**sbatch_diff_kwargs)
    # job_id = submit_sbatch(sbatch_fp)

    # print(job_id)
