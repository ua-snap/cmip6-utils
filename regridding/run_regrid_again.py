"""Script to make the batch file for a "second regridding" (aka regrid_again), 
where we are regridding a set of files that have already been regridded to a common grid.

Example usage:
    python run_regrid_again.py \
        --partition t2small \
        --conda_env_name cmip6-utils \
        --slurm_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/slurm \
        --regrid_script /center1/CMIP6/kmredilla/cmip6-utils/regridding/regrid.py \
        --interp_method bilinear \
        --target_grid_file /beegfs/CMIP6/kmredilla/downscaling/era5_target_slice.nc \
        --regridded_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/regrid \
        --regrid_again_batch_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/slurm/regrid_again_batch \
        --output_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/final_regrid
"""

import argparse
import logging
from pathlib import Path
from itertools import islice
from slurm import submit_sbatch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def parse_args():
    """Parse some command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        type=str,
        help="Slurm partition to use",
        default="t2small",
    )
    parser.add_argument(
        "--conda_env_name",
        type=str,
        help="Name of the conda environment to use",
        default="cmip6-utils",
    )
    parser.add_argument(
        "--slurm_dir",
        type=str,
        help="Path to directory where slurm files are stored",
    )
    parser.add_argument(
        "--regrid_script",
        type=str,
        help="Path to the regrid.py script",
    )
    parser.add_argument(
        "--interp_method",
        type=str,
        help="Interpolation method to use",
    )
    parser.add_argument(
        "--target_grid_file",
        type=str,
        help="Path to the target grid file",
    )
    parser.add_argument(
        "--regridded_dir",
        type=str,
        help="Path to directory where CMIP6 files are stored",
    )
    parser.add_argument(
        "--regrid_again_batch_dir",
        type=str,
        help="Path to directory where the regrid again batch files will be written",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory where the regridded files will be stored",
    )
    args = parser.parse_args()
    return (
        args.partition,
        args.conda_env_name,
        Path(args.slurm_dir),
        Path(args.regrid_script),
        args.interp_method,
        Path(args.target_grid_file),
        Path(args.regridded_dir),
        Path(args.regrid_again_batch_dir),
        Path(args.output_dir),
    )


def write_batch_files(src_fps, regrid_again_batch_dir):
    """Write the batch files for the regrid again job."""
    logging.info(
        f"Writing batch files for regridding again to {regrid_again_batch_dir}"
    )
    batch_size = 200
    batch_files = []
    for i, start in enumerate(range(0, len(src_fps), batch_size), start=1):
        batch_file = regrid_again_batch_dir.joinpath(f"regrid_again_batch_{i}.txt")
        with open(batch_file, "w") as f:
            for src_fp in islice(src_fps, start, start + batch_size):
                f.write(f"{src_fp}\n")
        batch_files.append(batch_file)
    logging.info(f"Batch files written to {regrid_again_batch_dir}")

    return batch_files


def write_config_file(
    config_path,
    batch_files,
):
    """Write a config file for the re-regridding slurm job script.
    This is used to split the job into a job array, one task per 200 files.

    Parameters
    ----------
    config_path : pathlib.PosixPath
        path to write the config file
    regrid_again_batch_dir : pathlib.PosixPath
        path to the directory where the regrid again batch files are stored

    Returns
    -------
    array_range : str
        string to use in the SLURM array
    """
    logging.info(f"Writing config file to {config_path}")
    array_list = []
    with open(config_path, "w") as f:
        f.write("array_id\tbatch_file\n")
        for array_id, batch_file in enumerate(batch_files, start=1):
            f.write(f"{array_id}\t{batch_file}\n")
            array_list.append(array_id)

    array_range = f"{min(array_list)}-{max(array_list)}"
    logging.info(f"Config file written to {config_path}")

    return array_range


def make_sbatch_head(array_range, partition, sbatch_out_file, conda_env_name):
    """Make a string of SBATCH commands that can be written into a .slurm script

    Args:
        array_range (str): string to use in the SLURM array
        partition (str): name of the partition to use
        sbatch_out_file (path_like): path to where sbatch stdout should be written
        conda_env_name (str): name of the conda environment to activate

    Returns:
        sbatch_head (str): string of SBATCH commands ready to be used as parameter in sbatch-writing functions.
        The following keys are left for filling with str.format:

            - output slurm filename
    """
    sbatch_head = (
        "#!/bin/sh\n"
        f"#SBATCH --array={array_range}%10\n"  # don't run more than 10 tasks
        f"#SBATCH --job-name=cmip6_dtr\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH -p {partition}\n"
        f"#SBATCH --output {sbatch_out_file}\n"
        # print start time
        "echo Start slurm && date\n"
        # prepare shell for using activate - Chinook requirement
        # this seems to work to initialize conda without init script
        'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"\n'
        f"conda activate {conda_env_name}\n"
    )

    return sbatch_head


def write_sbatch_regrid_again(
    partition,
    conda_env_name,
    slurm_dir,
    regrid_script,
    config_file,
    target_grid_file,
    output_dir,
    interp_method,
    array_range,
):
    """Write the sbatch file for the regrid again job."""
    sbatch_file = slurm_dir.joinpath("regrid_again.sbatch")
    sbatch_out_file = slurm_dir.joinpath("regrid_again_%j.out")

    sbatch_head = make_sbatch_head(
        array_range, partition, sbatch_out_file, conda_env_name
    )

    pycommands = "\n"
    pycommands += (
        # Extract the model and scenario to process for the current $SLURM_ARRAY_TASK_ID
        f"config={config_file}\n"
        "batch_file=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $2}' $config)\n"
        f"python {regrid_script} "
        f"-b $batch_file "
        f"-d {target_grid_file} "
        f"-o {output_dir} "
        f"--interp_method {interp_method}\n\n"
    )

    pycommands += f"echo End re-regridding && date\n\n"
    commands = sbatch_head + pycommands

    with open(sbatch_file, "w") as f:
        f.write(commands)

    logging.info(f"Wrote sbatch script to {sbatch_file}")

    return sbatch_file


if __name__ == "__main__":
    (
        partition,
        conda_env_name,
        slurm_dir,
        regrid_script,
        interp_method,
        target_grid_file,
        regridded_dir,
        regrid_again_batch_dir,
        output_dir,
    ) = parse_args()

    src_fps = list(regridded_dir.glob("**/*.nc"))
    regrid_again_batch_dir.mkdir(exist_ok=True)

    # write batch files for the regrid again job
    batch_files = write_batch_files(src_fps, regrid_again_batch_dir)

    # write the config file for the regrid again job
    config_path = slurm_dir.joinpath("regrid_again_config.txt")
    array_range = write_config_file(config_path, batch_files)

    # write the sbatch file for the regrid again job
    sbatch_file = slurm_dir.joinpath("regrid_again.sbatch")
    sbatch_kwargs = {
        "partition": partition,
        "conda_env_name": conda_env_name,
        "slurm_dir": slurm_dir,
        "regrid_script": regrid_script,
        "config_file": config_path,
        "target_grid_file": target_grid_file,
        "output_dir": output_dir,
        "interp_method": interp_method,
        "array_range": array_range,
    }
    sbatch_file = write_sbatch_regrid_again(**sbatch_kwargs)

    # submit the sbatch job
    logging.info(f"Submitting sbatch job to {partition} partition")
    # job_id = submit_sbatch(sbatch_file)

    job_id = "12345"
    print(job_id)
