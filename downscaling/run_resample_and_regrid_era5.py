"""Script to build a slurm file that runs resample_and_regrid_era5.py.


example usage:
    python /beegfs/CMIP6/kmredilla/repos/cmip6-utils/downscaling/run_resample_and_regrid_era5.py --conda_env_name snap-geo --runner_script /beegfs/CMIP6/kmredilla/cmip6-utils/downscaling/run_resample_and_regrid_era5.sh --wrf_era5_directory /beegfs/CMIP6/wrf_era5/04km --output_directory /beegfs/CMIP6/kmredilla/daily_era5_4km_3338/netcdf --slurm_directory /beegfs/CMIP6/kmredilla/daily_era5_4km_3338/netcdf --geo_file /beegfs/CMIP6/wrf_era5/geo_em.d02.nc --start_year 1965 --end_year 2022
"""

import argparse
import logging
import subprocess
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def submit_sbatch(sbatch_fp):
    """Submit a script to slurm via sbatch.

    Parameters
    ----------
    sbatch_fp : pathlib.PosixPath
        path to .slurm script to submit

    Returns
    -------
    str
        job id for submitted job
    """
    out = subprocess.check_output(["sbatch", str(sbatch_fp)])
    job_id = out.decode().replace("\n", "").split(" ")[-1]

    return job_id


def parse_args():
    """Parse some command line arguments.

    Returns
    -------
    conda_env_name : str
        Name of conda environment to activate
    runner_script : str
        Path to shell script that runs the resampling and reprojection of ERA5 data
    wrf_era5_directory : str
        Path to directory where WRF-downscaled ERA5 data is stored
    output_directory : str
        Path to directory where resampled and reprojected ERA5 data will be written
    slurm_directory : str
        Path to directory for writing slurm files
    geo_file : str
        Path to WRF geo_em file for projection information
    start_year : str
        Start year for processing
    end_year : str
        End year for processing
    no_clobber : bool
        Do not overwrite existing files in output_directory
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conda_env_name",
        type=str,
        help="Name of conda environment to activate",
        required=True,
    )
    parser.add_argument(
        "--runner_script",
        type=str,
        help="Path to script that runs the resampling and reprojection of ERA5 data",
        required=True,
    )
    parser.add_argument(
        "--wrf_era5_directory",
        type=str,
        help="Path to directory where WRF-downscaled ERA5 data is stored",
        required=True,
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        help="Path to directory where resampled and reprojected ERA5 data will be stored",
        required=True,
    )
    parser.add_argument(
        "--slurm_directory",
        type=str,
        help="Path to directory for writing slurm files",
        required=True,
    )
    parser.add_argument(
        "--geo_file",
        type=str,
        help="Path to WRF geo_em file for projection information",
        required=True,
    )
    parser.add_argument(
        "--start_year",
        type=str,
        help="Start year for processing",
        required=True,
    )
    parser.add_argument(
        "--end_year",
        type=str,
        help="End year for processing",
        required=True,
    )
    parser.add_argument(
        "--no_clobber",
        action="store_true",
        default=False,
        help="Do not overwrite existing files in output_directory",
    )
    args = parser.parse_args()

    return (
        # Path(args.conda_init_script),
        args.conda_env_name,
        Path(args.runner_script),
        Path(args.wrf_era5_directory),
        Path(args.output_directory),
        Path(args.slurm_directory),
        Path(args.geo_file),
        args.start_year,
        args.end_year,
        args.no_clobber,
    )


if __name__ == "__main__":

    (
        # conda_init_script,
        conda_env_name,
        runner_script,
        wrf_era5_directory,
        output_directory,
        slurm_directory,
        geo_file,
        start_year,
        end_year,
        no_clobber,
    ) = parse_args()

    output_directory.mkdir(exist_ok=True)
    slurm_directory.mkdir(exist_ok=True)
    process_era5_sbatch_file = slurm_directory.joinpath(
        "resample_and_regrid_era5.slurm"
    )
    process_era5_sbatch_out_file = str(process_era5_sbatch_file).replace(
        ".slurm", "_%j.out"
    )

    sbatch_text = (
        "#!/bin/sh\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --cpus-per-task=24\n"
        f"#SBATCH -p t2small\n"
        f"#SBATCH --time=08:00:00\n"
        f"#SBATCH --output {process_era5_sbatch_out_file}\n"
        "echo Start slurm && date\n"
        # this should work to initialize conda without init script
        'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"\n'
        f"conda activate {conda_env_name}\n"
        # call the shell script to iterate and try the python script
        # because iterating over years in the python script was causing hangups
        f"cd {runner_script.parent}\n"
        f"source {runner_script} "
        f"{wrf_era5_directory} "
        f"{output_directory} "
        f"{geo_file} "
        f"{start_year} {end_year}"
    )

    # save the sbatch text as a new slurm file in the repo directory
    with open(process_era5_sbatch_file, "w") as f:
        f.write(sbatch_text)

    logging.info(
        f"Submitting {process_era5_sbatch_file} to slurm (contents:)\n{sbatch_text}"
    )
    job_id = submit_sbatch(process_era5_sbatch_file)

    # print the job_id to stdout
    # there is no way to set an env var for the parent shell, so the only ways to
    # directly pass job ids are through stdout or a temp file
    print(job_id)
