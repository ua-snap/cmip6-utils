"""Script to build a slurm file that runs netcdf_to_zarr.py.


example usage:
    python run_netcdf_to_zarr.py --netcdf_dir /beegfs/CMIP6/kmredilla/daily_era5_4km_3338/ --year_str t2max/t2max_{year}_era5_4km_3338.nc --start_year 1965 --end_year 2014 --output_dir /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/optimized_inputs/
"""

import argparse
import logging
import subprocess
from pathlib import Path
import json
from netcdf_to_zarr import get_input_filepaths


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
    netcdf_dir : str
        Path to directory where netcdf files to be converted to zarr are stored
    glob_str : str
        Optional. Glob string for getting data files in data_dir. Required if files to optimize are not in the data_dir root. Cannot be used with year_str.
    year_str : str
        Optional. String to format with year value to create source netcdf filepaths. Must be used in conjunction with start_year and end_year.
    start_year : str
        Optional. Start year for processing. Must be used in conjunction with year_str and end_year.
    end_year : str
        Optional. End year for processing. Must be used in conjunction with year_str and start_year.
    chunks_dict : dict
        Optional. Dictionary of chunks to use for rechunking
    zarr_path : str
        Path to directory where zarr store will be written (with final name derived from source files)
    slurm_dir : str
        Path to directory for writing slurm files
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conda_env_name",
        type=str,
        help="Name of conda environment to activate",
        required=True,
    )
    parser.add_argument(
        "--script_dir",
        type=str,
        help="Path to directory containing conversion script to run.",
        required=True,
    )
    parser.add_argument(
        "--netcdf_dir",
        type=str,
        help="Path to directory containing data files to optimize.",
        required=True,
    )
    parser.add_argument(
        "--glob_str",
        type=str,
        help="Glob string for getting data files in data_dir. Required if files to optimize are not in the data_dir root.",
        default=None,
    )
    parser.add_argument(
        "--year_str",
        type=str,
        help="String for getting data files in data_dir based on start and end years. Requires both start_year and end_year to use.",
        default=None,
    )
    parser.add_argument(
        "--start_year",
        type=str,
        help="Starting year of data to optimize. Required if year_str is provided.",
        default=None,
    )
    parser.add_argument(
        "--end_year",
        type=str,
        help="Ending year of data to optimize. Required if year_str is provided.",
        default=None,
    )
    parser.add_argument(
        "--chunks_dict",  # this is just a template for now, in case we want to make this configurable
        type=str,
        help="Dictionary of chunks to use for rechunking",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        help="Path to write zarr store",
        required=True,
    )
    parser.add_argument(
        "--slurm_dir",
        type=str,
        help="Path to directory for writing slurm files",
        required=True,
    )
    args = parser.parse_args()

    return (
        args.conda_env_name,
        args.script_dir,
        args.netcdf_dir,
        args.glob_str,
        args.year_str,
        args.start_year,
        args.end_year,
        args.chunks_dict,
        Path(args.zarr_path),
        Path(args.slurm_dir),
    )


if __name__ == "__main__":

    (
        # conda_init_script,
        conda_env_name,
        script_dir,
        netcdf_dir,
        glob_str,
        year_str,
        start_year,
        end_year,
        chunks_dict,
        zarr_path,
        slurm_dir,
    ) = parse_args()

    assert (
        zarr_path.parent.exists()
    ), f"Parent directory of zarr_path, {zarr_path.parent}, does not exist. Aborting."
    slurm_dir.mkdir(exist_ok=True)
    # use the zarr path to get a name for the job
    job_name = zarr_path.name.replace(".zarr", "")
    conversion_job_file = slurm_dir.joinpath(f"convert_netcdf_to_zarr_{job_name}.slurm")
    conversion_job_out_file = str(conversion_job_file).replace(".slurm", "_%j.out")

    job_str = (
        "#!/bin/sh\n"
        f"#SBATCH --job-name=convert_netcdf_to_zarr_{job_name}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH -p t2small\n"
        f"#SBATCH --time=08:00:00\n"
        f"#SBATCH --output {conversion_job_out_file}\n"
        "echo Start slurm && date\n"
        # this should work to initialize conda without init script
        'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"\n'
        f"conda activate {conda_env_name}\n"
        f"cd {script_dir}\n"
        f"python netcdf_to_zarr.py --netcdf_dir {netcdf_dir} "
        # f"--glob_str {glob_str} --year_str {year_str} "
        # f"--output_dir {output_dir} --year $year  "
    )
    if glob_str:
        job_str += f"--glob_str {glob_str} "
    elif year_str:
        job_str += (
            f"--year_str {year_str} --start_year {start_year} --end_year {end_year} "
        )
    if chunks_dict:
        job_str += f"--chunks_dict '{chunks_dict}' "
    job_str += f"--zarr_path {zarr_path}\n"

    # save the sbatch text as a new slurm file in the repo directory
    with open(conversion_job_file, "w") as f:
        f.write(job_str)

    logging.info(f"Submitting {conversion_job_file} to slurm (contents:)\n{job_str}")
    job_id = submit_sbatch(conversion_job_file)

    # print the job_id to stdout
    # there is no way to set an env var for the parent shell, so the only ways to
    # directly pass job ids are through stdout or a temp file
    print(job_id)
