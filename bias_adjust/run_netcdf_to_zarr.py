"""Script to build a slurm file that runs netcdf_to_zarr.py on a suite of models and scenarios.


example usage:
    python run_netcdf_to_zarr.py --netcdf_dir /beegfs/CMIP6/kmredilla/daily_era5_4km_3338/ --year_str t2max/t2max_{year}_era5_4km_3338.nc --start_year 1965 --end_year 2014 --output_dir /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/optimized_inputs/
    python run_netcdf_to_zarr.py --worker_script /beegfs/CMIP6/kmredilla/cmip6-utils/bias_adjust/netcdf_to_zarr.py --conda_env_name cmip6-utils  --netcdf_dir /beegfs/CMIP6/kmredilla/cmip6_4km_3338/netcdf --models 'GFDL-ESM4 CESM2' --scenarios 'historical ssp585' --variables 'tasmax pr' --output_dir /beegfs/CMIP6/kmredilla/zarr_bias_adjust_inputs --partition t2small
"""

import argparse
from itertools import product
import logging
from pathlib import Path
from slurm import (
    write_netcdf_to_zarr_cmip6_config_file,
    write_sbatch_netcdf_to_zarr_cmip6,
    make_sbatch_array_head,
    submit_sbatch,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

target_dir_name = "zarr"


def validate_args(args):
    """Validate the arguments passed to the script."""
    args.worker_script = Path(args.worker_script)
    args.output_dir = Path(args.output_dir)
    if not args.output_dir.parent.exists():
        raise FileNotFoundError(
            f"Parent of output directory, {args.output_dir.parent}, does not exist. Aborting."
        )
    args.netcdf_dir = Path(args.netcdf_dir)
    if not args.netcdf_dir.exists():
        raise FileNotFoundError(
            f"Input directory, {args.netcdf_dir}, does not exist. Aborting."
        )

    args.models = args.models.split(" ")
    models_in_input_dir = [
        model
        for model in args.models
        if model
        in next(args.netcdf_dir.walk())[
            1
        ]  # this gets the list of subdirectories in the input directory
    ]
    if not any(models_in_input_dir):
        raise ValueError(
            f"No subdirectories in the input directory match the models provided. Aborting."
        )
    elif not all([model in models_in_input_dir for model in args.models]):
        logging.warning(
            f"Some models in the input directory do not have subdirectories: {models_in_input_dir}. Skipping these models."
        )

    # get list of model/scenario combinations in input directory
    args.scenarios = args.scenarios.split(" ")
    modscens_from_args = list(product(args.models, args.scenarios))
    modscens_from_input_dir = set(
        [(d.parent.name, d.name) for d in list((args.netcdf_dir.glob("*/*")))]
    )
    model_scenarios_in_input_dir = [
        (model, scenario)
        for model, scenario in modscens_from_args
        if (model, scenario) in modscens_from_input_dir
    ]
    if not any(model_scenarios_in_input_dir):
        raise ValueError(
            f"No subdirectories in the input directory match the scenarios provided. Model-scenario combinations specified in arguments: {modscens_from_args}. Model-scenario combinations in input directory: {modscens_from_input_dir}."
        )
    elif not all(
        [
            (model, scenario) in model_scenarios_in_input_dir
            for (model, scenario) in modscens_from_args
        ]
    ):
        logging.warning(
            f"Some specified model/scenario combinations were not found in the input directory: {set(modscens_from_args) - set(model_scenarios_in_input_dir)}. Skipping these model/scenario combinations."
        )
    args.variables = args.variables.split(" ")

    return args


def parse_args():
    """Parse some command line arguments.

    Returns
    -------
    worker_script : str
        Path to netcdf-to-zarr conversion script
    conda_env_name : str
        Name of conda environment to activate
    netcdf_dir : str
        Path to directory where netcdf files to be converted to zarr are stored
    models : list of str
        List of models to work on
    scenarios : list of str
        List of scenarios to work on
    variables : list of str
        List of variables to work on
    output_dir : str
        Path to directory where outputs will be written.
    chunks_dict : dict
        Optional. Dictionary of chunks to use for rechunking
    partition : str
        Slurm partition
    clear_out_files : bool
        Optional. Remove output files in the output directory before running the job
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker_script",
        type=str,
        help="Path to netcdf-to-zarr conversion script",
        required=True,
    )
    parser.add_argument(
        "--conda_env_name",
        type=str,
        help="Name of the conda environment to activate",
        required=True,
    )
    parser.add_argument(
        "--netcdf_dir",
        type=str,
        help="Path to directory of netcdf files to be converted to zarr",
        required=True,
    )
    parser.add_argument(
        "--models",
        type=str,
        help="' '-separated list of CMIP6 models to work on",
        required=True,
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="' '-separated list of scenarios to work on",
        required=True,
    )
    parser.add_argument(
        "--variables",
        type=str,
        help="' '-separated list of variables to work on",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory where outputs will be written.",
        required=True,
    )
    parser.add_argument(
        "--chunks_dict",
        type=str,
        help="Dictionary of chunks to use for rechunking",
        required=False,
        default=None,
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
        help="Remove output files in the output directory before running the job",
        default=True,
    )
    args = parser.parse_args()
    args = validate_args(args)

    return (
        args.worker_script,
        args.conda_env_name,
        args.netcdf_dir,
        args.models,
        args.scenarios,
        args.variables,
        args.output_dir,
        args.chunks_dict,
        args.partition,
        args.clear_out_files,
    )


if __name__ == "__main__":

    (
        worker_script,
        conda_env_name,
        netcdf_dir,
        models,
        scenarios,
        variables,
        output_dir,
        chunks_dict,
        partition,
        clear_out_files,
    ) = parse_args()

    output_dir.mkdir(exist_ok=True)
    target_dir = output_dir.joinpath(target_dir_name)
    target_dir.mkdir(exist_ok=True)

    slurm_dir = output_dir.joinpath("slurm")
    slurm_dir.mkdir(exist_ok=True)
    if clear_out_files:
        for file in slurm_dir.glob("*.out"):
            file.unlink()

    # filepath for slurm script
    sbatch_fp = slurm_dir.joinpath(f"convert_cmip6_netcdf_to_zarr.slurm")
    # filepath for slurm stdout
    sbatch_out_fp = slurm_dir.joinpath(sbatch_fp.name.replace(".slurm", "_%A-%a.out"))

    config_path = slurm_dir.joinpath("config.txt")
    array_range = write_netcdf_to_zarr_cmip6_config_file(
        config_path=config_path,
        models=models,
        scenarios=scenarios,
        variables=variables,
    )

    sbatch_head_kwargs = {
        "array_range": array_range,
        "partition": partition,
        "sbatch_out_fp": sbatch_out_fp,
        "conda_env_name": conda_env_name,
    }
    sbatch_head = make_sbatch_array_head(**sbatch_head_kwargs)

    sbatch_kwargs = {
        "sbatch_fp": sbatch_fp,
        "sbatch_out_fp": sbatch_out_fp,
        "worker_script": worker_script,
        "netcdf_dir": netcdf_dir,
        "target_dir": target_dir,
        "sbatch_head": sbatch_head,
        "config_file": config_path,
    }
    if chunks_dict is not None:
        sbatch_kwargs["chunks_dict"] = chunks_dict
    write_sbatch_netcdf_to_zarr_cmip6(**sbatch_kwargs)
    # job_id = submit_sbatch(sbatch_fp)

    # print(job_id)
