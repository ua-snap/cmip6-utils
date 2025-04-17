"""Script to build a slurm file that runs netcdf_to_zarr.py on a suite of models and scenarios.


example usage:
    python run_cmip6_netcdf_to_zarr.py \
        --partition t2small \
        --conda_env_name cmip6-utils \
        --worker_script /beegfs/CMIP6/kmredilla/cmip6-utils/bias_adjust/netcdf_to_zarr.py \
        --netcdf_dir /beegfs/CMIP6/kmredilla/cmip6_4km_3338/regrid \
        --output_dir /beegfs/CMIP6/kmredilla/cmip6_downscaling/optimized_inputs/ \
        --models 'GFDL-ESM4 CESM2' \
        --scenarios 'historical ssp585' \
        --variables 'tasmax pr' \
        --slurm_dir /beegfs/CMIP6/kmredilla/cmip6_downscaling/slurm
"""

import argparse
from itertools import product
import logging
from pathlib import Path
from slurm import (
    make_sbatch_head,
    submit_sbatch,
)
from utils import validate_path_arg
from config import (
    cmip6_netcdf_to_zarr_sbatch_tmp_fn,
    cmip6_regrid_tmp_fn,
    cmip6_zarr_tmp_fn,
    cmip6_netcdf_to_zarr_sbatch_tmp_fn,
)
from luts import cmip6_year_ranges


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def validate_args(args):
    """Validate the arguments passed to the script."""
    args.worker_script = Path(args.worker_script)
    args.netcdf_dir = Path(args.netcdf_dir)
    args.output_dir = Path(args.output_dir)
    args.slurm_dir = Path(args.slurm_dir)
    validate_path_arg(args.worker_script, "worker_script")
    validate_path_arg(args.netcdf_dir, "netcdf_dir")
    validate_path_arg(args.output_dir.parent, "parent of output_dir")
    validate_path_arg(args.slurm_dir.parent, "parent of slurm_dir")

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
    """Parse some command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        type=str,
        help="slurm partition",
        required=True,
    )
    parser.add_argument(
        "--conda_env_name",
        type=str,
        help="Name of the conda environment to activate",
        required=True,
    )
    parser.add_argument(
        "--worker_script",
        type=str,
        help="Path to netcdf-to-zarr conversion script",
        required=True,
    )
    parser.add_argument(
        "--netcdf_dir",
        type=str,
        help="Path to directory of netcdf files to be converted to zarr",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory where converted zarr stores will be written.",
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
        "--chunks_dict",
        type=str,
        help="Dictionary of chunks to use for rechunking",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--slurm_dir",
        type=str,
        help="Path to directory where slurm files and logs will be written.",
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
        args.partition,
        args.conda_env_name,
        args.worker_script,
        args.netcdf_dir,
        args.output_dir,
        args.models,
        args.scenarios,
        args.variables,
        args.chunks_dict,
        args.slurm_dir,
        args.clear_out_files,
    )


def write_netcdf_to_zarr_cmip6_config_file(
    config_path,
    models,
    scenarios,
    variables,
):
    """Write a config file for the Zarr conversion slurm job script.
    This is used to split the job into a job array, one task per model/scenario combination.

    Parameters
    ----------
    config_path : pathlib.PosixPath
        path to write the config file
    models : list of str
        list of models to process
    scenarios : list of str
        list of scenarios to process
    variables : list of str
        list of variables to process

    Returns
    -------
    array_range : str
        string to use in the SLURM array
    """
    array_list = []
    with open(config_path, "w") as f:
        f.write("array_id\tmodel\tscenario\tvariable\tstart_year\tend_year\n")
        for array_id, (model, scenario, variable) in enumerate(
            product(models, scenarios, variables), start=1
        ):
            start_year = cmip6_year_ranges[scenario]["start_year"]
            end_year = cmip6_year_ranges[scenario]["end_year"]
            f.write(
                f"{array_id}\t{model}\t{scenario}\t{variable}\t{start_year}\t{end_year}\n"
            )
            array_list.append(array_id)

    logging.info(f"Wrote config file to {config_path}")

    array_range = f"{min(array_list)}-{max(array_list)}"

    return array_range


def format_for_slurm_array_config(str):
    """Format a string for use in sbatch file for job array.

    Args:
        str (str): string to format

    Returns:
        str: formatted string
    """
    return str.replace("{", "${").replace("}", "}")


def write_sbatch_netcdf_to_zarr_cmip6(
    sbatch_path,
    worker_script,
    netcdf_dir,
    output_dir,
    sbatch_head,
    config_file,
):
    """Write an sbatch script for executing the bias adjustment script for a given model, scenario, and variable.
    Hardcoded for daily data.

    Args:
        sbatch_path (path_like): path to .slurm script to write sbatch commands to
        worker_script (path_like): path to the script to be called to run the netcdf-to-zarr conversion
        netcdf_dir (path-like): path to directory of netcdf files to be converted to zarr
        output_dir (path-like): directory to write the zarr data
        sbatch_head (dict): string for sbatch head script
        config_file (path_like): path to the config file for the slurm job array
    Returns:
        None, writes the commands to sbatch_path
    """
    # format the filename templates with the slurm array config variables
    regrid_fn_format = {
        "model": "${model}",
        "scenario": "${scenario}",
        "var_id": "${var_id}",
        "year": "{year}",
    }
    zarr_fn_format = regrid_fn_format.copy()
    del zarr_fn_format["year"]
    pycommands = "\n"
    pycommands += (
        # Extract the attrs for the current $SLURM_ARRAY_TASK_ID
        f"config={config_file}\n"
        "model=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $2}' $config)\n"
        "scenario=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $3}' $config)\n"
        "var_id=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $4}' $config)\n"
        "start_year=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $5}' $config)\n"
        "end_year=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $6}' $config)\n"
        f"python {worker_script} \\\n"
        f"--netcdf_dir {netcdf_dir} \\\n"
        f"--year_str $model/$scenario/day/$var_id/{cmip6_regrid_tmp_fn.format(**regrid_fn_format)} \\\n"
        f"--start_year $start_year \\\n"
        f"--end_year $end_year \\\n"
        # format this to get "${{variable}}${{model}}_${{scenario}}.zarr" for the slurm array
        f"--zarr_path {output_dir.joinpath(cmip6_zarr_tmp_fn.format(**zarr_fn_format))}\n"
    )

    pycommands += f"echo End netcdf-to-zarr conversion && date\n\n"
    commands = sbatch_head + pycommands

    with open(sbatch_path, "w") as f:
        f.write(commands)

    logging.info(f"Wrote sbatch script to {sbatch_path}")

    return


if __name__ == "__main__":

    (
        partition,
        conda_env_name,
        worker_script,
        netcdf_dir,
        output_dir,
        models,
        scenarios,
        variables,
        chunks_dict,
        slurm_dir,
        clear_out_files,
    ) = parse_args()

    output_dir.mkdir(exist_ok=True)
    slurm_dir.mkdir(exist_ok=True)
    if clear_out_files:
        for file in slurm_dir.glob(
            cmip6_netcdf_to_zarr_sbatch_tmp_fn.replace(".slurm", "*.out")
        ):
            file.unlink()

    # filepath for slurm script
    sbatch_path = slurm_dir.joinpath(cmip6_netcdf_to_zarr_sbatch_tmp_fn)
    # filepath for slurm stdout
    sbatch_out_path = slurm_dir.joinpath(
        sbatch_path.name.replace(".slurm", "_%A-%a.out")
    )

    config_path = slurm_dir.joinpath("cmip6_netcdf_to_zarr_config.txt")
    array_range = write_netcdf_to_zarr_cmip6_config_file(
        config_path=config_path,
        models=models,
        scenarios=scenarios,
        variables=variables,
    )

    sbatch_head_kwargs = {
        "array_range": array_range,
        "partition": partition,
        "sbatch_out_path": sbatch_out_path,
        "conda_env_name": conda_env_name,
        "job_name": "cmip6_netcdf_to_zarr",
    }
    sbatch_head = make_sbatch_head(**sbatch_head_kwargs)

    sbatch_kwargs = {
        "sbatch_path": sbatch_path,
        "worker_script": worker_script,
        "netcdf_dir": netcdf_dir,
        "output_dir": output_dir,
        "sbatch_head": sbatch_head,
        "config_file": config_path,
    }
    if chunks_dict is not None:
        sbatch_kwargs["chunks_dict"] = chunks_dict

    write_sbatch_netcdf_to_zarr_cmip6(**sbatch_kwargs)
    job_id = submit_sbatch(sbatch_path)

    print(job_id)
