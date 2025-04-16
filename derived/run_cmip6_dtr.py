"""Script for constructing slurm jobs for computing daily temperature range for CMIP6 data.

Example usage:
    python slurm_dtr.py \
        --worker_script /import/beegfs/CMIP6/kmredilla/cmip6-utils/derived/dtr.py \
        --conda_env_name cmip6-utils \
        --models "GFDL-ESM4 CESM2" \
        --scenarios "ssp245 ssp585" \
        --input_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/regrid \
        --output_dir /import/beegfs/CMIP6/kmredilla/cmip6_4km_downscaling/regrid \
        --slurm_dir /import/beegfs/CMIP6/kmredilla/cmip6_4km_downscaling/slurm \
        --partition t2small

Returns:
    Outputs are written in output_dir following the <model>/<scenario>/dtr/<files>*.nc convention.
    e.g. files for GFDL-ESM4 ssp245 would be written to <output_dir>/GFDL-ESM4/ssp245/dtr/<output files>
"""

import argparse
import subprocess
import logging
from pathlib import Path
from itertools import product
from config import dtr_sbatch_tmp_fn, dtr_sbatch_config_tmp_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def validate_args(args):
    """Validate the arguments passed to the script."""
    args.worker_script = Path(args.worker_script)

    args.input_dir = Path(args.input_dir)
    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Input directory, {args.input_dir}, does not exist. Aborting."
        )
    args.output_dir = Path(args.output_dir)
    if not args.output_dir.parent.exists():
        raise FileNotFoundError(
            f"Parent of output directory, {args.output_dir.parent}, does not exist. Aborting."
        )
    args.slurm_dir = Path(args.slurm_dir)
    if not args.outpuslurm_dirt_dir.parent.exists():
        raise FileNotFoundError(
            f"Parent of slurm directory, {args.slurm_dir.parent}, does not exist. Aborting."
        )

    args.models = args.models.split(" ")
    models_in_input_dir = [
        model
        for model in args.models
        if model
        in next(args.input_dir.walk())[
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
    # could go deeper here and check for day/tasmax/tasmin subdirectories
    args.scenarios = args.scenarios.split(" ")
    modscens_from_args = list(product(args.models, args.scenarios))
    modscens_from_input_dir = set(
        [(d.parent.name, d.name) for d in list((args.input_dir.glob("*/*")))]
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
        "--input_dir",
        type=str,
        help="Path to directory of simulated data files to be adjusted, with filepath structure <model>/<scenario>/day/<variable ID>/<files>",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory where outputs will be written.",
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
        args.models,
        args.scenarios,
        args.input_dir,
        args.output_dir,
        args.slurm_dir,
        args.partition,
        args.clear_out_files,
    )


def write_config_file(
    config_path,
    models,
    scenarios,
):
    """Write a config file for the DTR processing slurm job script.
    This is used to split the job into a job array, one task per model/scenario combination.

    Parameters
    ----------
    config_path : pathlib.PosixPath
        path to write the config file
    models : list of str
        list of models to process
    scenarios : list of str
        list of scenarios to process

    Returns
    -------
    array_range : str
        string to use in the SLURM array
    """
    array_list = []
    with open(config_path, "w") as f:
        f.write("array_id\tmodel\tscenario\n")
        for array_id, (model, scenario) in enumerate(
            product(models, scenarios), start=1
        ):
            f.write(f"{array_id}\t{model}\t{scenario}\n")
            array_list.append(array_id)

    array_range = f"{min(array_list)}-{max(array_list)}"

    return array_range


def make_sbatch_head(array_range, partition, sbatch_out_fp, conda_env_name):
    """Make a string of SBATCH commands that can be written into a .slurm script

    Args:
        array_range (str): string to use in the SLURM array
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
        f"#SBATCH --array={array_range}%10\n"  # don't run more than 10 tasks
        f"#SBATCH --job-name=cmip6_dtr\n"
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
    input_dir,
    output_dir,
    sbatch_head,
    config_file,
):
    """Write an sbatch array script for executing the dtr processing for a suite of models and scenarios.

    Args:
        sbatch_fp (path_like): path to .slurm script to write sbatch commands to
        sbatch_out_fp (path_like): path to where sbatch stdout should be written
        worker_script (path_like): path to the script to be called to run the dtr processing
        input_dir (path-like): path to directory of tasmax and tasmin files for all models and scenarios
        output_dir (path-like): directory to write the dtr data
        sbatch_head (dict): string for sbatch head script
        config_file (path_like): path to the config file for the slurm job array
    Returns:
        None, writes the commands to sbatch_fp
    """
    pycommands = "\n"
    pycommands += (
        # Extract the model and scenario to process for the current $SLURM_ARRAY_TASK_ID
        f"config={config_file}\n"
        "model=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $2}' $config)\n"
        "scenario=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $3}' $config)\n"
        f"python {worker_script} "
        f"--tmax_dir {input_dir}/$model/$scenario/day/tasmax "
        f"--tmin_dir {input_dir}/$model/$scenario/day/tasmin "
        f"--output_dir {output_dir}/$model/$scenario/day/dtr "
        f"--dtr_tmp_fn dtr_$model_$scenario_{{start_date}}_{{end_date}}.nc\n"
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
        models,
        scenarios,
        input_dir,
        output_dir,
        slurm_dir,
        partition,
        clear_out_files,
    ) = parse_args()

    output_dir.mkdir(exist_ok=True)
    slurm_dir.mkdir(exist_ok=True)
    if clear_out_files:
        for file in slurm_dir.glob(dtr_sbatch_tmp_fn.replace(".slurm", "*.out")):
            file.unlink()

    # filepath for slurm script
    sbatch_fp = slurm_dir.joinpath(dtr_sbatch_tmp_fn)
    # filepath for slurm stdout
    sbatch_out_fp = slurm_dir.joinpath(sbatch_fp.name.replace(".slurm", "_%A-%a.out"))

    config_path = slurm_dir.joinpath(dtr_sbatch_config_tmp_fn)
    array_range = write_config_file(
        config_path=config_path,
        models=models,
        scenarios=scenarios,
    )

    sbatch_head_kwargs = {
        "array_range": array_range,
        "partition": partition,
        "sbatch_out_fp": sbatch_out_fp,
        "conda_env_name": conda_env_name,
    }
    sbatch_head = make_sbatch_head(**sbatch_head_kwargs)

    sbatch_dtr_kwargs = {
        "sbatch_fp": sbatch_fp,
        "sbatch_out_fp": sbatch_out_fp,
        "worker_script": worker_script,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "sbatch_head": sbatch_head,
        "config_file": config_path,
    }
    write_sbatch_dtr(**sbatch_dtr_kwargs)
    job_id = submit_sbatch(sbatch_fp)

    print(job_id)
