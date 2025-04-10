"""Script to build slurm files that run bias_adjust.py on a suite of models, scenarios, and variables.

example usage:
    python run_bias_adjust.py \
        --partition t2small \
        --conda_env_name cmip6-utils \
        --worker_script /home/kmredilla/repos/cmip6-utils/bias_adjust.py \
        --input_dir /center1/CMIP6/kmredilla/zarr_bias_adjust_inputs/ \
        --models 'GFDL-ESM4 CESM2' \
        --scenarios 'historical ssp245' \
        --variables 'tasmax pr' \
        --output_dir /center1/CMIP6/kmredilla/cmip6_4km_3338_adjusted
"""

import argparse
import logging
import warnings
from itertools import product
from pathlib import Path
from slurm import (
    make_sbatch_head,
    submit_sbatch,
)
from config import (
    cmip6_zarr_tmp_fn,
    cmip6_adjusted_tmp_fn,
    trained_qm_tmp_fn,
    biasadjust_sbatch_tmp_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


target_dir_name = "zarr"


def get_sim_path(input_dir, model, scenario, var_id):
    """Get the filepath to a cmip6 file in input_dir given the attributes."""
    return input_dir.joinpath(
        cmip6_zarr_tmp_fn.format(model=model, scenario=scenario, var_id=var_id),
    )


def validate_args(args):
    """Validate the arguments passed to the script."""
    args.worker_script = Path(args.worker_script)
    args.output_dir = Path(args.output_dir)
    if not args.output_dir.parent.exists():
        raise FileNotFoundError(
            f"Parent of output directory, {args.output_dir.parent}, does not exist. Aborting."
        )
    args.input_dir = Path(args.input_dir)
    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Input directory, {args.input_dir}, does not exist. Aborting."
        )

    args.models = args.models.split(" ")
    args.scenarios = args.scenarios.split(" ")
    args.variables = args.variables.split(" ")
    expected_stores_in_input_dir = [
        get_sim_path(args.input_dir, model, scenario, var_id)
        for var_id, model, scenario in product(
            args.variables, args.models, args.scenarios
        )
    ]
    found_stores_in_input_dir = [
        store for store in expected_stores_in_input_dir if store.exists()
    ]
    if not any(found_stores_in_input_dir):
        raise ValueError(
            f"No zarr stores in the input directory ({args.input_dir}) match the models / scenarios / variables supplied. Aborting."
        )
    else:
        missing_stores = set(expected_stores_in_input_dir) - set(
            found_stores_in_input_dir
        )
        if not len(missing_stores) == 0:
            missing_stores_str = "\n".join(
                [f"- {str(store)}" for store in list(missing_stores)]
            )
            logging.warning(
                f"Some model / scenario / variable combinations were not found in the input directory and will be skipped: \n{missing_stores_str}\n"
            )

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
        "--input_dir",
        type=str,
        help="Path to directory containing all input zarr stores (including trained QM objects)",
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
        help="Path to directory where bias-adjusted data will be written.",
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
        args.input_dir,
        args.models,
        args.scenarios,
        args.variables,
        args.output_dir,
        args.clear_out_files,
    )


def write_sbatch_bias_adjust(
    input_dir,
    slurm_dir,
    target_dir,
    model,
    scenario,
    var_id,
    worker_script,
    sbatch_head_kwargs,
):
    """Write the sbatch file for bias adjustment for a given model, scenario, and variable."""
    train_path = input_dir.joinpath(
        trained_qm_tmp_fn.format(var_id=var_id, model=model)
    )
    if not train_path.exists():
        warnings.warn(
            f"Trained QM object {train_path} not found. Skipping {model} {scenario} {var_id}.",
            UserWarning,
        )
        return

    sim_path = get_sim_path(input_dir, model, scenario, var_id)
    if not sim_path.exists():
        warnings.warn(
            f"GCM data {sim_path} not found. Skipping {model} {scenario} {var_id}.",
            UserWarning,
        )
        return

    adj_path = target_dir.joinpath(
        cmip6_adjusted_tmp_fn.format(var_id=var_id, model=model, scenario=scenario)
    )

    # create the sbatch file
    sbatch_path = slurm_dir.joinpath(
        biasadjust_sbatch_tmp_fn.format(model=model, scenario=scenario, var_id=var_id)
    )
    sbatch_out_path = slurm_dir.joinpath(sbatch_path.name.replace(".sbatch", "_%j.out"))

    sbatch_head_kwargs.update(
        {
            "sbatch_out_path": sbatch_out_path,
            "job_name": f"bias_adjust_{model}_{scenario}_{var_id}",
        }
    )

    sbatch_head = make_sbatch_head(**sbatch_head_kwargs)

    pycommands = "\n"
    pycommands += (
        f"python {worker_script} \\\n"
        f"--train_path {train_path} \\\n"
        f"--sim_path {sim_path} \\\n"
        f"--adj_path {adj_path} \\\n"
    )

    pycommands += "\n\n"

    pycommands += f"echo End {var_id} bias adjustment && date\n" "echo Job Completed"
    commands = sbatch_head + pycommands

    with open(sbatch_path, "w") as f:
        f.write(commands)
        logging.info(f"sbatch file written to {sbatch_path}")

    return sbatch_path


def write_all_sbatch_bias_adjust(
    input_dir,
    target_dir,
    slurm_dir,
    worker_script,
    models,
    scenarios,
    variables,
    sbatch_head_kwargs,
):
    """Write the sbatch file for bias adjustment."""
    # create a list of all the combinations of models, scenarios, and variables
    sbatch_kwargs = {
        "input_dir": input_dir,
        "slurm_dir": slurm_dir,
        "target_dir": target_dir,
        "worker_script": worker_script,
        "sbatch_head_kwargs": sbatch_head_kwargs,
    }
    combinations = list(product(models, scenarios, variables))

    sbatch_paths = []
    for model, scenario, var_id in combinations:
        sbatch_kwargs.update({"model": model, "scenario": scenario, "var_id": var_id})
        sbatch_path = write_sbatch_bias_adjust(**sbatch_kwargs)
        if sbatch_path is not None:
            sbatch_paths.append(sbatch_path)

    return sbatch_paths


if __name__ == "__main__":

    (
        partition,
        conda_env_name,
        worker_script,
        input_dir,
        models,
        scenarios,
        variables,
        output_dir,
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

    sbatch_head_kwargs = {
        "partition": partition,
        "conda_env_name": conda_env_name,
    }
    all_sbatch_kwargs = {
        "input_dir": input_dir,
        "slurm_dir": slurm_dir,
        "target_dir": target_dir,
        "worker_script": worker_script,
        "sbatch_head_kwargs": sbatch_head_kwargs,
        "models": models,
        "scenarios": scenarios,
        "variables": variables,
    }
    sbatch_paths = write_all_sbatch_bias_adjust(**all_sbatch_kwargs)

    job_ids = [submit_sbatch(sbatch_path) for sbatch_path in sbatch_paths]
    print(job_ids)
