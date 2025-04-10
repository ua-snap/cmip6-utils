"""Script to build and submit slurm files that run train_qm.py on a suite of models, scenarios, and variables.

Notes:
- Writes the trained object to the input directory, as this remains the input directory for the bias adjustment effort.
- only trains on historical GCM data, no other CMIP6 "experiments" (i.e. SSPs) are used.

example usage:
    python run_train_qm.py \
        --partition t2small \
        --conda_env_name cmip6-utils \
        --worker_script /home/kmredilla/repos/cmip6-utils/train_qm.py \
        --input_dir /center1/CMIP6/kmredilla/zarr_bias_adjust_inputs/ \
        --models 'GFDL-ESM4 CESM2' \
        --variables 'tasmax pr' \
        --slurm_dir /center1/CMIP6/kmredilla/cmip6_qdm_downscaling/slurm \
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
    ref_zarr_tmp_fn,
    trained_qm_tmp_fn,
    train_qm_sbatch_tmp_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


target_dir_name = "zarr"


def get_hist_path(input_dir, model, var_id):
    """Get the filepath to a historical cmip6 file in input_dir given the attributes."""
    return input_dir.joinpath(
        cmip6_zarr_tmp_fn.format(model=model, scenario="historical", var_id=var_id),
    )


def validate_args(args):
    """Validate the arguments passed to the script."""
    args.worker_script = Path(args.worker_script)
    args.input_dir = Path(args.input_dir)
    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Input directory, {args.input_dir}, does not exist. Aborting."
        )
    args.slurm_dir = Path(args.slurm_dir)
    if not args.slurm_dir.exists():
        raise FileNotFoundError(
            f"Slurm directory, {args.slurm_dir}, does not exist. Aborting."
        )

    args.models = args.models.split(" ")
    args.variables = args.variables.split(" ")
    expected_stores_in_input_dir = [
        get_hist_path(args.input_dir, model, var_id)
        for var_id, model in product(args.variables, args.models)
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
                f"Some model / variable combinations were not found in the input directory and will be skipped: \n{missing_stores_str}\n"
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
        "--variables",
        type=str,
        help="' '-separated list of variables to work on",
        required=True,
    )
    parser.add_argument(
        "--slurm_dir",
        type=str,
        help="Path to directory where slurm scripts and logs will be written.",
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
        args.variables,
        args.slurm_dir,
        args.clear_out_files,
    )


def write_sbatch_train_qm(
    input_dir,
    slurm_dir,
    output_dir,
    model,
    var_id,
    worker_script,
    sbatch_head_kwargs,
):
    """Write the sbatch file for QM training for a given model and variable."""
    sim_path = get_hist_path(input_dir, model, var_id)
    if not sim_path.exists():
        warnings.warn(
            f"GCM data {sim_path} not found. Skipping {model} historical {var_id}.",
            UserWarning,
        )
        return
    ref_path = input_dir.joinpath(ref_zarr_tmp_fn.format(var_id=var_id))
    train_path = output_dir.joinpath(
        trained_qm_tmp_fn.format(var_id=var_id, model=model)
    )
    # create the sbatch file
    sbatch_path = slurm_dir.joinpath(
        train_qm_sbatch_tmp_fn.format(model=model, var_id=var_id)
    )
    sbatch_out_path = slurm_dir.joinpath(sbatch_path.name.replace(".sbatch", "_%j.out"))

    sbatch_head_kwargs.update(
        {
            "sbatch_out_path": sbatch_out_path,
            "job_name": f"train_qm_{model}_{var_id}",
        }
    )

    sbatch_head = make_sbatch_head(**sbatch_head_kwargs)

    pycommands = "\n"
    pycommands += (
        f"python {worker_script} "
        f"--sim_path {sim_path} "
        f"--ref_path {ref_path} "
        f"--train_path {train_path} "
    )

    pycommands += "\n\n"

    pycommands += f"echo End {var_id} QM training && date\n" "echo Job Completed"
    commands = sbatch_head + pycommands

    with open(sbatch_path, "w") as f:
        f.write(commands)
        logging.info(f"sbatch file written to {sbatch_path}")

    return sbatch_path


def write_all_sbatch_train_qm(
    input_dir,
    output_dir,
    slurm_dir,
    worker_script,
    models,
    variables,
    sbatch_head_kwargs,
):
    """Write the sbatch file for QM training."""
    # create a list of all the combinations of models, scenarios, and variables
    sbatch_kwargs = {
        "input_dir": input_dir,
        "slurm_dir": slurm_dir,
        "output_dir": output_dir,
        "worker_script": worker_script,
        "sbatch_head_kwargs": sbatch_head_kwargs,
    }
    combinations = list(product(models, variables))

    sbatch_paths = []
    for model, var_id in combinations:
        sbatch_kwargs.update({"model": model, "var_id": var_id})
        sbatch_path = write_sbatch_train_qm(**sbatch_kwargs)
        sbatch_paths.append(sbatch_path)

    return sbatch_paths


if __name__ == "__main__":

    (
        partition,
        conda_env_name,
        worker_script,
        input_dir,
        models,
        variables,
        slurm_dir,
        clear_out_files,
    ) = parse_args()

    output_dir = input_dir
    if clear_out_files:
        for file in slurm_dir.glob(
            train_qm_sbatch_tmp_fn.format(model="*", var_id="*").replace(
                ".sbatch", ".out"
            )
        ):
            file.unlink()

    sbatch_head_kwargs = {
        "partition": partition,
        "conda_env_name": conda_env_name,
    }
    all_sbatch_kwargs = {
        "input_dir": input_dir,
        "slurm_dir": slurm_dir,
        "output_dir": output_dir,
        "worker_script": worker_script,
        "sbatch_head_kwargs": sbatch_head_kwargs,
        "models": models,
        "variables": variables,
    }

    sbatch_paths = write_all_sbatch_train_qm(**all_sbatch_kwargs)

    # job_ids = [submit_sbatch(sbatch_path) for sbatch_path in sbatch_paths]
    job_ids = [123456, 234567, 345678]  # Mock job IDs for testing
    print(job_ids)
