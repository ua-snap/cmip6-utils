"""Script to build and submit slurm files that run train_qm.py on a suite of models, scenarios, and variables.

Notes:
- only trains on "historical" GCM data, no other CMIP6 "experiments" (i.e. SSPs) are used.

example usage:
    python run_train_qm.py \
        --partition t2small \
        --conda_env_name cmip6-utils \
        --worker_script /beegfs/CMIP6/kmredilla/cmip6-utils/bias_adjust/train_qm.py \
        --sim_dir /beegfs/CMIP6/kmredilla/cmip6_4km_downscaling/cmip6_zarr/ \
        --ref_dir /beegfs/CMIP6/kmredilla/cmip6_4km_downscaling/era5_zarr/ \
        --output_dir /beegfs/CMIP6/kmredilla/cmip6_downscaling/optimized_inputs/ \
        --tmp_dir /center1/CMIP6/kmredilla/tmp \
        --models 'GFDL-ESM4 CESM2' \
        --variables 'tasmax pr' \
        --slurm_dir /beegfs/CMIP6/kmredilla/cmip6_downscaling/slurm
"""

import argparse
import logging
from itertools import product
from pathlib import Path
from slurm import (
    make_sbatch_head,
    submit_sbatch,
)
from utils import validate_path_arg, check_for_input_data
from config import (
    cmip6_zarr_tmp_fn,
    era5_zarr_tmp_fn,
    trained_qm_tmp_fn,
    train_qm_sbatch_tmp_fn,
)
from luts import sim_ref_var_lu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def get_hist_path(sim_dir, model, var_id):
    """Get the filepath to a historical cmip6 file in sim_dir given the attributes."""
    return sim_dir.joinpath(
        cmip6_zarr_tmp_fn.format(model=model, scenario="historical", var_id=var_id),
    )


def validate_args(args):
    """Validate the arguments passed to the script."""
    args.worker_script = Path(args.worker_script)
    args.sim_dir = Path(args.sim_dir)
    args.ref_dir = Path(args.ref_dir)
    args.output_dir = Path(args.output_dir)
    args.tmp_dir = Path(args.tmp_dir)
    args.slurm_dir = Path(args.slurm_dir)
    validate_path_arg(args.worker_script, "worker_script")
    validate_path_arg(args.sim_dir, "sim_dir")
    validate_path_arg(args.ref_dir, "ref_dir")
    validate_path_arg(args.output_dir.parent, "parent of output_dir")
    validate_path_arg(args.tmp_dir, "tmp_dir")
    validate_path_arg(args.slurm_dir, "slurm_dir")

    args.models = args.models.split(" ")
    args.variables = args.variables.split(" ")
    expected_stores = [
        get_hist_path(args.sim_dir, model, var_id)
        for var_id, model in product(args.variables, args.models)
    ]
    check_for_input_data(expected_stores)

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
        "--sim_dir",
        type=str,
        help="Path to directory containing all biased/sim zarr stores",
        required=True,
    )
    parser.add_argument(
        "--ref_dir",
        type=str,
        help="Path to directory containing reference data zarr stores",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory where trained adjustment object datasets will be written.",
        required=True,
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        help="Path to directory where dask temporary files will be written.",
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
        args.sim_dir,
        args.ref_dir,
        args.output_dir,
        args.tmp_dir,
        args.models,
        args.variables,
        args.slurm_dir,
        args.clear_out_files,
    )


def write_sbatch_train_qm(
    sim_dir,
    ref_dir,
    slurm_dir,
    output_dir,
    tmp_dir,
    model,
    var_id,
    worker_script,
    sbatch_head_kwargs,
):
    """Write the sbatch file for QM training for a given model and variable."""
    sim_path = get_hist_path(sim_dir, model, var_id)
    if not sim_path.exists():
        logging.info(
            f"GCM data {sim_path} not found. Skipping {model} historical {var_id}.",
        )
        return
    ref_var_id = sim_ref_var_lu.get(var_id)
    ref_path = ref_dir.joinpath(era5_zarr_tmp_fn.format(var_id=ref_var_id))
    train_path = output_dir.joinpath(
        trained_qm_tmp_fn.format(var_id=var_id, model=model)
    )
    tmp_path = Path(tmp_dir)
    # create the sbatch file
    sbatch_path = slurm_dir.joinpath(
        train_qm_sbatch_tmp_fn.format(model=model, var_id=var_id)
    )
    sbatch_out_path = slurm_dir.joinpath(sbatch_path.name.replace(".slurm", "_%j.out"))

    sbatch_head_kwargs.update(
        {
            "sbatch_out_path": sbatch_out_path,
            "job_name": f"train_qm_{model}_{var_id}",
        }
    )

    sbatch_head = make_sbatch_head(**sbatch_head_kwargs)

    pycommands = "\n"
    pycommands += (
        f"python {worker_script} \\\n"
        f"--sim_path {sim_path} \\\n"
        f"--ref_path {ref_path} \\\n"
        f"--train_path {train_path} \\\n"
        f"--tmp_path {tmp_path} \n"
    )

    pycommands += "\n\n"

    pycommands += f"echo End {var_id} QM training && date\n" "echo Job Completed"
    commands = sbatch_head + pycommands

    with open(sbatch_path, "w") as f:
        f.write(commands)
        logging.info(f"sbatch file written to {sbatch_path}")

    return sbatch_path


def write_all_sbatch_train_qm(
    sim_dir,
    ref_dir,
    slurm_dir,
    output_dir,
    tmp_dir,
    worker_script,
    models,
    variables,
    sbatch_head_kwargs,
):
    """Write the sbatch file for QM training."""
    # create a list of all the combinations of models, scenarios, and variables
    sbatch_kwargs = {
        "sim_dir": sim_dir,
        "ref_dir": ref_dir,
        "slurm_dir": slurm_dir,
        "output_dir": output_dir,
        "tmp_dir": tmp_dir,
        "worker_script": worker_script,
        "sbatch_head_kwargs": sbatch_head_kwargs,
    }
    combinations = list(product(models, variables))

    sbatch_paths = []
    for model, var_id in combinations:
        sbatch_kwargs.update({"model": model, "var_id": var_id})
        sbatch_path = write_sbatch_train_qm(**sbatch_kwargs)
        if sbatch_path is not None:
            sbatch_paths.append(sbatch_path)

    return sbatch_paths


if __name__ == "__main__":

    (
        partition,
        conda_env_name,
        worker_script,
        sim_dir,
        ref_dir,
        output_dir,
        tmp_dir,
        models,
        variables,
        slurm_dir,
        clear_out_files,
    ) = parse_args()

    output_dir.mkdir(exist_ok=True)
    if clear_out_files:
        for file in slurm_dir.glob(
            train_qm_sbatch_tmp_fn.format(model="*", var_id="*").replace(
                ".slurm", ".out"
            )
        ):
            file.unlink()

    sbatch_head_kwargs = {
        "partition": partition,
        "conda_env_name": conda_env_name,
    }
    all_sbatch_kwargs = {
        "sim_dir": sim_dir,
        "ref_dir": ref_dir,
        "slurm_dir": slurm_dir,
        "output_dir": output_dir,
        "tmp_dir": tmp_dir,
        "worker_script": worker_script,
        "sbatch_head_kwargs": sbatch_head_kwargs,
        "models": models,
        "variables": variables,
    }

    sbatch_paths = write_all_sbatch_train_qm(**all_sbatch_kwargs)

    job_ids = [submit_sbatch(sbatch_path) for sbatch_path in sbatch_paths]
    _ = [print(job_id, end=" ") for job_id in job_ids]
