"""Script to build slurm files that run bias_adjust.py on a suite of models, scenarios, and variables.

example usage:
    python run_bias_adjust.py \
        --partition t2small \
        --conda_env_name cmip6-utils \
        --worker_script /beegfs/CMIP6/kmredilla/cmip6-utils/bias_adjust/bias_adjust.py \
        --sim_dir /beegfs/CMIP6/kmredilla/cmip6_4km_downscaling/cmip6_zarr/ \
        --train_dir /beegfs/CMIP6/kmredilla/cmip6_4km_downscaling/trained_datasets/ \
        --output_dir /beegfs/CMIP6/kmredilla/cmip6_4km_downscaling/adjusted \
        --tmp_dir /beegfs/CMIP6/kmredilla/cmip6_4km_downscaling/tmp \
        --models 'GFDL-ESM4 CESM2' \
        --scenarios 'historical ssp245' \
        --variables 'tasmax pr' \
        --slurm_dir /beegfs/CMIP6/kmredilla/cmip6_downscaling/slurm \
        --resolution 12km

Optional:
    --exclude 'sfcWind:CNRM-CM6-1-HR:historical sfcWind:EC-Earth3-Veg:historical'
    --skip_complete        skip combos whose output already passes the cheap completeness check
    --dry_run              write the .slurm files but do not submit
    --extra_args '--rank_main_only true --chunk_size 25'   passed through to bias_adjust.py
    --time_limit/--cpus_per_task/--mem   override config.SLURM_BIAS_ADJUST
"""

import argparse
import logging
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
    bias_adjust_sbatch_tmp_fn,
    SLURM_BIAS_ADJUST,
)
from utils import validate_path_arg, check_for_input_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def get_sim_path(sim_dir, model, scenario, var_id):
    """Get the filepath to a cmip6 file in sim_dir given the attributes."""
    return sim_dir.joinpath(
        cmip6_zarr_tmp_fn.format(model=model, scenario=scenario, var_id=var_id),
    )


def validate_args(args):
    """Validate the arguments passed to the script."""
    args.worker_script = Path(args.worker_script)
    args.output_dir = Path(args.output_dir)
    args.tmp_dir = Path(args.tmp_dir)
    args.sim_dir = Path(args.sim_dir)
    args.train_dir = Path(args.train_dir)
    args.slurm_dir = Path(args.slurm_dir)
    validate_path_arg(args.worker_script, "worker_script")
    validate_path_arg(args.sim_dir, "sim_dir")
    validate_path_arg(args.train_dir, "train_dir")
    validate_path_arg(args.output_dir.parent, "parent of output_dir")
    validate_path_arg(args.tmp_dir, "tmp_dir")
    validate_path_arg(args.slurm_dir, "slurm_dir")

    args.models = args.models.split(" ")
    args.scenarios = args.scenarios.split(" ")
    args.variables = args.variables.split(" ")
    args.exclude = set(args.exclude.split()) if args.exclude else set()
    expected_stores = [
        get_sim_path(args.sim_dir, model, scenario, var_id)
        for var_id, model, scenario in product(
            args.variables, args.models, args.scenarios
        )
    ]
    check_for_input_data(expected_stores)

    if args.time_limit is None:
        args.time_limit = SLURM_BIAS_ADJUST["time_limit"][args.resolution]
    if args.cpus_per_task is None:
        args.cpus_per_task = SLURM_BIAS_ADJUST["cpus_per_task"][args.resolution]
    if args.mem is None:
        args.mem = SLURM_BIAS_ADJUST["mem"].get((args.resolution, args.partition))
    elif args.mem.lower() == "none":
        args.mem = None

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
        help="Path to bias_adjust.py",
        required=True,
    )
    parser.add_argument(
        "--sim_dir",
        type=str,
        help="Path to directory containing sim (e.g. GCM) zarr stores",
        required=True,
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        help="Path to directory containing trained datasets zarr stores",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory where bias-adjusted data will be written.",
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
        "--slurm_dir",
        type=str,
        help="Path to directory where slurm files and logs will be written.",
        required=True,
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=sorted(SLURM_BIAS_ADJUST["time_limit"].keys()),
        default="12km",
        help="Resolution label; selects the default time limit, cpus and mem (config.SLURM_BIAS_ADJUST)",
    )
    parser.add_argument("--time_limit", type=str, default=None, help="HH:MM:SS")
    parser.add_argument("--cpus_per_task", type=int, default=None)
    parser.add_argument(
        "--mem",
        type=str,
        default=None,
        help="sbatch --mem value, e.g. 300G; 'none' to omit (whole exclusive node)",
    )
    parser.add_argument(
        "--extra_args",
        type=str,
        default="",
        help="Extra command-line arguments passed verbatim to the worker script",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="' '-separated var:model:scenario combos to skip",
    )
    parser.add_argument(
        "--skip_complete",
        action="store_true",
        help="Skip combos whose output store already passes the cheap completeness check",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write the slurm scripts but do not submit them",
    )
    parser.add_argument(
        "--clear_out_files",
        action="store_true",
        help="Remove previous bias_adjust .out logs in the slurm directory before running",
    )
    args = parser.parse_args()
    args = validate_args(args)
    return args


def output_is_complete(adj_path, sim_path):
    """Cheap completeness check (metadata only) via check_adjusted_complete."""
    from check_adjusted_complete import cheap_check

    result = cheap_check(adj_path, sim_path)
    return result["status"] == "OK"


def write_sbatch_bias_adjust(
    sim_dir,
    train_dir,
    slurm_dir,
    output_dir,
    tmp_dir,
    model,
    scenario,
    var_id,
    worker_script,
    sbatch_head_kwargs,
    extra_args="",
    exclude=(),
    skip_complete=False,
):
    """Write the sbatch file for bias adjustment for a given model, scenario, and variable."""
    if f"{var_id}:{model}:{scenario}" in exclude:
        logging.info(f"Excluded by request: {model} {scenario} {var_id}.")
        return

    train_path = train_dir.joinpath(
        trained_qm_tmp_fn.format(var_id=var_id, model=model)
    )
    if not train_path.exists():
        logging.info(
            f"Trained QM object {train_path} not found. Skipping {model} {scenario} {var_id}.",
        )
        return

    sim_path = get_sim_path(sim_dir, model, scenario, var_id)
    if not sim_path.exists():
        logging.info(
            f"GCM data {sim_path} not found. Skipping {model} {scenario} {var_id}.",
        )
        return

    adj_path = output_dir.joinpath(
        cmip6_adjusted_tmp_fn.format(var_id=var_id, model=model, scenario=scenario)
    )
    if skip_complete and adj_path.exists() and output_is_complete(adj_path, sim_path):
        logging.info(f"Output already complete, skipping: {adj_path.name}")
        return

    tmp_path = Path(tmp_dir)

    # create the sbatch file
    sbatch_path = slurm_dir.joinpath(
        bias_adjust_sbatch_tmp_fn.format(model=model, scenario=scenario, var_id=var_id)
    )
    sbatch_out_path = slurm_dir.joinpath(sbatch_path.name.replace(".slurm", "_%j.out"))

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
        f"--tmp_path {tmp_path}"
    )
    if extra_args:
        pycommands += f" \\\n{extra_args}"
    pycommands += "\n"

    pycommands += "\nstatus=$?\n\n"

    pycommands += (
        f"echo End {var_id} bias adjustment with exit status $status && date\n"
        "echo Job Completed\n"
        "exit $status\n"
    )
    commands = sbatch_head + pycommands

    with open(sbatch_path, "w") as f:
        f.write(commands)
        logging.info(f"sbatch file written to {sbatch_path}")

    return sbatch_path


def write_all_sbatch_bias_adjust(
    sim_dir,
    train_dir,
    output_dir,
    tmp_dir,
    slurm_dir,
    worker_script,
    models,
    scenarios,
    variables,
    sbatch_head_kwargs,
    extra_args="",
    exclude=(),
    skip_complete=False,
):
    """Write the sbatch file for bias adjustment."""
    # create a list of all the combinations of models, scenarios, and variables
    sbatch_kwargs = {
        "sim_dir": sim_dir,
        "train_dir": train_dir,
        "slurm_dir": slurm_dir,
        "output_dir": output_dir,
        "tmp_dir": tmp_dir,
        "worker_script": worker_script,
        "sbatch_head_kwargs": sbatch_head_kwargs,
        "extra_args": extra_args,
        "exclude": exclude,
        "skip_complete": skip_complete,
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
    args = parse_args()

    args.output_dir.mkdir(exist_ok=True)
    args.slurm_dir.mkdir(exist_ok=True)
    # Create subdirectory for bias adjustment slurm outputs
    bias_adjust_slurm_dir = args.slurm_dir.joinpath("bias_adjust")
    bias_adjust_slurm_dir.mkdir(exist_ok=True)

    if args.clear_out_files:
        for file in bias_adjust_slurm_dir.glob(
            bias_adjust_sbatch_tmp_fn.format(
                model="*", var_id="*", scenario="*"
            ).replace(".slurm", "_*.out")
        ):
            file.unlink()

    sbatch_head_kwargs = {
        "partition": args.partition,
        "conda_env_name": args.conda_env_name,
        "time_limit": args.time_limit,
        "cpus_per_task": args.cpus_per_task,
        "mem": args.mem,
    }
    logging.info(
        f"SLURM settings: partition={args.partition} time={args.time_limit} "
        f"cpus-per-task={args.cpus_per_task} mem={args.mem}"
    )
    all_sbatch_kwargs = {
        "sim_dir": args.sim_dir,
        "train_dir": args.train_dir,
        "slurm_dir": bias_adjust_slurm_dir,
        "output_dir": args.output_dir,
        "tmp_dir": args.tmp_dir,
        "worker_script": args.worker_script,
        "sbatch_head_kwargs": sbatch_head_kwargs,
        "models": args.models,
        "scenarios": args.scenarios,
        "variables": args.variables,
        "extra_args": args.extra_args,
        "exclude": args.exclude,
        "skip_complete": args.skip_complete,
    }
    sbatch_paths = write_all_sbatch_bias_adjust(**all_sbatch_kwargs)
    logging.info(f"{len(sbatch_paths)} job script(s) written")

    if args.dry_run:
        logging.info("Dry run: not submitting")
    else:
        job_ids = [submit_sbatch(sbatch_path) for sbatch_path in sbatch_paths]
        _ = [print(job_id, end=" ") for job_id in job_ids]
        print()
