r"""Script for constructing slurm jobs for computing the difference between arbitrary datasets.
Datasets are assumed to be in a flat file structure in the input directory and in zarr format. 

Example usage:
    # example for derived tasmin. 
    python run_cmip6_difference.py \
        --worker_script /home/kmredilla/repos/cmip6-utils/derived/difference.py \
        --conda_env_name cmip6-utils \
        --input_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/adjusted \
        --output_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/derived \
        --slurm_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/slurm \
        --minuend_tmp_fn tasmax_\${model}_\${scenario}_adjusted.zarr \
        --subtrahend_tmp_fn dtr_\${model}_\${scenario}_adjusted.zarr \
        --out_tmp_fn tasmin_\${model}_\${scenario}_adjusted.zarr \
        --new_var_id tasmin \
        --models "GFDL-ESM4 CESM2" \
        --scenarios "ssp245 ssp585" \
        --partition t2small
Notes:
  - notice the formatting of the minuend and subtrahend template file names with \${model} and \${scenario}.
  These are formatted so that model and scenario will be correctly substituted in the slurm job array.

Returns:
    Outputs are written in output_dir with out_tmp_fn formatted with the model and scenario names.
"""

import argparse
import subprocess
import logging
from pathlib import Path
from itertools import product
from config import diff_sbatch_tmp_fn, diff_sbatch_config_tmp_fn

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
    if not args.slurm_dir.parent.exists():
        raise FileNotFoundError(
            f"Parent of slurm directory, {args.slurm_dir.parent}, does not exist. Aborting."
        )

    args.models = args.models.split(" ")
    args.scenarios = args.scenarios.split(" ")

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
        "--minuend_tmp_fn",
        type=str,
        help="Template filename for the minuend data",
        required=True,
    )
    parser.add_argument(
        "--subtrahend_tmp_fn",
        type=str,
        help="template filename for the subtrahend data",
        required=True,
    )
    parser.add_argument(
        "--out_tmp_fn",
        type=str,
        help="Template filename for the output data",
        required=True,
    )
    parser.add_argument(
        "--new_var_id",
        type=str,
        help="New variable id for the output data",
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
        help="Path to directory containing CMIP6 data to be used as inputs",
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
        args.minuend_tmp_fn,
        args.subtrahend_tmp_fn,
        args.out_tmp_fn,
        args.new_var_id,
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
        f"#SBATCH --job-name=cmip6_difference\n"
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


def write_sbatch_diff(
    sbatch_fp,
    sbatch_out_fp,
    worker_script,
    input_dir,
    output_dir,
    minuend_tmp_fn,
    subtrahend_tmp_fn,
    out_tmp_fn,
    new_var_id,
    sbatch_head,
    config_file,
):
    """Write an sbatch array script for executing the dtr processing for a suite of models and scenarios.

    Args:
        sbatch_fp (path_like): path to .slurm script to write sbatch commands to
        sbatch_out_fp (path_like): path to where sbatch stdout should be written
        worker_script (path_like): path to the script to be called to run the dtr processing
        input_dir (path-like): path to parent directory containing CMIP6 data files
        output_dir (path-like): directory to write the derived data
        minuend_tmp_fn (str): template file name for the minuend data
        subtrahend_tmp_fn (str): template file name for the subtrahend data
        out_tmp_fn (str): template file name for the output data
        new_var_id (str): variable id for the output data
        sbatch_head (dict): string for sbatch head script
        config_file (path_like): path to the config file for the slurm job array
    Returns:
        None, writes the commands to sbatch_fp
    """

    pycommands = "\n"
    # these are template filepaths with $model and $scenario placeholders for slurm task array config
    minuend_store = input_dir.joinpath(minuend_tmp_fn)
    subtrahend_store = input_dir.joinpath(subtrahend_tmp_fn)
    output_store = output_dir.joinpath(out_tmp_fn)

    pycommands += (
        # Extract the model and scenario to process for the current $SLURM_ARRAY_TASK_ID
        f"config={config_file}\n"
        "model=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $2}' $config)\n"
        "scenario=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $3}' $config)\n"
        f"python {worker_script} "
        f"--minuend_store {minuend_store} "
        f"--subtrahend_store {subtrahend_store} "
        f"--output_store {output_store} "
        f"--new_var_id {new_var_id}\n"
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
        minuend_tmp_fn,
        subtrahend_tmp_fn,
        out_tmp_fn,
        new_var_id,
        models,
        scenarios,
        input_dir,
        output_dir,
        slurm_dir,
        partition,
        clear_out_files,
    ) = parse_args()

    output_dir.mkdir(exist_ok=True)
    if clear_out_files:
        for file in slurm_dir.glob(diff_sbatch_tmp_fn.replace(".slurm", "*.out")):
            file.unlink()

    # filepath for slurm script
    sbatch_fp = slurm_dir.joinpath(diff_sbatch_tmp_fn.format(new_var_id=new_var_id))
    # filepath for slurm stdout
    sbatch_out_fp = slurm_dir.joinpath(sbatch_fp.name.replace(".slurm", "_%A-%a.out"))

    config_path = slurm_dir.joinpath(
        diff_sbatch_config_tmp_fn.format(new_var_id=new_var_id)
    )
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

    sbatch_diff_kwargs = {
        "sbatch_fp": sbatch_fp,
        "sbatch_out_fp": sbatch_out_fp,
        "worker_script": worker_script,
        "minuend_tmp_fn": minuend_tmp_fn,
        "subtrahend_tmp_fn": subtrahend_tmp_fn,
        "out_tmp_fn": out_tmp_fn,
        "new_var_id": new_var_id,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "sbatch_head": sbatch_head,
        "config_file": config_path,
    }
    write_sbatch_diff(**sbatch_diff_kwargs)
    job_id = submit_sbatch(sbatch_fp)

    print(job_id)
