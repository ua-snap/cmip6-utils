"""Functions to assist with constructing slurm jobs"""

import argparse
import subprocess
from pathlib import Path
from config import output_dir_name


def make_sbatch_head(partition, conda_init_script, ncpus):
    """Make a string of SBATCH commands that can be written into a .slurm script

    Args:
        partition (str): name of the partition to use
        conda_init_script (path_like): path to a script that contains commands for initializing the shells on the compute nodes to use conda activate
        ncpus (int): number of cpus to request

    Returns:
        sbatch_head (str): string of SBATCH commands ready to be used as parameter in sbatch-writing functions. The following gaps are left for filling with .format:
            - ncpus
            - output slurm filename
    """
    sbatch_head = (
        "#!/bin/sh\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --cpus-per-task={ncpus}\n"
        "#SBATCH --mail-type=FAIL\n"
        f"#SBATCH -p {partition}\n"
        "#SBATCH --output {sbatch_out_fp}\n"
        # print start time
        "echo Start slurm && date\n"
        # prepare shell for using activate - Chinook requirement
        f"source {conda_init_script}\n"
        # okay this is not the desired way to do this, but Chinook compute
        # nodes are not working with anaconda-project, so we activate
        # this manually then run the python command
        f"conda activate cmip6-utils\n"
    )

    return sbatch_head


def write_sbatch_biasadjust(
    sbatch_fp,
    sbatch_out_fp,
    var_id,
    model,
    input_dir,
    reference_dir,
    biasadjust_script,
    adj_dir,
    sbatch_head,
    scenario=None,
):
    """Write an sbatch script for executing the bias adjustment script for a given model, scenario, and variable

    Args:
        sbatch_fp (path_like): path to .slurm script to write sbatch commands to
        sbatch_out_fp (path_like): path to where sbatch stdout should be written
        var_id (str): name of CMIP6 variable ID being adjusted
        model (str): model being adjusted
        scenario (str): scenario being adjusted
        input_dir (path-like): path to directory of files to be adjusted (likely the regridded files)
        biasadjust_script (path_like): path to the script to be called to run the bias adjustment
        adj_dir (path-like): directory to write the adjusted data
        sbatch_head (dict): string for sbatch head script

    Returns:
        None, writes the commands to sbatch_fp

    Notes:
        since these jobs seem to take on the order of 5 minutes or less, seems better to just run through all years once a node is secured for a job, instead of making a single job for every year / variable combination
    """
    pycommands = "\n"
    pycommands += (
        f"python {biasadjust_script} "
        f"--var_id {var_id} "
        f"--model {model} "
        f"--input_dir {input_dir} "
        f"--reference_dir {reference_dir} "
        f"--adj_dir {adj_dir} "
    )
    if scenario is not None:
        pycommands += f"--scenario {scenario} "

    pycommands += "\n\n"

    pycommands += f"echo End {var_id} bias adjustment && date\n" "echo Job Completed"
    commands = sbatch_head.format(sbatch_out_fp=sbatch_out_fp) + pycommands

    with open(sbatch_fp, "w") as f:
        f.write(commands)

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


def get_directories(working_dir, output_dir_name):
    """Get the output directory path which will contain all subfolders of outputs (data, slurm etc). Function for sharing."""
    output_dir = working_dir.joinpath(output_dir_name)
    adj_dir = output_dir.joinpath("netcdf")

    return output_dir, adj_dir


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--var_ids",
        type=str,
        help="' '-separated list of CMIP6 variable IDs to adjust",
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
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        help="Path to directory of reference data with filepath structure <variable ID>/<files>",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        help="Path to working directory, where outputs and ancillary files will be written",
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="slurm partition",
    )
    parser.add_argument(
        "--ncpus",
        type=str,
        help="CPUs per node",
    )
    args = parser.parse_args()

    return (
        args.var_ids.split(" "),
        args.models.split(" "),
        args.scenarios.split(" "),
        Path(args.input_dir),
        Path(args.reference_dir),
        Path(args.working_dir),
        args.partition,
        args.ncpus,
    )


if __name__ == "__main__":
    (
        var_ids,
        models,
        scenarios,
        input_dir,
        reference_dir,
        working_dir,
        partition,
        ncpus,
    ) = parse_args()

    working_dir.mkdir(exist_ok=True)
    output_dir, adj_dir = get_directories(working_dir, output_dir_name)
    adj_dir.mkdir(parents=True, exist_ok=True)

    # make batch files for each model / scenario / variable combination
    sbatch_dir = output_dir.joinpath("slurm")
    sbatch_dir.mkdir(exist_ok=True)
    _ = [fp.unlink() for fp in sbatch_dir.glob("*.slurm")]

    # sbatch head - replaces config.py params for now!
    sbatch_head_kwargs = {
        "partition": partition,
        "ncpus": ncpus,
        "conda_init_script": working_dir.joinpath(
            "cmip6-utils/bias_adjust/conda_init.sh"
        ),
    }

    biasadjust_script = working_dir.joinpath("cmip6-utils/bias_adjust/bias_adjust.py")

    job_ids = []
    for model in models:
        for scenario in scenarios:
            for var_id in var_ids:
                # filepath for slurm script
                sbatch_fp = sbatch_dir.joinpath(
                    f"{var_id}_{model}_{scenario}_biasadjust.slurm"
                )
                # filepath for slurm stdout
                sbatch_out_fp = sbatch_dir.joinpath(
                    sbatch_fp.name.replace(".slurm", "_%j.out")
                )
                # excluding node 138 until issue resolved
                sbatch_head = make_sbatch_head(**sbatch_head_kwargs)
                sbatch_biasadjust_kwargs = {
                    "sbatch_fp": sbatch_fp,
                    "sbatch_out_fp": sbatch_out_fp,
                    "var_id": var_id,
                    "model": model,
                    "scenario": scenario,
                    "input_dir": input_dir,
                    "biasadjust_script": biasadjust_script,
                    "reference_dir": reference_dir,
                    "adj_dir": adj_dir,
                    "sbatch_head": sbatch_head,
                }

                if scenario == "historical":
                    del sbatch_biasadjust_kwargs["scenario"]
                    sbatch_biasadjust_kwargs[
                        "biasadjust_script"
                    ] = working_dir.joinpath(
                        "cmip6-utils/bias_adjust/bias_adjust_historical.py"
                    )

                write_sbatch_biasadjust(**sbatch_biasadjust_kwargs)

                job_id = submit_sbatch(sbatch_fp)

                sbatch_out_fp_with_jobid = sbatch_dir.joinpath(
                    sbatch_out_fp.name.replace("%j", str(job_id))
                )
                job_ids.append(job_id)

    # currently printing Job IDs to give to next steps in prefect. Probably better ways to do this.
    print(job_ids)