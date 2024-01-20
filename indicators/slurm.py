"""Functions to assist with constructing slurm jobs"""

import argparse
import subprocess
from config import *


def make_sbatch_head(slurm_email, partition, conda_init_script, ncpus, exclude_nodes):
    """Make a string of SBATCH commands that can be written into a .slurm script

    Args:
        slurm_email (str): email address for slurm failures
        partition (str): name of the partition to use
        conda_init_script (path_like): path to a script that contains commands for initializing the shells on the compute nodes to use conda activate
        ncpus (int): number of cpus to request
        exclude_nodes (str): comma-separated string of nodes to exclude

    Returns:
        sbatch_head (str): string of SBATCH commands ready to be used as parameter in sbatch-writing functions. The following gaps are left for filling with .format:
            - ncpus
            - output slurm filename
    """
    sbatch_head = (
        "#!/bin/sh\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --exclude={exclude_nodes}\n"
        f"#SBATCH --cpus-per-task={ncpus}\n"
        "#SBATCH --mail-type=FAIL\n"
        f"#SBATCH --mail-user={slurm_email}\n"
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


def write_sbatch_indicators(
    sbatch_fp,
    sbatch_out_fp,
    indicator,
    model,
    scenario,
    input_dir,
    indicators_script,
    indicators_dir,
    no_clobber,
    sbatch_head,
):
    """Write an sbatch script for executing the indicators script for a given group and variable, executes for a given list of years

    Args:
        sbatch_fp (path_like): path to .slurm script to write sbatch commands to
        sbatch_out_fp (path_like): path to where sbatch stdout should be written
        indicators_script (path_like): path to the script to be called to run the indicators
        regrid_dir (pathlib.PosixPath): directory to write the regridded data to
        no_clobber (bool): do not overwrite regridded files if they exist in regrid_dir
        sbatch_head (dict): string for sbatch head script

    Returns:
        None, writes the commands to sbatch_fp

    Notes:
        since these jobs seem to take on the order of 5 minutes or less, seems better to just run through all years once a node is secured for a job, instead of making a single job for every year / variable combination
    """
    pycommands = "\n"
    pycommands += (
        f"python {indicators_script} "
        f"--indicators {indicator} "
        f"--model {model} "
        f"--scenario {scenario} "
        f"--input_dir {input_dir} "
        f"--backup_dir {backup_dir} "
        f"--out_dir {indicators_dir} "
    )
    if no_clobber:
        pycommands += "--no-clobber \n\n"
    else:
        pycommands += "\n\n"
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


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indicators",
        type=str,
        help="' '-separated list of indicators to compute, in quotes",
        required=True,
    )
    parser.add_argument(
        "--models",
        type=str,
        help="-seperated list of model names, as used in filepaths ex. 'CESM2 CanESM5'",
        required=True,
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="-seperated list of scenario names, as used in filepaths ex. 'ssp370 ssp585'",
        required=True,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to input directory having filepath structure <model>/<scenario>/day/<variable ID>/<files>",
        default=str(regrid_dir),
    )
    parser.add_argument(
        "--backup_dir",
        type=str,
        help="Path to backup input directory having filepath structure <model>/<scenario>/day/<variable ID>/<files>",
        default=str(cmip6_dir.parent.joinpath("regrid")),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to directory where indicators data should be written",
        required=True,
    )
    parser.add_argument(
        "--no-clobber",
        action="store_true",
        default=False,
        help="Do not overwrite files if they exists in out_dir",
    )
    args = parser.parse_args()

    return (
        args.indicators.split(" "),
        args.models.split(" "),
        args.scenarios.split(" "),
        Path(args.input_dir),
        Path(args.backup_dir),
        Path(args.out_dir),
        args.no_clobber,
    )


if __name__ == "__main__":
    (
        indicators,
        models,
        scenarios,
        input_dir,
        backup_dir,
        out_dir,
        no_clobber,
    ) = parse_args()

    # make batch files for each model / scenario / variable combination
    sbatch_fps = []
    sbatch_dir = slurm_dir.joinpath("indicators")
    _ = [fp.unlink() for fp in sbatch_dir.glob("*.slurm")]
    sbatch_dir.mkdir(exist_ok=True)

    for model in models:
        for scenario in scenarios:
            for indicator in indicators:
                # filepath for batch file

                # filepath for slurm script
                sbatch_fp = sbatch_dir.joinpath(
                    f"{indicator}_{model}_{scenario}_indicator.slurm"
                )
                # filepath for slurm stdout
                sbatch_out_fp = sbatch_dir.joinpath(
                    sbatch_fp.name.replace(".slurm", "_%j.out")
                )
                # excluding node 138 until issue resolved
                sbatch_head = make_sbatch_head(
                    **sbatch_head_kwargs, exclude_nodes="n138"
                )
                sbatch_indicators_kwargs = {
                    "sbatch_fp": sbatch_fp,
                    "sbatch_out_fp": sbatch_out_fp,
                    "indicator": indicator,
                    "model": model,
                    "scenario": scenario,
                    "input_dir": input_dir,
                    "indicators_script": indicators_script,
                    "indicators_dir": SCRATCH_DIR,
                    "no_clobber": no_clobber,
                    "sbatch_head": sbatch_head,
                }
                write_sbatch_indicators(**sbatch_indicators_kwargs)
                sbatch_fps.append(sbatch_fp)
