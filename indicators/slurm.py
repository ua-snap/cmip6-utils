"""Functions to assist with constructing slurm jobs"""

import argparse
import subprocess
from config import *


def make_sbatch_head(partition, sbatch_out_fp):
    """Make a string of SBATCH commands that can be written into a .slurm script

    Args:
        partition (str): name of the partition to use
        sbatch_out_fp (path_like): path to where sbatch stdout should be written

    Returns:
        sbatch_head (str): string of SBATCH commands ready to be used as parameter in sbatch-writing functions. The following gaps are left for filling with .format:
            - ncpus
            - output slurm filename
    """
    sbatch_head = (
        "#!/bin/sh\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --time=04:00:00\n"
        f"#SBATCH --exclusive\n"
        f"#SBATCH -p {partition}\n"
        f"#SBATCH --output {sbatch_out_fp}\n"
        # print start time
        "echo Start slurm && date\n"
        # prepare shell for using activate - Chinook requirement
        'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"\n'
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
        indicator (str): name of the indicator to be computed
        model (str): name of the model to be used
        scenario (str): name of the scenario to be used
        input_dir (pathlib.PosixPath): directory where input data is stored
        indicators_script (path_like): path to the script to be called to run the indicators
        indicators_dir (pathlib.PosixPath): directory where output data should be written
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
        f"--out_dir {indicators_dir} "
    )
    if no_clobber:
        pycommands += "--no-clobber \n\n"
    else:
        pycommands += "\n\n"

    pycommands += (
        f"echo End {indicator} indicator generation && date\n" "echo Job Completed"
    )
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
        "--working_dir",
        type=str,
        help="Path to directory where all underlying directories and files are written. Must have an up-to-date clone of cmip6-utils repo.",
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
        Path(args.working_dir),
        args.no_clobber,
    )


if __name__ == "__main__":
    (
        indicators,
        models,
        scenarios,
        input_dir,
        working_dir,
        no_clobber,
    ) = parse_args()

    # create working subdir with this name
    working_subdir = working_dir.joinpath("cmip6_indicators")
    working_subdir.mkdir(exist_ok=True)

    # make output_dir the place where we actually write indicators data
    output_dir = working_subdir.joinpath("netcdf")
    output_dir.mkdir(exist_ok=True)

    # make batch files for each model / scenario / variable combination
    sbatch_dir = working_subdir.joinpath("slurm")
    sbatch_dir.mkdir(exist_ok=True)
    _ = [fp.unlink() for fp in sbatch_dir.glob("*.slurm")]

    # make QC dir and "to-do" list for each model / scenario / indicator combination
    # the "w" accessor should overwrite any previous qc.txt files encountered
    qc_dir = working_subdir.joinpath("qc")
    qc_dir.mkdir(exist_ok=True)
    qc_file = qc_dir.joinpath("qc.csv")
    with open(qc_file, "w") as q:
        pass

    sbatch_head_kwargs = {
        "partition": "t2small",
    }

    indicators_script = f"{working_dir}/cmip6-utils/indicators/indicators.py"

    for model in models:
        for scenario in scenarios:
            for indicator in indicators:

                # filepath for slurm script
                sbatch_fp = sbatch_dir.joinpath(
                    f"{indicator}_{model}_{scenario}_indicator.slurm"
                )
                # filepath for slurm stdout
                sbatch_out_fp = sbatch_dir.joinpath(
                    sbatch_fp.name.replace(".slurm", "_%j.out")
                )
                sbatch_head_kwargs.update({"sbatch_out_fp": sbatch_out_fp})
                sbatch_head = make_sbatch_head(**sbatch_head_kwargs)
                sbatch_indicators_kwargs = {
                    "sbatch_fp": sbatch_fp,
                    "sbatch_out_fp": sbatch_out_fp,
                    "indicator": indicator,
                    "model": model,
                    "scenario": scenario,
                    "input_dir": input_dir,
                    "indicators_script": indicators_script,
                    "indicators_dir": output_dir,
                    "no_clobber": no_clobber,
                    "sbatch_head": sbatch_head,
                }
                write_sbatch_indicators(**sbatch_indicators_kwargs)
                job_id = submit_sbatch(sbatch_fp)

                # append indicator filepath and sbatch job filepath to qc file
                # build expected indicator output filepath using fp template directly from config (identical to how output fp is built in indicators.py) plus job ID from submit_sbatch() above
                indicator_fp = output_dir.joinpath(
                    model,
                    scenario,
                    indicator,
                    indicator_tmp_fp.format(
                        indicator=indicator, model=model, scenario=scenario
                    ),
                )

                sbatch_out_fp_with_jobid = sbatch_dir.joinpath(
                    sbatch_out_fp.name.replace("%j", str(job_id))
                )
                with open(qc_file, "a") as f:
                    f.write(f"{indicator},{indicator_fp},{sbatch_out_fp_with_jobid}\n")
