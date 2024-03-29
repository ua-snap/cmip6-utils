"""Script for constructing slurm jobs for computing daily temperature range for CMIP6 data.

Example usage:
    python slurm_dtr.py --models "GFDL-ESM4 CESM2" --scenarios "ssp245 ssp585" --input_dir /import/beegfs/CMIP6/arctic-cmip6/regrid --working_dir /import/beegfs/CMIP6/kmredilla --partition debug --ncpus 24

Returns:
    Outputs are written in a dtr_processing directory created as a subdirectory in working_dir, following the model/scenario/variable/<files>*.nc convention.
    e.g. files for GFDL-ESM4 ssp245 would be written to <working_dir>/dtr_processing/netcdf/GFDL-ESM4/ssp245/dtr/
"""

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


def write_sbatch_dtr(
    sbatch_fp,
    sbatch_out_fp,
    dtr_script,
    tasmax_dir,
    tasmin_dir,
    output_dir,
    sbatch_head,
):
    """Write an sbatch script for executing the bias adjustment script for a given model, scenario, and variable

    Args:
        sbatch_fp (path_like): path to .slurm script to write sbatch commands to
        sbatch_out_fp (path_like): path to where sbatch stdout should be written
        dtr_script (path_like): path to the script to be called to run the dtr processing
        tasmax_dir (path-like): path to directory of tasmax files
        tasmin_dir (path-like): path to directory of tasmin files (should correspond to files in tasmax_dir)
        output_dir (path-like): directory to write the dtr data
        sbatch_head (dict): string for sbatch head script

    Returns:
        None, writes the commands to sbatch_fp
    """
    pycommands = "\n"
    pycommands += (
        f"python {dtr_script} "
        f"--tasmax_dir {tasmax_dir} "
        f"--tasmin_dir {tasmin_dir} "
        f"--output_dir {output_dir}\n"
    )

    pycommands += f"echo End dtr processing && date\n\n"
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
        args.models.split(" "),
        args.scenarios.split(" "),
        Path(args.input_dir),
        Path(args.working_dir),
        args.partition,
        args.ncpus,
    )


def get_output_dir(working_dir, output_dir_name):
    """Get the output directory path which will contain all subfolders of outputs (data, slurm etc). Function for sharing."""
    return working_dir.joinpath(output_dir_name)


def get_dtr_dir(output_dir, model, scenario):
    """Get the DTR directory from output_dir, model, and scenario. Function for sharing."""
    return output_dir.joinpath("netcdf", model, scenario, "dtr")


def get_tmax_tmin_fps(tasmax_dir, tasmin_dir):
    """Helper function for getting tasmax and tasmin filepaths. Put in function for checking prior to slurming."""
    tasmax_fps = list(tasmax_dir.glob("tasmax*.nc"))
    tasmin_fps = list(tasmin_dir.glob("tasmin*.nc"))

    return tasmax_fps, tasmin_fps


if __name__ == "__main__":
    (
        models,
        scenarios,
        input_dir,
        working_dir,
        partition,
        ncpus,
    ) = parse_args()

    working_dir.mkdir(exist_ok=True)
    output_dir = get_output_dir(working_dir, output_dir_name)
    output_dir.mkdir(exist_ok=True)

    # make batch files for each model / scenario / variable combination
    sbatch_dir = output_dir.joinpath("slurm")
    sbatch_dir.mkdir(exist_ok=True)
    _ = [fp.unlink() for fp in sbatch_dir.glob("*.slurm")]

    qc_dir = output_dir.joinpath("qc")
    qc_dir.mkdir(exist_ok=True)

    # sbatch head - replaces config.py params for now!
    sbatch_head_kwargs = {
        "partition": partition,
        "ncpus": ncpus,
        "conda_init_script": working_dir.joinpath(
            "cmip6-utils/regridding/conda_init.sh"
        ),
    }

    dtr_script = working_dir.joinpath("cmip6-utils/derived/cmip6_dtr.py")

    job_ids = []
    for model in models:
        for scenario in scenarios:
            # get directories for tasmax and tasmin
            tasmax_dir = input_dir.joinpath(model, scenario, "day", "tasmax")
            tasmin_dir = input_dir.joinpath(model, scenario, "day", "tasmin")

            # if these don't exist, skip them
            try:
                assert tasmax_dir.exists()
                assert tasmin_dir.exists()
            except AssertionError:
                # wanted to have a print statement here, about skipping directories that don't have valid input data,
                #  but that would make it more messy to pass job IDs to next function in prefect
                continue

            try:
                tasmax_fps, tasmin_fps = get_tmax_tmin_fps(tasmax_dir, tasmin_dir)
                assert len(tasmax_fps) == len(tasmin_fps)
            except AssertionError:
                # again, print statement would interefere with printing job ids.
                continue

            # filepath for slurm script
            sbatch_fp = sbatch_dir.joinpath(f"{model}_{scenario}_process_dtr.slurm")
            # filepath for slurm stdout
            sbatch_out_fp = sbatch_dir.joinpath(
                sbatch_fp.name.replace(".slurm", "_%j.out")
            )
            # excluding node 138 until issue resolved
            sbatch_head = make_sbatch_head(**sbatch_head_kwargs)

            # create the nested output directory matching our model/scenario/variable convention
            # nesting this in a 'netcdf' subdir to keep the dir structure separate from slurm folder
            #  and other non-data outputs
            dtr_dir = get_dtr_dir(output_dir, model, scenario)
            dtr_dir.mkdir(exist_ok=True, parents=True)

            sbatch_dtr_kwargs = {
                "sbatch_fp": sbatch_fp,
                "sbatch_out_fp": sbatch_out_fp,
                "dtr_script": dtr_script,
                "tasmax_dir": tasmax_dir,
                "tasmin_dir": tasmin_dir,
                "output_dir": dtr_dir,
                "sbatch_head": sbatch_head,
            }
            write_sbatch_dtr(**sbatch_dtr_kwargs)
            job_id = submit_sbatch(sbatch_fp)

            sbatch_out_fp_with_jobid = sbatch_dir.joinpath(
                sbatch_out_fp.name.replace("%j", str(job_id))
            )
            job_ids.append(job_id)
    # currently printing Job IDs to give to next steps in prefect. Probably better ways to do this.
    print(job_ids)
