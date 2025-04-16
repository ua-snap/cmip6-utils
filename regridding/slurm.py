"""Functions to assist with constructing slurm jobs"""

import subprocess
import argparse
from pathlib import Path
from config import *


def make_sbatch_head(conda_init_script, conda_env_name, partition="t2small"):
    """Make a string of SBATCH commands that can be written into a .slurm script.

    Parameters
    ----------
    conda_init_script : pathlib.Path
        path to a script that contains commands for initializing the shells on the compute nodes to use conda activate
    conda_env_name : str
        name of the conda environment to activate
    partition : str
        slurm partition to use, default is t2small

    Returns
    -------
    str
        string of SBATCH commands ready to be used as parameter in sbatch-writing functions. The following gaps are left for filling with .format:
            - output slurm filename
    """
    sbatch_head = (
        "#!/bin/sh\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --cpus-per-task=24\n"
        f"#SBATCH -p {partition}\n"
        f"#SBATCH --time=01:00:00\n"
        f"#SBATCH --output {sbatch_out_fp}\n"
        # print start time
        "echo Start slurm && date\n"
        # prepare shell for using activate
        f"source {conda_init_script}\n"
        f"conda activate {conda_env_name}\n"
    )

    return sbatch_head


def write_sbatch_regrid(
    sbatch_fp,
    sbatch_out_fp,
    regrid_script,
    regrid_dir,
    regrid_batch_fp,
    dst_fp,
    no_clobber,
    interp_method,
    rasdafy,
    sbatch_head,
    src_sftlf_fp=None,
    dst_sftlf_fp=None,
):
    """Write an sbatch script for executing the restacking script for a given group and variable.
    Executes for a given list of years.

    Parameters
    ----------
    sbatch_fp : pathlib.Path
        path to .slurm script to write sbatch commands to
    sbatch_out_fp : pathlib.Path
        path to where sbatch stdout should be written
    regrid_script : pathlib.Path
        path to the script to be called to run the regridding
    regrid_dir : pathlib.Path
        path to directory where regridded files are written
    regrid_batch_fp : pathlib.Path
        path to the batch file containing paths of CMIP6 files to regrid
    dst_fp : pathlib.Path
        path to file being used as template / reference for destination grid
    no_clobber : bool
        if True, do not overwrite regridded files if they already exist
    interp_method : str
        method to use for regridding interpolation
    rasdafy : bool
        Do some Rasdaman-specific tweaks to the data
    sbatch_head : dict
        string for sbatch head script
    src_sftlf_fp : pathlib.Path
        path to the source grid sftlf file
    dst_sftlf_fp : pathlib.Path
        path to the destination grid sftlf file

    Note - since these jobs seem to take on the order of 5 minutes or less,
    seems better to just run through all years once a node is secured for a job,
    instead of making a single job for every year / variable combination.
    """
    pycommands = "\n"
    pycommands += (
        f"python {regrid_script} "
        f"-b {regrid_batch_fp} "
        f"-d {dst_fp} "
        f"-o {regrid_dir} "
        f"--interp_method {interp_method} "
    )
    if src_sftlf_fp is not None:
        pycommands += f"--src_sftlf_fp {src_sftlf_fp} "
    if dst_sftlf_fp is not None:
        pycommands += f"--dst_sftlf_fp {dst_sftlf_fp} "

    if rasdafy:
        pycommands += "--rasdafy "

    if no_clobber:
        pycommands += "--no-clobber \n\n"
    else:
        pycommands += "\n\n"

    commands = sbatch_head.format(sbatch_out_fp=sbatch_out_fp) + pycommands

    with open(sbatch_fp, "w") as f:
        f.write(commands)

    return


def submit_sbatch(sbatch_fp):
    """Submit a script to slurm via sbatch.

    Parameters
    ----------
    sbatch_fp : pathlib.PosixPath
        path to .slurm script to submit

    Returns
    -------
    str
        job id for submitted job
    """
    out = subprocess.check_output(["sbatch", str(sbatch_fp)])
    job_id = out.decode().replace("\n", "").split(" ")[-1]

    return job_id


def parse_args():
    """Parse some command line arguments.

    Returns
    -------
    slurm_dir : pathlib.Path
        path to directory where slurm files are written
    regrid_dir : pathlib.Path
        path to directory where regridded files are written
    regrid_batch_dir : pathlib.Path
        path to directory where batch files are stored
    conda_init_script : pathlib.Path
        path to conda init script
    conda_env_name : str
        name of conda environment to activate
    regrid_script : pathlib.Path
        path to main worker regrid.py script for processing
    target_grid_fp : pathlib.Path
        path to file used as the regridding target
    target_sftlf_fp : str
        path to the target grid sftlf file, must be supplied if any variables are land/sea
    no_clobber : bool
        do not overwrite existing regidded files
    interp_method : str
        method to use for regridding interpolation
    vars : str
        List of variables to generate batch files for
    freqs : str
        List of frequencies to use for generating batch files
    models : str
        List of models to use for generating batch files
    scenarios : str
        List of scenarios to use for generating batch files
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slurm_dir",
        type=str,
        help="Path to directory where slurm files are written",
        required=True,
    )
    parser.add_argument(
        "--regrid_dir",
        type=str,
        help="Path to directory where regridded files are written",
        required=True,
    )
    parser.add_argument(
        "--regrid_batch_dir",
        type=str,
        help="Path to directory where batch files are stored",
        required=True,
    )
    parser.add_argument(
        "--conda_init_script",
        type=str,
        help="Path to conda init script",
        required=True,
    )
    parser.add_argument(
        "--conda_env_name",
        type=str,
        help="Name of conda environment to activate",
        required=True,
    )
    parser.add_argument(
        "--regrid_script",
        type=str,
        help="Path to regrid.py script",
        required=True,
    )
    parser.add_argument(
        "--target_grid_fp",
        type=str,
        help="Path to file used as the regridding target",
        required=True,
    )
    parser.add_argument(
        "--target_sftlf_fp",
        type=str,
        help="Path to the target grid sftlf file, must be supplied if any variables are land/sea",
        required=False,
    )
    parser.add_argument(
        "--no_clobber",
        action="store_true",
        help="Do not overwrite existing regidded files",
    )
    parser.add_argument(
        "--interp_method",
        type=str,
        help="Method to use for regridding interpolation",
        required=True,
    )
    parser.add_argument(
        "--rasdafy",
        action="store_true",
        help="Do some Rasdaman-specific tweaks to the data",
    )
    parser.add_argument(
        "--vars",
        type=str,
        help="list of variables used in generating batch files",
        required=True,
    )
    parser.add_argument(
        "--freqs",
        type=str,
        help="list of frequencies (mon or day) used in generating batch files",
        required=True,
    )
    parser.add_argument(
        "--models",
        type=str,
        help="list of models used in generating batch files",
        required=True,
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="list of scenarios used in generating batch files",
        required=True,
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="partition to use for slurm jobs",
        default="t2small",
    )
    args = parser.parse_args()

    return (
        Path(args.slurm_dir),
        Path(args.regrid_dir),
        Path(args.regrid_batch_dir),
        Path(args.conda_init_script),
        args.conda_env_name,
        Path(args.regrid_script),
        Path(args.target_grid_fp),
        args.target_sftlf_fp,
        args.no_clobber,
        args.interp_method,
        args.rasdafy,
        args.vars,
        args.freqs,
        args.models,
        args.scenarios,
        args.partition,
    )


if __name__ == "__main__":
    (
        slurm_dir,
        regrid_dir,
        regrid_batch_dir,
        conda_init_script,
        conda_env_name,
        regrid_script,
        target_grid_fp,
        target_sftlf_fp,
        no_clobber,
        interp_method,
        rasdafy,
        vars,
        freqs,
        models,
        scenarios,
        partition,
    ) = parse_args()

    # make these dirs if they don't exist
    Path(regrid_dir).mkdir(exist_ok=True, parents=True)
    Path(slurm_dir).mkdir(exist_ok=True, parents=True)

    # build and write sbatch files
    sbatch_fps = []
    sbatch_dir = slurm_dir.joinpath("regrid")
    sbatch_dir.mkdir(exist_ok=True)

    # remove any existing sbatch files in this directory.
    #  Easier to keep track of things when the only jobs are those submitted from this directory.
    _ = [fp.unlink() for fp in sbatch_dir.glob("*.slurm")]

    for var in vars.split():
        for freq in freqs.split():
            for model in models.split():
                for scenario in scenarios.split():
                    # find the batch file for this model, scenario, variable, and frequency
                    # now that they are split up by model and scenario as well,
                    # most will only be one single file, but it's not garuanteed
                    for fp in regrid_batch_dir.glob(
                        f"batch_{model}*{scenario}*{freq}*{var}*.txt"
                    ):
                        sbatch_str = fp.name.split("batch_")[1].split(".txt")[0]
                        sbatch_fp = sbatch_dir.joinpath(f"regrid_{sbatch_str}.slurm")
                        # filepath for slurm stdout
                        sbatch_out_fp = sbatch_dir.joinpath(
                            sbatch_fp.name.replace(".slurm", "_%j.out")
                        )

                        sbatch_head = make_sbatch_head(
                            conda_init_script, conda_env_name, partition=partition
                        )
                        sbatch_regrid_kwargs = {
                            "sbatch_fp": sbatch_fp,
                            "sbatch_out_fp": sbatch_out_fp,
                            "regrid_script": regrid_script,
                            "regrid_dir": regrid_dir,
                            "regrid_batch_fp": fp,
                            "dst_fp": target_grid_fp,
                            "no_clobber": no_clobber,
                            "interp_method": interp_method,
                            "sbatch_head": sbatch_head,
                            "rasdafy": rasdafy,
                        }
                        if var in landsea_variables:
                            assert (
                                target_sftlf_fp is not None
                            ), "A target sftlf file must be supplied if any variables are land/sea"
                            try:
                                sbatch_regrid_kwargs["src_sftlf_fp"] = model_sftlf_lu[
                                    model
                                ]
                            except KeyError:
                                sbatch_regrid_kwargs["src_sftlf_fp"] = None
                            sbatch_regrid_kwargs["dst_sftlf_fp"] = target_sftlf_fp

                        write_sbatch_regrid(**sbatch_regrid_kwargs)
                        sbatch_fps.append(sbatch_fp)

    # remove existing slurm output files
    _ = [fp.unlink() for fp in sbatch_dir.glob("*.out")]

    # submit jobs
    job_ids = [submit_sbatch(fp) for fp in sbatch_fps]
    # print job ids as " "-separated string to parsed for prefect flow
    print(" ".join(job_ids))
