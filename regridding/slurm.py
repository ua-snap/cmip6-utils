"""Functions to assist with constructing slurm jobs"""

import subprocess
import argparse
from pathlib import Path
from config import *

def make_sbatch_head(slurm_email, conda_init_script):
    """Make a string of SBATCH commands that can be written into a .slurm script

    Args:
        slurm_email (str): email address for slurm failures
        conda_init_script (path_like): path to a script that contains commands for initializing the shells on the compute nodes to use conda activate

    Returns:
        sbatch_head (str): string of SBATCH commands ready to be used as parameter in sbatch-writing functions. The following gaps are left for filling with .format:
            - output slurm filename
    """
    sbatch_head = (
        "#!/bin/sh\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --exclude=138\n"
        f"#SBATCH --cpus-per-task=24\n"
        "#SBATCH --mail-type=FAIL\n"
        f"#SBATCH --mail-user={slurm_email}\n"
        f"#SBATCH -p t1small\n"
        "#SBATCH --output {sbatch_out_fp}\n"
        # print start time
        "echo Start slurm && date\n"
        # prepare shell for using activate
        f"source {conda_init_script}\n"
        f"conda activate cmip6-utils\n"
    )

    return sbatch_head


def write_sbatch_regrid(
    sbatch_fp,
    sbatch_out_fp,
    regrid_script,
    regrid_batch_dir,
    regrid_dir,
    regrid_batch_fp,
    dst_fp,
    no_clobber,
    sbatch_head,
):
    """Write an sbatch script for executing the restacking script for a given group and variable, executes for a given list of years

    Args:
        sbatch_fp (path_like): path to .slurm script to write sbatch commands to
        sbatch_out_fp (path_like): path to where sbatch stdout should be written
        regrid_script (path_like): path to the script to be called to run the regridding
        regrid_dir (pathlib.PosixPath): directory to write the regridded data to
        regrid_batch_fp (path_like): path to the batch file containing paths of CMIP6 files to regrid
        dst_fp (path_like): path to file being used as template / reference for destination grid
        no_clobber (str): if "true", do not overwrite regridded files if they already exist
        sbatch_head (dict): string for sbatch head script

    Returns:
        None, writes the commands to sbatch_fp

    Notes:
        since these jobs seem to take on the order of 5 minutes or less, seems better to just run through all years once a node is secured for a job, instead of making a single job for every year / variable combination
    """
    pycommands = "\n"
    pycommands += (
        f"python {regrid_script} "
        f"-r {regrid_batch_dir} "
        f"-b {regrid_batch_fp} "
        f"-d {dst_fp} "
        f"-o {regrid_dir} "
        f"--no_clobber {no_clobber}\n\n"
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
        help="Path to directory where batch files are written",
        required=True,
    )
    parser.add_argument(
        "--slurm_email",
        type=str,
        help="Email to send slurm messages to",
        required=True,
    )
    parser.add_argument(
        "--conda_init_script",
        type=str,
        help="Path to conda init script",
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
        "--no_clobber",
        type=str,
        help="If true, do not overwrite existing regidded files",
        required=True,
    )
    args = parser.parse_args()

    return (
        Path(args.slurm_dir),
        Path(args.regrid_dir),
        Path(args.regrid_batch_dir),
        args.slurm_email,
        Path(args.conda_init_script),
        Path(args.regrid_script),
        Path(args.target_grid_fp),
        args.no_clobber,
    )


if __name__ == "__main__":
    (slurm_dir,
     regrid_dir,
     regrid_batch_dir,
     slurm_email,
     conda_init_script,
     regrid_script,
     target_grid_fp,
     no_clobber) = parse_args()

    # make these dirs if they don't exist
    Path(regrid_dir).mkdir(exist_ok=True, parents=True)
    Path(slurm_dir).mkdir(exist_ok=True, parents=True)

    # build and write sbatch files
    sbatch_fps = []
    sbatch_dir = slurm_dir.joinpath("regrid")
    _ = [fp.unlink() for fp in sbatch_dir.glob("*.slurm")]
    sbatch_dir.mkdir(exist_ok=True)
    for fp in regrid_batch_dir.glob("*.txt"):
        sbatch_str = fp.name.split("batch_")[1].split(".txt")[0]
        sbatch_fp = sbatch_dir.joinpath(f"regrid_{sbatch_str}.slurm")
        # filepath for slurm stdout
        sbatch_out_fp = sbatch_dir.joinpath(sbatch_fp.name.replace(".slurm", "_%j.out"))
        # excluding node 138 until issue resolved
        sbatch_head = make_sbatch_head(slurm_email, conda_init_script)
        sbatch_regrid_kwargs = {
            "sbatch_fp": sbatch_fp,
            "sbatch_out_fp": sbatch_out_fp,
            "regrid_script": regrid_script,
            "regrid_batch_dir" : regrid_batch_dir,
            "regrid_dir": regrid_dir,
            "regrid_batch_fp": fp,
            "dst_fp": target_grid_fp,
            "no_clobber": no_clobber,
            "sbatch_head": sbatch_head,
        }
        write_sbatch_regrid(**sbatch_regrid_kwargs)
        sbatch_fps.append(sbatch_fp)

        # remove existing slurm output files
        _ = [fp.unlink() for fp in sbatch_dir.glob("*.out")]

        # submit jobs
        job_ids = [submit_sbatch(fp) for fp in sbatch_fps]