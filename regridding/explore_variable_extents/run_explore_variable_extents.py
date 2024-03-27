"""Script to build a slurm file that executes the explore_variable_extents.ipynb for multiple variables.

Usage: Run from the command line like so:

python run_explore_variable_extents.py --cmip6_directory /beegfs/CMIP6/arctic-cmip6/CMIP6 --repo_directory ~/cmip6-utils --conda_init_script ~/cmip6-utils/regridding/conda_init.sh --explore_variable_extents_notebook ~/cmip6-utils/regridding/explore_variable_extents/explore_variable_extents.ipynb --vars 'clt evspsbl hfls hfss hus huss mrro mrsol mrsos orog pr prsn ps psl rlds rls rsds rss sfcWind sfcWindmax sftlf sftof siconc sithick snd snw ta tas tasmax tasmin tos ts tsl ua uas va vas zg' --slurm_email jdpaul3@alaska.edu
"""
import argparse
from pathlib import Path
import subprocess

def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument(
        "--cmip6_directory",
        type=str,
        help="Path to directory where CMIP6 source files are stored",
        required=True,
    )
    parser.add_argument(
        "--repo_directory",
        type=str,
        help="Path to cmip6-utils repo directory",
        required=True,
    )
    parser.add_argument(
        "--conda_init_script",
        type=str,
        help="Path to script that initiates conda",
        required=True,
    )
    parser.add_argument(
        "--explore_variable_extents_notebook",
        type=str,
        help="Path to exploratory notebook",
        required=True,
    )
    parser.add_argument(
        "--vars",
        type=str,
        help="List of variables to QC, separated by whitespace (e.g. 'ta tas pr')",
        required=True,
    )
    parser.add_argument(
        "--slurm_email",
        type=str,
        help="Email address to send slurm messages to",
        required=True,
    )

    args = parser.parse_args()

    return (
        Path(args.cmip6_directory),
        Path(args.repo_directory),
        Path(args.conda_init_script),
        Path(args.explore_variable_extents_notebook),
        args.vars,
        args.slurm_email,
    )


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
        cmip6_directory,
        repo_directory,
        conda_init_script,
        explore_variable_extents_notebook,
        vars,
        slurm_email,
    ) = parse_args()

    # Create directory

    out_dir = repo_directory.joinpath("regridding", "explore_variable_extents")
    out_dir.mkdir(exist_ok=True)

    for var in vars.split(" "):

        output_nb = out_dir.joinpath(f"{var}.ipynb")

        # Create and submit job for variable

        sbatch_fp = out_dir.joinpath(str(explore_variable_extents_notebook.name).replace(".ipynb", f"_{var}.slurm"))
        sbatch_out_fp = out_dir.joinpath(str(explore_variable_extents_notebook.name).replace(".ipynb", f"_{var}_%j.out"))

        sbatch_text = (
            "#!/bin/sh\n"
            "#SBATCH --nodes=1\n"
            f"#SBATCH --cpus-per-task=24\n"
            "#SBATCH --mail-type=FAIL\n"
            f"#SBATCH --mail-user={slurm_email}\n"
            f"#SBATCH -p t2small\n"
            f"#SBATCH --output {sbatch_out_fp}\n"
            # print start time
            "echo Start slurm && date\n"
            # prepare shell for using activate
            f"source {conda_init_script}\n"
            f"conda activate cmip6-utils\n"
            # run the notebook
            f"cd {out_dir}\n"
            f"papermill {explore_variable_extents_notebook} {output_nb} -r cmip6_directory '{cmip6_directory}' -r repo_directory '{repo_directory}' -r var '{var}'\n"
        )

        # save the sbatch text as a new slurm file in the output directory
        with open(sbatch_fp, "w") as f:
            f.write(sbatch_text)
        # run the job
        submit_sbatch(sbatch_fp)