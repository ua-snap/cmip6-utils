"""Script to build a slurm file that runs generate_batch_files.py."""
import argparse
from pathlib import Path
import subprocess


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conda_init_script",
        type=str,
        help="Path to script that initiates conda",
        required=True,
    )
    parser.add_argument(
        "--generate_batch_files_script",
        type=str,
        help="Path to script that generates batch files",
        required=True,
    )
    parser.add_argument(
        "--cmip6_directory",
        type=str,
        help="Path to directory where CMIP6 files are stored",
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
        help="Email address to send slurm messages to",
        required=True,
    )
    parser.add_argument(
        "--vars",
        type=str,
        help="list of variables used in generating batch files",
        required=True,
    )

    args = parser.parse_args()

    return (
        Path(args.conda_init_script),
        Path(args.generate_batch_files_script),
        Path(args.cmip6_directory),
        Path(args.regrid_batch_dir),
        args.slurm_email,
        args.vars,
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
        conda_init_script,
        generate_batch_files_script,
        cmip6_directory,
        regrid_batch_dir,
        slurm_email,
        vars,
    ) = parse_args()

    regrid_batch_dir.mkdir(exist_ok=True, parents=True)


    generate_batch_files_sbatch_fp = str(generate_batch_files_script).replace(
        ".py", ".slurm"
    )
    generate_batch_files_sbatch_out_fp = str(generate_batch_files_script).replace(
        ".py", "_%j.out"
    )

    sbatch_text = (
        "#!/bin/sh\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --exclude=n138\n"
        f"#SBATCH --cpus-per-task=24\n"
        "#SBATCH --mail-type=FAIL\n"
        f"#SBATCH --mail-user={slurm_email}\n"
        f"#SBATCH -p t2small\n"
        f"#SBATCH --output {generate_batch_files_sbatch_out_fp}\n"
        # print start time
        "echo Start slurm && date\n"
        # prepare shell for using activate
        f"source {conda_init_script}\n"
        f"conda activate cmip6-utils\n"
        # run the generate batch files script
        f"python {generate_batch_files_script} --cmip6_directory '{cmip6_directory}' --regrid_batch_dir '{regrid_batch_dir}' --vars '{vars}'\n"
    )

    # save the sbatch text as a new slurm file in the repo directory
    with open(generate_batch_files_sbatch_fp, "w") as f:
        f.write(sbatch_text)

    submit_sbatch(generate_batch_files_sbatch_fp)
