"""Script to build a slurm file that runs qc.py and qc.ipynb."""
import argparse
from pathlib import Path
import subprocess


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_directory",
        type=str,
        help="Path to directory where regridded files are written",
        required=True,
    )
    parser.add_argument(
        "--cmip6_directory",
        type=str,
        help="Path to directory where CMIP6 source files are stored",
        required=True,
    )
    parser.add_argument(
        "--repo_regridding_directory",
        type=str,
        help="Path to regridding directory in cmip6-utils repo",
        required=True,
    )
    parser.add_argument(
        "--conda_init_script",
        type=str,
        help="Path to script that initiates conda",
        required=True,
    )
    parser.add_argument(
        "--qc_script",
        type=str,
        help="Path to regridding qc script",
        required=True,
    )
    parser.add_argument(
        "--visual_qc_notebook",
        type=str,
        help="Path to regridding visual qc notebook",
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
        Path(args.output_directory),
        Path(args.cmip6_directory),
        Path(args.repo_regridding_directory),
        Path(args.conda_init_script),
        Path(args.qc_script),
        Path(args.visual_qc_notebook),
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
        output_directory,
        cmip6_directory,
        repo_regridding_directory,
        conda_init_script,
        qc_script,
        visual_qc_notebook,
        vars,
        slurm_email,
    ) = parse_args()

    # Create and submit QC script

    qc_sbatch_fp = str(qc_script).replace(
        ".py", ".slurm"
    )
    qc_sbatch_out_fp = str(qc_script).replace(
        ".py", "_%j.out"
    )

    sbatch_text = (
        "#!/bin/sh\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --cpus-per-task=24\n"
        "#SBATCH --mail-type=FAIL\n"
        f"#SBATCH --mail-user={slurm_email}\n"
        f"#SBATCH -p t2small\n"
        f"#SBATCH --output {qc_sbatch_out_fp}\n"
        # print start time
        "echo Start slurm && date\n"
        # prepare shell for using activate
        f"source {conda_init_script}\n"
        f"conda activate cmip6-utils\n"
        # run the qc script
        f"python {qc_script} --output_directory {output_directory} --vars '{vars}'"
    )

    # save the sbatch text as a new slurm file in the repo directory
    with open(qc_sbatch_fp, "w") as f:
        f.write(sbatch_text)

    submit_sbatch(qc_sbatch_fp)

    # Create and submit notebook script

    output_nb = f"{output_directory}/qc/visual_qc_out.ipynb"

    visual_qc_notebook_sbatch_fp = str(visual_qc_notebook).replace(
        ".ipynb", "_nb.slurm"
    )

    vqc_sbatch_text = (
        "#!/bin/sh\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --cpus-per-task=24\n"
        "#SBATCH --mail-type=FAIL\n"
        f"#SBATCH --mail-user={slurm_email}\n"
        f"#SBATCH -p t2small\n"
        # print start time
        "echo Start slurm && date\n"
        # prepare shell for using activate
        f"source {conda_init_script}\n"
        f"conda activate cmip6-utils\n"
        # run the notebook
        f"cd {repo_regridding_directory}\n"
        f"papermill {visual_qc_notebook} {output_nb} -r output_directory '{output_directory}' -r cmip6_directory '{cmip6_directory}'\n"
        f"jupyter nbconvert --to html {output_nb}"
    )

    # save the sbatch text as a new slurm file in the repo directory
    with open(visual_qc_notebook_sbatch_fp, "w") as f:
        f.write(vqc_sbatch_text)

    submit_sbatch(visual_qc_notebook_sbatch_fp)