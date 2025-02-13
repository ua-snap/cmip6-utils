"""Script to build a slurm file that runs qc.py and qc.ipynb."""

import argparse
from pathlib import Path
from slurm import submit_sbatch


def parse_args():
    """Parse some command line arguments.

    Returns
    -------
    output_directory : pathlib.Path
        Path to directory where regridded files are written
    cmip6_directory : pathlib.Path
        Path to directory where CMIP6 files are stored
    repo_regridding_directory : pathlib.Path
        Path to regridding directory in cmip6-utils repo
    conda_init_script : pathlib.Path
        Path to script that initiates conda
    conda_env_name : str
        Name of conda environment to activate
    qc_notebook : pathlib.Path
        Path to regridding QC notebook
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
        "--conda_env_name",
        type=str,
        help="Name of conda environment to activate",
        required=True,
    )
    parser.add_argument(
        "--qc_notebook",
        type=str,
        help="Path to regridding QC notebook",
        required=True,
    )
    parser.add_argument(
        "--vars",
        type=str,
        help="List of variables to QC, separated by whitespace (e.g. 'ta tas pr')",
        required=True,
    )
    parser.add_argument(
        "--freqs",
        type=str,
        help="list of frequencies used in generating batch files",
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

    args = parser.parse_args()

    return (
        Path(args.output_directory),
        Path(args.cmip6_directory),
        Path(args.repo_regridding_directory),
        Path(args.conda_init_script),
        args.conda_env_name,
        Path(args.qc_notebook),
        args.vars,
        args.freqs,
        args.models,
        args.scenarios,
    )


if __name__ == "__main__":

    (
        output_directory,
        cmip6_directory,
        repo_regridding_directory,
        conda_init_script,
        conda_env_name,
        qc_notebook,
        vars,
        freqs,
        models,
        scenarios,
    ) = parse_args()

    # Create QC directory

    qc_dir = output_directory.joinpath("qc")
    qc_dir.mkdir(exist_ok=True)

    # Create and submit notebook script

    output_nb = qc_dir.joinpath("qc_out.ipynb")

    qc_notebook_sbatch_fp = qc_dir.joinpath(
        str(qc_notebook.name).replace(".ipynb", "_nb.slurm")
    )
    qc_notebook_sbatch_out_fp = qc_dir.joinpath(
        str(qc_notebook.name).replace(".ipynb", "_nb_%j.out")
    )

    qc_sbatch_text = (
        "#!/bin/sh\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --cpus-per-task=24\n"
        "#SBATCH --mail-type=FAIL\n"
        f"#SBATCH -p t2small\n"
        f"#SBATCH --time=02:00:00\n"
        f"#SBATCH --output {qc_notebook_sbatch_out_fp}\n"
        # print start time
        "echo Start slurm && date\n"
        # prepare shell for using activate
        f"source {conda_init_script}\n"
        f"conda activate {conda_env_name}\n"
        # run the notebook
        f"cd {repo_regridding_directory}\n"
        f"papermill {qc_notebook} {output_nb} -r output_directory"
        f" '{output_directory}' -r cmip6_directory '{cmip6_directory}'"
        f" -r vars '{vars}' -r freqs '{freqs}' -r models '{models}' -r scenarios '{scenarios}'\n"
        f"jupyter nbconvert --to html {output_nb}"
    )

    # save the sbatch text as a new slurm file in the QC directory
    with open(qc_notebook_sbatch_fp, "w") as f:
        f.write(qc_sbatch_text)

    job_id = submit_sbatch(qc_notebook_sbatch_fp)
    print(job_id)
