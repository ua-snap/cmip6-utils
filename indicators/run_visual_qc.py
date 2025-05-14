"""Script to create a slurm job to run the visual QC notebook on indicator outputs.

python run_visual_qc.py \
    --visual_qc_nb /beegfs/CMIP6/kmredilla/cmip6-utils/indicators/visual_qc.py \
    --output_nb /beegfs/CMIP6/kmredilla/cmip6_indicators/qc/visual_qc_out.ipynb \
    --repo_indicators_dir /beegfs/CMIP6/kmredilla/cmip6-utils/indicators/ \
    --working_dir /beegfs/CMIP6/kmredilla/cmip6_indicators/ \
    --input_dir /beegfs/CMIP6/arctic-cmip6/CMIP6_common_regrid/ \
    --slurm_dir /beegfs/CMIP6/kmredilla/cmip6_indicators/slurm
"""

import argparse
import logging
from itertools import product
from pathlib import Path
from slurm import (
    make_sbatch_head,
    submit_sbatch,
)

tmp_vis_qc_sbatch_fn = "indicators_visual_qc.slurm"


def parse_args():
    """Parse some command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visual_qc_nb",
        type=str,
        help="Path to visual_qc.py notebook",
        required=True,
    )
    parser.add_argument(
        "--output_nb",
        type=str,
        help="Path to write output notebook",
        required=True,
    )
    parser.add_argument(
        "--repo_indicators_dir",
        type=str,
        help="Path to the indicators directory of the cmip6-utils repo",
        required=True,
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        help="Path to the working directory",
        required=True,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to directory containing inputs for indicators",
        required=True,
    )
    parser.add_argument(
        "--slurm_dir",
        type=str,
        help="Path to directory where slurm scripts and logs will be written.",
        required=True,
    )
    args = parser.parse_args()

    return (
        args.visual_qc_nb,
        args.output_nb,
        args.repo_indicators_dir,
        args.working_dir,
        args.input_dir,
        args.slurm_dir,
    )


def write_sbatch_run_visual_qc(
    visual_qc_nb,
    output_nb,
    repo_indicators_dir,
    working_dir,
    input_dir,
    slurm_dir,
):
    """Write the sbatch file for the indicators visual QC."""
    # create the sbatch file
    sbatch_path = slurm_dir.joinpath(tmp_vis_qc_sbatch_fn)
    sbatch_out_path = slurm_dir.joinpath(sbatch_path.name.replace(".slurm", "_%j.out"))

    sbatch_head_kwargs = {
        "sbatch_out_path": sbatch_out_path,
        "partition": f"t2small",
    }
    sbatch_head = make_sbatch_head(**sbatch_head_kwargs)

    pycommands = "\n"

    pycommands += (
        f"cd {repo_indicators_dir}\n"
        # run the papermill execution
        f"papermill {visual_qc_nb} {output_nb} "
        f"-r working_directory '{working_dir}' "
        f"-r input_directory '{input_dir}'\n"
        # conver the output notebook to html
        f"jupyter nbconvert --to html {output_nb}"
    )

    pycommands += "\n\n"

    pycommands += (
        f"echo End visual indicators QC notebook && date\n" "echo Job Completed"
    )
    commands = sbatch_head + pycommands

    with open(sbatch_path, "w") as f:
        f.write(commands)
        logging.info(f"sbatch file written to {sbatch_path}")

    return sbatch_path


if __name__ == "__main__":

    (
        visual_qc_nb,
        output_nb,
        repo_indicators_dir,
        working_dir,
        input_dir,
        slurm_dir,
    ) = parse_args()

    # clear preexisting slurm output files
    for file in slurm_dir.glob(tmp_vis_qc_sbatch_fn.replace(".slurm", "*.out")):
        file.unlink()

    sbatch_kwargs = {
        "visual_qc_nb": visual_qc_nb,
        "output_nb": output_nb,
        "repo_indicators_dir": repo_indicators_dir,
        "working_dir": working_dir,
        "input_dir": input_dir,
        "slurm_dir": slurm_dir,
    }
    sbatch_path = write_sbatch_run_visual_qc(**sbatch_kwargs)

    job_id = [submit_sbatch(sbatch_path)]
    print(job_id)
