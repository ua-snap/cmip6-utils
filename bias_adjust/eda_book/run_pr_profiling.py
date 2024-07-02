"""Script to run the profiling 

This script is used to orchestrate the entire precipitation adjustment profiling effort by submitting profiling jobs for each model. 

It is designed to be run separately from the main adjustment pipeline, as it's purpose is to aid in determining the best parameters for the adjustment.

Example usage:
    python run_pr_profiling.py  --sim_dir /beegfs/CMIP6/kmredilla/cmip6_regridding/regrid --ref_dir /import/beegfs/CMIP6/arctic-cmip6/era5/daily_regrid --working_dir /beegfs/CMIP6/kmredilla/bias_adjust/profiling --partition t2small
"""

import argparse
import subprocess
from pathlib import Path


def make_sbatch_script(sbatch_kwargs):
    """Write the sbatch script for the bias adjustment profiling of a given model"""
    with open("./pr_profile_template.slurm") as f:
        template = f.read()

    sbatch_file = sbatch_kwargs["sbatch_file"]
    with open(sbatch_file, "w") as f:
        f.write(template.format(**sbatch_kwargs))

    if not sbatch_file.exists():
        raise FileNotFoundError(f"Failed to write sbatch file to {sbatch_kwargs}")

    return sbatch_file


def submit_sbatch(sbatch_file):
    """Submit a script to slurm via sbatch

    Args:
        sbatch_file (pathlib.PosixPath): path to .slurm script to submit

    Returns:
        job id for submitted job
    """
    out = subprocess.check_output(["sbatch", str(sbatch_file)])
    job_id = out.decode().replace("\n", "").split(" ")[-1]

    return job_id


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim_dir",
        type=str,
        help="Path to directory of simulated data files to be adjusted, with filepath structure <model>/<scenario>/day/<variable ID>/<files>",
    )
    parser.add_argument(
        "--ref_dir",
        type=str,
        help="Path to directory of reference data with filepath structure <variable ID>/<files>",
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
    args = parser.parse_args()

    return (
        Path(args.sim_dir),
        Path(args.ref_dir),
        Path(args.working_dir),
        args.partition,
    )


if __name__ == "__main__":
    (
        sim_dir,
        ref_dir,
        working_dir,
        partition,
    ) = parse_args()

    working_dir.mkdir(exist_ok=True)
    # make batch files for each model
    sbatch_dir = working_dir.joinpath("slurm")
    sbatch_dir.mkdir(exist_ok=True)
    _ = [fp.unlink() for fp in sbatch_dir.glob("*.slurm")]
    # summaries of adjusted data will be written to a subdir of working_dir
    adj_prof_dir = working_dir.joinpath("profiling_data")
    adj_prof_dir.mkdir(exist_ok=True)

    job_ids = []
    models = [
        "CESM2",
        "CESM2-WACCM",
        "CNRM-CM6-1-HR",
        "EC-Earth3-Veg",
        "GFDL-ESM4",
        "HadGEM3-GC31-LL",
        "HadGEM3-GC31-MM",
        "KACE-1-0-G",
        "MIROC6",
        "MPI-ESM1-2-LR",
        "NorESM2-MM",
        "TaiESM1",
    ]
    for model in models:
        # filepath for slurm script
        sbatch_file = sbatch_dir.joinpath(f"profile_adjustment_{model}.slurm")
        # filepath for slurm stdout
        sbatch_out_file = sbatch_dir.joinpath(
            sbatch_file.name.replace(".slurm", "_%j.out")
        )
        model_dir = sim_dir.joinpath(model)
        results_file = adj_prof_dir.joinpath(f"{model}_profiling_results.pkl")

        sbatch_kwargs = {
            "sbatch_file": sbatch_file,
            "sbatch_out_file": sbatch_out_file,
            "partition": "t2small",
            "model_dir": model_dir,
            "ref_dir": ref_dir,
            "working_dir": working_dir,
            "conda_init_script": "../conda_init.sh",
            "profile_script": "profile_pr.py",
            "results_file": results_file,
            "env_name": "new_cmip6_utils",
        }

        make_sbatch_script(sbatch_kwargs)
        job_id = submit_sbatch(sbatch_file)
        job_ids.append(job_id)

    print(f"Submitted jobs: {job_ids}")
