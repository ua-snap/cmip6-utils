"""Script to make the batch file for a "second regridding" (aka regrid_again), 
where we are regridding a set of files that have already been regridded to a common grid.

Example usage:
    python run_regrid_again.py \
        --partition t2small \
        --conda_env_name cmip6-utils \
        --slurm_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/slurm \
        --regrid_script /center1/CMIP6/kmredilla/cmip6-utils/regridding/regrid.py \
        --interp_method bilinear \
        --target_grid_file /beegfs/CMIP6/kmredilla/downscaling/era5_target_slice.nc \
        --regridded_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/regrid \
        --regrid_again_batch_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/slurm/regrid_again_batch \
        --output_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/final_regrid
"""

import argparse
import logging
from pathlib import Path
from itertools import islice
from slurm import submit_sbatch
from config import landsea_variables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def parse_args():
    """Parse some command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        type=str,
        help="Slurm partition to use",
        default="t2small",
    )
    parser.add_argument(
        "--conda_env_name",
        type=str,
        help="Name of the conda environment to use",
        default="cmip6-utils",
    )
    parser.add_argument(
        "--slurm_dir",
        type=str,
        help="Path to directory where slurm files are stored",
    )
    parser.add_argument(
        "--regrid_script",
        type=str,
        help="Path to the regrid.py script",
    )
    parser.add_argument(
        "--interp_method",
        type=str,
        help="Interpolation method to use",
    )
    parser.add_argument(
        "--target_grid_file",
        type=str,
        help="Path to the target grid file",
    )
    parser.add_argument(
        "--regridded_dir",
        type=str,
        help="Path to directory where CMIP6 files are stored",
    )
    parser.add_argument(
        "--regrid_again_batch_dir",
        type=str,
        help="Path to directory where the regrid again batch files will be written",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory where the regridded files will be stored",
    )
    parser.add_argument(
        "--stage",
        type=str,
        help="Regridding stage identifier (e.g., 'second' or 'final')",
        required=True,
        choices=["second", "final"],
    )
    parser.add_argument(
        "--sftlf_dir",
        type=str,
        help="Path to directory containing model-specific sftlf files for land-sea masking (optional)",
        required=False,
    )
    args = parser.parse_args()
    return (
        args.partition,
        args.conda_env_name,
        Path(args.slurm_dir),
        Path(args.regrid_script),
        args.interp_method,
        Path(args.target_grid_file),
        Path(args.regridded_dir),
        Path(args.regrid_again_batch_dir),
        Path(args.output_dir),
        args.stage,
        Path(args.sftlf_dir) if args.sftlf_dir else None,
    )


def extract_model_from_path(filepath):
    """Extract model name from regridded file path.

    Expected structure: .../model/scenario/frequency/variable/file.nc

    Parameters
    ----------
    filepath : Path
        Path to regridded file

    Returns
    -------
    str or None
        Model name if found, None otherwise
    """
    parts = filepath.parts
    # Model should be 5th from the end: model/scenario/freq/var/file.nc
    if len(parts) >= 5:
        return parts[-5]
    return None


def write_batch_files_with_models(src_fps, regrid_again_batch_dir):
    """Write batch files grouped by model for model-specific sftlf handling.

    Parameters
    ----------
    src_fps : list of Path
        List of source file paths
    regrid_again_batch_dir : Path
        Directory to write batch files

    Returns
    -------
    list of dict
        List of batch info dicts with keys: 'batch_file', 'model', 'files'
    """
    logging.info(
        f"Grouping files by model and writing batch files to {regrid_again_batch_dir}"
    )

    # Group files by model
    files_by_model = {}
    for fp in src_fps:
        model = extract_model_from_path(fp)
        if model:
            if model not in files_by_model:
                files_by_model[model] = []
            files_by_model[model].append(fp)
        else:
            logging.warning(f"Could not extract model from path: {fp}")

    # Create batch files for each model
    batch_size = 200
    batch_infos = []
    batch_num = 1

    for model, model_files in sorted(files_by_model.items()):
        logging.info(f"  Model {model}: {len(model_files)} files")

        # Split model files into batches
        for i in range(0, len(model_files), batch_size):
            batch_files = model_files[i : i + batch_size]
            batch_file = regrid_again_batch_dir.joinpath(
                f"batch_{batch_num}_{model}.txt"
            )

            with open(batch_file, "w") as f:
                for src_fp in batch_files:
                    f.write(f"{src_fp}\n")

            batch_infos.append(
                {
                    "batch_file": batch_file,
                    "model": model,
                    "file_count": len(batch_files),
                }
            )
            batch_num += 1

    logging.info(
        f"Created {len(batch_infos)} batch files for {len(files_by_model)} models"
    )
    return batch_infos


def write_config_file_with_models(config_path, batch_infos):
    """Write config file with model information for array job.

    Parameters
    ----------
    config_path : Path
        Path to write config file
    batch_infos : list of dict
        List of batch info dicts from write_batch_files_with_models

    Returns
    -------
    str
        Array range string for SLURM
    """
    logging.info(f"Writing config file with model info to {config_path}")

    with open(config_path, "w") as f:
        f.write("array_id\tbatch_file\tmodel\n")
        for array_id, batch_info in enumerate(batch_infos, start=1):
            f.write(f"{array_id}\t{batch_info['batch_file']}\t{batch_info['model']}\n")

    array_range = f"1-{len(batch_infos)}"
    logging.info(f"Config file written with array range: {array_range}")
    return array_range


def write_batch_files(src_fps, regrid_again_batch_dir):
    """Write the batch files for the regrid again job."""
    logging.info(
        f"Writing batch files for regridding again to {regrid_again_batch_dir}"
    )
    batch_size = 200
    batch_files = []
    for i, start in enumerate(range(0, len(src_fps), batch_size), start=1):
        batch_file = regrid_again_batch_dir.joinpath(f"batch_{i}.txt")
        with open(batch_file, "w") as f:
            for src_fp in islice(src_fps, start, start + batch_size):
                f.write(f"{src_fp}\n")
        batch_files.append(batch_file)
    logging.info(f"Batch files written to {regrid_again_batch_dir}")

    return batch_files


def write_config_file(
    config_path,
    batch_files,
):
    """Write a config file for the re-regridding slurm job script.
    This is used to split the job into a job array, one task per 200 files.

    Parameters
    ----------
    config_path : pathlib.PosixPath
        path to write the config file
    regrid_again_batch_dir : pathlib.PosixPath
        path to the directory where the regrid again batch files are stored

    Returns
    -------
    array_range : str
        string to use in the SLURM array
    """
    logging.info(f"Writing config file to {config_path}")
    array_list = []
    with open(config_path, "w") as f:
        f.write("array_id\tbatch_file\n")
        for array_id, batch_file in enumerate(batch_files, start=1):
            f.write(f"{array_id}\t{batch_file}\n")
            array_list.append(array_id)

    array_range = f"{min(array_list)}-{max(array_list)}"
    logging.info(f"Config file written to {config_path}")

    return array_range


def make_sbatch_head(array_range, partition, sbatch_out_file, conda_env_name):
    """Make a string of SBATCH commands that can be written into a .slurm script

    Args:
        array_range (str): string to use in the SLURM array
        partition (str): name of the partition to use
        sbatch_out_file (path_like): path to where sbatch stdout should be written
        conda_env_name (str): name of the conda environment to activate

    Returns:
        sbatch_head (str): string of SBATCH commands ready to be used as parameter in sbatch-writing functions.
        The following keys are left for filling with str.format:

            - output slurm filename
    """
    sbatch_head = (
        "#!/bin/sh\n"
        f"#SBATCH --array={array_range}%10\n"  # don't run more than 10 tasks
        f"#SBATCH --job-name=regrid_cmip6_again\n"
        f"#SBATCH --time=04:00:00\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH -p {partition}\n"
        f"#SBATCH --output {sbatch_out_file}\n"
        # print start time
        "echo Start slurm && date\n"
        # prepare shell for using activate - Chinook requirement
        # this seems to work to initialize conda without init script
        'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"\n'
        f"conda activate {conda_env_name}\n"
    )

    return sbatch_head


def write_sbatch_regrid_again(
    partition,
    conda_env_name,
    slurm_subdir,
    regrid_script,
    config_file,
    target_grid_file,
    output_dir,
    interp_method,
    array_range,
    stage,
    sftlf_dir=None,
):
    """Write the sbatch file for the regrid again job.

    Parameters
    ----------
    sftlf_dir : Path or None
        Directory containing model-specific sftlf files (e.g., second_regrid_target_sftlf_CESM2.nc)
    """
    sbatch_file = slurm_subdir.joinpath(f"regrid_{stage}.slurm")
    sbatch_out_file = slurm_subdir.joinpath(f"regrid_{stage}_%A_%a.out")

    sbatch_head = make_sbatch_head(
        array_range, partition, sbatch_out_file, conda_env_name
    )

    pycommands = "\n"

    # Extract batch file and model from config
    pycommands += (
        f"config={config_file}\n"
        "batch_file=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $2}' $config)\n"
        "model=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $3}' $config)\n"
        "\n"
    )

    # Build regrid command
    regrid_cmd = (
        f"python {regrid_script} "
        f"-b $batch_file "
        f"-d {target_grid_file} "
        f"-o {output_dir} "
        f"--interp_method {interp_method}"
    )

    # Add model-specific sftlf if sftlf_dir provided
    if sftlf_dir:
        pycommands += (
            f"# Construct model-specific sftlf path\n"
            f'sftlf_file="{sftlf_dir}/{stage}_regrid_target_sftlf_${{model}}.nc"\n'
            f"\n"
            f"# Check if sftlf file exists for this model\n"
            f'if [ -f "$sftlf_file" ]; then\n'
            f'  echo "Using model-specific sftlf: $sftlf_file"\n'
            f'  sftlf_args="--src_sftlf_fp $sftlf_file --dst_sftlf_fp $sftlf_file"\n'
            f"else\n"
            f'  echo "No sftlf file found for model $model, proceeding without land masking"\n'
            f'  sftlf_args=""\n'
            f"fi\n"
            f"\n"
        )
        regrid_cmd += " $sftlf_args"

    regrid_cmd += "\n\n"

    pycommands += regrid_cmd
    pycommands += f"echo End re-regridding && date\n\n"
    commands = sbatch_head + pycommands

    with open(sbatch_file, "w") as f:
        f.write(commands)

    logging.info(f"Wrote sbatch script to {sbatch_file}")

    return sbatch_file


def precreate_output_directories(src_fps, output_dir):
    """Pre-create all output directories to avoid race conditions in parallel jobs.

    When multiple array jobs run simultaneously, they may all try to create the same
    directory structure, leading to FileExistsError even with exist_ok=True due to
    timing issues with parents=True. Pre-creating avoids this race condition.

    Parameters
    ----------
    src_fps : list of Path
        List of source file paths to be regridded
    output_dir : Path
        Root output directory
    """
    logging.info("Pre-creating output directory structure to avoid race conditions...")

    # Extract unique directory paths that will be needed
    unique_dirs = set()
    for fp in src_fps:
        # Parse the path structure: .../model/scenario/frequency/variable/file.nc
        parts = fp.parts
        # Find indices of key directories by working backwards from filename
        if len(parts) >= 5:
            # Last 5 parts before filename: model/scenario/frequency/variable/filename
            model = parts[-5]
            scenario = parts[-4]
            frequency = parts[-3]
            variable = parts[-2]

            out_dir = output_dir / model / scenario / frequency / variable
            unique_dirs.add(out_dir)

    # Create all directories
    created_count = 0
    for dir_path in sorted(unique_dirs):
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            created_count += 1
        except FileExistsError:
            # Safe to ignore - directory already exists
            pass

    logging.info(f"Pre-created {created_count} output directories")


def check_if_needs_landmask(src_fps):
    """Check if any of the source files contain land-sea variables.

    Parameters
    ----------
    src_fps : list of Path
        List of source file paths

    Returns
    -------
    bool
        True if any files contain land-sea variables that need masking
    """
    for fp in src_fps:
        # Extract variable from path: .../model/scenario/frequency/VARIABLE/file.nc
        if len(fp.parts) >= 2:
            variable = fp.parts[-2]
            if variable in landsea_variables:
                logging.info(
                    f"Detected land-sea variable '{variable}' - land masking will be used if sftlf files provided"
                )
                return True
    return False


if __name__ == "__main__":
    (
        partition,
        conda_env_name,
        slurm_dir,
        regrid_script,
        interp_method,
        target_grid_file,
        regridded_dir,
        regrid_again_batch_dir,
        output_dir,
        stage,
        sftlf_dir,
    ) = parse_args()

    # Create stage-specific subdirectory for organized slurm outputs
    slurm_subdir = slurm_dir.joinpath(f"{stage}_regrid")
    slurm_subdir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Using slurm subdirectory: {slurm_subdir}")

    # Create stage-specific batch directory
    batch_subdir = slurm_subdir.joinpath("batch")
    batch_subdir.mkdir(exist_ok=True, parents=True)

    src_fps = list(regridded_dir.glob("**/*.nc"))

    # Check if land masking is needed
    needs_landmask = check_if_needs_landmask(src_fps)
    if needs_landmask and sftlf_dir:
        logging.info(
            f"Land-sea masking will be applied using model-specific files from: {sftlf_dir}"
        )
    elif needs_landmask and not sftlf_dir:
        logging.warning(
            "Land-sea variables detected but no sftlf directory provided - masking will not be applied"
        )
    else:
        logging.info("No land-sea variables detected in batch")

    # Pre-create all output directories to avoid race conditions in parallel array jobs
    precreate_output_directories(src_fps, output_dir)

    # Write batch files grouped by model for model-specific sftlf handling
    batch_infos = write_batch_files_with_models(src_fps, batch_subdir)

    # Write the config file with model information
    config_path = slurm_subdir.joinpath(f"regrid_{stage}_config.txt")
    array_range = write_config_file_with_models(config_path, batch_infos)

    # write the sbatch file for the regrid again job
    sbatch_kwargs = {
        "partition": partition,
        "conda_env_name": conda_env_name,
        "slurm_subdir": slurm_subdir,
        "regrid_script": regrid_script,
        "config_file": config_path,
        "target_grid_file": target_grid_file,
        "output_dir": output_dir,
        "interp_method": interp_method,
        "array_range": array_range,
        "stage": stage,
        "sftlf_dir": sftlf_dir,
    }
    sbatch_file = write_sbatch_regrid_again(**sbatch_kwargs)

    # submit the sbatch job
    logging.info(f"Submitting sbatch job to {partition} partition")
    job_id = submit_sbatch(sbatch_file)

    print(job_id)
