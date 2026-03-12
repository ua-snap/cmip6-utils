"""Functions to assist with constructing slurm jobs"""

import subprocess
import argparse
import sys
from pathlib import Path
from config import *


def make_sbatch_head(partition, sbatch_out_file, conda_env_name, array_range=None):
    """Make a string of SBATCH commands that can be written into a .slurm script.

    Parameters
    ----------
    partition : str
        slurm partition to use, default is t2small
    sbatch_out_file : str
        path to where sbatch stdout should be written
    conda_env_name : str
        name of the conda environment to activate
    array_range : str, optional
        array range for array jobs (e.g., "1-20")

    Returns
    -------
    str
        string of SBATCH commands ready to be used as parameter in sbatch-writing functions.
    """
    sbatch_lines = [
        "#!/bin/sh\n",
    ]

    if array_range:
        sbatch_lines.append(
            f"#SBATCH --array={array_range}%10\n"
        )  # max 10 concurrent tasks
        sbatch_lines.append("#SBATCH --job-name=regrid_cmip6\n")

    sbatch_lines.extend(
        [
            "#SBATCH --nodes=1\n",
            f"#SBATCH --cpus-per-task=24\n",
            f"#SBATCH -p {partition}\n",
            f"#SBATCH --time=04:00:00\n",
            f"#SBATCH --output {sbatch_out_file}\n",
            # print start time
            "echo Start slurm && date\n",
            # prepare shell for using activate
            'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"\n',
            f"conda activate {conda_env_name}\n",
        ]
    )

    return "".join(sbatch_lines)


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

    commands = sbatch_head + pycommands

    with open(sbatch_fp, "w") as f:
        f.write(commands)

    return


def write_config_file(config_path, batch_files_info):
    """Write a config file for array job mapping task IDs to batch files.

    Parameters
    ----------
    config_path : pathlib.PosixPath
        path to write the config file
    batch_files_info : list of tuples
        list of (var, model, scenario, freq, batch_file_path) tuples

    Returns
    -------
    array_range : str
        string to use in SLURM array (e.g., "1-20")
    """
    array_list = []
    with open(config_path, "w") as f:
        f.write(
            "array_id\tvar\tmodel\tscenario\tfreq\tbatch_file\tsrc_sftlf\tdst_sftlf\n"
        )
        for array_id, (
            var,
            model,
            scenario,
            freq,
            batch_fp,
            src_sftlf,
            dst_sftlf,
        ) in enumerate(batch_files_info, start=1):
            src_sftlf_str = src_sftlf if src_sftlf else "NONE"
            dst_sftlf_str = dst_sftlf if dst_sftlf else "NONE"
            f.write(
                f"{array_id}\t{var}\t{model}\t{scenario}\t{freq}\t{batch_fp}\t{src_sftlf_str}\t{dst_sftlf_str}\n"
            )
            array_list.append(array_id)

    array_range = f"{min(array_list)}-{max(array_list)}"
    print(f"Wrote config file to {config_path}", file=sys.stderr)
    return array_range


def write_sbatch_array_regrid(
    sbatch_fp,
    config_file,
    regrid_script,
    regrid_dir,
    target_grid_fp,
    no_clobber,
    interp_method,
    rasdafy,
    sbatch_head,
):
    """Write an sbatch script for array job regridding.

    Parameters
    ----------
    sbatch_fp : pathlib.Path
        path to .slurm script to write
    config_file : pathlib.Path
        path to config file with array task mappings
    regrid_script : pathlib.Path
        path to regrid.py script
    regrid_dir : pathlib.Path
        output directory for regridded files
    target_grid_fp : pathlib.Path
        target grid file
    no_clobber : bool
        if True, don't overwrite existing files
    interp_method : str
        interpolation method
    rasdafy : bool
        Rasdaman-specific tweaks
    sbatch_head : str
        sbatch header commands
    """
    pycommands = "\n"
    pycommands += (
        # Extract task info from config file
        f"config={config_file}\n"
        "var=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $2}' $config)\n"
        "model=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $3}' $config)\n"
        "scenario=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $4}' $config)\n"
        "freq=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $5}' $config)\n"
        "batch_file=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $6}' $config)\n"
        "src_sftlf=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $7}' $config)\n"
        "dst_sftlf=$(awk -v array_id=$SLURM_ARRAY_TASK_ID '$1==array_id {print $8}' $config)\n"
        "\n"
        # Build sftlf arguments conditionally
        'sftlf_args=""\n'
        'if [ "$src_sftlf" != "NONE" ]; then sftlf_args="--src_sftlf_fp $src_sftlf"; fi\n'
        'if [ "$dst_sftlf" != "NONE" ]; then sftlf_args="$sftlf_args --dst_sftlf_fp $dst_sftlf"; fi\n'
        "\n"
        # Run the python regridding command
        f"python {regrid_script} "
        f"-b $batch_file "
        f"-d {target_grid_fp} "
        f"-o {regrid_dir} "
        f"--interp_method {interp_method} "
        "$sftlf_args"
    )

    if rasdafy:
        pycommands += " --rasdafy"
    if no_clobber:
        pycommands += " --no-clobber"

    pycommands += "\n\necho End regridding && date\n"

    commands = sbatch_head + pycommands

    with open(sbatch_fp, "w") as f:
        f.write(commands)

    print(f"Wrote sbatch array script to {sbatch_fp}", file=sys.stderr)
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

    Raises
    ------
    Exception
        If sbatch submission fails
    """
    try:
        out = subprocess.check_output(
            ["sbatch", str(sbatch_fp)], stderr=subprocess.STDOUT
        )
        job_id = out.decode().replace("\n", "").split(" ")[-1]
        print(f"  Submitted {sbatch_fp.name}: job ID {job_id}", file=sys.stderr)
        return job_id
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Failed to submit {sbatch_fp}: {e.output.decode() if e.output else str(e)}"
        )
        print(f"ERROR: {error_msg}", file=sys.stderr)
        raise Exception(error_msg)


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
    _ = [fp.unlink() for fp in sbatch_dir.glob("*.out")]

    print(f"Searching for batch files in {regrid_batch_dir}...", file=sys.stderr)
    expected_jobs = 0
    batch_files_info = (
        []
    )  # Will store (var, model, scenario, freq, batch_fp, src_sftlf, dst_sftlf)

    for var in vars.split():
        for freq in freqs.split():
            for model in models.split():
                for scenario in scenarios.split():
                    # find the batch file for this model, scenario, variable, and frequency
                    pattern = f"batch_{model}*{scenario}*{freq}*{var}*.txt"
                    expected_jobs += 1
                    matches = list(regrid_batch_dir.glob(pattern))

                    if not matches:
                        print(
                            f"WARNING: No batch file found for {model}/{scenario}/{freq}/{var} (pattern: {pattern})",
                            file=sys.stderr,
                        )
                        continue

                    for fp in matches:
                        # Determine sftlf files if needed
                        src_sftlf = None
                        dst_sftlf = None
                        if var in landsea_variables:
                            assert (
                                target_sftlf_fp is not None
                            ), "A target sftlf file must be supplied if any variables are land/sea"
                            try:
                                src_sftlf = model_sftlf_lu[model]
                            except KeyError:
                                src_sftlf = None
                            dst_sftlf = target_sftlf_fp

                        batch_files_info.append(
                            (var, model, scenario, freq, fp, src_sftlf, dst_sftlf)
                        )

    # Report batch file discovery results
    print(f"\nBatch file discovery complete:", file=sys.stderr)
    print(f"  Expected jobs: {expected_jobs}", file=sys.stderr)
    print(f"  Found batch files: {len(batch_files_info)}", file=sys.stderr)

    if len(batch_files_info) == 0:
        raise Exception("No batch files found!")

    if len(batch_files_info) != expected_jobs:
        print(
            f"WARNING: Mismatch between expected ({expected_jobs}) and found ({len(batch_files_info)}) batch files!",
            file=sys.stderr,
        )

    # Write config file for array job
    config_file = sbatch_dir.joinpath("regrid_config.txt")
    array_range = write_config_file(config_file, batch_files_info)

    # Create single array job script
    sbatch_fp = sbatch_dir.joinpath("regrid_array.slurm")
    sbatch_out_fp = sbatch_dir.joinpath(
        "regrid_array_%A_%a.out"
    )  # %A = job ID, %a = array task ID

    sbatch_head = make_sbatch_head(
        partition, sbatch_out_fp, conda_env_name, array_range=array_range
    )

    write_sbatch_array_regrid(
        sbatch_fp=sbatch_fp,
        config_file=config_file,
        regrid_script=regrid_script,
        regrid_dir=regrid_dir,
        target_grid_fp=target_grid_fp,
        no_clobber=no_clobber,
        interp_method=interp_method,
        rasdafy=rasdafy,
        sbatch_head=sbatch_head,
    )

    # Submit the array job
    print(
        f"\nSubmitting array job with {len(batch_files_info)} tasks...", file=sys.stderr
    )
    try:
        job_id = submit_sbatch(sbatch_fp)
        print(
            f"\nSubmission complete: Job ID {job_id} with {len(batch_files_info)} array tasks",
            file=sys.stderr,
        )
        # Print single job ID for prefect flow parsing (stdout only)
        print(job_id)
    except Exception as e:
        print(f"ERROR: Failed to submit array job: {e}", file=sys.stderr)
        raise
