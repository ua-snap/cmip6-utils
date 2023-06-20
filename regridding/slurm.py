"""Functions to assist with constructing slurm jobs"""

import subprocess


def make_sbatch_head(slurm_email, partition, conda_init_script, ncpus):
    """Make a string of SBATCH commands that can be written into a .slurm script
    
    Args:
        slurm_email (str): email address for slurm failures
        partition (str): name of the partition to use
        conda_init_script (path_like): path to a script that contains commands for initializing the shells on the compute nodes to use conda activate
        ncpus (int): number of cpus to request
            
    Returns:
        sbatch_head (str): string of SBATCH commands ready to be used as parameter in sbatch-writing functions. The following gaps are left for filling with .format:
            - ncpus
            - output slurm filename
    """
    sbatch_head = (
        "#!/bin/sh\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --cpus-per-task={ncpus}\n"
        "#SBATCH --mail-type=FAIL\n"
        f"#SBATCH --mail-user={slurm_email}\n"
        f"#SBATCH -p {partition}\n"
        "#SBATCH --output {sbatch_out_fp}\n"
        # print start time
        "echo Start slurm && date\n"
        # prepare shell for using activate - Chinook requirement
        f"source {conda_init_script}\n"
        # okay this is not the desired way to do this, but Chinook compute
        # nodes are not working with anaconda-project, so we activate
        # this manually then run the python command
        f"conda activate cmip6-utils\n"
    )

    return sbatch_head


def write_sbatch_regrid(
    sbatch_fp,
    sbatch_out_fp,
    regrid_script,
    regrid_dir,
    regrid_batch_fp,
    dst_fp,
    sbatch_head
):
    """Write an sbatch script for executing the restacking script for a given group and variable, executes for a given list of years 
    
    Args:
        sbatch_fp (path_like): path to .slurm script to write sbatch commands to
        sbatch_out_fp (path_like): path to where sbatch stdout should be written
        regrid_script (path_like): path to the script to be called to run the regridding
        regrid_dir (pathlib.PosixPath): directory to write the regridded data to
        regrid_batch_fp (path_like): path to the batch file containing paths of CMIP6 files to regrid
        dst_fp (path_like): path to file being used as template / reference for destination grid
        sbatch_head (dict): string for sbatch head script
        
    Returns:
        None, writes the commands to sbatch_fp
        
    Notes:
        since these jobs seem to take on the order of 5 minutes or less, seems better to just run through all years once a node is secured for a job, instead of making a single job for every year / variable combination
    """
    pycommands = "\n"
    pycommands += (
        f"python {regrid_script} "
        f"-b {regrid_batch_fp} "
        f"-d {dst_fp} "
        f"-o {regrid_dir}\n\n"
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
