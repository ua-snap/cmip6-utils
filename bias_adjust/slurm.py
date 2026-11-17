"""Functions to assist with constructing slurm jobs."""

import logging
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def make_sbatch_head(
    partition,
    sbatch_out_path,
    conda_env_name,
    job_name,
    array_range=None,
    time_limit="04:00:00",
    cpus_per_task=None,
    mem=None,
    extra_sbatch=None,
):
    """Make a string of SBATCH commands that can be written into a .slurm script

    Args:
        partition (str): name of the partition to use
        sbatch_out_path (path_like): path to where sbatch stdout should be written
        conda_env_name (str): name of the conda environment to activate
        job_name (str): name of the job to use in sbatch
        array_range (str): string to use in the SLURM array
        time_limit (str): time limit for the job in HH:MM:SS format
        cpus_per_task (int): CPUs to allocate (keeps Slurm from packing multi-threaded
            jobs onto a single core on shared partitions such as 'analysis')
        mem (str): memory request, e.g. "250G"; None to omit (whole node on exclusive
            partitions). Defaults to 250G on the 'analysis' partition.
        extra_sbatch (list[str]): additional raw "#SBATCH ..." lines

    Returns:
        sbatch_head (str): string of SBATCH commands ready to be used as parameter in sbatch-writing functions.
    """
    sbatch_head = (
        "#!/bin/sh\n"
        f"#SBATCH --job-name={job_name}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --output={sbatch_out_path}\n"
        f"#SBATCH --time={time_limit}\n"
    )
    if mem is None and partition == "analysis":
        mem = "250G"
    if mem:
        sbatch_head += f"#SBATCH --mem={mem}\n"
    if cpus_per_task:
        sbatch_head += f"#SBATCH --cpus-per-task={cpus_per_task}\n"
    if array_range is not None:
        sbatch_head += f"#SBATCH --array={array_range}%10\n"
    for line in extra_sbatch or []:
        sbatch_head += line.rstrip("\n") + "\n"
    sbatch_head += (
        # print start time and node diagnostics (CPU binding, local scratch)
        "echo Start slurm && date\n"
        'echo "node=$(hostname) nproc=$(nproc) cpuset=$(taskset -cp $$ 2>/dev/null | cut -d: -f2) '
        'SLURM_TMPDIR=${SLURM_TMPDIR:-unset} SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset}"\n'
        "df -h /tmp ${SLURM_TMPDIR:-/tmp} 2>/dev/null | tail -n +2\n"
        # return freed memory to the OS promptly (dask 'unmanaged memory high' mitigation)
        "export MALLOC_TRIM_THRESHOLD_=0\n"
        # $HOME is not writable from jobs submitted through tclaude: give numba (xclim's
        # JIT cache) and matplotlib a node-local scratch dir or `import xclim` fails.
        "export NUMBA_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/${USER}/numba_cache_${SLURM_JOB_ID}\n"
        "export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/${USER}/mpl_${SLURM_JOB_ID}\n"
        "mkdir -p $NUMBA_CACHE_DIR $MPLCONFIGDIR\n"
        # prepare shell for using activate - Chinook requirement
        # this seems to work to initialize conda without init script
        'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"\n'
        f"conda activate {conda_env_name}\n"
    )

    return sbatch_head


def submit_sbatch(sbatch_path):
    """Submit a script to slurm via sbatch.

    Parameters
    ----------
    sbatch_path : pathlib.PosixPath
        path to .slurm script to submit

    Returns
    -------
    str
        job id for submitted job
    """
    out = subprocess.check_output(["sbatch", str(sbatch_path)])
    job_id = out.decode().replace("\n", "").split(" ")[-1]

    return job_id
