"""Submit the QC for a resolution: one Slurm array task per (variable, model), each running
qc_stats.py then qc_report.py, plus a dependent job that writes the summary PDF.

usage: python run_qc.py --res 12km [--vars pr dtr] [--models MIROC6] [--skip_done] [--dry_run]
"""

import argparse
import subprocess
from datetime import datetime

from qc_common import HERE, jobs_for, load_config, out_dir

HEAD = """#!/bin/sh
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --partition={partition}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --output={log_dir}/{name}_%A_%a.out
export NUMBA_CACHE_DIR=${{SLURM_TMPDIR:-/tmp}}/${{USER}}_numba_${{SLURM_JOB_ID}}
export MPLCONFIGDIR=${{SLURM_TMPDIR:-/tmp}}/${{USER}}_mpl_${{SLURM_JOB_ID}}
mkdir -p $NUMBA_CACHE_DIR $MPLCONFIGDIR
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate {env}
cd {code}
echo "node=$(hostname) start=$(date)"
"""

ARRAY = (
    HEAD
    + """set -- $(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {tasks})
var=$1; model=$2
echo "task $SLURM_ARRAY_TASK_ID: $var $model"
python qc_stats.py --res {res} --var $var --model $model && \\
python qc_report.py model --res {res} --var $var --model $model
status=$?
echo "end=$(date) status=$status"
exit $status
"""
)

SUMMARY = (
    HEAD
    + """python qc_report.py summary --res {res}
status=$?
echo "end=$(date) status=$status"
exit $status
"""
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--res", required=True)
    ap.add_argument("--vars", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument(
        "--skip_done",
        action="store_true",
        help="skip (var, model) pairs whose PDF exists",
    )
    ap.add_argument("--array_limit", type=int, default=20)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    od = out_dir(cfg, a.res)
    jobs = jobs_for(cfg, a.res, a.vars, a.models)
    if a.skip_done:
        jobs = [
            (v, m)
            for v, m in jobs
            if not (od / "pdf" / f"qc_{v}_{m}_{a.res}.pdf").exists()
        ]
    if not jobs:
        print("nothing to do")
        return
    sdir = od / "slurm"
    log_dir = sdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    tasks = sdir / f"tasks_{stamp}.txt"
    tasks.write_text("".join(f"{v} {m}\n" for v, m in jobs))
    common = dict(
        cfg["resolutions"][a.res]["slurm"],
        env=cfg["conda_env"],
        code=HERE,
        log_dir=log_dir,
        res=a.res,
    )
    arr = sdir / f"qc_array_{stamp}.slurm"
    arr.write_text(ARRAY.format(name=f"qc_{a.res}", tasks=tasks, **common))
    summ = sdir / f"qc_summary_{stamp}.slurm"
    summ.write_text(
        SUMMARY.format(
            name=f"qc_summary_{a.res}",
            **{**common, "mem": "32G", "cpus_per_task": 2, "time": "00:30:00"},
        )
    )
    print(f"{len(jobs)} tasks -> {tasks}")
    cmd = [
        "sbatch",
        "--parsable",
        "--chdir=/tmp",
        f"--array=0-{len(jobs) - 1}%{a.array_limit}",
        str(arr),
    ]
    if a.dry_run:
        print(" ".join(cmd))
        return
    jid = (
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        .stdout.strip()
        .split(";")[0]
    )
    print(f"array job {jid}")
    jid2 = subprocess.run(
        [
            "sbatch",
            "--parsable",
            "--chdir=/tmp",
            f"--dependency=afterany:{jid}",
            str(summ),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"summary job {jid2} (after {jid})")


if __name__ == "__main__":
    main()
