"""Collect the RUN_SUMMARY json lines written by bias_adjust.py job logs into one CSV.

Example usage:
    python summarize_runs.py --log_dirs /beegfs/.../tests/slurm_legacy/bias_adjust \
        /beegfs/.../tests/slurm_windowed_c25/bias_adjust \
        --out_csv /beegfs/.../tests/diagnostics/timing_calibration.csv
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

LINE = re.compile(r"RUN_SUMMARY (\{.*\})\s*$", re.MULTILINE)


def collect(log_dirs, pattern="*.out"):
    rows = []
    for d in log_dirs:
        for f in sorted(Path(d).glob(pattern)):
            text = f.read_text(errors="replace")
            m = LINE.findall(text)
            if not m:
                continue
            row = json.loads(m[-1])
            phase = row.pop("phase_s", {}) or {}
            for k, v in phase.items():
                row[f"phase_{k}"] = v
            peaks = row.pop("worker_peak_gb", {}) or {}
            if peaks:
                row["worker_peak_gb_max"] = max(peaks.values())
                row["worker_peak_gb_mean"] = round(sum(peaks.values()) / len(peaks), 2)
            row["log"] = str(f)
            # the nanny kills workers at normal shutdown ("still alive after ...,
            # killing"); only count those that happen before the cluster is closed
            pre_close = text.split("Closing Dask cluster")[0]
            row["n_restart_warnings"] = len(
                re.findall(r"Worker process still alive after", pre_close)
            ) + len(re.findall(r"Restarting worker|KilledWorker|Compute Failed", text))
            row["n_memory_warnings"] = len(
                re.findall(r"Unmanaged memory use is high|Pausing worker", text)
            )
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log_dirs", nargs="+", required=True)
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()
    df = collect(args.log_dirs)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    cols = [
        c
        for c in [
            "run_label",
            "sim",
            "resolution",
            "rank_main_only",
            "chunk_size",
            "n_workers",
            "threads_per_worker",
            "memory_limit_gb",
            "host",
            "cpus_on_node",
            "phase_adjust_compute_s",
            "phase_squeeze_s",
            "phase_write_s",
            "phase_total_s",
            "cluster_peak_gb",
            "worker_peak_gb_max",
            "client_maxrss_gb",
            "n_restart_warnings",
            "n_memory_warnings",
            "success",
        ]
        if c in df.columns
    ]
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(
            df[cols].to_string(index=False) if len(df) else "no RUN_SUMMARY lines found"
        )
