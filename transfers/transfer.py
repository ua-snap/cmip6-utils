"""Script to run the globus batch transfer from LLNL ESGF node to ACDN. Accepts arguement for temporal frequency and variable name, and looks in batch_files/ for the corresponding file.

Example usage: `python transfer.py -v tas`
"""

import argparse
import time
import sys
from pathlib import Path
from subprocess import check_output
from config import llnl_ep, acdn_ep


def arguments(argv):
    """Parse some args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--varname",
        type=str,
        help="Name of variable to transfer. If no variable is supplied, all batch files are used.",
        default="all_variables",
    )
    parser.add_argument(
        "-f",
        "--freq",
        type=str,
        help="Temporal frequency. Either 'day' (default) or 'mon'.",
        default="day",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Set the --dry-run switch in globus transfer",
    )
    args = parser.parse_args()

    return args.varname, args.freq, args.dry_run


if __name__ == "__main__":
    varname, freq, dry_run = arguments(sys.argv)

    if varname == "all_variables":
        # wildcard in front of frequency to handle different ones like day, Oday, SIday etc
        batch_fps = Path("batch_files").glob(f"batch_llnl_*{freq}_*.txt")
        all_fp = Path(f"/tmp/batch_llnl_{freq}_all_variables.txt")
        batch_fp = str(all_fp)
        with open(batch_fp, "w") as batch_file:
            for fp in batch_fps:
                with open(fp) as infile:
                    batch_file.write(infile.read())
    else:
        batch_fp = f"batch_files/batch_llnl_{freq}_{varname}.txt"

    command = [
        "globus",
        "transfer",
        llnl_ep,
        acdn_ep,
        "--label",
        f"Batch {freq} {varname}",
        "--batch",
        batch_fp,
        "--sync-level",
        "mtime",
    ]

    if dry_run:
        command += ["--dry-run"]

    out = check_output(command)
    print(out.decode("utf-8"))

    # delete the all-variable batch file if created
    try:
        all_fp.unlink()
    except:
        pass
