"""Script to run the globus batch transfer from LLNL ESGF node to ACDN. Accepts arguement for table ID and variable name, and looks in batch_files/ for the corresponding file.

Example usage: `python transfer.py -v tas`
"""

import argparse
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
        "--table_id",
        type=str,
        help="Temporal frequency. Should be a valid table ID such as 'day', 'Eday', 'Amon', 'Lmon', etc. If no frequency is supplied, all are used.",
        default="*",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Set the --dry-run switch in globus transfer",
    )
    args = parser.parse_args()

    return args.varname, args.table_id, args.dry_run


if __name__ == "__main__":
    varname, table_id, dry_run = arguments(sys.argv)

    if varname == "all_variables":
        # wildcard in front of frequency to handle different ones like day, Oday, SIday etc
        batch_fps = Path("batch_files").glob(f"batch_llnl_{table_id}_*.txt")
        all_fp = Path(f"/tmp/batch_llnl_{table_id}_all_variables.txt")
        batch_fp = str(all_fp)
        with open(batch_fp, "w") as batch_file:
            for fp in batch_fps:
                with open(fp) as infile:
                    batch_file.write(infile.read())
    else:
        batch_fp = f"batch_files/batch_llnl_{table_id}_{varname}.txt"

    command = [
        "globus",
        "transfer",
        llnl_ep,
        acdn_ep,
        "--label",
        f"Batch {table_id} {varname}",
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
