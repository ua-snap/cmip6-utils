"""Script to run the globus batch transfer from LLNL ESGF node to ACDN. Accepts arguement for temporal frequency and variable name, and looks in batch_files/ for the corresponding file.

Example usage: `python transfer.py -v tas`
"""

import argparse
import time
import sys
from subprocess import check_output
from config import llnl_ep, acdn_ep


def arguments(argv):
    """Parse some args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--varname", type=str, help="Name of variable to transfer", required=True
    )
    parser.add_argument(
        "-f",
        "--freq",
        type=str,
        help="Temporal frequency (currently only option is 'day' and this is default)",
        default="day",
    )
    args = parser.parse_args()
    
    return args.varname, args.freq

                                     
if __name__ == "__main__":
    varname, freq = arguments(sys.argv)
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
    ]
    
    out = check_output(command)
    print(out.decode("utf-8"))
                                     