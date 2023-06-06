"""Generate batch files for transferring production mirror data.

This script is used to generate the batch_files/batch_<ESGF node>_(day|Amon)_<variable ID>.txt files that contain the filepaths (<source filepath (on ESGF node)> <destination filepath (on ACDN)>) to transfer to the Arctic Climate Data Node.

Sample usage: 
    python generate_batch_files.py --node llnl
"""

import argparse
import sys
import pandas as pd
import luts
from config import *


def arguments(argv):
    """Parse some args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=str, help="ESGF node to query", required=True)
    args = parser.parse_args()
    esgf_node = args.node

    return esgf_node


def generate_transfer_paths(row, freq):
    """Generate the paths for transferring between LLNL ESGF node and ACDN
    
    Args:
        row (pandas.core.series.Series): a single row series from pandas.DataFrame.iterrows() on dataframe of desired data filenames
        freq (str): temporal frequency to generate transfer paths for, should be either "day", or "Amon"

    Returns:
        transfer_tpl (tuple): has format (<remote path>, <target path>) for the file in row["filename"]
    """
    activity = "CMIP" if row["scenario"] == "historical" else "ScenarioMIP"
    model = row["model"]
    institution = luts.model_inst_lu[model]
    group_path = Path().joinpath(
        activity,
        institution,
        model,
        row["scenario"],
        row["variant"],
        freq,
        row["variable"],
        row["grid_type"],
        row["version"],
    )
    
    fn = row["filename"]
    fp = group_path.joinpath(fn)
    transfer_tpl = (llnl_prefix.joinpath(fp), acdn_prefix.joinpath(fp))

    return transfer_tpl


def write_batch_file(freq, varname, transfer_paths):
    """Write the batch file for a particular variable and scenario group"""
    batch_file = batch_dir.joinpath(batch_tmp_fn.format(esgf_node=ESGF_NODE, freq=freq, varname=varname))
    with open(batch_file, "w") as f:
        for paths in transfer_paths:
            f.write(f"{paths[0]} {paths[1]}\n")


if __name__ == "__main__":
    # this script only runs for a single ESGF node
    ESGF_NODE = arguments(sys.argv)
    
    # use the manifest file for generating batch files
    manifest = pd.read_csv(
        manifest_tmp_fn.format(esgf_node=ESGF_NODE),
    )
    
    # group batch files by variable name and 
    for freq in prod_freqs:
        for varname in prod_vars:
            transfer_paths = []
            
            query_str = f"frequency == '{freq}' & variable == '{varname}'"
            for i, row in manifest.query(query_str).iterrows():
                transfer_paths.append(generate_transfer_paths(row, freq))
            
            # only write batch file if transfer paths were found
            if transfer_paths != []:
                write_batch_file(freq, varname, transfer_paths)
