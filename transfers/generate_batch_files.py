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
        transfer_list (list): has format [(<remote path>, <target path>), ...] for all files in row["filenames"]
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
    
    transfer_list = []
    for fn in row["filenames"]:
        fp = group_path.joinpath(fn.replace("'", ""))
        transfer_list.append((llnl_prefix.joinpath(fp), acdn_prefix.joinpath(fp)))
        
    return transfer_list


def write_batch_file(freq, varname, transfer_paths):
    """Write the batch file for a particular variable and scenario group"""
    batch_file = batch_dir.joinpath(batch_tmp_fn.format(esgf_node=ESGF_NODE, freq=freq, varname=varname))
    with open(batch_file, "w") as f:
        for paths in transfer_paths:
            f.write(f"{paths[0]} {paths[1]}\n")


if __name__ == "__main__":
    # this script only runs for a single ESGF node
    ESGF_NODE = arguments(sys.argv)
    
    holdings = pd.read_csv(
        holdings_tmp_fn.format(esgf_node=ESGF_NODE),
        # filenames column should be list for each row
        converters={"filenames": lambda x: x.strip("[]").split(", ")}
    )
    # ignore rows where data not on LLNL node for now
    holdings = holdings.query("~n_files.isnull()")

    for freq in prod_freqs:
        for varname in prod_vars:
            transfer_paths = []
            # holdings table is created from production scenarios only, so all scenarios in here should be included
            # iterate over model so that we can subset by the correct variant to be mirrored:
            for model in prod_models:
                # subset to the variant we will be mirroring
                variant = luts.prod_variant_lu[model]
                query_str = f"model == '{model}' & variant == '{variant}' & frequency == '{freq}' & variable == '{varname}'"
                for i, row in holdings.query(query_str).iterrows():
                    transfer_paths.extend(generate_transfer_paths(row, freq))

            write_batch_file(freq, varname, transfer_paths)

