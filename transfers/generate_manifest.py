"""Use this script to generate the manifest of complete filepaths to mirror for a given ESGF node. Creates a CSV file named "<ESGF node>_manifest.csv", that contains the filepaths from the rows in the corresponding "holdings" table that contain target production data for mirroring on the ACDN. This is generated from the different "production" attributes (model variants, variables, frequencies, etc) in the config file.

Sample usage: 
    python generate_manifest.py --node llnl
"""

import argparse
from datetime import datetime
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


def get_ymd_from_str(ymd_str):
    """Get the year, month, and dates from a string of format %Y%m or %Y%m%d"""
    if len(ymd_str) == 6:
        dt = datetime.strptime(ymd_str, "%Y%m")
        ymd_tuple = dt.year, dt.month, None
    elif len(ymd_str) == 8:
        dt = datetime.strptime(ymd_str, "%Y%m%d")
        ymd_tuple = dt.year, dt.month, dt.day

        
    return ymd_tuple


def split_by_filenames(row):
    row_di = row.to_dict()
    start_end_times = [fn.split("_")[-1].split(".nc")[0].split("-") for fn in row_di["filenames"]]
    start_ymd = [get_ymd_from_str(year_tuple[0]) for year_tuple in start_end_times]
    end_ymd = [get_ymd_from_str(year_tuple[1]) for year_tuple in start_end_times]
    row_di["filename"] = [fn.replace("'", "") for fn in row_di["filenames"]]
    row_di["start_year"] = [ymd[0] for ymd in start_ymd]
    row_di["start_month"] = [ymd[1] for ymd in start_ymd]
    row_di["start_day"] = [ymd[2] for ymd in start_ymd]
    row_di["end_year"] = [ymd[0] for ymd in end_ymd]
    row_di["end_month"] = [ymd[1] for ymd in end_ymd]
    row_di["end_day"] = [ymd[2] for ymd in end_ymd]

    del row_di["filenames"]
    del row_di["n_files"]
    
    return pd.DataFrame(row_di)


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

    pre_manifest = []
    for freq in prod_freqs:
        for varname in prod_vars:
            transfer_paths = []
            # holdings table is created from production scenarios only, so all scenarios in here should be included
            # iterate over model so that we can subset by the correct variant to be mirrored:
            for model in prod_models:
                # subset to the variant we will be mirroring
                variant = luts.prod_variant_lu[model]
                query_str = f"model == '{model}' & variant == '{variant}' & frequency == '{freq}' & variable == '{varname}'"
                pre_manifest.append(holdings.query(query_str))

    pre_manifest = pd.concat(pre_manifest)

    manifest = pre_manifest.apply(lambda row: split_by_filenames(row), axis=1)
    manifest = pd.concat(manifest.to_list())
    
    manifest.to_csv(manifest_tmp_fn.format(esgf_node=ESGF_NODE), index=False)
    