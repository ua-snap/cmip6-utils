"""Use this script to generate the manifest of complete filepaths to mirror for a given ESGF node. Creates a CSV file named "<ESGF node>_manifest.csv", that contains the filepaths from the rows in the corresponding "holdings" table that contain target production data for mirroring on the ACDN. This is generated from the different "production" attributes (model variants, variables, frequencies, etc) in the config file.

Sample usage: 
    python generate_manifest.py --node llnl
    
    or

    python generate_manifest.py --node llnl --wrf
"""

import argparse
from datetime import datetime
import sys
import pandas as pd
from config import *


def arguments(argv):
    """Parse some args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=str, help="ESGF node to query", required=True)
    parser.add_argument(
        "--wrf",
        action="store_true",
        help="Whether or not to generate the manifest for the WRF variables, at sub-daily resolutions.",
    )
    args = parser.parse_args()
    esgf_node, wrf_vars = args.node, args.wrf

    return esgf_node, wrf_vars


def get_ymd_from_str(ymd_str):
    """Get the year, month, and dates from a string of format %Y%m or %Y%m%d"""
    if len(ymd_str) == 6:
        dt = datetime.strptime(ymd_str, "%Y%m")
        ymd_tuple = dt.year, dt.month, None
    elif len(ymd_str) == 8:
        dt = datetime.strptime(ymd_str, "%Y%m%d")
        ymd_tuple = dt.year, dt.month, dt.day
    elif len(ymd_str) == 12:
        dt = datetime.strptime(ymd_str, "%Y%m%d%H%M")
        ymd_tuple = dt.year, dt.month, dt.day
    else:
        print(ymd_str)
        exit()

    return ymd_tuple


def split_by_filenames(row, variable_lut):
    row_di = row.to_dict()
    row_di["filename"] = [fn.replace("'", "") for fn in row_di["filenames"]]
    if variable_lut[row_di["variable"]]["freqs"][0] in ["fx", "Ofx"]:
        # these variables do not have time ranges
        row_di["start_year"] = [None]
        row_di["start_month"] = [None]
        row_di["start_day"] = [None]
        row_di["end_year"] = [None]
        row_di["end_month"] = [None]
        row_di["end_day"] = [None]
    else:
        start_end_times = [
            fn.split("_")[-1].split(".nc")[0].split("-") for fn in row_di["filenames"]
        ]
        start_ymd = [get_ymd_from_str(year_tuple[0]) for year_tuple in start_end_times]
        end_ymd = [get_ymd_from_str(year_tuple[1]) for year_tuple in start_end_times]
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
    ESGF_NODE, wrf_vars = arguments(sys.argv)

    # make the holdings table
    if wrf_vars:
        variable_lut = wrf_variables
        #  we need to add a "freqs" key to each child dict in the WRF variable dict.
        #  We will do so using the main list of all possible subdaily table IDs.
        for var_id in variable_lut:
            variable_lut[var_id]["freqs"] = subdaily_table_ids
        suffix = "_wrf"
        # for WRF, we only are after two models, for now:
        models = ["CNRM-CM6-1-HR", "MIROC6"]
    else:
        variable_lut = variables
        suffix = ""
        # we want all models in prod_variant_lu if not WRF
        models = list(prod_variant_lu.keys())

    holdings = pd.read_csv(
        holdings_tmp_fn.format(esgf_node=ESGF_NODE, suffix=suffix),
        # filenames column should be list for each row
        converters={"filenames": lambda x: x.strip("[]").split(", ")},
    )
    # ignore rows where data not on LLNL node for now
    holdings = holdings.query("~n_files.isnull()")

    pre_manifest = []

    # group batch files by variable name and
    for var_id in variable_lut:
        for freq in variable_lut[var_id]["freqs"]:
            transfer_paths = []
            # holdings table is created from production scenarios only, so all scenarios in here should be included
            # iterate over model so that we can subset by the correct variant to be mirrored:
            for model in prod_variant_lu:
                # subset to the variant we will be mirroring
                variant = prod_variant_lu[model]
                query_str = f"model == '{model}' & variant == '{variant}' & frequency == '{freq}' & variable == '{var_id}'"
                pre_manifest.append(holdings.query(query_str))

    pre_manifest = pd.concat(pre_manifest)

    manifest = pre_manifest.apply(
        lambda row: split_by_filenames(row, variable_lut), axis=1
    )
    manifest = pd.concat(manifest.to_list())

    # ignore these files, for one reason or another!
    # this is just a grouping of all the split up files, doesn't occur for any other variable!
    manifest = manifest.query(
        "filename != 'psl_day_CESM2-WACCM_historical_r1i1p1f1_gn_18500101-20150101.nc'"
    )

    manifest.to_csv(
        manifest_tmp_fn.format(esgf_node=ESGF_NODE, suffix=suffix), index=False
    )
