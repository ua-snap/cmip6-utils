"""Generate batch files for transferring production mirror data.

This script is used to generate the batch_files/batch_<ESGF node>_(day|Amon)_<variable ID>.txt files that contain the filepaths (<source filepath (on ESGF node)> <destination filepath (on ACDN)>) to transfer to the Arctic Climate Data Node.

Sample usage: 
    python generate_batch_files.py
"""

import argparse
import sys
import pandas as pd
from config import *


def arguments(argv):
    """Parse some args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrf",
        action="store_true",
        help="Whether or not to generate the batch files for the WRF variables, at sub-daily resolutions.",
    )
    args = parser.parse_args()
    do_wrf = args.wrf

    return do_wrf


def generate_transfer_paths(row, table_id):
    """Generate the paths for transferring between LLNL ESGF node and ACDN

    Args:
        row (pandas.core.series.Series): a single row series from pandas.DataFrame.iterrows() on dataframe of desired data filenames
        table_id (str): table ID to generate transfer paths for

    Returns:
        transfer_tpl (tuple): has format (<remote path>, <target path>) for the file in row["filename"]
    """
    activity = "CMIP" if row["scenario"] == "historical" else "ScenarioMIP"
    model = row["model"]
    if isinstance(model_inst_lu[model], list):
        # if more than one inst for the model, choose the first if historical and second if scenario
        institution = (
            model_inst_lu[model][0] if activity == "CMIP" else model_inst_lu[model][1]
        )
    else:
        institution = model_inst_lu[model]
    group_path = Path().joinpath(
        activity,
        institution,
        model,
        row["scenario"],
        row["variant"],
        table_id,
        row["variable"],
        row["grid_type"],
        row["version"],
    )

    fn = row["filename"]
    fp = group_path.joinpath(fn)
    esgf_prefix = e3sm_prefix if model in e3sm_models_of_interest else llnl_prefix
    transfer_tpl = (esgf_prefix.joinpath(fp), acdn_prefix.joinpath(fp))

    return transfer_tpl


def write_batch_file(table_id, var_id, transfer_paths):
    """Write the batch file for a particular variable and scenario group"""
    batch_file = batch_dir.joinpath(
        batch_tmp_fn.format(esgf_node="llnl", table_id=table_id, var_id=var_id)
    )
    with open(batch_file, "w") as f:
        for paths in transfer_paths:
            f.write(f"{paths[0]} {paths[1]}\n")


if __name__ == "__main__":
    # this script only runs for a single ESGF node
    do_wrf = arguments(sys.argv)

    # use the manifest file for generating batch files
    suffix = "_wrf" if do_wrf else ""
    manifest = pd.read_csv(
        manifest_tmp_fn.format(esgf_node="llnl", suffix=suffix),
    )

    # group batch files by variable name and
    for var_id, var_df in manifest.groupby("variable"):
        for table_id, freq_df in var_df.groupby("table_id"):

            # skipping fx variables for now
            if table_id in ["fx", "Ofx"]:
                continue

            transfer_paths = []
            for i, row in freq_df.iterrows():
                transfer_paths.append(generate_transfer_paths(row, table_id))

            # only write batch file if transfer paths were found
            if transfer_paths != []:
                write_batch_file(table_id, var_id, transfer_paths)
