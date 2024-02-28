"""Generate a reference table of CMIP6 holdings on a given ESGF node.

The table resulting from this should have the following columns: model, scenario, variant, table_id, variable, grid_type, version, n_files, filenames

Usage:
    python esgf_holdings.py --node llnl --ncpus 24
    
    or

    python esgf_holdings.py --node llnl --ncpus 24 --wrf
"""

import argparse
import sys
from itertools import product
from multiprocessing import Pool
import globus_sdk
import numpy as np
import pandas as pd
from config import *
import utils


def arguments(argv):
    """Parse some args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=str, help="ESGF node to query", required=True)
    parser.add_argument(
        "--ncpus", type=int, help="Number of cores to use", required=False, default=8
    )
    parser.add_argument(
        "--wrf",
        action="store_true",
        help="Whether or not to audit holdings for WRF variables, at sub-daily resolutions.",
    )
    args = parser.parse_args()
    esgf_node, ncpus, do_wrf = args.node, args.ncpus, args.wrf

    return esgf_node, ncpus, do_wrf


def list_variants(tc, node_ep, node_prefix, activity, model, scenario):
    """List the different variants available on a particular ESGF node for the given activity, model, and scenario"""
    scenario_path = node_prefix.joinpath(
        activity, model_inst_lu[model], model, scenario
    )

    variants = utils.operation_ls(tc, node_ep, scenario_path)

    if isinstance(variants, int):
        return {}
    elif isinstance(variants, list):
        return {"model": model, "scenario": scenario, "variants": variants}


def make_model_variants_lut(tc, node_ep, node_prefix, models, scenarios, ncpus):
    """Create a lookup table of all variants available for each model/ scenario combination. Uses an existing TransferClient object.

    Returns a pandas DataFrame with a list of variants available for each model/scenario
    """
    # find what variants are available for each model / scenario combination
    args = list(product(["CMIP"], models, ["historical"])) + list(
        product(["ScenarioMIP"], models, scenarios)
    )
    # include the TransferClient object for each query
    args = [[tc, node_ep, node_prefix] + list(a) for a in args]

    with Pool(ncpus) as pool:
        rows = pool.starmap(list_variants, args)

    df = pd.DataFrame(rows)

    return df.dropna()


def get_filenames(
    tc, node_ep, node_prefix, activity, model, scenario, variant, table_id, varname
):
    """Get the file names for a some combination of model, scenario, and variable."""
    # the subdirectory under the variable name is the grid type.
    #  This is almost always "gn", meaning the model's native grid, but it could be different.
    #  So we have to check it instead of assuming. I have only seen one model where this is different (gr1, GFDL-ESM4)
    var_path = node_prefix.joinpath(
        activity,
        model_inst_lu[model],
        model,
        scenario,
        variant,
        table_id,
        varname,
    )
    var_id_ls = utils.operation_ls(tc, node_ep, var_path)
    empty_row = {
        "model": model,
        "scenario": scenario,
        "variant": variant,
        # table ID is essentially frequency, but there are different codes for different variables,
        #  e.g. Eday and day, both for the "day" frequency
        "table_id": table_id,
        "variable": varname,
        "grid_type": None,
        "version": None,
        "n_files": None,
        "filenames": None,
    }
    if isinstance(var_id_ls, int) or (len(var_id_ls) == 0):
        # error if int
        # there is no data for this particular combination.
        # or if variable folder exists but is empty, also should give empty row
        row_di = empty_row
    elif len(var_id_ls) > 1:
        # if this ever happens, we will need to think about how to handle it, so just exit for now :)
        exit(
            f"Unexpected result, multiple grid types found for ls operation on {var_path}"
        )
    elif len(var_id_ls) == 1:
        grid_type = var_id_ls[0]
        grid_path = var_path.joinpath(grid_type)
        grid_type_ls = utils.operation_ls(tc, node_ep, grid_path)

        if isinstance(grid_type_ls, int):
            print(f"Unexpected result, ls error on supposed valid path: {grid_path}")
            row_di = empty_row.update({"grid_type": grid_type})
        elif len(grid_type_ls) > 0:
            use_version = sorted(grid_type_ls)[-1]
            version_path = var_path.joinpath(grid_type, use_version)
            fns_ls = utils.operation_ls(tc, node_ep, version_path)

            # handle possible missing files, even though version exists? new observation as of 2/14/24
            if isinstance(fns_ls, int):
                print(
                    f"Unexpected result, ls error on supposed valid path: {version_path}"
                )
                row_di = empty_row.update(
                    {"grid_type": grid_type, "version": use_version}
                )
            else:
                n_files = len(fns_ls)

                row_di = {
                    "model": model,
                    "scenario": scenario,
                    "variant": variant,
                    "table_id": table_id,
                    "variable": varname,
                    "grid_type": grid_type,
                    "version": use_version,
                    "n_files": n_files,
                    "filenames": fns_ls,
                }

    return row_di


def make_holdings_table(tc, node_ep, node_prefix, variant_lut, ncpus, variable_lut):
    """Create a table of filename availability for all models, scenarios, variants, and variable names"""
    # generate lists of arguments from all combinations of variables, models, and scenarios
    args = []
    for i, row in variant_lut.iterrows():
        activity = "CMIP" if row["scenario"] == "historical" else "ScenarioMIP"
        for var_id in variable_lut:
            for t_id in variable_lut[var_id]["table_ids"]:
                args.extend(
                    # make these into lists so we can iterate over variables/table IDs and add
                    product(
                        [tc],
                        [node_ep],
                        [node_prefix],
                        [activity],
                        [row["model"]],
                        [row["scenario"]],
                        row["variants"],
                        [t_id],
                        [var_id],
                    )
                )

    with Pool(ncpus) as pool:
        rows = pool.starmap(get_filenames, args)

    filenames_lu = pd.DataFrame(rows)

    return filenames_lu


if __name__ == "__main__":
    esgf_node, ncpus, do_wrf = arguments(sys.argv)

    # create an authorization client for Globus
    auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    tc = utils.login_and_get_transfer_client(auth_client)

    print("Log in successful. Running the audit now.")

    # check if we need to grant conesnt for ACDN
    utils.check_for_consent_required(tc, auth_client, acdn_ep)

    # I think this activation has to be done?
    node_ep = globus_esgf_endpoints[esgf_node]["ep"]
    tc.endpoint_autoactivate(node_ep)
    node_prefix = Path(globus_esgf_endpoints[esgf_node]["prefix"])

    # specify the models we are interested in
    # currently this is just everything in the config.model_inst_lu table
    models = list(model_inst_lu.keys())

    # get a dataframe of variants available for each model and scenario
    variant_lut = make_model_variants_lut(
        tc, node_ep, node_prefix, models, prod_scenarios, ncpus
    )

    # Check that we won't get a particular error which pops up when the user has not logged into the ESGF node via Globus
    try:
        _ = variant_lut.iloc[0]["scenario"]
    except KeyError:
        print(
            "Key error. Check that you have logged into the endpoint via the Globus app."
        )

    # make the holdings table
    if do_wrf:
        variable_lut = wrf_variables
        # to keep consistent with process for auditing standard variables,
        #  we need to add a "table_id" key to each child dict in the WRF variable dict.
        #  We will do so using the main list of all possible subdaily table IDs.
        for var_id in variable_lut:
            variable_lut[var_id]["table_ids"] = subdaily_table_ids
        outfn_suffix = "_wrf"
    else:
        variable_lut = variables
        outfn_suffix = ""

    holdings_df = make_holdings_table(
        tc=tc,
        node_ep=node_ep,
        node_prefix=node_prefix,
        variant_lut=variant_lut,
        ncpus=ncpus,
        variable_lut=variable_lut,
    )

    holdings_df.to_csv(f"{esgf_node}_esgf_holdings{outfn_suffix}.csv", index=False)
