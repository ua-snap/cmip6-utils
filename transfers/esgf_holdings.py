"""Generate a reference table of CMIP6 holdings on a given ESGF node.

The table resulting from this should have the following columns: model, scenario, variant, table_id, variable, grid_type, version, n_files, filenames

Usage:
    python esgf_holdings.py --node llnl --ncpus 24
    
    or

    python esgf_holdings.py --node llnl --ncpus 24 --wrf
"""

import argparse
import sys
from itertools import product, chain
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
    """List the different variants available on a particular ESGF node for the given activity, model, and scenario.
    Returns a list of rows to allow for more than one institution per model."""
    if isinstance(model_inst_lu[model], list):
        rows = []
        for inst in model_inst_lu[model]:
            scenario_path = node_prefix.joinpath(
            activity, inst, model, scenario
            )

            variants = utils.operation_ls(tc, node_ep, scenario_path)

            if isinstance(variants, int):
                rows.append({})
            elif isinstance(variants, list):
                rows.append({"model": model, "scenario": scenario, "variants": variants})
        return rows
    
    else:
        scenario_path = node_prefix.joinpath(
            activity, model_inst_lu[model], model, scenario
        )

        variants = utils.operation_ls(tc, node_ep, scenario_path)

        if isinstance(variants, int):
            return [{}]
        elif isinstance(variants, list):
            return [{"model": model, "scenario": scenario, "variants": variants}]


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
        lists_of_rows = pool.starmap(list_variants, args)
    rows = list((chain(*lists_of_rows)))

    df = pd.DataFrame(rows)
    print(df)

    return df.dropna()


def get_filenames(
    tc, node_ep, node_prefix, activity, model, scenario, variant, table_id, varname
):
    """Get the file names for a some combination of model, scenario, and variable."""
    # the subdirectory under the variable name is the grid type.
    #  This is almost always "gn", meaning the model's native grid, but it could be different.
    #  So we have to check it instead of assuming. As of 3/29/24, we now know in multiple models some variables have multiple grids.
    
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

    # return row dicts as items in a list (allows for multiple row returns if >1 grid type)
    list_of_row_dicts = []

    # check if there is more than one institution listed for the model; list the filenames for each inst
    if isinstance(model_inst_lu[model], str):
        insts = [model_inst_lu[model]]
    else:
        insts = model_inst_lu[model]
        
    for inst in insts:
        var_path = node_prefix.joinpath(
            activity,
            inst,
            model,
            scenario,
            variant,
            table_id,
            varname,
        )

        var_id_ls = utils.operation_ls(tc, node_ep, var_path)
        if isinstance(var_id_ls, int) or (len(var_id_ls) == 0):
            # error if int (indicates a http status code, probably error)
            # or if there is no data for this particular combination,
            # or if variable folder exists but is empty, also should give empty row
            list_of_row_dicts.append(empty_row)
        else:
            for grid_type in var_id_ls:
                grid_path = var_path.joinpath(grid_type)
                grid_type_ls = utils.operation_ls(tc, node_ep, grid_path)

                # handle possible missing version, even though grid exists? new observation as of 3/29/24
                if isinstance(grid_type_ls, int) or (len(grid_type_ls) == 0):
                    print(f"Unexpected result, ls error on supposed valid path: {grid_path}")
                    print("ls operation in grid directory returned: ")
                    print(grid_type_ls)
                    list_of_row_dicts.append(empty_row.update({"grid_type": grid_type}))
                else:
                    use_version = sorted(grid_type_ls)[-1]
                    version_path = var_path.joinpath(grid_type, use_version)
                    fns_ls = utils.operation_ls(tc, node_ep, version_path)

                    # handle possible missing files, even though version exists? new observation as of 2/14/24
                    if isinstance(fns_ls, int) or (len(fns_ls) == 0):
                        print(
                            f"Unexpected result, ls error on supposed valid path: {version_path}"
                        )
                        print("ls operation in most recent version directory returned: ")
                        print(fns_ls)
                        list_of_row_dicts.append(empty_row.update({"grid_type": grid_type, "version": use_version}))
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

                        list_of_row_dicts.append(row_di)
    
    if len(list_of_row_dicts) == 0:
        list_of_row_dicts.append(empty_row)

    return list_of_row_dicts


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
        lists_of_row_dicts = pool.starmap(get_filenames, args)

    # holdings data items are returned in lists, with some lists representing >1 row
    # we need to combine these together as one master list of items to write to rows
    rows = list(chain(*lists_of_row_dicts))
    # do a final check for any non-dict items that will not translate to dataframe rows
    # remove any items that are not dicts, and print error message
    to_remove = []
    for row in rows:
        if not isinstance(row, dict):
            print(f"Removing a row that is not a dict: {row}")
            to_remove.append(row)
    rows_for_df = [i for i in rows if i not in to_remove]

    filenames_lu = pd.DataFrame(rows_for_df)

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
