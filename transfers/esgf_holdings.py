"""Generate a reference table of CMIP6 holdings on a given ESGF node.

The table resulting from this should have the following columns: model, scenario, variant, frequency, variable, grid_type, version, n_files, filenames

Usage:
    python esgf_holdings.py --node llnl
"""

import argparse
import sys
from itertools import product
from multiprocessing import Pool
import numpy as np
import pandas as pd
from config import *
import luts
import utils


def arguments(argv):
    """Parse some args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=str, help="ESGF node to query", required=True)
    args = parser.parse_args()
    esgf_node = args.node

    return esgf_node


def list_variants(tc, esgf_node, activity, model, scenario):
    """List the different variants available on a particular ESGF node for the given activity, model, and scenario"""
    node_ep = luts.globus_esgf_endpoints[esgf_node]["ep"]
    node_prefix = Path(luts.globus_esgf_endpoints[esgf_node]["prefix"])
    scenario_path = node_prefix.joinpath(
        activity, luts.model_inst_lu[model], model, scenario
    )
    
    variants = utils.operation_ls(tc, node_ep, scenario_path)
    
    if isinstance(variants, int):
        return {}
    elif isinstance(variants, list):
        return {"model": model, "scenario": scenario, "variants": variants}
    
    
def make_model_variants_lut(tc, esgf_node, models, scenarios):
    """Create a lookup table of all variants available for each model/ scenario combination. Uses an existing TransferClient object. 
    
    Returns a pandas DataFrame with a list of variants available for each model/scenario
    """
    # find what variants are available for each model / scenario combination
    args = list(
        product(["CMIP"], models, ["historical"])
    ) + list(
        product(["ScenarioMIP"], models, scenarios)
    )
    # include the TransferClient object for each query
    args = [[tc, esgf_node] + list(a) for a in args]
    
    with Pool(32) as pool:
        rows = pool.starmap(list_variants, args)
        
    df = pd.DataFrame(rows)
    
    return df.dropna()


def get_filenames(tc, esgf_node, activity, model, scenario, variant, frequency, varname):
    """Get the file names for a some combination of model, scenario, and variable."""
    node_ep = luts.globus_esgf_endpoints[esgf_node]["ep"]
    node_prefix = Path(luts.globus_esgf_endpoints[esgf_node]["prefix"])
    
    # the subdirectory under the variable name is the grid type.
    #  This is almost always "gn", meaning the model's native grid, but it could be different. 
    #  So we have to check it instead of assuming. I have only seen one model where this is different (gr1, GFDL-ESM4)
    var_path = node_prefix.joinpath(
        activity, luts.model_inst_lu[model], model, scenario, variant, frequency, varname
    )
    grid_type = utils.operation_ls(tc, node_ep, var_path)

    if isinstance(grid_type, int):
        # error if int
        # there is no data for this particular combination.
        row_di = {
            "model": model,
            "scenario": scenario,
            "variant": variant,
            "frequency": frequency,
            "variable": varname,
            "grid_type": None,
            "version": None,
            "n_files": None,
            "filenames": None,
        }

    else:
        # combo does exist, return all filenames
        grid_type = grid_type[0]
        versions = utils.operation_ls(tc, node_ep, var_path.joinpath(grid_type))
        # go with newer version
        use_version = sorted(versions)[-1]
        fns = utils.operation_ls(tc, node_ep, var_path.joinpath(grid_type, use_version))
        row_di = {
            "model": model,
            "scenario": scenario,
            "variant": variant,
            "frequency": frequency,
            "variable": varname,
            "grid_type": grid_type,
            "version": use_version,
            "n_files": len(fns),
            "filenames": fns,
        }
    
    return row_di


def make_holdings_table(tc, esgf_node, models, scenarios, variant_lut, varnames):
    """Create a table of filename availability for all models, scenarios, variants, and variable names"""
    # generate lists of arguments from all combinations of variables, models, and scenarios
    freqs = ["Amon", "day"]
    args = []
    for i, row in variant_lut.iterrows():
        activity = "CMIP" if row["scenario"] == "historical" else "ScenarioMIP"
        args.extend(
            product(
                [tc], 
                [esgf_node], 
                [activity], 
                [row["model"]], 
                [row["scenario"]], 
                row["variants"],
                freqs, 
                varnames
            )
        )

    with Pool(32) as pool:
        rows = pool.starmap(get_filenames, args)

    filenames_lu = pd.DataFrame(rows)
    
    return filenames_lu


if __name__ == "__main__":
    esgf_node = arguments(sys.argv)
    
    # create an authorization client for Globus
    auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    tc = utils.login_and_get_transfer_client(auth_client)
    
    print("Log in successful. Running the audit now.")
    
    # check if we need to grant conesnt for ACDN
    utils.check_for_consent_required(tc, auth_client, acdn_ep)
    
    # specify the models we are interested in
    # currently this is just everything in the luts.model_inst_lu table
    models = list(luts.model_inst_lu.keys())
    
    # get a dataframe of variants available for each model and scenario
    variant_lut = make_model_variants_lut(tc, esgf_node, models, prod_scenarios)
    
    # amke the holdings table
    holdings_df = make_holdings_table(tc, esgf_node, models, prod_scenarios, variant_lut, prod_vars)

    holdings_df.to_csv(f"{esgf_node}_esgf_holdings.csv", index=False)
