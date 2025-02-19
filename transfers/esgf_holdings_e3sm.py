"""Generate a reference table of E3SM CMIP6 holdings on a given ESGF node, following from esgf_holdings.py.
Doing a separate script for E3SM because ScenarioMIP data comes from a different root. 

Usage:
    python esgf_holdings_e3sm.py --node llnl --ncpus 24
"""

import argparse
import sys
from itertools import product, chain
from multiprocessing import Pool
import globus_sdk
import numpy as np
import pandas as pd
from esgf_holdings import *
from config import *
import utils


def arguments(argv):
    """Parse some args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=str, help="ESGF node to query", required=True)
    parser.add_argument(
        "--ncpus", type=int, help="Number of cores to use", required=False, default=8
    )
    args = parser.parse_args()
    esgf_node, ncpus = args.node, args.ncpus

    return esgf_node, ncpus


if __name__ == "__main__":
    esgf_node, ncpus = arguments(sys.argv)

    # create an authorization client for Globus
    auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    tc = utils.login_and_get_transfer_client(auth_client)

    print("Log in successful. Running the audit now.")

    # check if we need to grant conesnt for ACDN
    utils.check_for_consent_required(tc, auth_client, acdn_ep)

    # I think this activation has to be done?
    node_ep = globus_esgf_endpoints[esgf_node]["ep"]
    tc.endpoint_autoactivate(node_ep)
    # overriding this as global since it is used in the imported functions
    models = e3sm_models_of_interest

    # We know
    variant_lut = make_model_variants_lut(
        tc, node_ep, e3sm_prefix, model_inst_lu, models, prod_scenarios, ncpus
    )

    # Check that we won't get a particular error which pops up when the user has not logged into the ESGF node via Globus
    try:
        _ = variant_lut.iloc[0]["scenario"]
    except KeyError:
        print(
            "Key error. Check that you have logged into the endpoint via the Globus app."
        )

    e3sm_holdings_df = make_holdings_table(
        tc=tc,
        node_ep=node_ep,
        node_prefix=e3sm_prefix,
        variant_lut=variant_lut,
        ncpus=ncpus,
        variable_lut=variables,
        model_inst_lu=model_inst_lu,
    )

    e3sm_holdings_df.to_csv(f"{esgf_node}_esgf_holdings_e3sm.csv", index=False)
