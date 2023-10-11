"""Use this script to run a quick ls on a given ESGF node. No error handling.

Example usage:
    python quick_ls.py --node llnl --path /css03_data/CMIP6/ScenarioMIP/NCAR/CESM2
"""

import argparse
import os
import sys
from pathlib import Path
import utils
import luts
import globus_sdk
from globus_sdk.scopes import TransferScopes
from config import CLIENT_ID


def arguments(argv):
    """Parse some args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=str, help="ESGF node to query", required=True)
    parser.add_argument(
        "--path",
        type=str,
        help="Path on ESGF node globus collection to run ls operation for",
        required=True,
    )

    args = parser.parse_args()
    esgf_node, ls_path = args.node, args.path

    return esgf_node, ls_path


if __name__ == "__main__":
    esgf_node, ls_path = arguments(sys.argv)

    # create an authorization client for Globus
    auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    tc = utils.login_and_get_transfer_client(auth_client)
    node_ep = luts.globus_esgf_endpoints[esgf_node]["ep"]

    # I think this activation has to be done?
    tc.endpoint_autoactivate(node_ep)

    out = utils.operation_ls(tc, node_ep, ls_path)
    [print(x) for x in out]
