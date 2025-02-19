"""Module containing functions for working with globus CLI and generally helping to orchestrate CMIP6 transfers

Some of the code in here is adapted from scripts supplied by ESGF/LLNL (with note "Revised version by Matt Pritchard, CEDA/STFC to work with globus-cli")

The general structure is retained: use a dict of {endpoint: [list, of, filepaths]} to transfer files to our endpoint. 
"""

import time
from pathlib import Path
import globus_sdk
from globus_sdk.scopes import TransferScopes


def login_and_get_transfer_client(auth_client, scopes=TransferScopes.all):
    """This forces a login each time but is a way to create a TransferClient easily.

    From globus_sdk docs: https://globus-sdk-python.readthedocs.io/en/stable/examples/minimal_transfer_script/index.html#example-minimal-transfer
    """
    # note that 'requested_scopes' can be a single scope or a list
    # this did not matter in previous examples but will be leveraged in
    # this one
    auth_client.oauth2_start_flow(requested_scopes=scopes)
    authorize_url = auth_client.oauth2_get_authorize_url()
    print(f"Please go to this URL and login:\n\n{authorize_url}\n")
    time.sleep(1)

    auth_code = input("Please enter the code here: ").strip()
    tokens = auth_client.oauth2_exchange_code_for_tokens(auth_code)
    transfer_tokens = tokens.by_resource_server["transfer.api.globus.org"]

    # return the TransferClient object, as the result of doing a login
    return globus_sdk.TransferClient(
        authorizer=globus_sdk.AccessTokenAuthorizer(transfer_tokens["access_token"])
    )


def check_for_consent_required(tc, auth_client, target):
    """Function to check if consent is required

    Modified from globus_sdk docs
    """
    # now, try an ls on the source and destination to see if ConsentRequired
    # errors are raised
    consent_required_scopes = []
    try:
        tc.operation_ls(target, path="/")
    # catch all errors and discard those other than ConsentRequired
    # e.g. ignore PermissionDenied errors as not relevant
    except globus_sdk.TransferAPIError as err:
        if err.info.consent_required:
            consent_required_scopes.extend(err.info.consent_required.required_scopes)

    # handle ConsentRequired with a new login
    if consent_required_scopes:
        print(
            "One of your endpoints requires consent in order to be used.\n"
            "You must login a second time to grant consents.\n\n"
        )
        tc = login_and_get_transfer_client(auth_client, scopes=consent_required_scopes)

    return tc


def operation_ls(tc, ep, path):
    """ls operation on the specified endpoint:path. Wrapper for the globus_sdk.services.transfer.client.TransferClient.operation_ls function.

    Args:
        tc (globus_sdk.services.transfer.client.TransferClient): transfer client object
        ep (str): endpoint ID
        path (str): path on the specified endpoint to run ls

    Returns:
        contents (list): list of contents (as strings) in the directory or HTTP error code
    """
    try:
        r = tc.operation_ls(ep, path)
    except globus_sdk.TransferAPIError as exc:
        return exc.http_status

    contents = [item["name"] for item in r]
    return contents
