"""Module containing functions for working with globus CLI and generally helping to orchestrate CMIP6 transfers

Some of the code in here is adapted from scripts supplied by ESGF/LLNL (with note "Revised version by Matt Pritchard, CEDA/STFC to work with globus-cli")

The general structure is retained: use a dict of {endpoint: [list, of, filepaths]} to transfer files to our endpoint. 
"""

import re
from pathlib import Path
import subprocess
import globus_sdk
from globus_sdk.scopes import TransferScopes


def check_transfer_token(token):
    """Checks a transfer token for validity"""

    
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
        
    


def make_globus_client():
    """Make a globus client object. """
    client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    client.oauth2_start_flow()
    
    
def authorize_client(client):
    """Authorize the Globus client. Returns the tokens needed for using the globus_sdk package for transfers etc."""
    authorize_url = client.oauth2_get_authorize_url()
    print(f"Please go to this URL and login:\n\n{authorize_url}\n")

    auth_code = input("Please enter the code you get after login here: ").strip()
    token_response = client.oauth2_exchange_code_for_tokens(auth_code)

    globus_auth_data = token_response.by_resource_server["auth.globus.org"]
    globus_transfer_data = token_response.by_resource_server["transfer.api.globus.org"]

    # most specifically, you want these tokens as strings
    AUTH_TOKEN = globus_auth_data["access_token"]
    TRANSFER_TOKEN = globus_transfer_data["access_token"]
    
    return TRANSFER_TOKEN, AUTH_TOKEN


def make_transfer_client(token):
    """Make a TransferClient using a token. Will """
    authorizer = globus_sdk.AccessTokenAuthorizer(token)
    tc = globus_sdk.TransferClient(authorizer=authorizer)
    
    try:
        r = tc.get_endpoint(config.acdn_ep)
    except globus_sdk.TransferAPIError as exc:
        if exc.http_status == 401:
            # need to re-authorize
            authorize_client()
            
    return token


def write_token_info():
    """"""


def activate_endpoint(tc, ep):
    """Activate an endpoint, and handle case where manual activation is required.
    
    Args:
        tc (globus_sdk.services.transfer.client.TransferClient): transfer client object
        ep (str): endpoint ID
    
    This borrowed from globus_sdk docs: https://globus-sdk-python.readthedocs.io/_/downloads/en/3.0.0a1/pdf/
    """
    # cannot autoactivate the ESGF endpoints, but can use the function to help get there
    r = tc.endpoint_autoactivate(ep, if_expires_in=3600)
    while (r["code"] == "AutoActivationFailed"):
        print(
            "Endpoint requires manual activation, please open "
            "the following URL in a browser to activate the "
            "endpoint:"
            f"https://app.globus.org/file-manager?origin_id={ep}"
        )
        input("Press ENTER after activating the endpoint:")
        r = tc.endpoint_autoactivate(ep, if_expires_in=3600)
        
    return


def get_contents(ep, path):
    """Get the contents of a Globus directory as a list of string values"""
    ep_path = f"{ep}:{path}"
    
    command = ["globus", "ls", ep_path]
    try:
        out = subprocess.check_output(command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        out_list = re.split("\n|\s\s\s\s\s", exc.output.decode("utf-8"))
        
        try:
            http_status = out_list[out_list.index("HTTP status:") + 1]
        except ValueError:
            print(exc.output.decode("utf-8"))
            print(command)
            exit()
        return int(http_status)
        
    contents = [c for c in out.decode("utf-8").split("\n") if c != ""]
    
    return contents


def operation_ls(tc, ep, path):
    """ls operation on the specified endpoint:path. Wrapper for the globus_sdk.services.transfer.client.TransferClient.operation_ls function.
    
    Args:
        tc (globus_sdk.services.transfer.client.TransferClient): transfer client object
        ep (str): endpoint ID
        path (str): path on the specified endpoint to run ls
    
    Returns:
        contents (list): list of contents (as strings) in the directory or HTTP error code
    """
    n = 5
    for retry in range(n):
        try:
            r = tc.operation_ls(ep, path)
        except globus_sdk.TransferAPIError as exc:
            if retry < n-1: continue
            else: return exc.http_status

    contents = [item["name"] for item in r]
    return contents
    

def list_endpoints(gendpoint_dict):
    end_names = list(gendpoint_dict.keys())
    print("Endpoints involved:")
    for the_end_name in end_names:
        print(this_end_name)
        

def arguments(argv):

    parser = argparse.ArgumentParser(description = \
        '''To use this script, you must have the Globus Command Line Interface
        tools installed locally (see https://docs.globus.org/cli/)
        The host where you install these tools does
        NOT need to be one of the endpoints in the transfer.
        This script makes use of the Globus CLI 'transfer' command.
        You need to ensure the endpoints involved are activated, see "Endpoints
        to be activated" in output (use "globus endpoint activate")
        By default, the transfer command will:
        - verify the checksum of the transfer
        - encrypt the transfer
        - and delete any fies at the user endpoint with the same name.'''
            )
    parser.add_argument('-u', '--username', type=str, help='your Globus username', required=True)
    parser.add_argument('-p', '--path', type=str, help='the path on your endpoint where you want files to be downloaded to', default='/~/')
    parser.add_argument('-l', '--list-endpoints', help='List the endpoints to be activated and exit (no transfer attempted)', action='store_true')
    parser._optionals.title = 'required and optional arguments'
    args = parser.parse_args()

    username = args.username
    upath = args.path
    listonly = args.list_endpoints

    if '#' in upath:
        print("The '#' character is invalid in your path, please re-enter")
        sys.exit()
    if upath[0] != '/' and upath != '/~/':
        upath = '/' + upath

    return (username, upath, listonly)


def make_batch_file(ep_di, u_ep, username, base_dir, batch_fp):
    
    base_dir = Path(base_dir)
    eps = list(ep_di.keys())
    for ep in eps:
        file_list = ep_di[ep]
        file = open(batch_fp, "w")

        for file in file_list:
            file = Path(file)
            fn = file.name
            
            
            remote = file
            local = upath.joinpath(fn)
            
            
            file.write(str(remote) + ' ' + str(local) + '\n')

        file.close()

        # os.s ystem("globus transfer "+thisEndName+" "+uendpoint+" --label \"CLI Batch\" --batch "+transferFile)

        os.remove(transferFile)

    return


