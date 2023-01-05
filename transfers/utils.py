"""Module containing functions for working with globus CLI and generally helping to orchestrate CMIP6 transfers

Some of the code in here is adapted from scripts supplied by ESGF/LLNL (with note "Revised version by Matt Pritchard, CEDA/STFC to work with globus-cli")

The general structure is retained: use a dict of {endpoint: [list, of, filepaths]} to transfer files to our endpoint. 
"""

import re
from pathlib import Path
import subprocess

def get_contents(ep, path):
    """Get the contents of a Globus directory as a list of string values"""
    ep_path = f"{ep}:{path}"
    
    try:
        out = check_output(["globus", "ls", ep_path], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        out_list = re.split("\n|\s\s\s\s\s", exc.output.decode("utf-8"))
        http_status = out_list[out_list.index("HTTP status:") + 1]
        return int(http_status)
        
    contents = [c for c in out.decode("utf-8").split("\n") if c != ""]
    
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



def generate_local_path(src_fp, base_dir):
    """Create a path to save """


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


