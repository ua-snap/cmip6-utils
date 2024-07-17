"""Submit transfer jobs for all files in batch_files/ folder."""

import argparse
import sys
import globus_sdk
from globus_sdk.scopes import TransferScopes
from config import *
import utils


def arguments(argv):
    """Parse some args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=str, help="ESGF node to query", required=True)
    parser.add_argument(
        "--wrf",
        action="store_true",
        help="Whether or not to generate the batch files for the WRF variables, at sub-daily resolutions.",
    )
    args = parser.parse_args()
    esgf_node, do_wrf = args.node, args.wrf

    return esgf_node, do_wrf


def create_transfer_from_batch_file(task_data, batch_fp):
    """Add the individual file transfer items represented by each row in a batch file to a transfer task"""
    with open(batch_fp) as f:
        for line in f.readlines():
            src_fp, dst_fp = line.replace("\n", "").split(" ")
            task_data.add_item(src_fp, dst_fp)

    return task_data


if __name__ == "__main__":
    ESGF_NODE, do_wrf = arguments(sys.argv)
    source_ep_id = globus_esgf_endpoints[ESGF_NODE]["ep"]

    # create an authorization client for Globus
    auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    tc = utils.login_and_get_transfer_client(auth_client)

    print("Log in successful. Submitting the transfer jobs now.")

    # check if we need to grant conesnt for ACDN
    utils.check_for_consent_required(tc, auth_client, acdn_ep)

    if do_wrf:
        batch_fps = []
        [
            batch_fps.extend(batch_dir.glob(f"*{ESGF_NODE}_{t_id}_*.txt"))
            for t_id in subdaily_table_ids
        ]
    else:
        # batch files to iterate over, and create a submission for each
        batch_fps = batch_dir.glob(f"*{ESGF_NODE}*.txt")

    for fp in batch_fps:
        # think this needs to be initialized for each task
        task_data = globus_sdk.TransferData(
            source_endpoint=source_ep_id, destination_endpoint=acdn_ep, sync_level=2
        )
        task_data = create_transfer_from_batch_file(task_data, fp)

        try:
            task_doc = tc.submit_transfer(task_data)
        except globus_sdk.services.transfer.errors.TransferAPIError as exc:
            print(f"Transfer for {fp.name} failed. Message: \n")
            print(exc.message, end="\n")
            print("Continuing to next file..")

        task_id = task_doc["task_id"]
        print(f"submitted transfer for {fp.name}, task_id={task_id}")
