"""Functions for downloading CMIP6 data via the CDS API"""

import logging
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import cdsapi
import pandas as pd


def run_retrieve(api_di, fp, log_fp, dataset="projections-cmip6"):
    """Wrapper for the cdsapi.Client.retrieve function
    to run the download request for a given set of parameters.
    
    Args:
        api_di (dict): dict of kwargs for the request arg of the 
            retrieve function
        fp (path-like): path to write downloaded data
        log_fp (path-like): path to text file where download logging
            messages should be written
        dataset (str): name of the CDS dataset to access
    
    Returns:
        out (str): path to download path on CDS if success, error 
            message if the download failed
    """
    c = cdsapi.Client(progress=False)
    # this allows the logging outputs to be written to a file
    output_file_handler = logging.FileHandler(log_fp)
    # set the format for logging messages (same as in cdsapi)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    output_file_handler.setFormatter(formatter)
    c.logger.addHandler(output_file_handler)
    c.logger.propagate = False
    
    try:
        result = c.retrieve(dataset, api_di, str(fp))
        out = result.location
    except Exception as exc:
        out = exc.args[0]
    return out


def download(row, skip_existing=True):
    """Attempt to download the data specified by the options in a row
    of the download tracker table
    
    Args:
        row (pandas.core.series.Series): row of the tracking table containing 
            information for running a retrieval request via the CDS API
        skip_existing (bool): skip the download if it already exists in base_dir
            
    Returns:
        updated row with success
    """
    if skip_existing:
        if row["result"] == "pass":
            return row
        
    api_di = {
        "format": "zip",
        "temporal_resolution": row["t_res"],
        "experiment": row["scenario"],
        "level": "single_levels",
        "variable": row["variable"],
        "model": row["model"],
    }
    # if bbox is not NaN then convert to list and use for "area" arg
    if row["bbox"] != "":
        api_di["area"] = eval(row["bbox"])
    zip_fp = row["zip_path"]
    out = run_retrieve(api_di, zip_fp, row["log_path"])
    if "http" in out:
        row["result"] = "pass"
    else:
        row["result"] = "fail"
        row["fail_reason"] = out
    return row


def make_expected_base_fp(zip_fp, zip_fns, dest_dir):
    """make expected base filepath from file inside zip and 
    "west" or "east" hemisphere specified in zip path
    
    Args:
        zip_fp (pathlib.PosixPath): filepath for zip archive download
        zip_fns (list): list of strings coresponding to filenames inside of 
            the downloaded zip file at zip_path. these are extracted as a 
            test that the zip downloaded successfully
        dest_dir (pathlib.PosixPath): path to the directory where the downloads
            should be extracted to
    
    Returns:
        pathlib.PosixPath object for the path to the expected data file in base_dir
    """
    expected_base_fn = [fn for fn in zip_fns if ".nc" in fn][0]
    hemisphere = zip_fp.name.split("_")[-1].split(".zip")[0]
    expected_base_fn = expected_base_fn.replace(".nc", f"_{hemisphere}.nc")
    expected_base_fp = dest_dir.joinpath(expected_base_fn)
    
    return expected_base_fp


def update_tracker(tracker_fp, raw_dir):
    """Update the tracking table based on presence of files. 
    May be executed repeatedly without change if there are no updates to be made.
    
    Args:
        tracker_fp (path-like): path to download tracking table
        raw_dir (pathlib.PosixPath): path to directory where raw CMIP6 data
            file should be extracted to
            
    Returns:
        Nothing, prints the number of records updated
    """
    # de-duplicating some code just for use in here 
    def mosaic_exists(mosaic_fp):
        """update mosaic_exists column"""
        if Path(mosaic_fp).exists():
            df.at[row_id, "mosaic_exists"] = True
        return None
                
    def update_mosaic():
        """Update mosaic path columns"""
        if row["hemisphere"] != "":
            mosaic_fp = str(expected_base_fp).replace(
                f"_{row['hemisphere']}.nc", ".nc"
            )
            df.at[row_id, "mosaic_path"] = mosaic_fp
        else:
            return None
        mosaic_exists(mosaic_fp)
        return None
    
    df = pd.read_csv(tracker_fp, keep_default_na=False)
    
    # check if data present in scratch_dir but not in base_dir
    for row_id, row in df.iterrows():
        zip_fp = Path(row["zip_path"])
        if zip_fp.exists():
            try:
                # try to open the zipfile to see if success
                with ZipFile(zip_fp, "r") as zip_ref:
                    zip_fns = zip_ref.namelist()
            except BadZipFile:
                df.at[row_id, "result"] = "fail"
                df.at[row_id, "fail_reason"] = "BadZipFile"
                continue

            if row["result"] == "":
                # if we're this far then the download was a success
                df.at[row_id, "result"] = "pass"
            elif row["result"] == "fail":
                # this should never happen and would indicate 
                #  corrupt / partial zip or something weird
                if df.at[row_id, "fail_reason"] == "":
                    df.at[row_id, "fail_reason"] = "Good zip file, but failed?"
                else:
                    df.at[row_id, "result"] = "pass"
                    df.at[row_id, "fail_reason"] = "originally failed but now passing"
                continue
                
            # should only be one netcdf in there
            dest_dir = raw_dir.joinpath(row["data_group"])
            expected_base_fp = make_expected_base_fp(zip_fp, zip_fns, dest_dir)
            base_fp = Path(row["base_path"])
            if str(base_fp) != "":
                #not sure if this check is necessary
                if expected_base_fp == base_fp:
                    if base_fp.exists():
                        if row["fail_reason"] == "BadZipFile":
                            df.at[row_id, "fail_reason"] = ""
                        else:
                            update_mosaic()
                    else:
                        # don't need to save base_fp return in this case
                        # _ = unzip(zip_fp, dest_dir)
                        pass
                elif expected_base_fp.exists():
                    df.at[row_id, "base_path"] = str(expected_base_fp)
                    update_mosaic()
                else:
                    # df.at[row_id, "base_path"] = unzip(zip_fp, dest_dir)
                    pass
            elif expected_base_fp.exists():
                # zip present with data .nc file, and expected data
                #  present in base_dir, set base_path
                df.at[row_id, "base_path"] = str(expected_base_fp)
                update_mosaic()
        elif row["base_path"] != "":
            update_mosaic()
        
    df.to_csv(tracker_fp, index=False)
    return df


def unzip(zip_fp, base_fp):
    """Extract contents of a zip file to a destination directory
    
    Args:
        zip_fp (path-like): path to zip file
        base_fp (pathlib.PosixPath): path to extract the data file to
        
    Returns:
        base_fp (pathlib.PosixPath): New path to data file in $BASE_DIR
    """
    with ZipFile(zip_fp, "r") as zip_ref:
        zip_info = zip_ref.infolist()
        # iterate across files and change names to have specific hemisphere
        for info in zip_info:
            if ".nc" in info.filename:
                info.filename = base_fp.name
                dest_dir = base_fp.parent
                zip_ref.extract(info, dest_dir)

    return base_fp



def batch_unzip(tracker_fp, raw_dir, try_badzip=True):
    """Iterate over the rows of the tracker table and
    attempt to extract any data files that have not been 
    extracted yet
    
    Args:
        tracker_fp (path-like): path to download tracking table
        raw_dir (pathlib.PosixPath): path to directory where raw CMIP6 data
            file should be extracted to
        
    Returns:
        list of new data files extracted to base_dir
    """
    df = update_tracker(tracker_fp, raw_dir)
    
    new_base_fps = []
    for row_id, row in df.iterrows():
        zip_fp = Path(row["zip_path"])
        if zip_fp.exists():
            if (row["fail_reason"] != "BadZipFile") or try_badzip:
                with ZipFile(zip_fp, "r") as zip_ref:
                    zip_fns = zip_ref.namelist()
                dest_dir = raw_dir.joinpath(row["data_group"])
                expected_base_fp = make_expected_base_fp(zip_fp, zip_fns, dest_dir)
                if not expected_base_fp.exists():
                    try:
                        # unzip and update dataframe
                        base_fp = unzip(zip_fp, expected_base_fp)
                    except BadZipFile as exc:
                        df.at[row_id, "fail_reason"] = exc.args[0]
                        continue

                    new_base_fps.append(base_fp)
                    df.at[row_id, "base_path"] = base_fp
                
    # save changes to tracker df
    df.to_csv(tracker_fp, index=False)
    print(f"{len(new_base_fps)} data files extracted from zips")
    return df
