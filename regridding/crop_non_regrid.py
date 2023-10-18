"""Crop the files which were not regridded, and fix the time axis to be consistent with the regridded schema. 
Ensure that the generate_batch_files.py script has been run and that the files_to_crop.csv table exists.

Usage:
    python crop_non_regrid.py
"""

from multiprocessing import Pool
import tqdm
import numpy as np
import pandas as pd
from config import *
from regrid import (
    generate_regrid_filepath,
    open_and_crop_dataset,
    fix_time_and_write,
    prod_lat_slice,
)


def crop_dataset(args):
    """Crop a dataset to a domain as set in open_and_crop_dataset, write the cropped dataset to netCDF file in out_dir

    Args:
        args (tuple): tuple of (cmip6_fp, out_dir) where cmip6_fp is the CMIP6 filepath to be cropped (pathlib.Path) and out_dir is the output directory (pathlib.Path). Set up as a tuple for Pool.imap processing.

    Returns:
        out_fp (pathlib.Path): path where the cropped dataset was written
    """
    cmip6_fp, out_dir = args

    out_fp = generate_regrid_filepath(cmip6_fp, out_dir)
    # make sure the parent dirs exist
    out_fp.parent.mkdir(exist_ok=True, parents=True)

    ds = open_and_crop_dataset(cmip6_fp, lat_slice=prod_lat_slice)
    var_id = out_fp.parent.name
    assert var_id in list(ds.data_vars)
    assert var_id == ds.attrs["variable_id"]

    out_fps = fix_time_and_write(ds, ds, out_fp)

    return out_fps


def run_crop_datasets(fps, out_dir, ncpus):
    """Run the cropping of all files in fps using multiprocessing and with a progress bar.

    Args:
        fps (list): list of files to crop
        out_dir (pathlib.Path): path to where the cropped files should be written
        ncpus (int): number of CPUs to use with multiprocessing.Pool
        no_clobber (bool): Do not crop a file if the file already exists in out_dir

    Returns:
        out_fps (list): list of filepaths of cropped files
    """
    out_fps = []
    with Pool(ncpus) as pool:
        for out_fp in tqdm.tqdm(
            pool.imap_unordered(crop_dataset, [(fp, out_dir) for fp in fps]),
            total=len(fps),
        ):
            out_fps.append(out_fp)

    return out_fps


if __name__ == "__main__":
    # Load the table of files to crop created from the generate_batch_files.py script
    crop_df = pd.read_csv("files_to_crop.csv")
    crop_cmip6_fps = crop_df.fp.values

    out_fps = run_crop_datasets(crop_cmip6_fps, regrid_dir, 24)
