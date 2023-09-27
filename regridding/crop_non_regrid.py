import argparse
from multiprocessing import Pool
import tqdm
import numpy as np
import pandas as pd
import xarray as xr
from config import *
from regrid import (
    generate_regrid_filepath,
    open_and_crop_dataset,
    prod_lat_slice,
)
from generate_batch_files import max_time, read_grids


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-clobber",
        dest="no_clobber",
        action="store_true",
        default=False,
        help="Do not crop a file if the file already exists in out_dir",
    )
    args = parser.parse_args()

    return args.no_clobber


def generate_cmip6_filepath_from_regrid_filename(fn):
    """Get the path to the original CMIP6 filename from a regridded file name."""
    var_id, freq, model, scenario, _, timespan = fn.split(".nc")[0].split("_")
    institution = model_inst_lu[model]
    experiment_id = "ScenarioMIP" if scenario in prod_scenarios else "CMIP"
    # Construct the original CMIP6 filepath from the filename.
    # Need to use glob because of the "grid type" filename attribute that we do not have a lookup for.
    var_dir = cmip6_dir.joinpath(f"{experiment_id}/{institution}/{model}/{scenario}")
    glob_str = (
        f"*/{freq}/{var_id}/*/*/{var_id}_{freq}_{model}_{scenario}_*_{timespan}.nc"
    )
    fp = list(var_dir.glob(glob_str))[0]

    return fp


def get_source_filepaths_from_batch_files(regrid_batch_dir):
    """Get all of source filepaths selected for regridding from the batch files in the regrid_batch_dir"""
    src_fps = []
    for fp in regrid_batch_dir.glob("*.txt"):
        with open(fp) as f:
            src_fps.extend([Path(line.replace("\n", "")) for line in f.readlines()])

    return src_fps


def crop_dataset(args):
    """Crop a dataset to a domain as set in open_and_crop_dataset, write the cropped dataset to netCDF file in out_dir

    Args:
        args (tuple): tuple of (cmip6_fp, out_dir) where cmip6_fp is the CMIP6 filepath to be cropped (pathlib.Path) and out_dir is the output directory (pathlib.Path). Set up as a tuple for Pool.imap processing.

    Returns:
        out_fp (pathlib.Path): path where the cropped dataset was written
    """
    cmip6_fp, out_dir, no_clobber = args

    out_fp = generate_regrid_filepath(cmip6_fp, out_dir)
    # make sure the parent dirs exist
    out_fp.parent.mkdir(exist_ok=True, parents=True)

    if no_clobber:
        if out_fp.exists():
            return out_fp

    ds = open_and_crop_dataset(cmip6_fp, lat_slice=prod_lat_slice)
    var_id = out_fp.parent.name
    assert var_id in list(ds.data_vars)
    assert var_id == ds.attrs["variable_id"]
    ds.to_netcdf(out_fp)

    return out_fp


def run_crop_datasets(fps, out_dir, ncpus, no_clobber=False):
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
            pool.imap_unordered(
                crop_dataset, [(fp, out_dir, no_clobber) for fp in fps]
            ),
            total=len(fps),
        ):
            out_fps.append(out_fp)

    return out_fps


if __name__ == "__main__":
    no_clobber = parse_args()

    # get non-regridded filenames from set difference on standardized filenames
    #  between all CMIP6 files and all filepaths listed in batch files
    cmip6_fps = list(cmip6_dir.glob("**/*.nc"))
    src_fps = get_source_filepaths_from_batch_files(regrid_batch_dir)
    # standardize filenames for set comparison by simple renaming to expected regridded filenanme
    cmip6_std_fns = set(
        [generate_regrid_filepath(fp, regrid_dir).name for fp in cmip6_fps]
    )
    src_std_fns = set([generate_regrid_filepath(fp, regrid_dir).name for fp in src_fps])
    # standardized filenames for files which have not been regridded, need to be converted
    #  back to CMIP6 filepaths for cropping
    non_regrid_fns = cmip6_std_fns - src_std_fns
    crop_cmip6_fps = [
        generate_cmip6_filepath_from_regrid_filename(fn) for fn in non_regrid_fns
    ]

    # borrowed code from generate_batch_files.py to ensure we are only cropping files which
    #  are already on the target grid (including the temporal axis!)
    results_df = pd.DataFrame(read_grids(crop_cmip6_fps))
    # the grid of the file chosen as the target template grid
    cesm2_grid = results_df.query(f"fp == @target_grid_fp").grid.values[0]
    # crop files that are on this grid
    crop_df = results_df.query("grid == @cesm2_grid")
    # only crop files if their start date is less than or equal to 2101-01-01
    crop_cmip6_fps = crop_df.query("start_time < @max_time").fp.values

    out_fps = run_crop_datasets(crop_cmip6_fps, regrid_dir, 24, no_clobber)
