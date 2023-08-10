import argparse
import re
import time
from multiprocessing import Pool
import tqdm
import numpy as np
import xarray as xr
from config import *
from regrid import rename_file, open_and_crop_dataset, parse_cmip6_fp


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


def crop_dataset(args):
    """Crop a dataset to a domain as set in open_and_crop_dataset, write the cropped dataset to netCDF file in out_dir

    Args:
        args (tuple): tuple of (fp, out_dir) where fp is the filepath to be cropped (pathlib.Path) and out_dir is the output directory (pathlib.Path). Set up as a tuple for Pool.imap processing.

    Returns:
        out_fp (pathlib.Path): path where the cropped dataset was written
    """
    fp, out_dir, no_clobber = args

    # get output filepath from input file
    fp_attrs = parse_cmip6_fp(fp)
    varname = fp_attrs["varname"]
    model = fp_attrs["model"]
    scenario = fp_attrs["scenario"]
    frequency = fp_attrs["frequency"]
    # rename the file by simply switching out the existing grid type component with "regrid"
    #  should only be one of three options: gr, gr1, or gn
    rep = {"_gr_": "_regrid_", "_gr1_": "_regrid_", "_gn_": "_regrid_"}
    fn = rename_file(fp.name, rep)
    out_fp = out_dir.joinpath(model, scenario, frequency, varname, fn)
    # make sure the parent dirs exist
    out_fp.parent.mkdir(exist_ok=rue, parents=True)

    if no_clobber:
        if out_fp.exists():
            time.sleep(1)
            return out_fp

    ds = open_and_crop_dataset(fp)
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
    regrid_fps = list(regrid_dir.glob("**/*.nc"))
    cmip6_fps = list(cmip6_dir.glob("**/*.nc"))

    rep = {"_gr_": "_", "_gr1_": "_", "_gn_": "_"}
    cmip6_fns = set([rename_file(fp.name, rep) for fp in cmip6_fps])
    regrid_fns = set([fp.name.replace("_regrid_", "_") for fp in regrid_fps])

    # get non-regridded filenames from set difference on standardized filenames
    non_regrid_fns = cmip6_fns - regrid_fns

    if len(non_regrid_fns) == 0:
        print(
            "All source CMIP6 files accounted for in regridding output dir. No files to crop."
        )
        pass

    else:
        non_regrid_fps = []
        for fn in non_regrid_fns:
            var_id, freq, model, scenario, variant, timespan = fn.split(".nc")[0].split(
                "_"
            )
            fp = list(
                cmip6_dir.glob(
                    f"*/*/{model}/{scenario}/{variant}/{freq}/{var_id}/*/*/{var_id}_{freq}_{model}_{scenario}_{variant}_*_{timespan}.nc"
                )
            )[0]
            non_regrid_fps.append(fp)

        out_fps = run_crop_datasets(non_regrid_fps, regrid_dir, 24, no_clobber)
