"""Script for regridding a batch of files listed in a text file

Note - this script first crops the dataset to the panarctic domain of 50N and up. 
"""

import argparse
import re
import time
from pathlib import Path
import dask
import numpy as np
import xesmf as xe
import xarray as xr
from tqdm import tqdm

# ignore serializationWarnings from xarray for datasets with multiple FillValues
import warnings

warnings.filterwarnings("ignore", category=xr.SerializationWarning)


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-b",
        dest="regrid_batch_fp",
        type=str,
        help="Batch file containing filepaths to be regridded",
        required=True,
    )
    parser.add_argument(
        "-d", dest="dst_fp", type=str, help="Destination grid filepath", required=True
    )
    parser.add_argument(
        "-o",
        dest="out_dir",
        type=str,
        help="Path to directory where regridded data should be written",
        required=True,
    )
    parser.add_argument(
        "--no-clobber",
        dest="no_clobber",
        action="store_true",
        default=False,
        help="Do not regrid a file if the regridded file already exists in out_dir",
    )
    args = parser.parse_args()

    return args.regrid_batch_fp, args.dst_fp, Path(args.out_dir), args.no_clobber


def init_regridder(src_ds, dst_ds):
    """Initialize the regridder object for a single dataset. All source files listed in the batch files should have the same grid, and so we should only need to initiate this for a single file.
    
    Args:
        src_ds (xarray.DataSet): Source dataset for initializing a regridding object. This should have the same grid as all files in the batch file being worked on.
        dst_ds (xarray.DataSet): Destination dataset for initializing a regridding object. This should be a cropped version of the pipeline's target grid dataset.

    Returns:
        regridder (xesmf.Regridder): a regridder object
    """
    regridder = xe.Regridder(src_ds, dst_ds, "bilinear", unmapped_to_nan=True)

    return regridder


def parse_cmip6_fp(fp):
    """Pull some data attributes from an ESGF CMIP6 filepath.
    
    Args:
        fp (pathlib.Path): CMIP6 path mirrored from ESGF

    Returns:
        attr_di (dict): dict of modeling-relevant attributes of the filepath and filename
    """
    varname = fp.parent.parent.parent.name
    frequency = fp.parent.parent.parent.parent.name
    scenario = fp.parent.parent.parent.parent.parent.parent.name
    model = fp.parent.parent.parent.parent.parent.parent.parent.name
    timeframe = fp.name.split("_")[-1].split(".nc")[0]

    attr_di = {
        "model": model,
        "scenario": scenario,
        "frequency": frequency,
        "varname": varname,
        "timeframe": timeframe,
    }

    return attr_di


def rename_file(fn, rep):
    """Renames a file by swapping out patterns based on mappings in rep

    Args:
        fn (str): filename to rename
        rep (dict): mapping from potential patterns to replacement text

    Returns:
        new_fn (str): new filename string
    """
    pattern = re.compile("|".join([k for k in rep]))
    new_fn = pattern.sub(lambda m: rep[re.escape(m.group(0))], fn)

    return new_fn


def open_and_crop_dataset(fp):
    """Open the connection to a dataset and crop it to a panarctic domain of 50N and up.
    
    Args:
        fp (pathlib.Path): path to file to be opened as xarray.Dataset and cropped to a panarctic extent

    Returns:
        src_ds (xarray.Dataset): xarray Dataset (chunked with Dask) cropped to a panarctic domain
    """
    # chunk_spec = get_time_chunking(fp)
    src_ds = xr.open_dataset(fp, chunks={"time": 100}).sel(lat=slice(50, 90))

    return src_ds


# @dask.delayed
def regrid_dataset(fp, regridder, out_fp):
    """Regrid a dataset using a regridder object initiated using the target grid with a latitude domain of 50N and up.
    
    Args:
        fp (pathlib.Path): path to file to be regridded
        regridder (xesmf.Regridder): regridder object initialized on source dataset that has the same grid as dataset as read from fp
        out_fp (pathlib.Path): Path to output regridded file

    Returns:
        out_fp (pathlib.Path): Path to output regridded file (just to return something)
    """
    # open the source dataset
    src_ds = open_and_crop_dataset(fp)

    regrid_task = regridder(src_ds, keep_attrs=True)
    regrid_ds = regrid_task.compute()

    regrid_ds.to_netcdf(out_fp)

    return out_fp


if __name__ == "__main__":
    # parse args
    regrid_batch_fp, dst_fp, out_dir, no_clobber = parse_args()

    # get the paths of files to regrid from the batch file
    with open(regrid_batch_fp) as f:
        lines = f.readlines()
    src_fps = [Path(line.replace("\n", "")) for line in lines]

    # open destination dataset for regridding to
    # dst_ds = xr.open_dataset(dst_fp)
    dst_ds = open_and_crop_dataset(dst_fp)
    # do the same for one of the source datasets to configure the regridder object
    src_init_ds = open_and_crop_dataset(src_fps[0])

    # use one of the source files to be regridded and the destination grid file to create a regridder object
    regridder = init_regridder(src_init_ds, dst_ds)

    # now iterate over files in batch file and run the regridding
    print(f"Regridding {len(src_fps)} files", flush=True)
    tic = time.perf_counter()

    results = []
    for fp in src_fps:
        fp_attrs = parse_cmip6_fp(fp)
        # create custom filepath for regridded data
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
        out_fp.parent.mkdir(exist_ok=True, parents=True)

        if no_clobber:
            if not out_fp.exists():
                results.append(regrid_dataset(fp, regridder, out_fp))
            else:
                continue
        else:
            results.append(regrid_dataset(fp, regridder, out_fp))

    # dask.compute(results)

    print(
        f"done, {len(results)} files regridded in {np.round((time.perf_counter() - tic) / 60, 1)}m"
    )
