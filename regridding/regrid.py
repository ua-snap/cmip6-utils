"""Script for regridding a batch of files listed in a text file

Note - this script first crops the dataset to the panarctic domain of 50N and up. 
"""

import argparse
import re
import time
from pathlib import Path
import numpy as np
import xesmf as xe
import xarray as xr

# ignore serializationWarnings from xarray for datasets with multiple FillValues
import warnings

warnings.filterwarnings("ignore", category=xr.SerializationWarning)


# define the production latitude domain slice
prod_lat_slice = slice(50, 90)


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
    model, scenario, variant, frequency, variable_id, grid_type = fp.parts[-8:-2]
    timeframe = fp.name.split("_")[-1].split(".nc")[0]

    attr_di = {
        "model": model,
        "scenario": scenario,
        "variant": variant,
        "frequency": frequency,
        "variable_id": variable_id,
        "grid_type": grid_type,
        "timeframe": timeframe,
    }

    return attr_di


def generate_regrid_filepath(fp, out_dir):
    """Generates the name for a regridded file using info parsed from source filepath

    Args:
        fp (str): CMIP6 filepath to generate name for
        out_dir (pathlib.Path): path to the root of the output directory for regridded files

    Returns:
        new_fn (str): new filename string
    """
    fp_attrs = parse_cmip6_fp(fp)

    # drop the variant and grid type from existing filename to get the regrid filename
    fn = fp.name.replace(f"_{fp_attrs['variant']}_", "_").replace(
        f"_{fp_attrs['grid_type']}_", "_regrid_"
    )
    # construct the output filepath
    regrid_fp = out_dir.joinpath(
        *(fp_attrs[k] for k in ["model", "scenario", "frequency", "variable_id"])
    ).joinpath(fn)

    return regrid_fp


def open_and_crop_dataset(fp, lat_slice):
    """Open the connection to a dataset and crop it to a panarctic domain of 50N and up.

    Args:
        fp (pathlib.Path): path to file to be opened as xarray.Dataset and cropped to a panarctic extent
        lat_slice (slice): slice object for cropping latitude dimension of dataset to

    Returns:
        src_ds (xarray.Dataset): xarray Dataset (chunked with Dask) cropped to a panarctic domain
    """
    # we are cropping the dataset using the .sel method as we do not need to regrid the entire grid,
    #  only the part that will evetually wind up in the dataset.
    src_ds = xr.open_dataset(fp, chunks={"time": 100}).sel(lat=lat_slice)

    return src_ds


def regrid_dataset(fp, regridder, out_fp, lat_slice):
    """Regrid a dataset using a regridder object initiated using the target grid with a latitude domain of 50N and up.

    Args:
        fp (pathlib.Path): path to file to be regridded
        regridder (xesmf.Regridder): regridder object initialized on source dataset that has the same grid as dataset as read from fp
        out_fp (pathlib.Path): Path to output regridded file
        lat_slice (slice): slice object for cropping latitude dimension of dataset to

    Returns:
        out_fp (pathlib.Path): Path to output regridded file (just to return something)
    """
    # open the source dataset
    # open the "exteneded" latitude domain so the regridding effort includes the minimum production latitude.
    src_ds = open_and_crop_dataset(fp, lat_slice=lat_slice)

    regrid_task = regridder(src_ds, keep_attrs=True)
    regrid_ds = regrid_task.compute()

    # subset to the production latitude slice after regridding.
    regrid_ds.to_netcdf(out_fp)

    return out_fp


if __name__ == "__main__":
    # parse args
    regrid_batch_fp, dst_fp, out_dir, no_clobber = parse_args()

    # get the paths of files to regrid from the batch file
    with open(regrid_batch_fp) as f:
        lines = f.readlines()
    src_fps = [Path(line.replace("\n", "")) for line in lines]

    # open destination dataset for regridding to.
    dst_ds = open_and_crop_dataset(dst_fp, lat_slice=prod_lat_slice)
    # do the same for one of the source datasets to configure the regridder object
    # defining an "extended" latitude slice, so that grids encoompass the entire
    #  production latitude extent before regridding (e.g. a grid will have domain [49.53, 90] instead of [50.75, 90],
    #  so this is probably always going to give just one more "row" of grid cells for interpolation.)
    ext_lat_slice = slice(49, 90)
    src_init_ds = open_and_crop_dataset(src_fps[0], lat_slice=ext_lat_slice)

    # use one of the source files to be regridded and the destination grid file to create a regridder object
    regridder = init_regridder(src_init_ds, dst_ds)

    # now iterate over files in batch file and run the regridding
    print(f"Regridding {len(src_fps)} files", flush=True)
    tic = time.perf_counter()

    results = []
    for fp in src_fps:
        out_fp = generate_regrid_filepath(fp, out_dir)
        # make sure the parent dirs exist
        out_fp.parent.mkdir(exist_ok=True, parents=True)

        if no_clobber:
            if not out_fp.exists():
                results.append(regrid_dataset(fp, regridder, out_fp, ext_lat_slice))
            else:
                continue
        else:
            print("clobber!")
            results.append(regrid_dataset(fp, regridder, out_fp, ext_lat_slice))

    print(
        f"done, {len(results)} files regridded in {np.round((time.perf_counter() - tic) / 60, 1)}m"
    )
