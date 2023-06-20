"""Script for regridding a batch of files listed in a batch file"""

import argparse
import re
import time
from pathlib import Path
import dask
import xesmf as xe
import xarray as xr

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
        required=True
    )
    parser.add_argument(
        "-d",
        dest="dst_fp",
        type=str,
        help="Destination grid filepath",
        required=True
    )
    parser.add_argument(
        "-o",
        dest="out_dir",
        type=str,
        help="Path to directory where regridded data should be written",
        required=True
    )
    args = parser.parse_args()
    
    return args.regrid_batch_fp, args.dst_fp, Path(args.out_dir)


def init_regridder(src_fp, dst_ds):
    src_ds = xr.open_dataset(src_fp)
    regridder = xe.Regridder(src_ds, dst_ds, 'bilinear', unmapped_to_nan=True)

    return regridder


def parse_cmip6_fp(fp):
    """pull some data attributes from a CMIP6 filepath"""
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
        "timeframe": timeframe
    }
    
    return attr_di
    

@dask.delayed
def regrid_file(fp, regridder, out_fp):
    src_ds = xr.open_dataset(fp)
    regrid_ds = regridder(src_ds)
    regrid_ds.to_netcdf(out_fp)
    
    return out_fp


if __name__ == "__main__":
    # parse args
    regrid_batch_fp, dst_fp, out_dir = parse_args()
    
    # get the paths of files to regrid from the batch file
    with open(regrid_batch_fp) as f:
        lines = f.readlines()
    src_fps = [Path(line.replace("\n", "")) for line in lines]
    
    # open destination dataset for regridding to
    dst_ds = xr.open_dataset(dst_fp)
    
    # use one of the source files to be regridded and the destination grid file to create a regridder object
    regridder = init_regridder(src_fps[0], dst_ds)
    
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
        rep = {"gr": "regrid", "gr1": "regrid", "gn": "regrid"}
        pattern = re.compile("gr|gr1|gn")
        fn = pattern.sub(lambda m: rep[re.escape(m.group(0))], fp.name)
        out_fp = out_dir.joinpath(model, scenario, frequency, varname, fn)
        # make sure the parent dirs exist
        out_fp.parent.mkdir(exist_ok=True, parents=True)
        results.append(regrid_file(fp, regridder, out_fp))
     
    # processing results with dask
    print("Running the regridding with Dask", end="...")
    tic = time.perf_counter()
    dask.compute(results)
    print(f"done, {np.round((time.pref_counter() - tic) / 60, 1)}m")
