"""Script for regridding a batch of files listed in a text file"""

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


def get_time_chunking(fp):
    """Determine the time axis chunking for opening a dataset to be regridded.
    This was done to alleviate memory issues with regridding some of the larger datasets ( ~10 GB on disk).
    
    Args:
        fp (pathlike): path to file being worked on
        
    Returns:
        chunk_spec (dict): Chunking spec in form {"time": <chunk size>
    """
    # from some quick experimentation, it looks like arrays that were
    #  under < 5 GB were regridded without memory errors.
    # we will implement a simple switch option then, with < 5GB getting no chunking, 
    #  and > 5gb getting time: 100 chunking.
    ds = xr.open_dataset(fp)
    da = ds[list(ds.data_vars)[0]]
    assert str(da.dtype) == "float32"
    # float32 is 4 bytes
    memory = (da.size * 4) / (1028 ** 3)
    if memory < 5:
        chunk_spec = None
    else:
        chunk_spec = {"time": 100}
    
    return chunk_spec
    

@dask.delayed
def regrid_file(fp, regridder, out_fp):
    chunk_spec = get_time_chunking(fp)
    src_ds = xr.open_dataset(fp, chunks=chunk_spec)
    regrid_ds = regridder(src_ds)
    regrid_ds.to_netcdf(out_fp)
    del regrid_ds
    
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
        rep = {"_gr_": "_regrid_", "_gr1_": "_regrid_", "_gn_": "_regrid_"}
        fn = rename_file(fp.name, rep)
        out_fp = out_dir.joinpath(model, scenario, frequency, varname, fn)
        # make sure the parent dirs exist
        out_fp.parent.mkdir(exist_ok=True, parents=True)
        results.append(regrid_file(fp, regridder, out_fp))
     
    # processing results with dask
    print("Running the regridding with Dask", end="...", flush=True)
    tic = time.perf_counter()
    dask.compute(results)
    print(f"done, {np.round((time.pref_counter() - tic) / 60, 1)}m")
