from pathlib import Path
import xarray as xr
import glob
from multiprocessing import Pool
import tqdm
import argparse

#set the global keep_attrs to True, to avoid losing longitude attributes during computation
xr.set_options(keep_attrs=True)

def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regrid_dir",
        dest="regrid_dir",
        type=str,
        help="Directory of regridded files to be processed",
        required=True,
    )
    args = parser.parse_args()

    return args.regrid_dir

def list_nonfixed_nc_files(regrid_dir):
    fps = list(regrid_dir.glob('**/*.nc'))
    removed_fps = []
    for fp in fps:
        if "fx" in fps[0].parts[-3]:
            fps.remove(fp)
            removed_fps.append(fp)
    return fps, removed_fps


def check_longitude(fp):
    try:
        ds = xr.open_dataset(fp)
        # check that longitude is scaled 0-360, return True if so

        # first, make sure the max is between 180 and 360, and the min is greater than 0
        # also change the sign of the min to positive and measure the range;
        # it should be over 350 for a 0-360 scale, and would be closer to 0 if a -180 to 180 scale
        max = ds.lon.values.max()
        min = ds.lon.values.min()

        if 180 < max <= 360 and \
            min >= 0 and \
            len(range(int(abs(min)), int(max))):
            return True, ds
        else:
            return False, None

    except:
        print(f"Could not open {fp}!")
        return False, None


def convert_to_standard_longitude(ds):
    #copy original encoding (not persisted thru computations)
    lon_enc = ds['lon'].encoding
    #subtract from 0-360 lon coords to get -180 to 180 lon coords, and reapply encoding
    ds['lon'] = ds['lon'] - 180
    ds['lon'].encoding = lon_enc
    #sort and verify
    ds = ds.sortby(ds.lon, ascending=True)

    return ds


def check_longitude_and_convert(fp):
    status, ds = check_longitude(fp)
    if status==True:
        ds_out = convert_to_standard_longitude(ds).copy()
        ds.close()
        ds_out.to_netcdf(fp)
        #return ds_out ### TODO: option to pass the dataset to new "add CRS" function before writing??
        return None
    else:
        print(f"File not converted: {fp}")
        return None


if __name__ == '__main__':

    regrid_dir = parse_args()
    fps, removed_fps = list_nonfixed_nc_files(Path(regrid_dir))
    if len(removed_fps) > 0:
        print(f"Ignoring {len(removed_fps)} files with fixed frequencies...")
    print(f"Checking longitude of {len(fps)} regridded files in {regrid_dir}...")
    print("Attempting to convert to standard longitude...")

    # TODO: figure out how to use multiprocessing here!
    # with Pool(24) as pool:
    #     pool.imap_unordered(check_longitude_and_convert, fps)

    for fp in tqdm.tqdm(fps):
        check_longitude_and_convert(fp)
