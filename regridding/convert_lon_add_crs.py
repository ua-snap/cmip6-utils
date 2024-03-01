"""Script to find all .nc files in the regrid directory and convert longitudes from 0 to 360 scale to standard -180 to 180 scale.
CF-compliant CRS information is added to the dataset, which will allow most software to read the CRS as WGS84.
Files are modified "in place", ie the original files are overwritten by a new dataset.

Since some of the "fx" variables are on an Antarctic grid, we do not want to apply the standard longitude conversion to those.
For now, this script will check file paths for fixed "fx" variables and ignore them. 
"""

from pathlib import Path
import xarray as xr
import glob
from multiprocessing import Pool
import tqdm
import argparse
from pyproj import Proj, CRS
import numpy as np


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
    #load the dataset entirely into memory, so we can free up the original file for overwriting
    try:
        with xr.open_dataset(fp) as file_ds:
            ds = file_ds.load()
    except:
        return False, None, msg_a

    # affirm that there is a "lon" coordinate by trying to extract min/max
    try:
        max = ds.lon.values.max()
        min = ds.lon.values.min()
    except:
        return False, None, msg_b

    # check that longitude is scaled 0-360
    # first, make sure the max is between 180 and 360, and the min is greater than 0
    # also change the sign of the min to positive and measure the range;
    # it should be over 350 for a 0-360 scale, and would be closer to 0 if a -180 to 180 scale

    if 180 < max <= 360 and \
        min >= 0 and \
        len(range(int(abs(min)), int(max))):
        return True, ds, None
    else:
        return False, ds, msg_c


def convert_to_standard_longitude(ds):
    
    try:
        #copy original encoding (not persisted thru computations)
        lon_enc = ds['lon'].encoding
        #subtract from 0-360 lon coords to get -180 to 180 lon coords, and reapply encoding
        ds['lon'] = ds['lon'] - 180
        ds['lon'].encoding = lon_enc
        #sort and verify
        ds = ds.sortby(ds.lon, ascending=True)
        return ds, None
    except:
        return ds, msg_d


def apply_wgs84(ds):    
    #get CF-compliant crs attribute dict
    cf_crs = CRS.from_epsg(4326).to_cf()

    try:

        #create a spatial_ref coordinate, which is an empty array but has the CF-compliant crs attribute dict
        ds = ds.assign_coords({
            "spatial_ref": ([],np.array(0), cf_crs)
            })
        
        #add a second attribute "spatial_ref" identical to "crs_wkt" (matches test rioxarray output)
        ds["spatial_ref"].attrs["spatial_ref"] = cf_crs['crs_wkt']

        #manually link spatial_ref attributes to the data variable via "grid_mapping" encoding
        #assumes dataset will only have one data variable!
        var = list(ds.data_vars)[0]
        ds[var].encoding["grid_mapping"] = "spatial_ref"
        return ds, None

    except:
        return ds, msg_h




def convert_longitude_and_apply_wgs84(fp):
    #collect messages in this list
    msgs = []

    status, ds, msg = check_longitude(fp)
    if msg is not None:
        msgs.append(msg)
    if status==True:

        ds, msg = convert_to_standard_longitude(ds)
        if msg is not None:
            msgs.append(msg)

        ds, msg = apply_wgs84(ds)
        if msg is not None:
            msgs.append(msg)

        try:
            ds.to_netcdf(fp, mode="w", format="NETCDF4")
            msgs.append(msg_e)
        except:
            msgs.append(msg_f)
            

    elif status==False and ds is not None:
        ds, msg = apply_wgs84(ds)
        if msg is not None:
            msgs.append(msg)
        try:
            ds.to_netcdf(fp, mode="w", format="NETCDF4")
            msgs.append(msg_e)
        except:
            msgs.append(msg_f)
    else:
        msgs.append(msg_g)

    return str(fp), msgs


if __name__ == '__main__':

    #establish basic status messages as global variables
    msg_a = "ERROR: Could not open file. Aborted."
    msg_b = "ERROR: Could not find 'lon' coordinate. Aborted."
    msg_c = "WARNING: Standard 'lon' coordinates already exists; attempting to add CRS."
    msg_d = "ERROR: Longitude could not be converted."
    msg_e = "SUCCESS: Longitude converted and/or CRS added; file overwritten."
    msg_f = "ERROR: Dataset modified, but file could not be overwritten."
    msg_g = "ERROR: File not modified."
    msg_h = "ERROR: Could not assign CRS to file."

    regrid_dir = parse_args()
    fps, removed_fps = list_nonfixed_nc_files(Path(regrid_dir))
    if len(removed_fps) > 0:
        print(f"Ignoring {len(removed_fps)} files with fixed frequencies...")
    print(f"Processing {len(fps)} regridded files in {regrid_dir}...")
    print("Attempting to convert to standard longitude, apply CRS info, and overwrite all files...")

    # TODO: figure out how to use multiprocessing here!
    # below does not work, provides no error messages??
    # with Pool(24) as pool:
    #     pool.imap_unordered(convert_longitude_and_apply_wgs84, fps)

    results_dict={}

    for fp in tqdm.tqdm(fps):
        fp_str, msgs = convert_longitude_and_apply_wgs84(fp)
        results_dict[fp_str] = msgs

    errs = []
    wins = []
    for result in results_dict.keys():
        if any(x in [msg_a, msg_b, msg_d, msg_f, msg_g, msg_h] for x in results_dict[result]):
            errs.append(result)
        elif msg_e in results_dict[result]:
            wins.append(result)

    print(f"Number of files successfully modified and overwritten: {len(wins)}")
    print(f"Number of files with errors: {len(wins)}")

