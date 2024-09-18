#!/usr/bin/env python
import xarray as xr
import glob
import os

import warnings
warnings.filterwarnings("ignore")

cmip6_dir = "/beegfs/CMIP6/arctic-cmip6/CMIP6"

def find_duplicates(root_dir):
    # Get list of model names from subdirectories
    subdirs = glob.glob(os.path.join(root_dir, '*/*'))
    models = []
    for subdir in subdirs:
        subdir_split = subdir.split('/')
        model = subdir_split[7]
        models.append(model)

    # Iterate through models, get the size of the first (alphabetical) file from
    # each subdirectory, and if multiple files from this model have the same
    # filesize, do an xarray equals() comparison to check if the files are the same.
    # Comparing files in this way is necessary because two NetCDF files can have
    # identical data even if the md5 sums are different.
    for model in models:
        sizes = {}
        subdirs = glob.glob(os.path.join(root_dir, f"*/{model}/*/*/*/*/*/*"))
        for subdir in subdirs:
            for root, dirs, files in os.walk(subdir):
                if files == []:
                    continue
                files.sort()
                first_file = files[0]
                file_path = os.path.join(root, first_file)
                file_size = os.path.getsize(file_path)
                sizes[file_size] = sizes.get(file_size, []) + [file_path]

        for size, files in sizes.items():
            for i in range(len(files)):
                ds = xr.open_dataset(files[i])
                for j in range(i+1, len(files)):
                    ds2 = xr.open_dataset(files[j])
                    if ds.equals(ds2):
                        print(f"Files {files[i]} and {files[j]} are the same.")
                    ds2.close()

# Historical
find_duplicates(f"{cmip6_dir}/CMIP")

# Projected
find_duplicates(f"{cmip6_dir}/ScenarioMIP")
