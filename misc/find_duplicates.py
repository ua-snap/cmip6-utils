#!/usr/bin/env python
import xarray as xr
import glob
import os
import warnings

warnings.filterwarnings("ignore")

cmip6_dir = "/beegfs/CMIP6/arctic-cmip6/CMIP6"
byte_buffer = 20


def find_duplicates(root_dir):
    # Get list of model names from subdirectories.
    subdirs = glob.glob(os.path.join(root_dir, "*/*"))
    models = []
    for subdir in subdirs:
        subdir_split = subdir.split("/")
        model = subdir_split[7]
        models.append(model)

    # Iterate through models, get the size of the first (alphabetical) file
    # from each subdirectory, and if multiple files from this model have
    # similar file sizes, do an xarray equals() comparison to check if the data
    # are the same. Comparing files in this way is necessary because two NetCDF
    # files can have identical data even if the md5 sums are different.
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

        for current_size in sizes.keys():
            # Find any other files within byte_buffer bytes of the current
            # file size, since it's conceivable that the data are the same even
            # if the file size differs slightly.
            files_to_compare = []
            for size, files in sizes.items():
                if (
                    size > current_size - byte_buffer
                    and size < current_size + byte_buffer
                ):
                    files_to_compare += files

            for i in range(len(files_to_compare)):
                ds = xr.open_dataset(files_to_compare[i])
                for j in range(i + 1, len(files_to_compare)):
                    ds2 = xr.open_dataset(files_to_compare[j])
                    if ds.equals(ds2):
                        print(
                            f"The following files have the same data:\n"
                            f"{files_to_compare[i]}\n"
                            f"{files_to_compare[j]}\n"
                            "-----------------------------------"
                        )
                    ds2.close()


# Historical
find_duplicates(f"{cmip6_dir}/CMIP")

# Projected
find_duplicates(f"{cmip6_dir}/ScenarioMIP")
