"""Generate text files ("batch" files) containing all of the files we want to regrid broken up by model and scenario. It utilizes code from the explore_grids.ipynb notebook to select the files which need to be regridded."""


from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from math import radians
from multiprocessing import Pool
import tqdm

from config import *

# ignore serializationWarnings from xarray for datasets with multiple FillValues
import warnings

warnings.filterwarnings("ignore", category=xr.SerializationWarning)


GRID_VARS = ["lat", "lon"]
max_time = np.datetime64("2101-01-01T12:00:00.0000")


def fp_to_attrs(fp):
    """pull the data attributes from a filepath"""
    var_id = fp.parent.parent.parent.name
    frequency = fp.parent.parent.parent.parent.name
    scenario = fp.parent.parent.parent.parent.parent.parent.name
    model = fp.parent.parent.parent.parent.parent.parent.parent.name
    timeframe = fp.name.split("_")[-1].split(".nc")[0]

    attr_di = {
        "model": model,
        "scenario": scenario,
        "frequency": frequency,
        "var_id": var_id,
        "timeframe": timeframe,
    }

    return attr_di


def get_grid(fp):
    """Read the info from a grid for a single file"""
    grid_di = {}
    with xr.open_dataset(fp) as ds:
        for var_id in GRID_VARS:
            if var_id in ds.dims:
                grid_di[f"{var_id}_min"] = ds[var_id].values.min()
                grid_di[f"{var_id}_max"] = ds[var_id].values.max()
                grid_di[f"{var_id}_size"] = ds[var_id].values.shape[0]
                grid_di[f"{var_id}_step"] = np.diff(ds[var_id].values)[0]
            else:
                grid_di[f"{var_id}_min"] = None
                grid_di[f"{var_id}_max"] = None
                grid_di[f"{var_id}_size"] = None
                grid_di[f"{var_id}_step"] = None

    # create a new column that is a concatenation of all of these values
    grid_di["grid"] = "_".join([str(grid_di[key]) for key in grid_di.keys()])
    # pull out file attributes (model scenario etc)
    grid_di.update(fp_to_attrs(fp))
    # also keep the filename for reference
    grid_di["fp"] = fp
    # want to save the earliest time, because we will just ignore projections greater than 2100, for now at least.
    ts_min = ds.time.values.min()
    if not isinstance(ts_min, np.datetime64):
        ts_min = np.datetime64(ts_min.strftime("%Y-%m-%d"))
    grid_di["start_time"] = ts_min
    # save file size for better chunking into batches
    grid_di["filesize"] = fp.stat().st_size / (1e3**3)

    return grid_di


def read_grids(fps):
    """Read the grid info from all files in fps, using multiprocessing and with a progress bar"""
    grids = []
    with Pool(8) as pool:
        for grid_di in tqdm.tqdm(pool.imap_unordered(get_grid, fps), total=len(fps)):
            grids.append(grid_di)

    return grids


def write_batch_files(group_df, model, scenario):
    """Write the batch file for a particular model and scenario group. Breaks up into multiple jobs if file count exceeds 500"""

    def generate_grid_names(df):
        """Helper function to give the unique grids within group_df a useful name for file naming"""
        grids = df.grid.unique()
        grid_name_lu = {
            grid: f"gr{i}" for grid, i in zip(grids, np.arange(grids.shape[0]))
        }
        df["grid_name"] = [grid_name_lu[grid] for grid in df["grid"]]

        return df

    def chunk_fp_list(df, max_size, max_count):
        """Helper function to chunk lists of files for appropriately-sized batches"""
        # split filepaths into chunks such that sum total of sizes is less than max_size
        #  and no more than max_count paths are included
        fp_chunks = []
        # initialize counter for tallying sizes and chunk list
        k = 0
        chunk = []
        for i, row in df.iterrows():
            if ((k + row["filesize"]) > max_size) or (len(chunk) >= max_count):
                fp_chunks.append(chunk)
                k = 0
                # re-initialize with current filepath
                chunk = [row["fp"]]
            else:
                chunk.append(row["fp"])

        if len(chunk) > 0:
            fp_chunks.append(chunk)

        return fp_chunks

    # iterate over the types of grid within a single model scenario so that only files
    #  with same grid are worked on in the same batch file
    # first, classify the different grid types with a name to be included in the batch file name
    group_df = generate_grid_names(group_df)

    for grid_name, df in group_df.groupby("grid_name"):
        fp_chunks = chunk_fp_list(df, max_size=50, max_count=200)

        for i, chunk in enumerate(fp_chunks):
            batch_file = regrid_batch_dir.joinpath(
                batch_tmp_fn.format(
                    model=model, scenario=scenario, grid_name=grid_name, count=i
                )
            )
            with open(batch_file, "w") as f:
                for fp in chunk:
                    f.write(f"{fp}\n")

    return


if __name__ == "__main__":
    # read the grid info from all files
    results = []
    for inst_model in inst_models:
        inst, model = inst_model.split("_")
        fps = []
        for exp_id in ["ScenarioMIP", "CMIP"]:
            fps.extend(list(cmip6_dir.joinpath(exp_id).glob(f"{inst}/{model}/**/*.nc")))
        results.append(read_grids(fps))

    results_df = pd.concat([pd.DataFrame(rows) for rows in results])
    results_df.to_csv("test.csv")

    # here we will exclude some files.
    # we are only going to worry about regridding those which have a latitude variable for now.
    results_df = results_df.query("~lat_min.isnull()")
    # we are also going to exclude files which cannot form a panarctic result (very few so far).
    results_df = results_df.query("lat_max > 50")
    # only regrid files if their starting date is less than or equal to 2101-01-01
    results_df = results_df.query("start_time < @max_time")
    # the grid of the file chosen as the target template grid
    cesm2_grid = results_df.query(f"fp == @target_grid_fp").grid.values[0]
    # subset to all files that do have this grid first.
    # save this as a table for use with crop_non_regrid.py.
    # These are the files that have the right grid already but need to be cropped to a panarctic extent.
    results_df.query("grid == @cesm2_grid").to_csv("files_to_crop.csv")
    # now, subset to all qualifying files that do not have this grid
    regrid_df = results_df.query("grid != @cesm2_grid")

    for name, group_df in regrid_df.groupby(["model", "scenario"]):
        # make sure that there are not multiple grids within one model/scenario at this point
        model, scenario = name
        write_batch_files(group_df, model, scenario)
