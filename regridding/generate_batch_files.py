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


GRID_VARS = ["lat", "lon", "lat_bnds", "lon_bnds"]

def fp_to_attrs(fp):
    """pull the data attributes from a filepath"""
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


def get_grid(fp):
    """Read the info from a grid for a single file"""
    grid_di = {}
    with xr.open_dataset(fp) as ds:
        for varname in GRID_VARS:
            if (varname in ds.dims) or (varname in ds.data_vars):
                grid_di[f"{varname}_min"] = ds[varname].values.min()
                grid_di[f"{varname}_max"] = ds[varname].values.max()
                grid_di[f"{varname}_size"] = ds[varname].values.shape[0]
                grid_di[f"{varname}_step"] = np.diff(ds[varname].values)[0]
            else:
                grid_di[f"{varname}_min"] = None
                grid_di[f"{varname}_max"] = None
                grid_di[f"{varname}_size"] = None
                grid_di[f"{varname}_step"] = None
            
    # create a new column that is a concatenation of all of these values
    grid_di["grid"] = "_".join([str(grid_di[key]) for key in grid_di.keys()])
    # pull out file attributes (model scenario etc)
    grid_di.update(fp_to_attrs(fp))
    # also keep the filename for reference
    grid_di["fp"] = fp
        
    return grid_di


def read_grids(fps):
    """Read the grid info from all files in fps, using multiprocessing and with a progress bar"""
    grids = []
    with Pool(32) as pool:
        for grid_di in tqdm.tqdm(
            pool.imap_unordered(get_grid, fps), total=len(fps)
        ):
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
    
    def chunk_fp_list(fp_list, n):
        """Helper function to chunk lists of files for appropriately-sized batches"""
        for i in range(0, len(fp_list), n): 
            yield fp_list[i:i + n]

    # iterate over the types of grid within a single model scenario so that only files 
    #  with same grid are worked on in the same batch file
    # first, classify the different grid types with a name to be included in the batch file name
    group_df = generate_grid_names(group_df)
    for grid_name, df in group_df.groupby("grid_name"):
        # bumped this down from 200 to 100 after getting some memory errors with dask
        fp_chunks = chunk_fp_list(df.fp.values, 100)
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
            fps.extend(list(cmip6_dir.joinpath(exp_id).glob(f"{inst}/{model}/*/*/*/*/*/*/*.nc")))
        results.append(read_grids(fps))
        
    results_df = pd.concat([pd.DataFrame(rows) for rows in results])
    
    # the grid of the file chosen as the target template grid
    cesm2_grid = results_df.query(f"fp == @target_grid_fp").grid.values[0]
    regrid_df = results_df.query("grid != @cesm2_grid")
    
    for name, group_df in regrid_df.groupby(["model", "scenario"]):
        # make sure that there are not multiple grids within one model/scenario at this point
        model, scenario = name
        write_batch_files(group_df, model, scenario)
        