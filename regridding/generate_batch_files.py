"""Generate text files ("batch" files) containing all of the files we want to regrid broken up by frequency, model, scenario, and variable. 
It utilizes code from the explore_grids.ipynb notebook to select the files which need to be regridded.


"""

import numpy as np
import pandas as pd
import xarray as xr
import tqdm
import argparse
from multiprocessing import Pool, set_start_method
from pathlib import Path
from config import *

import concurrent.futures

# ignore serializationWarnings from xarray for datasets with multiple FillValues
import warnings

warnings.filterwarnings("ignore", category=xr.SerializationWarning)


GRID_VARS = ["lat", "lon"]
max_year = 2101
min_year = 1950


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
    try:
        # using the h5netcdf engine because it seems faster and might help prevent pool hanging
        ds = xr.open_dataset(fp, engine="h5netcdf")
    except:
        # this seems to have only failed due to some files (KACE model) being written in netCDF3 format
        ds = xr.open_dataset(fp)

    grid_di = {}
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
    ts_min = ds.time.values.min()
    ts_max = ds.time.values.max()

    # trying to help multiprocessing not hang, not ideal of course
    ds.close()
    del ds

    # create a new column that is a concatenation of all of these values
    grid_di["grid"] = "_".join([str(grid_di[key]) for key in grid_di.keys()])
    # pull out file attributes (model scenario etc)
    grid_di.update(fp_to_attrs(fp))
    # also keep the filename for reference
    grid_di["fp"] = fp

    # only want to process files that have data from 1950-2100
    # want to save the earliest time, because we will just ignore projections greater than 2100, for now at least.
    # ts_min = ds.time.values.min()
    # using pd.Timestamp here because numpy datetime64 can hav OOB errors for large timestamps
    if isinstance(ts_min, np.datetime64):
        start_year = ts_min.astype("datetime64[Y]").astype(int) + 1970
    else:
        # assumes cf time object
        start_year = ts_min.year
    grid_di["start_year"] = start_year

    # Same thing as above but for last time in dataset
    # ts_max = ds.time.values.max()
    # using pd.Timestamp here because numpy datetime64 can hav OOB errors for large timestamps
    if isinstance(ts_max, np.datetime64):
        end_year = ts_max.astype("datetime64[Y]").astype(int) + 1970
    else:
        # assumes cf time object
        end_year = ts_max.year
    grid_di["end_year"] = end_year

    # save file size for better chunking into batches
    grid_di["filesize"] = fp.stat().st_size / (1e3**3)

    return grid_di


def read_grids(fps, pool, progress=False):
    """Read the grid info from all files in fps, using multiprocessing and with a progress bar"""

    grid_futures = [pool.submit(get_grid, fp) for fp in fps]
    grids = []
    for grid in tqdm.tqdm(
        concurrent.futures.as_completed(grid_futures), total=len(grid_futures)
    ):
        grids.append(grid.result())

    # using fewer cores might help prevent hanging with multiprocessing?
    # with Pool(10) as pool:
    #     if progress:
    #         grids = []
    #         for grid_di in tqdm.tqdm(
    #             pool.imap_unordered(get_grid, fps), total=len(fps)
    #         ):
    #             grids.append(grid_di)
    #     else:
    #         grids = pool.map(get_grid, fps)

    return grids


def chunk_list_of_files(fps, max_count):
    """Helper function to chunk lists of files for appropriately-sized batches"""
    fp_chunks = []
    chunk = []
    for fp in fps:
        if len(chunk) >= max_count:
            fp_chunks.append(chunk)
            # re-initialize with current filepath
            chunk = [fp]
        else:
            chunk.append(fp)

    if len(chunk) > 0:
        fp_chunks.append(chunk)

    return fp_chunks


def write_batch_files(group_df, model, scenario, var_id, frequency, regrid_batch_dir):
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
                    model=model,
                    scenario=scenario,
                    var_id=var_id,
                    frequency=frequency,
                    grid_name=grid_name,
                    count=i,
                )
            )
            with open(batch_file, "w") as f:
                for fp in chunk:
                    f.write(f"{fp}\n")

    return


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cmip6_directory",
        type=str,
        help="Path to directory where CMIP6 files are stored",
        required=True,
    )
    parser.add_argument(
        "--regrid_batch_dir",
        type=str,
        help="Path to directory where batch files are written",
        required=True,
    )
    parser.add_argument(
        "--vars",
        type=str,
        help="list of variables used in generating batch files",
        required=True,
    )
    parser.add_argument(
        "--freqs",
        type=str,
        help="list of frequencies used in generating batch files",
        required=True,
    )
    parser.add_argument(
        "--models",
        type=str,
        help="list of models used in generating batch files",
        required=True,
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="list of scenarios used in generating batch files",
        required=True,
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars (best used for interactive jobs, default is False)",
    )
    args = parser.parse_args()

    return (
        Path(args.cmip6_directory),
        Path(args.regrid_batch_dir),
        args.vars,
        args.freqs,
        args.models,
        args.scenarios,
        args.progress,
    )


if __name__ == "__main__":
    (cmip6_dir, regrid_batch_dir, vars, freqs, models, scenarios, progress) = (
        parse_args()
    )

    # hopefully
    set_start_method("spawn")

    # read the grid info from all files
    fps = []
    for exp_id in ["ScenarioMIP", "CMIP"]:
        # add only daily and monthly files
        for var in vars.split():
            for freq in freqs.split():
                for model in models.split():
                    inst = model_inst_lu[model]
                    for scenario in scenarios.split():
                        fps.extend(
                            list(
                                cmip6_dir.joinpath(exp_id, inst, model, scenario).glob(
                                    f"*/*{freq}/{var}/**/*.nc"
                                )
                            )
                        )

    grids = []
    # pool seems to be more likely to hang on larger batches of inputs, so we will break up into smaller batches
    # also using smaller max workers might help
    fps = chunk_list_of_files(fps, 1000)
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        for i, batch in enumerate(fps):
            grids.extend(read_grids(batch, pool=pool, progress=progress))

    results_df = pd.DataFrame(grids)

    # here we will exclude some files.
    # we are only going to worry about regridding those which have a latitude variable for now.
    results_df = results_df.query("~lat_min.isnull()")
    # we are also going to exclude files which cannot form a panarctic result (very few so far).
    results_df = results_df.query("lat_max > 50")
    # drop any subdaily frequencies.
    results_df = results_df.query(
        "frequency.str.contains('day') | frequency.str.contains('mon')"
    )
    # only regrid files if their starting date is less than or equal to 2101-01-01.
    results_df = results_df.query("start_year < @max_year")
    results_df = results_df.query("end_year >= @min_year")

    # remove all batch files. We will be generating only those which contain files to be regridded based on flow parameters.
    _ = [fp.unlink() for fp in regrid_batch_dir.glob("*.txt")]

    for name, group_df in results_df.groupby(
        ["model", "scenario", "var_id", "frequency"]
    ):
        # make sure that there are not multiple grids within one model/scenario at this point
        model, scenario, var_id, frequency = name
        write_batch_files(
            group_df, model, scenario, var_id, frequency, regrid_batch_dir
        )
