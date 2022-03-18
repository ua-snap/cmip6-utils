"""Functions to perform mosaicking of CMIP6 data"""

import xarray as xr
from luts import api_lu


def open_tile_datasets(df, mosaic_fp):
    """Open connections to the datasets corresponding to the mosaic filepath,
    store in dict
    
    Args:
        df (pandas.DataFrame): main tracker dataframe
        mosaic_fp (str): file path to get corresponding tile file paths for

    Returns:
        tile_ds_di (dict): dict of open tile datasets
    """
    # dict for storing open datasets
    tile_ds_di = {}
    for hemisphere in ["west", "east"]:
        fp = df.query(
            f"mosaic_path == '{mosaic_fp}' & hemisphere == '{hemisphere}'"
        ).iloc[0]["base_path"]
        tile_ds_di[f"{hemisphere}_ds"] = xr.open_dataset(fp, decode_times=False)

    return tile_ds_di


def prep_east_tile_ds(east_ds):
    """Current mosaicking strategy involves east and west hemisphere downloads.
    This function adjusts the longitude coordinates and related lon_bnds variable
    of the east dataset to the [-360, 0] scale for proper combination of datasets.
    
    Args:
        east_ds (xarray.dataset): East hemisphere tile dataset
    
    Returns:
        modified version of east_ds with lon data on [-360, 0] scale
    """
    east_ds = east_ds.assign_coords({"lon": east_ds.lon.values - 360})
    if "lon_bnds" in east_ds.variables:
        east_ds["lon_bnds"].values = east_ds["lon_bnds"].values - 360
    
    return east_ds


def mosaic(args):
    """Creates a mosaic dataset and writes it to mosaic_fp.
    
    Args:
        df (pandas.DataFrame): main tracker dataframe
        mosaic_fp (str): file path to write mosaicked data to
    
    Returns:
        None - mosaicked data is written to file path provided
    """
    def get_target_var(ds):
        """Get the target variable name from a dataset"""
        target_var = [
            varname for varname in ds.variables if varname in api_lu["varnames"].keys()
        ][0]
        
        return target_var

    df, mosaic_fp = args
    tile_ds_di = open_tile_datasets(df, mosaic_fp)
    tile_ds_di["east_ds"] = prep_east_tile_ds(tile_ds_di["east_ds"])
    target_var = get_target_var(tile_ds_di["east_ds"])
    combine_vars = [target_var]
    if "lon_bnds" in tile_ds_di["east_ds"].variables:
        combine_vars.append("lon_bnds")

    try:
        mosaic_ds = xr.combine_by_coords(
            tile_ds_di.values(), data_vars=combine_vars, combine_attrs="no_conflicts"
        )
    except xr.MergeError:
        # find conflicting attrs and append as "<hemisphere>_<attribute>" on combine
        conflict_keys = []
        for key in tile_ds_di["west_ds"].attrs:
            if tile_ds_di["west_ds"].attrs[key] != tile_ds_di["west_ds"].attrs[key]:
                conflict_keys.append(key)
        mosaic_ds = xr.combine_by_coords(
            tile_ds_di.values(), data_vars=combine_vars, combine_attrs="drop_conflicts"
        )
        for key in conflict_keys:
            for hemi in ["east", "west"]:
                mosaic_ds.attrs[f"{hemi}_{key}"] = tile_ds_di[f"{hemi}_ds"].attrs[key]
    
    mosaic_ds.to_netcdf(mosaic_fp)
    
    return mosaic_fp