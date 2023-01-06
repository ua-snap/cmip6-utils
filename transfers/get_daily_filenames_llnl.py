"""Generate a reference table of daily CMIP6 holdings of interest for SNAP on the LLNL ESGF node"""

from itertools import product
from multiprocessing import Pool
import numpy as np
import pandas as pd
from config import *
import luts
import utils


def get_filenames(args):
    """Get the file names for a some combination of model, scenario, and variable."""
    activity, model, scenario, varname = args
    variant = luts.model_inst_lu[model]["variant"]
    
    # the subdirectory under the variable name is the grid type.
    #  This is almost always "gn", meaning the model's native grid, but it could be different. 
    #  So we have to check it instead of assuming. I have only seen one model where this is different (gr1, GFDL-ESM4)
    var_path = llnl_prefix.joinpath(
        activity, luts.model_inst_lu[model]["institution"], model, scenario, variant, "day", varname
    )
    grid_type = utils.get_contents(llnl_ep, var_path)

    if isinstance(grid_type, int):
        # there is no data for this particular combination.
        row_di = {
            "model": model,
            "scenario": scenario,
            "variant": variant,
            "variable": varname,
            "grid_type": None,
            "version": None,
            "n_files": None,
            "filenames": None,
        }

    else:
        # combo does exist, return all filenames
        grid_type = grid_type[0].replace("/", "")
        versions = utils.get_contents(llnl_ep, var_path.joinpath(grid_type))
        # go with newer version
        use_version = sorted([v.replace("/", "") for v in versions])[-1]
        # add "v" back in
        fns = utils.get_contents(llnl_ep, var_path.joinpath(grid_type, use_version))
        row_di = {
            "model": model,
            "scenario": scenario,
            "variant": variant,
            "variable": varname,
            "grid_type": grid_type,
            "version": use_version,
            "n_files": len(fns),
            "filenames": fns,
        }
    
    return row_di


if __name__ == "__main__":
    
    # put all variables of interest into single list
    varnames = list(luts.vars_tier1.keys()) + list(luts.vars_tier2.keys())
    
    # generate lists of arguments from all combinations of variables, models, and scenarios
    args = list(
        product(["CMIP"], luts.model_inst_lu, ["historical"], varnames)
    ) + list(
        product(["ScenarioMIP"], luts.model_inst_lu, scenarios, varnames)
    )
    
    with Pool(32) as pool:
        rows = pool.map(get_filenames, args)
    
    # create dataframe from results and save to this folder
    df = pd.DataFrame(rows)
    df.to_csv("llnl_esgf_day_filenames.csv")
