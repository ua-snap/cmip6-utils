from itertools import product
import pytest
import pandas as pd
from config import regrid_dir, regrid_batch_dir, variables, prod_scenarios, inst_models
from regrid import generate_regrid_filepath
from crop_non_regrid import get_source_filepaths_from_batch_files


def convert_src_filename_to_regrid_filenames(fn):
    timeframe = fn.split("_")[-1].split(".nc")[0]
    tmp_fn = "_".join(fn.split("_")[:-1]) + "_{}.nc"
    freq = fn.split("_")[1]
    new_time_strings = []
    if "day" in freq:
        start_date, end_date = pd.to_datetime(timeframe.split("-"))
        # handle cases where file starts on last day of year (those were simply dropped in regridding)
        if start_date.month == 12:
            start_date = pd.to_datetime(f"{start_date.year + 1}0101")
        # handle cases where file ends on first day of year (also dropped in regridding)
        if end_date.month == 1:
            end_date = pd.to_datetime(f"{end_date.year - 1}1231")
        for year in range(start_date.year, end_date.year + 1):
            new_time_strings.append(f"{year}0101-{year}1231")
    elif "mon" in freq:
        if len(timeframe.split("-")[0]) == 8:
            start_date, end_date = pd.to_datetime(timeframe.split("-"))
        else:
            start_date, end_date = pd.to_datetime(
                [x + "01" for x in timeframe.split("-")]
            )
        if start_date.month == 12:
            start_date = pd.to_datetime(f"{start_date.year + 1}01")
        for year in range(start_date.year, end_date.year + 1):
            new_time_strings.append(f"{year}01-{year}12")

    dst_fns = [tmp_fn.format(time_str) for time_str in new_time_strings]

    return dst_fns


# all source files slated for regridding
all_src_fps = get_source_filepaths_from_batch_files(regrid_batch_dir)

var_ids = list(variables.keys())
models = [mi.split("_")[1] for mi in inst_models]


# only need to run tests for which there are combinations in the bacth files
# use this function below to elimintate nujll combinations
def filename_matches_var_model_scenario(fn, var_id, model, scenario):
    """Check that a filename is for a given variable, model, and scenario"""
    fn_parts = fn.split("_")
    fn_var_id = fn_parts[0]
    fn_model = fn_parts[2]
    fn_scenario = fn_parts[3]
    return (var_id == fn_var_id) & (model == fn_model) & (scenario == fn_scenario)


params = []
for var_id, model, scenario in product(var_ids, models, prod_scenarios):
    src_fps = [
        fp
        for fp in all_src_fps
        if filename_matches_var_model_scenario(fp.name, var_id, model, scenario)
    ]
    if len(src_fps) != 0:
        params.append((var_id, model, scenario))


@pytest.mark.parametrize("var_id,model,scenario", params)
def test_regrid_completion(var_id, model, scenario):
    regrid_fps = list(regrid_dir.glob(f"**/{var_id}_*_{model}_{scenario}*.nc"))

    # Since we renamed the files by replacing the grid type component of the original
    #  filename with "regrid" upon saving the regridded files, we must do this again
    #  to compare the source file names with the regridded filenames:
    src_fns = set(
        [
            generate_regrid_filepath(fp, regrid_dir).name
            for fp in all_src_fps
            if filename_matches_var_model_scenario(fp.name, var_id, model, scenario)
        ]
    )
    regrid_fns = set([fp.name for fp in regrid_fps])

    # Now split the source filenames up into the yearly filenames:
    src_regrid_fns = []
    for fn in src_fns:
        src_regrid_fns.extend(convert_src_filename_to_regrid_filenames(fn))

    assert set(src_regrid_fns) == set(regrid_fns)
