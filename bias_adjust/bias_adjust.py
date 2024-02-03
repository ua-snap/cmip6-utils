""""""

import argparse
from itertools import Product
from pathlib import Path
from luts import sim_ref_var_lu


def generate_adjusted_filepaths(output_dir, var_ids, models, scenarios, years):
    """Generate the adjusted filepaths

    Args:
        output_dir (pathlib.Path): path to parent output directory
        var_ids (list): list of variable IDs (str)
        models (list): list of models (str)
        scenarios (list): list of scenarios (str)
        years (list): list of years because the data will be written to one file per year

    Returns:
        list of adjusted filepaths
    """
    tmp_fn = "{var_id}_day_{model}_{scenario}_adjusted_{year}0101-{year}1231.nc"
    adj_fps = [
        output_dir.joinpath(
            model,
            scenario,
            var_id,
            year,
            tmp_fn.format(model=model, scenario=scenario, var_id=var_id, year=year),
        )
        for model, scenario, var_id, year in Product(models, scenarios, var_ids, years)
    ]

    return adj_fps


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--var_id",
        type=str,
        help="Variable ID to adjust",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to adjust",
        required=True,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario to adjust",
        required=True,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to directory of simulated data files to be adjusted, with filepath structure <model>/<scenario>/day/<variable ID>/<files>",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        help="Path to directory of reference data with filepath structure <variable ID>/<files>",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to working directory, where outputs and ancillary files will be written",
    )
    parser.add_argument(
        "--no-clobber",
        action="store_true",
        default=False,
        help="Do not overwrite files if they exists in out_dir",
    )
    args = parser.parse_args()

    return (
        args.var_ids.split(" "),
        args.models.split(" "),
        args.scenarios.split(" "),
        Path(args.input_dir),
        Path(args.reference_dir),
        Path(args.output_dir),
        args.no_clobber,
    )


if __name__ == "__main__":
    (
        var_id,
        model,
        scenario,
        input_dir,
        reference_dir,
        output_dir,
        no_clobber,
    ) = parse_args()

    ref_var_id = sim_ref_var_lu[var_id]
    ref_start_year = 1993
    ref_end_year = 2022
    ref_years = list(range(ref_start_year, ref_end_year + 1))
    ref_fps = [
        reference_dir.joinpath(ref_var_id).joinpath(
            f"era5_daily_regrid_{ref_var_id}_{year}.nc"
        )
        for year in ref_years
    ]

    hist_start_year = 1993
    hist_end_year = 2014
    hist_years = list(range(hist_start_year, hist_end_year + 1))

    cmip6_tmp_fn = "{var_id}_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"

    hist_fps = [
        input_dir.joinpath(
            model,
            "historical",
            "day",
            var_id,
            cmip6_tmp_fn.format(
                var_id=var_id, model=model, scenario="historical", year=year
            ),
        )
        for year in hist_years
    ]

    # these are the years we will be including from ScenarioMIP projections to match the reference period
    sim_ref_start_year = 2015
    sim_ref_end_year = 2022
    sim_ref_years = list(range(sim_ref_start_year, sim_ref_end_year + 1))

    sim_ref_fps = [
        input_dir.joinpath(
            model,
            scenario,
            "day",
            var_id,
            cmip6_tmp_fn.format(
                var_id=var_id, model=model, scenario=scenario, year=year
            ),
        )
        for year in sim_ref_years
    ]

    adj_fps = generate_adjusted_filepaths(
        output_dir, [var_id], [model], [scenario], ref_years
    )
    print(adj_fps)
