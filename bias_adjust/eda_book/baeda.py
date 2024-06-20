# module for re-using code for bias adjustment exploratory data analysis

from pathlib import Path
import matplotlib.pyplot as plt

# some constants for the paths to the data and such
cmip6_tmp_fn = "{var_id}_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"
era5_dir = Path("/import/beegfs/CMIP6/arctic-cmip6/era5/daily_regrid")
cmip6_dir = Path("/beegfs/CMIP6/kmredilla/cmip6_regridding/regrid")
cmip6_tmp_fn = "{var_id}_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"


# we will be getting filepaths in a similar way many times. Make some functions to make this easier
def get_cmip6_fps(model, scenario, var_id, start_year, end_year):
    # need to supply scenario for projected years overlapping historical reference years
    return [
        cmip6_dir.joinpath(
            model,
            scenario,
            "day",
            var_id,
            cmip6_tmp_fn.format(
                var_id=var_id, model=model, scenario=scenario, year=year
            ),
        )
        for year in range(start_year, end_year + 1)
    ]


def get_era5_fps(var_id, start_year, end_year):
    return [
        era5_dir.joinpath(var_id).joinpath(f"era5_daily_regrid_{var_id}_{year}.nc")
        for year in range(start_year, end_year + 1)
    ]


# we will always be pulling out the dataarray from a dataset and rechunking, so make a function for that
def get_rechunked_da(ds, var_id, lat_chunk=20, lon_chunk=20):
    return ds[var_id].chunk({"time": -1, "lat": lat_chunk, "lon": lon_chunk})


# function to plot averaged time series of ref, sim, and scen
# ('scen' is xclim's name for the adjusted data)
def plot_avg_ts(ref, sim, scen, gb_str="time.dayofyear"):
    ref.groupby(gb_str).mean().plot(label="Reference")
    sim.groupby(gb_str).mean().plot(label="Model - simulated")
    scen.groupby(gb_str).mean().plot(label="Model - adjusted")
    plt.legend()
