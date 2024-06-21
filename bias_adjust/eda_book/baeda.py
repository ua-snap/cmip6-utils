# module for re-using code for bias adjustment exploratory data analysis

from pathlib import Path
import matplotlib.pyplot as plt
from xclim import units, sdba
import numpy as np


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


# here is a function to print the estimated adjustment factor for the q15 quantile for a given day of year
# this will be the 0.15 quantile of ref divided by hist
def print_q15_ratio(ref, hist, doy):
    hist_doy = units.convert_units_to(
        hist.sel(time=hist.time.dt.dayofyear.isin([doy])), "m d-1"
    ).values
    hist_doy_q15 = np.quantile(hist_doy, 0.15)
    ref_doy = ref.sel(time=ref.time.dt.dayofyear.isin([doy])).values
    ref_doy_q15 = np.quantile(ref_doy, 0.15)

    print(f"doy {doy}, model: {hist_doy_q15:.9f}")
    print(f"doy {doy}, reference: {ref_doy_q15:.9f}")
    print(f"Ratio (ref / hist): {ref_doy_q15 / hist_doy_q15:.2f}")


def compute_rmse(da1, da2):
    # copilot code
    # Ensure the two DataArrays have the same shape
    assert da1.shape == da2.shape, "DataArrays must have the same shape"

    # Compute the squared difference between the two DataArrays
    squared_diff = (da1 - da2) ** 2

    # Compute the mean of the squared difference
    mean_squared_diff = squared_diff.mean()

    # Compute the square root of the mean squared difference
    rmse = np.sqrt(mean_squared_diff)

    return rmse


def run_adjust(ref, hist, ju_thresh, adapt_freq_thresh, det):
    if ref.attrs["units"] == "m":
        ref.attrs["units"] = "m d-1"
    ref_ju = sdba.processing.jitter_under_thresh(ref, thresh=ju_thresh)
    hist_ju = sdba.processing.jitter_under_thresh(hist, thresh=ju_thresh)

    dqm = sdba.DetrendedQuantileMapping.train(
        ref_ju,
        hist_ju,
        nquantiles=50,
        group="time.dayofyear",
        window=31,
        kind="*",
        adapt_freq_thresh=adapt_freq_thresh,
    )

    scen = dqm.adjust(hist_ju, extrapolation="constant", interp="nearest", detrend=det)

    return scen.compute()


def doy_means(da, dims=None):
    doy_means = da.groupby("time.dayofyear").mean()

    return doy_means.compute()
