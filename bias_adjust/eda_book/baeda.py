# module for re-using code for bias adjustment exploratory data analysis

from pathlib import Path
import matplotlib.pyplot as plt
from xclim import units, sdba, indices
import numpy as np
import xarray as xr


# some constants for the paths to the data and such
cmip6_tmp_fn = "{var_id}_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"
era5_dir = Path("/import/beegfs/CMIP6/arctic-cmip6/era5/daily_regrid")
cmip6_dir = Path("/beegfs/CMIP6/kmredilla/cmip6_regridding/regrid")

# lookups for functions and indices
doy_func_lu = {"min": np.min, "max": np.max, "mean": np.mean}
indices_lu = {
    "mcdd": indices.maximum_consecutive_dry_days,
    "dstl": indices.dry_spell_total_length,
    "mcwd": indices.maximum_consecutive_wet_days,
    "wstl": indices.wet_spell_total_length,
}

# we will be getting filepaths in a similar way many times. Make some functions to make this easier
def get_cmip6_fps(cmip6_dir, model, scenario, var_id, start_year, end_year):
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


def get_era5_fps(era5_dir, var_id, start_year, end_year):
    return [
        era5_dir.joinpath(var_id).joinpath(f"era5_daily_regrid_{var_id}_{year}.nc")
        for year in range(start_year, end_year + 1)
    ]


def get_all_hist_fps(
    model_dir,
    var_id,
    hist_start_year=1993,
    hist_end_year=2014,
    sim_ref_start_year=2015,
    sim_ref_end_year=2022,
):
    """Get all historical filepaths for a given model and variable."""
    var_id = "pr"
    model = model_dir.parts[-1]
    cmip6_dir = model_dir.parent
    hist_start_year = 1993
    hist_end_year = 2014
    hist_fps = get_cmip6_fps(
        cmip6_dir, model, "historical", var_id, hist_start_year, hist_end_year
    )

    scenario = "ssp585"
    sim_ref_start_year = 2015
    sim_ref_end_year = 2022
    sim_ref_fps = get_cmip6_fps(
        cmip6_dir, model, scenario, var_id, sim_ref_start_year, sim_ref_end_year
    )

    return hist_fps, sim_ref_fps


# we will always be pulling out the dataarray from a dataset and rechunking, so make a function for that
def get_rechunked_da(ds, var_id, lat_chunk=60, lon_chunk=60):
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


def rmse(da1, da2):
    # copilot code
    # Ensure the two DataArrays have the same shape
    if not da1.shape == da2.shape:
        da2 = da2.transpose(*da1.dims)

    # Compute the squared difference between the two DataArrays
    squared_diff = (da1.values - da2.values) ** 2

    # Compute the mean of the squared difference
    mean_squared_diff = squared_diff.mean()

    # Compute the square root of the mean squared difference
    rmse = np.sqrt(mean_squared_diff)

    return rmse


def run_adjust(ref, hist, det, ju_thresh, adapt_freq_thresh="1 mm d-1"):
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


def doy_stats(da, stat_func):
    doy_stats = stat_func(da.groupby("time.dayofyear"))
    return doy_stats.compute()


def run_adjust_and_compute_doy_rmse(
    ref,
    hist,
    det,
    ju_thresh="0.01 mm d-1",
    adapt_freq_thresh="1 mm d-1",
    doy_stat_func=np.mean,
):
    """Run an adjustment and compute the RMSE of DOY stats between the reference and adjusted data."""
    scen = run_adjust(ref, hist, det, ju_thresh, adapt_freq_thresh)

    ref_doy = doy_stats(ref, stat_func=doy_stat_func)
    scen_doy = doy_stats(scen, stat_func=doy_stat_func)

    return rmse(ref_doy.values, scen_doy.values)


def summarize(da):
    """Function for summarizing and packaging results for a given model over DOY and indices."""
    results = {}

    results["doy"] = {}
    for func in doy_func_lu:
        results["doy"][func] = doy_stats(da, doy_func_lu[func]).compute()

    results["indices"] = {}
    for func in indices_lu:
        results["indices"][func] = indices_lu[func](da).rename(func).compute()

    return results


def get_hist(model_dir, var_id, chunks={"lat_chunk": 60, "lon_chunk": 60}):
    """Get the dataarray of historical simulations for a given model and variable."""
    hist_fps, sim_ref_fps = get_all_hist_fps(model_dir, var_id)
    hist_ds = xr.open_mfdataset(hist_fps + sim_ref_fps)
    hist = get_rechunked_da(hist_ds, var_id, **chunks)

    return hist
