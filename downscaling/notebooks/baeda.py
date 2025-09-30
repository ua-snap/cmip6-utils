# module for re-using code for bias adjustment exploratory data analysis

from pathlib import Path
from warnings import warn
import matplotlib.pyplot as plt
from xclim import units, sdba, indices
from xclim.core.calendar import percentile_doy
import numpy as np
import xarray as xr
from pyproj import Proj
import pandas as pd
import seaborn as sns

# some constants for the paths to the data and such
cmip6_tmp_fn = "{var_id}_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"
era5_dir = Path("/import/beegfs/CMIP6/arctic-cmip6/era5/daily_regrid")
cmip6_dir = Path("/beegfs/CMIP6/kmredilla/cmip6_regridding/regrid")

tmp_era5_discrete_idx_fp = "{var_id}_idx_era5_discrete.zarr"


# full domain config stuff
# template filename for downscaled outputs
tmp_window_fn = "qmw{qm_window}_{var_id}_{model}_{scenario}.zarr"
tmp_adapt_freq_fn = "{adapt_freq}_{var_id}_{model}_{scenario}.zarr"
tmp_nquantiles_fn = "nq{nquantiles}_{var_id}_{model}_{scenario}.zarr"
# parameter values for profiling
window_sizes = [31, 45, 61, 91]
adapt_freq_threhsolds = ["0.05 mm d-1", "0.254 mm d-1", "1 mm d-1", "2 mm d-1"]
n_quantiles_list = [50, 100, 150, 200, 250]


# lookups for functions and indices
doy_func_lu = {"min": np.min, "max": np.max, "mean": np.mean}


# these are some different detrending methods for the DQM
detrend_configs = {
    "Loess DQM": sdba.detrending.LoessDetrend(
        group="time.dayofyear", d=0, niter=1, f=0.2, weights="tricube"
    ),
    "Mean DQM": sdba.detrending.MeanDetrend(group="time.dayofyear"),
    "Poly1 DQM": sdba.detrending.PolyDetrend(group="time.dayofyear", degree=1),
    "Poly3 DQM": sdba.detrending.PolyDetrend(group="time.dayofyear", degree=3),
}

# simply window sizes
window_configs = [31, 61, 91, 121]

quantile_sizes = [50, 75, 100, 150, 200]

# This is a dict of locations with lat/lon coords for rapid testing
coords = {
    "Fairbanks": {"lat": 64.8401, "lon": -147.72},
    "Anchorage": {"lat": 61.2176, "lon": -149.8997},
    "Nome": {"lat": 64.5006, "lon": -165.4086},
    "Yakutat": {"lat": 59.5453, "lon": -139.7268},
    "Utqiagvik": {"lat": 71.2906, "lon": -156.7886},
}


# projected coordinates for discrete analysis, copied and pasted from setup notebook
projected_coords = {
    "warmest": {"x": 599436.2987255983, "y": 1730624.4491167169},
    "coldest": {"x": 880053.9955060182, "y": 1321724.3766652476},
    "avg_warmest": {"x": 407012.7352190246, "y": 1197450.8252339188},
    "avg_coldest": {"x": 731727.4986363677, "y": 1265600.8373091638},
    "wettest": {"x": 723709.850156927, "y": 1229521.419151681},
    "driest": {"x": 739745.1471158082, "y": 2428159.8668280467},
    "extra_1": {"x": 483900.0, "y": 1265000.0},
    "Fairbanks": {"x": 297504.537742375, "y": 1667301.4672914068},
    "Anchorage": {"x": 219385.09124775976, "y": 1255247.7055300924},
    "Nome": {"x": -545085.3109182257, "y": 1662288.1529547903},
    "Yakutat": {"x": 798186.4505862442, "y": 1147549.237487728},
    "Utqiagvik": {"x": -102347.93849425799, "y": 2368027.8649091986},
    "near_McGrath": {"x": -158391.91330927753, "y": 1377871.2528303924},
    "near_Arctic_Village": {"x": 340012.1339986391, "y": 2118363.789483479},
}


era5_var_id_lu = {"t2max": "tasmax", "pr": "pr"}
jitter_under_thresh_lu = {"pr": "0.01 mm d-1", "dtr": "1e-4 K"}
adapt_freq_thresh_lu = {"pr": "1 mm d-1"}
units_lu = {"pr": "mm d-1", "tasmax": "degC", "dtr": "degC"}
varid_adj_kind_lu = {"tasmax": "+", "tasmin": "+", "dtr": "*", "pr": "*"}


def wsdi(tasmax, hist_da):
    """Warm spell duration index."""
    # use the ERA5 as baseline reference for WSDI percentiles
    tasmax_per = percentile_doy(hist_da, per=90).sel(percentiles=90)
    da = indices.warm_spell_duration_index(tasmax, tasmax_per, window=6).drop(
        "percentiles"
    )
    da.name = "wsdi"
    da.attrs["long_name"] = "Warm spell duration index"
    return da


def ice_days(
    tasmax,
):
    """Icing days."""
    da = indices.tx_days_below(tasmax, thresh="0.0 degC")
    da.name = "id"
    da.attrs["long_name"] = "Icing days"
    return da


def summer_days(tasmax):
    """Summer days."""
    # threshold for summer days is 20.0 degC because Alaska
    da = indices.tx_days_above(tasmax, thresh="20.0 degC")
    da.name = "su"
    da.attrs["long_name"] = "Summer days"
    return da


def tx90p(tasmax, hist_da):
    tasmax_per = percentile_doy(hist_da, per=90).sel(percentiles=90)
    da = indices.tx90p(tasmax, tasmax_per)
    da.name = "tx90p"
    da.attrs["long_name"] = "Days above 90th percentile"
    return da


def rx1day(pr):
    """Maximum 1-day precipitation."""
    da = indices.max_1day_precipitation_amount(pr)
    da.name = "rx1day"
    da.attrs["long_name"] = "Maximum 1-day precipitation"
    return da


def rx5day(pr):
    """Maximum 5-day precipitation."""
    da = indices.max_n_day_precipitation_amount(pr, window=5)
    da.name = "rx5day"
    da.attrs["long_name"] = "Maximum 5-day precipitation"
    return da


def dpi(pr):
    """Daily precipitation intensity."""
    da = indices.daily_pr_intensity(pr)
    da.name = "dpi"
    da.attrs["long_name"] = "Daily precipitation intensity"
    return da


def cdd(pr):
    """Consecutive dry days."""
    da = indices.maximum_consecutive_dry_days(pr)
    da.name = "cdd"
    da.attrs["long_name"] = "Maximum consecutive dry days"
    return da


def cwd(pr):
    """Consecutive wet days."""
    da = indices.maximum_consecutive_wet_days(pr)
    da.name = "cwd"
    da.attrs["long_name"] = "Maximum consecutive wet days"
    return da


indices_lu = {
    "pr": {
        "rx1day": rx1day,
        "rx5day": rx5day,
        "dpi": dpi,
        "cdd": cdd,
        "cwd": cwd,
    },
    #     "dstl": indices.dry_spell_total_length,
    #     "mcwd": indices.maximum_consecutive_wet_days,
    #     "wstl": indices.wet_spell_total_length,
    # },
    "tasmax": {
        "su": summer_days,
        "id": ice_days,
        "wsdi": wsdi,
        "tx90p": tx90p,
    },
}

# indices that require additional  historical reference data
hist_ref_indices = ["wsdi", "tx90p"]


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


### everything below here is from the more recent explore_qdm.ipynb efforts
# for the bias adjustment piece of statistical downscaling


def get_projected_coords(zarr_store, coords):
    """Get the projected coordinates for the given coordinates."""
    with xr.open_zarr(zarr_store) as ds:

        # Define the WKT projection for the ERA5 data
        wkt_projection = ds.spatial_ref.attrs["crs_wkt"]

        proj = Proj(wkt_projection)

        # Convert lat, lon to the projection system
        projected_coords = {}
        for point, latlon in coords.items():
            x, y = proj(latlon["lon"], latlon["lat"])
            projected_coords[point] = {"x": x, "y": y}

    return projected_coords


def extract_time_series(ds, var_id, projected_coords=None):
    """Extracts the time series of the given coordinates from the dataset. returns a dictionary of time series."""
    if projected_coords is not None:
        time_series = {}
        for location, coord in projected_coords.items():
            x, y = coord["x"], coord["y"]
            extr = ds[var_id].sel(x=x, y=y, method="nearest").drop_vars(["x", "y"])
            if extr.dropna("time").size == 0:
                warn(
                    (
                        f"All-nan extraction encountered for location: {location}.\n"
                        f"Coordinates supplied: x: {projected_coords[location]['x']}, y: {projected_coords[location]['y']}. "
                        f"Dataset connection: {ds}"
                    )
                )
            time_series[location] = extr

        # Combine the time series into a single dataset
        combined_time_series = xr.concat(time_series.values(), dim="location")
        combined_time_series["location"] = list(time_series.keys())
    else:
        combined_time_series = ds[var_id]

    return combined_time_series


def adjust_and_combine_allqm(qm_train, dqm_train, qdm_train, sim):
    """Run adjustments for all iterations of detrending, plus the qdm method, and combine into a single dataset.
    Triggers computation of returned dataarray."""
    adj_das = []
    for det_name, det in detrend_configs.items():
        scen = dqm_train.adjust(
            sim,
            extrapolation="constant",
            interp="nearest",
            detrend=det,
        )
        adj_das.append(scen)

    # run the QDM adjustment
    qdm_da = qdm_train.adjust(
        sim,
        extrapolation="constant",
        interp="nearest",
    )
    adj_das.append(qdm_da)

    # run the QM adjustment
    qm_da = qm_train.adjust(
        sim,
        extrapolation="constant",
        interp="nearest",
    )
    adj_das.append(qm_da)

    # combine the DQM and QDM data
    adj_da = xr.concat(adj_das, dim="Method").rename(sim.name)
    adj_da["Method"] = list(detrend_configs.keys()) + ["QDM", "QM"]
    adj_da = adj_da

    return adj_da


def drop_non_coord_vars(ds):
    """Function to drop all coordinates from xarray dataset which are not coordinate variables, i.e. which are not solely indexed by a dimension of the same name

    Args:
        ds (xarray.Dataset): dataset to drop non-coordinate-variables from

    Returns:
        ds (xarray.Dataset): dataset with only dimension coordinates
    """
    coords_to_drop = [coord for coord in ds.coords if ds[coord].dims != (coord,)]
    # some datasets have a variables such as spatial_ref that are indexed by time and should be dropped.
    vars_to_drop = [var for var in ds.data_vars if len(ds[var].dims) < 3]
    ds = ds.drop_vars(coords_to_drop + vars_to_drop)

    return ds


def open_era5_dataset(era5_stores):
    """Open the ERA5 dataset from the given zarr stores."""
    # open and convert calendar to noleap
    era5_ds = xr.merge(
        [xr.open_zarr(store) for store in era5_stores.values()]
    ).convert_calendar("noleap")

    era5_ds = era5_ds.assign_coords(time=era5_ds.time.dt.floor("D"))

    # set the correct compatible precipitation units for ERA5 if precip
    for era5_var_id in era5_stores.keys():
        var_id = era5_var_id_lu[era5_var_id]
        # first rename the variable to the shared CMIP6 name
        era5_ds = era5_ds.rename({era5_var_id: var_id})
        # Make units proper if needed
        if era5_ds[var_id].attrs["units"] == "mm":
            era5_ds[var_id].attrs["units"] = "mm d-1"
        # convert to shared units
        era5_ds[var_id] = units.convert_units_to(era5_ds[var_id], units_lu[var_id])

    return era5_ds


def extract_era5_time_series(era5_ds, projected_coords):
    """Extract time series from ERA5 dataset for the given coordinates."""
    era5_ds = drop_non_coord_vars(era5_ds)
    return xr.merge(
        [
            extract_time_series(era5_ds, var_id, projected_coords)
            for var_id in era5_ds.data_vars
        ]
    )


def extract_time_series_from_zarr(zarr_dir, model, scenario, var_id, coords):
    """Extract time series from zarr stores for ERA5, historical, and future data for a given model and scenario."""
    # open historical and future data
    hist_scenario = "historical"
    hist_store = zarr_dir.joinpath(f"{var_id}_{model}_{hist_scenario}.zarr")
    sim_store = zarr_dir.joinpath(f"{var_id}_{model}_{scenario}.zarr")
    hist_ds = xr.open_zarr(hist_store)
    sim_ds = xr.open_zarr(sim_store)
    hist_ds = drop_non_coord_vars(hist_ds)
    sim_ds = drop_non_coord_vars(sim_ds)

    # ensure the time coordinates have 0 for the hour
    hist_ds = hist_ds.assign_coords(time=hist_ds.time.dt.floor("D")).sel(
        time=slice("1965", "2014")
    )
    sim_ds = sim_ds.assign_coords(time=sim_ds.time.dt.floor("D")).sel(
        time=slice("2015", "2100")
    )

    target_unit = units_lu[var_id]
    # Extract time series for each location
    hist_extr = units.convert_units_to(
        extract_time_series(hist_ds, var_id, coords), target_unit
    )
    sim_extr = units.convert_units_to(
        extract_time_series(sim_ds, var_id, coords), target_unit
    )

    hist_extr = hist_extr.assign_coords(
        Method=f"{model}", experiment=hist_scenario
    ).expand_dims(["Method", "experiment"])
    sim_extr = sim_extr.assign_coords(
        Method=f"{model}", experiment=scenario
    ).expand_dims(["Method", "experiment"])

    hist_extr.attrs["source_id"] = model
    sim_extr.attrs["source_id"] = model
    hist_extr.attrs["experiment_id"] = hist_scenario
    sim_extr.attrs["experiment_id"] = scenario

    return hist_extr, sim_extr


def run_bias_adjustment_and_package_data(hist_extr, sim_extr, era5_extr):
    """returns bias adjusted data and non-adjusted data relevant for plotting comparisons.
    assumes all relevant zarr stores are in the same directory.
    """
    var_id = hist_extr.name
    # need to supply the correct variable to the train function
    if isinstance(era5_extr, xr.Dataset):
        era5_extr = era5_extr[var_id]

    # Since it is easy to process both quantile delta mapping and detrended quantile mapping, we will do so
    train_kwargs = dict(
        ref=era5_extr,
        # think having experiment coordinate may quietly prevent
        # adjustment of data with different coordinates (e.g. ssp's)
        hist=hist_extr.isel(Method=0, experiment=0).drop_vars(["Method", "experiment"]),
        nquantiles=50,
        group="time.dayofyear",
        window=31,
        kind=varid_adj_kind_lu[var_id],
    )
    if var_id in adapt_freq_thresh_lu:
        # do the adapt frequency thingy for precipitation data
        train_kwargs.update(
            adapt_freq_thresh=adapt_freq_thresh_lu[var_id],
            jitter_under_thresh_value=jitter_under_thresh_lu[var_id],
        )

    qm_train = sdba.EmpiricalQuantileMapping.train(**train_kwargs)
    dqm_train = sdba.DetrendedQuantileMapping.train(**train_kwargs)
    qdm_train = sdba.QuantileDeltaMapping.train(**train_kwargs)

    # adjust and combine the historical data
    hist_adj_da = adjust_and_combine_allqm(
        qm_train,
        dqm_train,
        qdm_train,
        hist_extr.isel(Method=0, experiment=0).drop_vars(["Method", "experiment"]),
    )
    sim_adj_da = adjust_and_combine_allqm(
        qm_train,
        dqm_train,
        qdm_train,
        sim_extr.isel(Method=0, experiment=0).drop_vars(["Method", "experiment"]),
    )

    adj_da = xr.merge([hist_extr, sim_extr, hist_adj_da, sim_adj_da])[var_id]

    del adj_da.attrs["experiment_id"]

    return adj_da


def run_bias_adjustment_profile_window(hist_extr, era5_extr):
    """returns bias adjusted data and non-adjusted data relevant for plotting comparisons.
    assumes all relevant zarr stores are in the same directory.
    """
    var_id = hist_extr.name
    # need to supply the correct variable to the train function
    if isinstance(era5_extr, xr.Dataset):
        era5_extr = era5_extr[var_id]

    hist = hist_extr.isel(Method=0, experiment=0).drop_vars(["Method", "experiment"])

    adj_das = []
    for window in window_configs:
        train_kwargs = dict(
            ref=era5_extr,
            # think having experiment coordinate may quietly prevent
            # adjustment of data with different coordinates (e.g. ssp's)
            hist=hist_extr.isel(Method=0, experiment=0).drop_vars(
                ["Method", "experiment"]
            ),
            nquantiles=50,
            group="time.dayofyear",
            window=window,
            kind=varid_adj_kind_lu[var_id],
        )
        if var_id in adapt_freq_thresh_lu:
            # do the adapt frequency thingy for precipitation data
            train_kwargs.update(
                adapt_freq_thresh=adapt_freq_thresh_lu[var_id],
                jitter_under_thresh_value=jitter_under_thresh_lu[var_id],
            )

        # will only do QDM for this window analysis
        qdm_train = sdba.QuantileDeltaMapping.train(**train_kwargs)

        qdm_da = qdm_train.adjust(
            hist,
            extrapolation="constant",
            interp="nearest",
        )
        adj_das.append(qdm_da)

    # combine the DQM and QDM data
    adj_da = xr.concat(adj_das, dim="window_size").rename(hist.name)
    adj_da["window_size"] = [str(w) for w in window_configs]

    return adj_da


def run_bias_adjustment_profile_quantiles(hist_extr, era5_extr):
    """returns bias adjusted data and non-adjusted data relevant for plotting comparisons.
    assumes all relevant zarr stores are in the same directory.
    """
    var_id = hist_extr.name
    # need to supply the correct variable to the train function
    if isinstance(era5_extr, xr.Dataset):
        era5_extr = era5_extr[var_id]

    hist = hist_extr.isel(Method=0, experiment=0).drop_vars(["Method", "experiment"])

    adj_das = []
    for n_quantiles in quantile_sizes:
        train_kwargs = dict(
            ref=era5_extr,
            # think having experiment coordinate may quietly prevent
            # adjustment of data with different coordinates (e.g. ssp's)
            hist=hist_extr.isel(Method=0, experiment=0).drop_vars(
                ["Method", "experiment"]
            ),
            nquantiles=n_quantiles,
            group="time.dayofyear",
            window=31,
            kind=varid_adj_kind_lu[var_id],
        )
        if var_id in adapt_freq_thresh_lu:
            # do the adapt frequency thingy for precipitation data
            train_kwargs.update(
                adapt_freq_thresh=adapt_freq_thresh_lu[var_id],
                jitter_under_thresh_value=jitter_under_thresh_lu[var_id],
            )

        # will only do QDM for this window analysis
        qdm_train = sdba.QuantileDeltaMapping.train(**train_kwargs)

        qdm_da = qdm_train.adjust(
            hist,
            extrapolation="constant",
            interp="nearest",
        )
        adj_das.append(qdm_da)

    # combine the DQM and QDM data
    adj_da = xr.concat(adj_das, dim="n_quantiles").rename(hist.name)
    adj_da["n_quantiles"] = [str(w) for w in quantile_sizes]

    return adj_da


def run_indicators(da, hist_da=None, indices=None):
    """Run a set of indicators on the given dataframe."""
    var_id = da.name
    # if hist_da is not provided, use the same as da
    if hist_da is None:
        hist_da = da

    if indices == None:
        indices = indices_lu[var_id]

    indicator_das = []
    for index in indices:
        if index in hist_ref_indices:
            indicator_das.append(indices_lu[var_id][index](da, hist_da))
        else:
            indicator_das.append(indices_lu[var_id][index](da))

    indicators = xr.merge(indicator_das)
    indicators = drop_non_coord_vars(indicators)

    return indicators


### plotting functions


def plot_kde(df, var_id):
    """Plots a KDE of the given variable, location, and method."""

    g = sns.displot(
        data=df,
        x=var_id,
        hue="Method",
        col="location",
        kind="kde",
        col_wrap=3,
        facet_kws={"sharex": False, "sharey": False, "legend_out": False},
        common_norm=False,
    )

    sns.move_legend(g, "upper left", bbox_to_anchor=(0.75, 0.45), frameon=False)

    # pr distros have long tails, so limit the x axis to 25% of inital xlimit for better visualization
    if var_id == "pr":
        for ax in g.axes.flat:
            ax.set_xlim(0, ax.get_xlim()[1] * 0.25)


def indicator_boxplot_by_location(indicators, indicator, hue="Method"):
    """Create a seaborn catplot of an indicator faceted by location and colored by method"""
    historical_indicators_df = indicators.to_dataframe()

    g = sns.catplot(
        historical_indicators_df,
        kind="box",
        y=indicator,
        hue=hue,
        col="location",
        sharey=False,
        col_wrap=5,
        height=3,
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.8, 0.3), frameon=False)
    g.figure.suptitle(indicators[indicator].attrs["long_name"])
    plt.tight_layout()
    plt.show()


def indicator_deltas_by_method_location(proj_indicators, hist_indicators, indicator):
    """Create a seaborn catplot of an indicator faceted by location and colored by method"""
    proj_da = proj_indicators.sel(time=slice("1970", "2100")).mean("time")
    hist_da = hist_indicators.sel(time=slice("1985", "2014")).mean("time")
    delta_df = (proj_da - hist_da).to_dataframe()

    g = sns.catplot(
        delta_df,
        kind="bar",
        y=indicator,
        hue="Method",
        col="location",
        sharey=False,
        col_wrap=5,
        height=3,
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.8, 0.3), frameon=False)
    g.figure.suptitle(
        f"Projected - Historical {proj_indicators[indicator].attrs['long_name']}"
    )
    plt.tight_layout()
    plt.show()


def run_full_adjustment_and_summarize(hist_extr, sim_extr, era5_extr, results):
    """Run the full adjustment and summarize the results."""
    adj_da = run_bias_adjustment_and_package_data(hist_extr, sim_extr, era5_extr)

    var_id = hist_extr.name
    assert var_id == sim_extr.name == era5_extr.name
    model = adj_da.attrs["source_id"]
    scenario = sim_extr.attrs["experiment_id"]
    results[model][var_id]["adj"] = {
        "historical": adj_da.sel(experiment="historical")
        .dropna(dim="time")
        .drop_vars("experiment"),
        scenario: adj_da.sel(experiment=scenario)
        .dropna(dim="time")
        .drop_vars("experiment"),
    }

    # run the historical indicators separate so we can merge with ERA5 indicators for later steps
    historical_indicators = run_indicators(
        results[model][var_id]["adj"]["historical"],
    )
    future_indicators = run_indicators(
        results[model][var_id]["adj"][scenario],
        results[model][var_id]["adj"]["historical"],
    )
    # merge with ERA5 indicators for plotting
    era5_indicators = results["ERA5"][var_id]["indicators"]
    historical_indicators = xr.concat(
        [era5_indicators, historical_indicators], dim="Method"
    )
    indicators_dict = {
        "historical": historical_indicators,
        scenario: future_indicators,
    }
    results[model][var_id]["indicators"] = indicators_dict

    return results


def run_window_profile_adjustment_and_summarize(hist_extr, era5_extr, results):
    """Run the full adjustment and summarize the results."""
    adj_da = run_bias_adjustment_profile_window(hist_extr, era5_extr)

    var_id = hist_extr.name
    tmp_name = "qdm_windows"
    model = adj_da.attrs["source_id"]
    results[model][var_id][tmp_name] = {"historical": adj_da}

    # run the historical indicators separate so we can merge with ERA5 indicators for later steps
    historical_indicators = run_indicators(
        results[model][var_id][tmp_name]["historical"],
    )
    era5_indicators = results["ERA5"][var_id]["indicators"]
    historical_indicators = xr.concat(
        [
            era5_indicators.rename(Method="window_size"),
            historical_indicators,
        ],
        dim="window_size",
    )

    indicators_dict = {
        "historical": historical_indicators,
    }
    results[model][var_id][tmp_name]["indicators"] = indicators_dict

    return results


def run_quantile_profile_adjustment_and_summarize(hist_extr, era5_extr, results):
    """Run the full adjustment and summarize the results."""
    adj_da = run_bias_adjustment_profile_quantiles(hist_extr, era5_extr)

    var_id = hist_extr.name
    tmp_name = "qdm_quantiles"
    model = adj_da.attrs["source_id"]
    results[model][var_id][tmp_name] = {"historical": adj_da}

    # run the historical indicators separate so we can merge with ERA5 indicators for later steps
    historical_indicators = run_indicators(
        results[model][var_id][tmp_name]["historical"],
    )
    era5_indicators = results["ERA5"][var_id]["indicators"]
    historical_indicators = xr.concat(
        [
            era5_indicators.rename(Method="n_quantiles"),
            historical_indicators,
        ],
        dim="n_quantiles",
    )

    indicators_dict = {
        "historical": historical_indicators,
    }
    results[model][var_id][tmp_name]["indicators"] = indicators_dict

    return results


def plot_sxs(da1, da2, title1, title2, plot1_kwargs, plot2_kwargs, main_title):

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7), sharey=True)
    da1.load()
    da2.load()
    # Determine the common color scale
    vmin = min(da1.min().values, da2.min().values)
    vmax = max(da1.max().values, da2.max().values)

    # Plot first time slice from cmip6_ds
    im1 = da1.plot(ax=axes[0], vmin=vmin, vmax=vmax, **plot1_kwargs)
    axes[0].set_title(title1)

    # Plot first time slice from down_ds
    im2 = da2.plot(ax=axes[1], vmin=vmin, vmax=vmax, **plot2_kwargs)
    axes[1].set_title(title2)

    # Adjust layout to make space for the colorbar
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Add a single colorbar for all subplots
    cbar = fig.colorbar(
        im1, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.02, pad=0.04
    )
    cbar.set_label(f'{da1.attrs["units"]}')

    axes[1].yaxis.set_visible(False)
    plt.suptitle(main_title)
    plt.show()
