# lookup to crosswalk CMIP6 variable names to ERA5 variable names
sim_ref_var_lu = {
    "tasmax": "t2max",
    "tasmin": "t2min",
    "dtr": "dtr",
    "pr": "pr",
    "hurs": "rh2_mean",
    "hursmin": "rh2_min",
    "snw": "snow_sum",
    "sfcWind": "wspd10_mean",
}

# lookup for "kind" arg of DQM training function
varid_adj_kind_lu = {
    "tasmax": "+",
    "tasmin": "+",
    "dtr": "*",
    "pr": "*",
    "hurs": "+",
    "hursmin": "+",
    "snw": "*",
    "sfcWind": "*",
}

jitter_under_lu = {"pr": "0.01 mm d-1", "dtr": "1e-4 K", "snw": "0.01 kg m-2"}

adapt_freq_thresh_lu = {"pr": "0.254 mm d-1", "snw": "0.254 kg m-2"}

future_start_year = 2015
future_end_year = 2100
era5_start_year = 1965
era5_end_year = 2014
cmip6_year_ranges = {
    "historical": {
        "start_year": era5_start_year,
        "end_year": era5_end_year,
    },
    "ssp126": {
        "start_year": future_start_year,
        "end_year": future_end_year,
    },
    "ssp245": {
        "start_year": future_start_year,
        "end_year": future_end_year,
    },
    "ssp370": {
        "start_year": future_start_year,
        "end_year": future_end_year,
    },
    "ssp585": {
        "start_year": future_start_year,
        "end_year": future_end_year,
    },
}
