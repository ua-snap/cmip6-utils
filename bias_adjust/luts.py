# this should be referenced in downscaling module
sim_ref_var_lu = {"tasmax": "t2max", "tasmin": "t2min", "dtr": "dtr", "pr": "tp"}

# lookup for "kind" arg of DQM training function
varid_adj_kind_lu = {"tasmax": "+", "tasmin": "+", "dtr": "*", "pr": "*"}

jitter_under_lu = {"pr": "0.01 mm d-1", "dtr": "1e-4 K"}

# min / max possible ranges
expected_value_ranges = {
    # this value was determined in exploration
    #  max temps for panarctic domain were ~45 C, min temps around -60 C
    #  within day differences of 100 C are extremely unlikely
    #  (units for data are kelvin but )
    "dtr": {"minimum": 0, "maximum": 100},
    # ~130 mm in 24 hrs is close to maximum for SE AK.
    # safe max is probably 150 mm, or in 1 kg m-2 s-1
    "pr": {"minimum": 0, "maximum": 1},
}

future_start_year = 1965
future_end_year = 2100
cmip6_year_ranges = {
    "historical": {
        "start_year": 1965,
        "end_year": 2014,
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
