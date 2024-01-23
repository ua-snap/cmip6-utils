# map from model variable names to possible index variable names
# helps with possible optimization of computing indicators that rely on the same datasets
varid_idx_lu = {
    # "pr": ["rx1day", "rx5day", "r10mm", "cwd", "cdd"],
    "pr": ["rx1day", "rx5day", "r10mm", "cwd", "cdd"],
    # "prsn": ["hsd"],
    "tasmax": ["hd", "su", "wsdi"],
    "tasmin": ["cd", "dw", "csdi"],
    # "sfcWind": ["wndd"],
}

# this is the reverse lookup, for mapping indicators to the list of CMIP6 variable ID's that are needed
idx_varid_lu = {
    "su": ["tasmax"],
    "su2": ["tasmax"],
    "dw": ["tasmin"],
    "ftc": ["tasmax", "tasmin"],
    "rx1day": ["pr"],
}

# units dict for each indicator, used for QC
# empty values are unitless indices, most likely a simple count (# of days, etc)
units_lu = {
    "rx1day": {"units": "mm"},
    "su": {},
    "dw": {},
    "ftc": {}
}

# ranges dict for each indicator, used for QC
# range references:
#rx1day: max recorded in historical record is <400mm(16") https://journals.ametsoc.org/view/journals/bams/95/8/bams-d-13-00027.1.xml#:~:text=The%20National%20Climatic%20Data%20Center,single%20calendar%2Dday%20precipitation%20amount.
ranges_lu = {
    "rx1day": {"min": 0,
               "max": 500},
    "su": {"min": 0,
            "max": 100},
    "dw": {"min": 0,
            "max": 100},
    "ftc": {"min": 0,
            "max": 100},
}