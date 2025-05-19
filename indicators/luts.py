# map from model variable names to possible index variable names
# helps with possible optimization of computing indicators that rely on the same datasets
varid_idx_lu = {
    "pr": ["rx1day", "rx5day", "r10mm", "cwd", "cdd"],
    "tasmax": ["hd", "su", "wsdi"],
    "tasmin": ["cd", "dw", "csdi"],
}

# this is the reverse lookup, for mapping indicators to the list of CMIP6 variable ID's that are needed
idx_varid_lu = {
    "su": ["tasmax"],
    "dw": ["tasmin"],
    "ftc": ["tasmax", "tasmin"],
    "rx1day": ["pr"],
    "rx5day": ["pr"],
    "r10mm": ["pr"],
    "cdd": ["pr"],
    "cwd": ["pr"],
    "hd": ["tasmax"],
    "cd": ["tasmin"],
}

# units str for each indicator, used for QC
units_lu = {
    "rx1day": "mm",
    "rx5day": "mm",
    "r10mm": "days",
    "cdd": "days",
    "cwd": "days",
    "su": "d",
    "dw": "d",
    "ftc": "d",
    "hd": "degrees C",
    "cd": "degrees C",
}

# ranges dict for each indicator, used for QC
# range references:
# rx1day: max recorded in historical record is <400mm(16") https://journals.ametsoc.org/view/journals/bams/95/8/bams-d-13-00027.1.xml#:~:text=The%20National%20Climatic%20Data%20Center,single%20calendar%2Dday%20precipitation%20amount.
# rx5day: use 5 times the rx1day value
# r10mm: Yakutat is the rainiest place in AK, and ~7 months of the year (~210 days) would average >10mm of rain per day https://en.wikipedia.org/wiki/Yakutat,_Alaska#Climate
# hd: highest temp recorded in the Arctic is 38C https://wmo.int/media/news/wmo-recognizes-new-arctic-temperature-record-of-380c#:~:text=A%20temperature%20of%2038%C2%B0,World%20Meteorological%20Organization%20(WMO).
# cd: lowest temp ever recorded in the northern hemisphere is -69.6C https://wmo.int/asu-map?map=Temp_005#:~:text=Discussion,%25C2%25B0c%2Dgr%E2%80%A6

ranges_lu = {
    "rx1day": {"min": 0, "max": 500},
    "rx5day": {"min": 0, "max": 2500},
    "r10mm": {"min": 0, "max": 250},
    "cdd": {"min": 0, "max": 365},
    "cwd": {"min": 0, "max": 365},
    "su": {"min": 0, "max": 200},
    "dw": {"min": 0, "max": 275},
    "ftc": {"min": 0, "max": 250},
    "hd": {"min": 0, "max": 45},
    "cd": {"min": -80, "max": 20},
}

# lookup table of frequencies by variable id
# copied from the transfers/config.py
varid_freqs = {
    "tas": ["day"],
    "tasmax": ["day"],
    "tasmin": ["day"],
    "pr": ["day"],
    "psl": ["day"],
    "huss": ["day"],
    "uas": ["day"],
    "vas": ["day"],
    "ta": ["day"],
    "ua": ["day"],
    "va": ["day"],
    "sfcWind": ["day"],
    "sfcWindmax": ["day"],
    "hus": ["day"],
    "evspsbl": ["day"],
    "mrro": ["day"],
    "mrsos": ["day"],
    "prsn": ["day"],
    "snd": ["Eday"],
    "snw": ["day"],
    "rlds": ["day"],
    "rsds": ["day"],
    "rls": ["day"],
    "rss": ["day"],
    "orog": ["fx"],
    "sftlf": ["fx"],
    "sftof": ["Ofx"],
    "clt": ["day"],
    "tos": ["Oday"],
    "siconc": ["SIday"],
    "sithick": ["SIday"],
    "hfls": ["day"],
    "hfss": ["day"],
}


#### Lookup tables for writing metadata to .nc file attributes

indicator_lu = {
    "rx1day": {
        "title": "Yearly Maxmimum 1-day Precipitation",
        "long_name": "yearly_maximum_1_day_precipitation",
        "description": "Maxmimum 1-day Precipitation, calculated over a yearly frequency using xclim.indices.max_n_day_precipitation_amount().",
    },
    "rx5day": {
        "title": "Yearly Maximum 5-day Precipitation",
        "long_name": "yearly_maximum_5_day_precipitation",
        "description": "Maximum 5-day Precipitation, calculated over a yearly frequency using xclim.indices.max_n_day_precipitation_amount().",
    },
    "r10mm": {
        "title": "Yearly Number of Days with Precipitation >= 10mm",
        "long_name": "yearly_days_with_precipitation_10mm",
        "description": "Number of Days with Precipitation >= 10mm, calculated over a yearly frequency using xclim.indices._threshold.tg_days_above().",
    },
    "cdd": {
        "title": "Yearly Number of Consecutive Days with Precipitation < 1mm",
        "long_name": "yearly_consecutive_dry_days",
        "description": "Number of Consecutive Days with Precipitation < 1mm, calculated over a yearly frequency using xclim.indices.maximum_consecutive_dry_days().",
    },
    "cwd": {
        "title": "Yearly Number of Consecutive Days with Precipitation > 1mm",
        "long_name": "yearly_consecutive_wet_days",
        "description": "Number of Consecutive Days with Precipitation > 1mm, calculated over a yearly frequency using xclim.indices.maximum_consecutive_wet_days().",
    },
    "dw": {
        "title": "Yearly Number of Deep Winter Days (-30C threshold)",
        "long_name": "yearly_deep_winter_days_-30C",
        "description": "Number of Deep Winter Days, calculated over a yearly frequency with a daily minimum temperature threshold of -30C using xclim.indices.tn_days_below().",
    },
    "su": {
        "title": "Yearly Number of Summer Days (25C threshold)",
        "long_name": "yearly_summer_days_25C",
        "description": "Number of Summer Days, calculated over a yearly frequency with a daily maximum temperature threshold of 25C using xclim.indices.tx_days_above().",
    },
    "ftc": {
        "title": "Yearly Number of Freeze-Thaw Cycles",
        "long_name": "yearly_freeze_thaw_cycles",
        "description": "Number of Freeze Thaw Cycles, calculated over a yearly frequency using xclim.indicators.atmos.daily_freezethaw_cycles().",
    },
    "hd": {
        "title": "Hot Day Threshold",
        "long_name": "hot_day_threshold",
        "description": "the highest observed daily maximum 2m air temperature such that there are 5 other observations equal to or greater than this value.",
    },
    "cd": {
        "title": "Cold Day Threshold",
        "long_name": "cold_day_threshold",
        "description": "the lowest observed daily minimum 2m air temperature such that there are 5 other observations equal to or less than this value.",
    },
}

model_lu = {
    "CESM2": {
        "institution": "NCAR",
        "institution_name": "National Center for Atmospheric Research, Climate and Global Dynamics Laboratory",
    },
    "CNRM-CM6-1-HR": {
        "institution": "CNRM-CERFACS",
        "institution_name": "Centre National de Recherches Meteorologiques, Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique",
    },
    "E3SM-1-1": {
        "institution": "U.S. Department of Energy",
        "institution_name": "U.S. Department of Energy, Office of Science, Office of Biological and Environmental Research Earth Systems Model Development Program area of Earth and Environmental System Modeling",
    },
    "E3SM-2-0": {
        "institution": "U.S. Department of Energy",
        "institution_name": "U.S. Department of Energy, Office of Science, Office of Biological and Environmental Research Earth Systems Model Development Program area of Earth and Environmental System Modeling",
    },
    "EC-Earth3-Veg": {
        "institution": "EC-Earth-Consortium",
        "institution_name": "EC-Earth, Rossby Center, Swedish Meteorological and Hydrological Institute",
    },
    "GFDL-ESM4": {
        "institution": "NOAA-GFDL",
        "institution_name": "National Oceanic and Atmospheric Administration, Geophysical Fluid Dynamics Laboratory",
    },
    "HadGEM3-GC31-LL": {
        "institution": "MOHC",
        "institution_name": "Met Office Hadley Centre for Climate Science and Services",
    },
    "HadGEM3-GC31-MM": {
        "institution": "MOHC",
        "institution_name": "Met Office Hadley Centre for Climate Science and Services",
    },
    "KACE-1-0-G": {
        "institution": "NIMS-KMA",
        "institution_name": "National Institute of Meteorological Sciences/Korea Meteorological Administration, Climate Research Division",
    },
    "MIROC6": {
        "institution": "MIROC",
        "institution_name": "Japan Agency for Marine-Earth Science and Technology; Atmosphere and Ocean Research Institute, The University of Tokyo; National Institute for Environmental Studies; RIKEN Center for Computational Science",
    },
    "MPI-ESM1-2-HR": {
        "institution": "MPI-M",
        "institution_name": "Max Planck Institute for Meteorology",
    },
    "MRI-ESM2-0": {
        "institution": "MRI",
        "institution_name": "Meteorological Research Institute",
    },
    "NorESM2-MM": {
        "institution": "NCC",
        "institution_name": "NorESM Climate Modeling Consortium",
    },
    "TaiESM1": {
        "institution": "AS_RCEC",
        "institution_name": "Research Center for Environmental Changes, Academia Sinica",
    },
}
