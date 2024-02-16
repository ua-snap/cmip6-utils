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

# units str for each indicator, used for QC
units_lu = {"rx1day": "mm", "su": "d", "dw": "d", "ftc": "d"}

# ranges dict for each indicator, used for QC
# range references:
# rx1day: max recorded in historical record is <400mm(16") https://journals.ametsoc.org/view/journals/bams/95/8/bams-d-13-00027.1.xml#:~:text=The%20National%20Climatic%20Data%20Center,single%20calendar%2Dday%20precipitation%20amount.
ranges_lu = {
    "rx1day": {"min": 0, "max": 500},
    "su": {"min": 0, "max": 200},
    "dw": {"min": 0, "max": 275},
    "ftc": {"min": 0, "max": 250},
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
        "references": "A list of references TBD!", #TODO: list references for indicators / sources / etc
        },
    "dw": {
        "title": "Yearly Number of Deep Winter Days (-30C threshold)",
        "long_name": "yearly_deep_winter_days_-30C",
        "description": "Number of Deep Winter Days, calculated over a yearly frequency with a daily minimum temperature threshold of -30C using xclim.indices.tn_days_below().",
        "references": "A list of references TBD!", #TODO: list references for indicators / sources / etc
        },
    "su": {
        "title": "Yearly Number of Summer Days (25C threshold)",
        "long_name": "yearly_summer_days_25C",
        "description": "Number of Summer Days, calculated over a yearly frequency with a daily maximum temperature threshold of 25C using xclim.indices.tx_days_above().",
        "references": "A list of references TBD!", #TODO: list references for indicators / sources / etc
        },
    "ftc": {
        "title": "Yearly Number of Freeze-Thaw Cycles",
        "long_name": "yearly_freeze_thaw_cycles",
        "description": "Number of Freeze Thaw Cycles, calculated over a yearly frequency using xclim.indicators.atmos.daily_freezethaw_cycles().",
        "references": "A list of references TBD!", #TODO: list references for indicators / sources / etc
        },
}

model_lu = {
    "ACCESS-CM2": {
        "institution": "CSIRO-ARCCSS",
        "institution_name": "Commonwealth Scientific and Industrial Research Organisation, Australian Research Council Centre of Excellence for Climate System Science",
    },
    "CESM2": {
        "institution": "NCAR",
        "institution_name": "National Center for Atmospheric Research, Climate and Global Dynamics Laboratory",
    },
    "CESM2-WACCM": {
        "institution": "NCAR",
        "institution_name": "National Center for Atmospheric Research, Climate and Global Dynamics Laboratory",
    },
    "CNRM-CM6-1-HR": {
        "institution": "CNRM-CERFACS",
        "institution_name": "Centre National de Recherches Meteorologiques, Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique",
    },
    "EC-Earth3-Veg": {
        "institution": "EC-Earth-Consortium",
        "institution_name":"EC-Earth, Rossby Center, Swedish Meteorological and Hydrological Institute",
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
    "MPI-ESM1-2-LR": {
        "institution": "MPI-M",
        "institution_name": "Max Planck Institute for Meteorology",
    },
    "MPI-ESM1-2-HR": {
        "institution": "MPI-M",
        "institution_name": "Max Planck Institute for Meteorology",
    },
    "MPI-ESM1-0": {
        "institution": "MPI-M",
        "institution_name": "Max Planck Institute for Meteorology",
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

scenario_lu = {
    "historical": {
        "ssp": "NA",
        "forcing_level":"NA",
    },
    "ssp126": {
        "ssp": "ssp1",
        "forcing_level": "2.6",
    },
    "ssp245": {
        "ssp": "ssp2",
        "forcing_level": "4.5",
    },
    "ssp370": {
        "ssp": "ssp3",
        "forcing_level": "7.0",
    },
    "ssp585": {
        "ssp": "ssp5",
        "forcing_level": "8.5",
    },
}