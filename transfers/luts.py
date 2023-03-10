"""Lookup tables for CMIP6 transfers"""


# format model: institution
model_inst_lu = {
    "ACCESS-CM2": {"institution": "CSIRO-ARCCSS", "variant": "r1i1p1f1"},
    "CESM2": {"institution": "NCAR", "variant": "r1i1p1f1"},
    "CNRM-CM6-1-HR": {"institution": "CNRM-CERFACS", "variant": "r1i1p1f2"},
    "EC-Earth3-Veg": {"institution": "EC-Earth-Consortium", "variant": "r1i1p1f1"},
    "GFDL-ESM4": {"institution": "NOAA-GFDL", "variant": "r1i1p1f1"},
    "HadGEM3-GC31-LL": {"institution": "MOHC", "variant": "r1i1p1f3"},
    "HadGEM3-GC31-MM": {"institution": "MOHC", "variant": "r1i1p1f3"},
    "KACE-1-0-G": {"institution": "NIMS-KMA", "variant": "r1i1p1f1"},
    "MIROC6": {"institution": "MIROC", "variant": "r1i1p1f1"},
    "MPI-ESM1-2-LR": {"institution": "MPI-M", "variant": "r10i1p1f1"},
    "MRI-ESM2-0": {"institution": "MPI-M", "variant": "r1i1p1f1"},
    "NorESM2-MM": {"institution": "NCC", "variant": "r1i1p1f1"},
}

# highest priority variables 
vars_tier1 = {
    "tas": "near_surface_air_temperature",
    "pr": "precipitation",
    "psl": "sea_level_pressure",
    "uas": "eastward_wind",
    "vas": "northward_wind",
}

# lower priority variables
vars_tier2 = {
    "evspsbl": "evaporation_including_sublimation_and_transpiration",
    "mrro": "total_runoff",
    "mrsos": "moisture_in_upper_portion_of_soil_column",
    "prsn": "snowfall_flux",
    "snd": "surface_snow_thickness",
    "snw": "surface_snow_amount",
    
}

const_variables = {
    "orog": "surface_altitude",
    "sftlf": "percentage_of_the_grid_cell_occupied_by_land_including_lakes",
    "sftof": "sea_area_percentage",
}

globus_esgf_endpoints = {
    "llnl": {
        "ep": "415a6320-e49c-11e5-9798-22000b9da45e",
        "prefix": "/css03_data/CMIP6"
    }
}