"""Lookup tables for CMIP6 regridding. 
Since we are providing our config via Prefect, these were copied from the transfers/config.py file
to avoid using system environment variables."""

# names of the ScenarioMIP scenarios that we are interested in,
#  matching directory names in ESGF archives
prod_scenarios = [
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585",
]

# institution model strings (<institution>_<model>, from mirrored data) that we will be regridding
inst_models = [
    "NOAA-GFDL_GFDL-ESM4",
    "NIMS-KMA_KACE-1-0-G",
    "CNRM-CERFACS_CNRM-CM6-1-HR",
    "NCC_NorESM2-MM",
    "AS-RCEC_TaiESM1",
    "MOHC_HadGEM3-GC31-MM",
    "MOHC_HadGEM3-GC31-LL",
    "MIROC_MIROC6",
    "EC-Earth-Consortium_EC-Earth3-Veg",
    "NCAR_CESM2",
    "NCAR_CESM2-WACCM",
    "MPI-M_MPI-ESM1-2-LR",
]

model_inst_lu = {
    "ACCESS-CM2": "CSIRO-ARCCSS",
    "CESM2": "NCAR",
    "CNRM-CM6-1-HR": "CNRM-CERFACS",
    "EC-Earth3-Veg": "EC-Earth-Consortium",
    "GFDL-ESM4": "NOAA-GFDL",
    "HadGEM3-GC31-LL": "MOHC",
    "HadGEM3-GC31-MM": "MOHC",
    "KACE-1-0-G": "NIMS-KMA",
    "MIROC6": "MIROC",
    "MRI-ESM2-0": "MPI-M",
    "NorESM2-MM": "NCC",
    "TaiESM1": "AS-RCEC",
    "CESM2-WACCM": "NCAR",
    # Another oddity - MPI-ESM1-2-* models have different representation among the institutions, or "Institution ID".
    # the -HR version is apparently mostly available under "DKRZ". The -LR version is mostly available under "MPI-M".
    # There is apparently mixing, too, as the -HR version has historical data under "MPI-M", and the -LR version has
    #  data available under "DKRZ". We will just go with the institution which has the majority for each, for now.
    "MPI-ESM1-2-HR": "DKRZ",
    "MPI-ESM1-2-LR": "MPI-M",
}

variables = {
    "tas": {"name": "near_surface_air_temperature", "table_ids": ["Amon", "day"]},
    "ts": {"name": "surface_temperature", "table_ids": ["Amon", "day"]},
    "tasmax": {
        "name": "maximum_near_surface_air_temperature",
        "table_ids": ["Amon", "day"],
    },
    "tasmin": {
        "name": "minimum_near_surface_air_temperature",
        "table_ids": ["Amon", "day"],
    },
    "pr": {"name": "precipitation", "table_ids": ["Amon", "day"]},
    "psl": {"name": "sea_level_pressure", "table_ids": ["Amon", "day"]},
    "huss": {"name": "near_surface_specific humidity", "table_ids": ["Amon", "day"]},
    "uas": {"name": "near_surface_eastward_wind", "table_ids": ["Amon", "day"]},
    "vas": {"name": "near_surface_northward_wind", "table_ids": ["Amon", "day"]},
    "ta": {"name": "air_temperature", "table_ids": ["Amon", "day"]},
    "ua": {"name": "eastward_wind", "table_ids": ["Amon", "day"]},
    "va": {"name": "northward_wind", "table_ids": ["Amon", "day"]},
    "sfcWind": {"name": "near_surface_wind_speed", "table_ids": ["Amon", "day"]},
    "sfcWindmax": {
        "name": "maximum_near_surface_wind_speed",
        "table_ids": ["Amon", "day"],
    },
    "hus": {"name": "specific_humidity", "table_ids": ["Amon", "day"]},
    "evspsbl": {
        "name": "evaporation_including_sublimation_and_transpiration",
        "table_ids": ["Amon", "Eday"],
    },
    "mrro": {"name": "total_runoff", "table_ids": ["Lmon", "day"]},
    "mrsos": {
        "name": "moisture_in_upper_portion_of_soil_column",
        "table_ids": ["Lmon", "day"],
    },
    "mrsol": {
        "name": "moisture_in_upper_portion_of_soil_column",
        "table_ids": ["Emon", "Eday"],
    },
    "prsn": {
        "name": "snowfall_flux",
        "table_ids": ["Amon", "Omon", "day"],
    },  # some models use Omon for table ID
    "snd": {"name": "surface_snow_thickness", "table_ids": ["LImon", "Eday"]},
    "snw": {"name": "surface_snow_amount", "table_ids": ["LImon", "day"]},
    "rlds": {
        "name": "surface_downwelling_longwave_flux_in_air",
        "table_ids": ["Amon", "day"],
    },
    "rsds": {
        "name": "surface_downwelling_shortwave_flux_in_air",
        "table_ids": ["Amon", "day"],
    },
    "rls": {
        "name": "surface_net_downward_longwave_flux",
        "table_ids": ["Emon", "day"],
    },
    "rss": {
        "name": "surface_net_downward_shortwave_flux",
        "table_ids": ["Emon", "day"],
    },
    "orog": {"name": "surface_altitude", "table_ids": ["fx"]},
    "sftlf": {
        "name": "percentage_of_the_grid_cell_occupied_by_land_including_lakes",
        "table_ids": ["fx"],
    },
    "sftof": {"name": "sea_area_percentage", "table_ids": ["Ofx"]},
    "clt": {"name": "cloud_area_fraction", "table_ids": ["Amon", "day", "Eday"]},
    "tos": {"name": "sea_surface_temperature", "table_ids": ["Omon", "Oday"]},
    "siconc": {"name": "sea_ice_area_fraction", "table_ids": ["SImon", "SIday"]},
    # there is also "siconca", which has the same name, but the files are generally much smaller,
    #  so they are likely a subset os summary in some way of the siconc data
    "sithick": {"name": "sea_ice_thickness", "table_ids": ["SImon", "SIday"]},
    "hfls": {
        "name": "surface_upward_latent_heat_flux",
        "table_ids": ["Amon", "day", "Eday"],
    },
    "hfss": {
        "name": "surface_upward_sensible_heat_flux",
        "table_ids": ["Amon", "day", "Eday"],
    },
}

