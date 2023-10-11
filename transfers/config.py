"""Config file to assist with Globus transfers. This includes lookup tables, to simplify things that were once linked between this file and a separate luts.py."""

import os
from pathlib import Path


# Globus Client ID requried to complete an OAuth2 flow to get tokens
# https://globus-sdk-python.readthedocs.io/en/stable/tutorial.html
CLIENT_ID = os.getenv("CLIENT_ID")

# the endpoint string for the Arctic Climate Data Node.
# this points to the "UAF Arctic CMIP6" collection
acdn_ep = "7235217a-be50-46ba-be31-70bffe2b5bf4"

acdn_prefix = Path("/CMIP6")

# the endpoint for the LLNL ESGF node
llnl_ep = "1889ea03-25ad-4f9f-8110-1ce8833a9d7e"

ceda_ep = "ee3aa1a0-7e4c-11e6-afc4-22000b92c261"

# path prefix for LLNL ESGF CMIP6 data
llnl_prefix = Path("/css03_data/CMIP6")

# template name for an  ESGF holdings audit table
holdings_tmp_fn = "{esgf_node}_esgf_holdings.csv"

# template name for an  ESGF holdings audit table
manifest_tmp_fn = "{esgf_node}_manifest.csv"

# name of directory in home folder for writing batch files
batch_dir = Path("batch_files")
batch_dir.mkdir(exist_ok=True)
# template name for batch files
batch_tmp_fn = "batch_{esgf_node}_{freq}_{var_id}.txt"

# names of the ScenarioMIP scenarios that we are interested in,
#  matching directory names in ESGF archives
prod_scenarios = [
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585",
]

# production models and variants to mirror
# determined using select_variants.ipynb
prod_variant_lu = {
    "CESM2": "r11i1p1f1",
    "CNRM-CM6-1-HR": "r1i1p1f2",
    "EC-Earth3-Veg": "r1i1p1f1",
    "GFDL-ESM4": "r1i1p1f1",
    "HadGEM3-GC31-LL": "r1i1p1f3",
    "HadGEM3-GC31-MM": "r1i1p1f3",
    "KACE-1-0-G": "r1i1p1f1",
    "MIROC6": "r1i1p1f1",
    "MPI-ESM1-2-LR": "r1i1p1f1",
    "MRI-ESM2-0": "r1i1p1f1",
    "NorESM2-MM": "r1i1p1f1",
    "TaiESM1": "r1i1p1f1",
    "CESM2-WACCM": "r1i1p1f1",
}

# this lookup includes all models of interest, including some that may not be transferred, left in here for compatability with any exploratory efforts
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
    "MPI-ESM1-2-LR": "MPI-M",
    "MRI-ESM2-0": "MPI-M",
    "NorESM2-MM": "NCC",
    "MPI-ESM1-2-HR": "MPI-M",
    "TaiESM1": "AS-RCEC",
    "CESM2-WACCM": "NCAR",
}

# we will just use this dict as the reference for production variables.
variables = {
    "tas": {"name": "near_surface_air_temperature", "freqs": ["Amon", "day"]},
    "tasmax": {
        "name": "maximum_near_surface_air_temperature",
        "freqs": ["Amon", "day"],
    },
    "tasmin": {
        "name": "minimum_near_surface_air_temperature",
        "freqs": ["Amon", "day"],
    },
    "pr": {"name": "precipitation", "freqs": ["Amon", "day"]},
    "psl": {"name": "sea_level_pressure", "freqs": ["Amon", "day"]},
    "huss": {"name": "near_surface_specific humidity", "freqs": ["Amon", "day"]},
    "uas": {"name": "near_surface_eastward_wind", "freqs": ["Amon", "day"]},
    "vas": {"name": "near_surface_northward_wind", "freqs": ["Amon", "day"]},
    "ta": {"name": "air_temperature", "freqs": ["Amon", "day"]},
    "ua": {"name": "eastward_wind", "freqs": ["Amon", "day"]},
    "va": {"name": "northward_wind", "freqs": ["Amon", "day"]},
    "sfcWind": {"name": "near_surface_wind_speed", "freqs": ["Amon", "day"]},
    "sfcWindmax": {"name": "maximum_near_surface_wind_speed", "freqs": ["Amon", "day"]},
    "hus": {"name": "specific_humidity", "freqs": ["Amon", "day"]},
    "evspsbl": {
        "name": "evaporation_including_sublimation_and_transpiration",
        "freqs": ["Amon", "day"],
    },
    "mrro": {"name": "total_runoff", "freqs": ["Amon", "day"]},
    "mrsos": {
        "name": "moisture_in_upper_portion_of_soil_column",
        "freqs": ["Amon", "day"],
    },
    "prsn": {"name": "snowfall_flux", "freqs": ["Amon", "day"]},
    "snd": {"name": "surface_snow_thickness", "freqs": ["Llmon", "Eday"]},
    "snw": {"name": "surface_snow_amount", "freqs": ["Amon", "day"]},
    "rlds": {
        "name": "surface_downwelling_longwave_flux_in_air",
        "freqs": ["Amon", "day"],
    },
    "rsds": {
        "name": "surface_downwelling_shortwave_flux_in_air",
        "freqs": ["Amon", "day"],
    },
    "rls": {
        "name": "surface_net_downward_longwave_flux",
        "freqs": ["Emon", "day"],
    },
    "rss": {
        "name": "surface_net_downward_shortwave_flux",
        "freqs": ["Emon", "day"],
    },
    "orog": {"name": "surface_altitude", "freqs": ["fx"]},
    "sftlf": {
        "name": "percentage_of_the_grid_cell_occupied_by_land_including_lakes",
        "freqs": ["fx"],
    },
    "sftof": {"name": "sea_area_percentage", "freqs": ["Ofx"]},
    "clt": {"name": "cloud_area_fraction", "freqs": ["Amon", "day"]},
    "tos": {"name": "sea_surface_temperature", "freqs": ["Omon", "Oday"]},
    "siconc": {"name": "sea_ice_area_fraction", "freqs": ["SImon", "SIday"]},
    "sithick": {"name": "sea_ice_thickness", "freqs": ["SImon", "SIday"]},
    "hfls": {"name": "surface_upward_latent_heat_flux", "freqs": ["Amon", "day"]},
    "hfss": {"name": "surface_upward_sensible_heat_flux", "freqs": ["Amon", "day"]},
}

globus_esgf_endpoints = {
    "llnl": {
        "ep": "1889ea03-25ad-4f9f-8110-1ce8833a9d7e",
        "prefix": "/css03_data/CMIP6",
    }
}
