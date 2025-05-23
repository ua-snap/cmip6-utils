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

# path prefix for LLNL ESGF CMIP6 data
llnl_prefix = Path("/css03_data/CMIP6")

# E3SM scenarioMIP data has different prefix
e3sm_prefix = Path("/user_pub_work/CMIP6")

# template name for an  ESGF holdings audit table
# wrf versions will have _wrf suffix
holdings_tmp_fn = "{esgf_node}_esgf_holdings{suffix}.csv"

# template name for an  ESGF holdings audit table
manifest_tmp_fn = "{esgf_node}_manifest{suffix}.csv"

# name of directory in home folder for writing batch files
batch_dir = Path("batch_files")
batch_dir.mkdir(exist_ok=True)
# template name for batch files
batch_tmp_fn = "batch_{esgf_node}_{table_id}_{var_id}.txt"

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
    "CNRM-CM6-1-HR": "r1i1p1f2",
    "EC-Earth3-Veg": "r1i1p1f1",
    "GFDL-ESM4": "r1i1p1f1",
    "HadGEM3-GC31-LL": "r1i1p1f3",
    "HadGEM3-GC31-MM": "r1i1p1f3",
    "KACE-1-0-G": "r1i1p1f1",
    "MIROC6": "r1i1p1f1",
    "MPI-ESM1-2-HR": "r1i1p1f1",
    "CESM2": "r11i1p1f1",
    "MRI-ESM2-0": "r1i1p1f1",
    "NorESM2-MM": "r1i1p1f1",
    "TaiESM1": "r1i1p1f1",
    # E3SM variants were chosen from looking at globus-available data in the MetaGrid app
    "E3SM-1-1": "r1i1p1f1",
    "E3SM-2-0": "r1i1p1f1",
}

# production models and grids to mirror
# determined using select_grids.ipynb
prod_grid_lu = {
    "CNRM-CM6-1-HR": ["gn", "gr"],
    "EC-Earth3-Veg": ["gn", "gr"],
    "GFDL-ESM4": "gr1",
    "HadGEM3-GC31-LL": "gn",
    "HadGEM3-GC31-MM": "gn",
    "KACE-1-0-G": "gr",
    "MIROC6": "gn",
    "MPI-ESM1-2-HR": "gn",
    "MPI-ESM1-2-LR": "gn",
    "MRI-ESM2-0": "gn",
    "NorESM2-MM": "gn",
    "TaiESM1": "gn",
    "CESM2": "gn",
    # only one E3SM grid is available
    "E3SM-1-1": "gr",
    "E3SM-2-0": "gr",
}

# this lookup includes all models of interest, including some that will NOT be transferred,
# left in here for compatability with any exploratory efforts
model_inst_lu = {
    "ACCESS-CM2": "CSIRO-ARCCSS",
    "CESM2": "NCAR",
    "CESM2-WACCM": "NCAR",
    "CNRM-CM6-1-HR": "CNRM-CERFACS",
    "EC-Earth3-Veg": "EC-Earth-Consortium",
    "GFDL-ESM4": "NOAA-GFDL",
    "HadGEM3-GC31-LL": "MOHC",
    "HadGEM3-GC31-MM": "MOHC",
    "KACE-1-0-G": "NIMS-KMA",
    "MIROC6": "MIROC",
    "MRI-ESM2-0": "MRI",
    "NorESM2-MM": "NCC",
    "TaiESM1": "AS-RCEC",
    # # Another oddity - MPI-ESM1-2-* models have different representation among the institutions, or "Institution ID":
    # # The -HR version is apparently mostly available under "DKRZ", except for the historical data which is all under "MPI-M".
    # # We will need to transfer data from both of these instiutions to have both historical and ScenarioMIP data.
    "MPI-ESM1-2-HR": [
        "MPI-M",
        "DKRZ",
    ],  # historical and ScenarioMIP data for MPI-ESM1-2-HR, respectively. Historical must be first in the list!
    # # The -LR version is mostly available under "MPI-M", but has some ssp119 data available under "DKRZ".
    # # We will only transfer from "MPI-M" in this case.
    "MPI-ESM1-2-LR": "MPI-M",
    # # We will also look for E3SM Project data
    "E3SM-1-0": "E3SM-Project",
    "E3SM-1-1": "E3SM-Project",
    "E3SM-1-1-ECA": "E3SM-Project",
    "E3SM-2-0": "E3SM-Project",
    "E3SM-2-0-NARRM": "E3SM-Project",
}

# lists of models of interest
# these are all of the models we want to check holdings for
models_of_interest = [
    "ACCESS-CM2",
    "CESM2",
    "CNRM-CM6-1-HR",
    "EC-Earth3-Veg",
    "GFDL-ESM4",
    "HadGEM3-GC31-LL",
    "HadGEM3-GC31-MM",
    "KACE-1-0-G",
    "MIROC6",
    "MRI-ESM2-0",
    "NorESM2-MM",
    "TaiESM1",
    "MPI-ESM1-2-HR",
    "CESM2-WACCM",
    "MPI-ESM1-2-LR",
]
# E3SM-1-1-NARRM not available on globus according to MetaGrid
e3sm_models_of_interest = ["E3SM-1-0", "E3SM-1-1", "E3SM-2-0", "E3SM-1-1-ECA"]

# we will just use this dict as the reference for production variables.
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
        "table_ids": ["Amon", "day"],
    },
    "snd": {"name": "surface_snow_thickness", "table_ids": ["LImon", "Eday"]},
    "snw": {"name": "surface_snow_amount", "table_ids": ["LImon", "day"]},
    "rlds": {
        "name": "surface_downwelling_longwave_flux_in_air",
        "table_ids": ["Amon", "day"],
    },
    "rlus": {
        "name": "surface_upwelling_longwave_flux_in_air",
        "table_ids": ["Amon", "day"],
    },
    "rsds": {
        "name": "surface_downwelling_shortwave_flux_in_air",
        "table_ids": ["Amon", "day"],
    },
    "rsus": {
        "name": "surface_upwelling_shortwave_flux_in_air",
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
    "tsl": {"name": "soil_temperature", "table_ids": ["Lmon", "Eday"]},
    "mlotst": {
        "name": "ocean_mixed_layer_thickness_defined_by_sigma_t",
        "table_ids": ["Omon", "Eday"],
    },
}

# This dict is for auditing WRF variables!
# It was constructed using the ESGF MetaGrid application to figure out what is available via ESGF,
#  using this table of inputs in the WRF docs to inform it:
#  https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/wps.html#required-input-for-running-wrf

# For now we will just have a separate workflow for the WRF variables at a subdaily frequency
#  instead of trying to wrap it into existing workflow for daily and monthly data.

# The first inconsistency among modeling groups to note is that
#  so if there is more than one listed in the "table_ids" list, that measn this inconsistency was seen in MetaGrid
# was originall working with only 6-hour variables, but some had more availability for 3hr

# there is a complete mess of "table IDs" used for each of these that are fairaly inconsistent across models and variables.
# The file naming scheme / directory structure is based off of this.
# so, we will provide a list of all possible subdaily table ID names, so that all may be checked every variable.

# For example, for 6hr ta (3D air temperature), the MPI-ESM1-2-* models use 6hrPlevPt and
#  CESM2 uses 6hrLev.
# MPI-ESM1-2-* models use 6hrPlev for hurs (relative humidity) and 6hrPlevPt for ta (temperature)

# here is the list of all possible subdaily table ID's
subdaily_table_ids = [
    "6hrLev",
    "6hrPlev",
    "6hrPlevPt",
    "3hr",
    "E3hr",
    "E3hrPt",
    "CF3hr",
]

# the downlscalers ended up providing a list of CMIP6 variables for WRF during development of this one.
#  So, I have commented out some variables that are mentioned in the WRF inputs spec but which were not
#   included in their list, for visibility purposes

wrf_variables = {
    "ta": {
        "name": "air_temperature",
    },
    # It doesn't looke like there is any 3d relative humidity at subdaily, but included anyway
    # "hur": {"name": "relative_humidity"}, # not listed by downscalers
    "hus": {"name": "specific_humidity"},
    "ua": {
        "name": "eastward_wind",
    },
    "va": {
        "name": "northward_wind",
    },
    "zg": {"name": "geopotential_height"},
    # air pressure - this one is out for now, only available for a couple of models which are not in out ensemble (at subdaily)
    "ps": {"name": "surface_air_pressure"},
    "psl": {"name": "sea_level_pressure"},
    # I think this is what we want for "skin temperature"
    "ts": {"name": "surface_temperature"},
    # what is "soil height"? Surface altitude, orog? might be a fixed variable, ignoring for now
    # not sure if you can get "2m temperature" from "air_temperature", but will include tas as separate for now
    # Also this one is called "air_temperature" like the 3-D version, which is confusing
    # Definitely called near surface air temp in current tas files we have for monthly
    "tas": {"name": "air_temperature"},
    # unlike the 3D version, there does appear to be data for near surface relative humidity
    # no 3hr hurs though
    # "hurs": {"name": "relative_humidity"}, # not listed by downscalers
    "huss": {"name": "specific_humidity"},
    "uas": {"name": "eastward_wind"},
    "vas": {"name": "northward_wind"},
    # there are multiple soil moisture possibilities
    # mrso, total_soil_moisture_content, is not available in subdaily, also not listed by downscalers
    # this one was not selected by downscaling group
    # "mrsos": {
    #     "name": "moisture_in_upper_portion_of_soil_column",
    # },
    "mrsol": {
        # this one has three names used for the same variable ID: moisture_content_of_soil_layer,
        #  total_water_content_of_soil_layer, and mass_content_of_water_in_soil_layer
        # mass_content_of_water_in_soil_layer is the most common one apparently, so we will go with that for now
        "name": "mass_content_of_water_in_soil_layer",
    },
    # will not have the snw and siconc variables in here as we know there is no subdaily
}

globus_esgf_endpoints = {
    "llnl": {
        "ep": llnl_ep,
        "prefix": "/css03_data/CMIP6",
    }
}

# these are full filepaths of specific files we want to add to the manifest;
# these do not get captured by our current audit workflow and so are hardcoded here
# since these are all under the css03_data/CMIP6 directory, we can use the llnl_ep endpoint
# adding the directory components directly to the manifest should be fine
add_to_manifest = [
    # sftlf
    # these are the only two models that have additional sftlf data available on the LLNL ESGF node; KACE-1-0-G has no sftlf data, and MRI-ESM2-0 directories are empty (probably not accessible without DoD permissions)
    "http://aims3.llnl.gov/thredds/dodsC/css03_data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/fx/sftlf/gn/v20190709/sftlf_fx_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn.nc",
    "http://esgf-data1.llnl.gov/thredds/dodsC/css03_data/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/piControl/r1i1p1f1/fx/sftlf/gn/v20200108/sftlf_fx_HadGEM3-GC31-MM_piControl_r1i1p1f1_gn.nc",
    # sftof
    # these are the only two models that have additional sftof data available on the LLNL ESGF node; CNRM-CM6-1-HR, GFDL-ESM4, KACE-1-0-G, TaiESM1, E3SM-1-1 and E3SM-2-0 have no sftof data, and MRI-ESM2-0 directories are empty (probably not accessible without DoD permissions)
    "http://aims3.llnl.gov/thredds/dodsC/css03_data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/Ofx/sftof/gn/v20190709/sftof_Ofx_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn.nc",
    "http://esgf-data1.llnl.gov/thredds/dodsC/css03_data/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/piControl/r1i1p1f1/Ofx/sftof/gn/v20200108/sftof_Ofx_HadGEM3-GC31-MM_piControl_r1i1p1f1_gn.nc",
]
