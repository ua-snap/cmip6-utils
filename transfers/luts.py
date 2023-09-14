"""Lookup tables for CMIP6 transfers"""

# lookup for the variants to mirror in production
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

varname_lu = {
    "tas": "near_surface_air_temperature",
    "tasmax": "maximum_near_surface_air_temperature",
    "tasmin": "minimum_near_surface_air_temperature",
    "pr": "precipitation",
    "psl": "sea_level_pressure",
    "huss": "near_surface_specific humidity",
    "uas": "near_surface_eastward_wind",
    "vas": "near_surface_northward_wind",
    "ta": "air_temperature",
    "ua": "eastward_wind",
    "va": "northward_wind",
    "sfcWind": "near_surface_wind_speed",
    "sfcWindmax": "maximum_near_surface_wind_speed",
    "hus": "specific_humidity",
    "evspsbl": "evaporation_including_sublimation_and_transpiration",
    "mrro": "total_runoff",
    "mrsos": "moisture_in_upper_portion_of_soil_column",
    "prsn": "snowfall_flux",
    "snd": "surface_snow_thickness",
    "snw": "surface_snow_amount",
    "rlds": "surface_downwelling_longwave_flux_in_air",
    "rsds": "surface_downwelling_shortwave_flux_in_air",
    "orog": "surface_altitude",
    "sftlf": "percentage_of_the_grid_cell_occupied_by_land_including_lakes",
    "sftof": "sea_area_percentage",
    "clt": "cloud_area_fraction",
    "sot": "sea_surface_temperature",
    "sic": "sea_ice_area_fraction",
    "zmlo": "ocean_mixed_layer_thickness",
    "hfls": "surface_upward_latent_heat_flux",
    "hfss": "surface_upward_sensible_heat_flux",
    "rsntp": "net_downward_shortwave_flux_in_air",
    "rlntp": "net_upward_longwave_flux_in_air",
}

globus_esgf_endpoints = {
    "llnl": {
        "ep": "1889ea03-25ad-4f9f-8110-1ce8833a9d7e",
        "prefix": "/css03_data/CMIP6",
    }
}
