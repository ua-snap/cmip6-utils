"""Lookup tables for CMIP6 regridding. 
Since we are providing our config via Prefect, these were copied from the transfers/config.py file
to avoid using system environment variables."""

# batch file naming template
batch_tmp_fn = "batch_{model}_{scenario}_{frequency}_{var_id}_{grid_name}_{count}.txt"

# names of the ScenarioMIP scenarios that we are interested in,
#  matching directory names in ESGF archives
prod_scenarios = [
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585",
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
    "MRI-ESM2-0": "MRI",
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

landsea_variables = {
    "mrro": "land",
    "mrsos": "land",
    "mrsol": "land",
    "snd": "land",
    "snw": "land",
    "prsn": "land",
    "siconc": "sea",
}

# lookup for the sftlf file paths for each model, hardcoded paths for now
model_sftlf_lu = {
    "GFDL-ESM4": "/beegfs/CMIP6/arctic-cmip6/CMIP6/ScenarioMIP/NOAA-GFDL/GFDL-ESM4/ssp370/r1i1p1f1/fx/sftlf/gr1/v20180701/sftlf_fx_GFDL-ESM4_ssp370_r1i1p1f1_gr1.nc",
    "CNRM-CM6-1-HR": "/beegfs/CMIP6/arctic-cmip6/CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1-HR/historical/r1i1p1f2/fx/sftlf/gr/v20191021/sftlf_fx_CNRM-CM6-1-HR_historical_r1i1p1f2_gr.nc",
    "NorESM2-MM": "/beegfs/CMIP6/arctic-cmip6/CMIP6/CMIP/NCC/NorESM2-MM/historical/r1i1p1f1/fx/sftlf/gn/v20191108/sftlf_fx_NorESM2-MM_historical_r1i1p1f1_gn.nc",
    "TaiESM1": "/beegfs/CMIP6/arctic-cmip6/CMIP6/CMIP/AS-RCEC/TaiESM1/historical/r1i1p1f1/fx/sftlf/gn/v20200624/sftlf_fx_TaiESM1_historical_r1i1p1f1_gn.nc",
    "HadGEM3-GC31-MM": "/beegfs/CMIP6/arctic-cmip6/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/piControl/r1i1p1f1/fx/sftlf/gn/v20200108/sftlf_fx_HadGEM3-GC31-MM_piControl_r1i1p1f1_gn.nc",
    "HadGEM3-GC31-LL": "/beegfs/CMIP6/arctic-cmip6/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/fx/sftlf/gn/v20190709/sftlf_fx_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn.nc",
    "MIROC6": "/beegfs/CMIP6/arctic-cmip6/CMIP6/CMIP/MIROC/MIROC6/historical/r1i1p1f1/fx/sftlf/gn/v20190311/sftlf_fx_MIROC6_historical_r1i1p1f1_gn.nc",
    "EC-Earth3-Veg": "/beegfs/CMIP6/arctic-cmip6/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3-Veg/historical/r1i1p1f1/fx/sftlf/gr/v20211207/sftlf_fx_EC-Earth3-Veg_historical_r1i1p1f1_gr.nc",
    "CESM2": "/beegfs/CMIP6/arctic-cmip6/CMIP6/CMIP/NCAR/CESM2/historical/r11i1p1f1/fx/sftlf/gn/v20190514/sftlf_fx_CESM2_historical_r11i1p1f1_gn.nc",
    "MPI-ESM1-2-HR": "/beegfs/CMIP6/arctic-cmip6/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/fx/sftlf/gn/v20190710/sftlf_fx_MPI-ESM1-2-HR_historical_r1i1p1f1_gn.nc",
    "MRI-ESM2-0": "/beegfs/CMIP6/arctic-cmip6/CMIP6/CMIP/MRI/MRI-ESM2-0/historical/r1i1p1f1/fx/sftlf/gn/v20190603/sftlf_fx_MRI-ESM2-0_historical_r1i1p1f1_gn.nc",
    # no sftlf files for E3SM models or KACE-1-0-G
}
