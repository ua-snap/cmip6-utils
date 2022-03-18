# initialize a lookup dict for CDS API
api_lu = {}

# full names of models (for reference at this point)
models = [
    "ACCESS-CM2",
    "CESM2",
    "CNRM-CM6-1-HR",
    "EC-Earth3-Veg-LR",
    "GFDL-ESM4",
    "HadGEM3-GC31-LL",
    "HadGEM3-GC31-MM",
    "KACE-1-0-G",
    "MIROC6",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NorESM2-MM",
]
# api names for models are simply lower case and underscores instead of hyphens
api_lu["models"] = {model: model.lower().replace("-", "_") for model in models}

# subset of models for the water balanace data
wb_models = [
    "CESM2",
    "EC-Earth3-Veg-LR",
    "GFDL-ESM4",
    "MIROC6",
    "MRI-ESM2-0",
    "NorESM2-MM",
]

scenarios = [
    "historical",
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP5-8.5",
]

# water balance scenarios
wb_scenarios = [
    "historical",
    "SSP2-4.5",
    "SSP5-8.5",
]

# lookups for API
api_lu["scenarios"] = {
    "historical": "historical",
    "SSP1-2.6": "ssp1_2_6",
    "SSP2-4.5": "ssp2_4_5",
    "SSP5-8.5": "ssp5_8_5",
}

met_varnames = ["tas", "pr", "psl"]
land_varnames = ["orog", "sftlf", "sftof"]
wb_varnames = ["evspsbl", "mrsos", "prsn", "snw", "mrro"]

# variable names
api_lu["varnames"] = {
    "tas": "near_surface_air_temperature",
    "pr": "precipitation",
    "psl": "sea_level_pressure",
    "orog": "surface_altitude",
    "sftlf": "percentage_of_the_grid_cell_occupied_by_land_including_lakes",
    "sftof": "sea_area_percentage",
    "evspsbl": "evaporation_including_sublimation_and_transpiration",
    "mrsos": "moisture_in_upper_portion_of_soil_column",
    "prsn": "snowfall_flux",
    "snw": "surface_snow_amount",
    "mrro": "total_runoff",
}
