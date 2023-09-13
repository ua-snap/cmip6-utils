"""Config file to assist with Globus transfers"""

import os
from pathlib import Path
import globus_sdk
import luts


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
batch_tmp_fn = "batch_{esgf_node}_{freq}_{varname}.txt"

# Production mirror model list
#  assumed to be interested in this entire list of models for all temporal frequencies, etc.
prod_models = [
    "CESM2",
    "CNRM-CM6-1-HR",
    "EC-Earth3-Veg",
    "GFDL-ESM4",
    "HadGEM3-GC31-LL",
    "HadGEM3-GC31-MM",
    "KACE-1-0-G",
    "MIROC6",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NorESM2-MM",
    "TaiESM1",
    "CESM2-WACCM",
]

# names of the ScenarioMIP scenarios that we are interested in,
#  matching directory names in ESGF archives
prod_scenarios = [
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585",
]

# production variables
# This list should contain all of the variables in the specs doc
# Current workflow assumes we will request all combinations prod_vars x prod_freqs
prod_vars = [
    "tas",
    "tasmax",
    "tasmin",
    "pr",
    "psl",
    "huss",
    "uas",
    "vas",
    "ta",
    "ua",
    "va",
    "hus",
    "evspsbl",
    "mrro",
    "mrsos",
    "prsn",
    "sfcWind",
    "sfcWindmax",
    "snd",
    "snw",
    "rlds",
    "rsds",
    "clt",
    "sot",
    "sic",
    "zmlo",
    "hfls",
    "hfss",
    "rsntp",
    "rlntp",
]

# make sure that all production variables have a long name lookup in the lut
try:
    missing_vars = [v for v in prod_vars if v not in list(luts.varname_lu.keys())]
    assert len(missing_vars) == 0
except AssertionError:
    print(
        f"One or more production variable IDs is missing long name in lut: {missing_vars}"
    )


# constant variables for each model
prod_const_vars = ["orog", "sftlf", "sftof"]

prod_freqs = ["day", "Amon"]
