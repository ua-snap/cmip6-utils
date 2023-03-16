"""Config file to assist with Globus transfers"""

import os
from pathlib import Path
import globus_sdk


# Globus Client ID requried to complete an OAuth2 flow to get tokens
# https://globus-sdk-python.readthedocs.io/en/stable/tutorial.html
CLIENT_ID = os.getenv("CLIENT_ID")

# names of the ScenarioMIP scenarios that we are interested in, matching directory names in ESGF archives
scenarios = [
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp585",
]

# the endpoint string for the Arctic Climate Data Node.
# this points to the "UAF Arctic CMIP6" collection
acdn_ep = "7235217a-be50-46ba-be31-70bffe2b5bf4"

acdn_prefix = Path("/CMIP6")

# the endpoint for the LLNL ESGF node
llnl_ep = "415a6320-e49c-11e5-9798-22000b9da45e"

ceda_ep = "ee3aa1a0-7e4c-11e6-afc4-22000b92c261"

# path prefix for LLNL ESGF CMIP6 data
llnl_prefix = Path("/css03_data/CMIP6")
