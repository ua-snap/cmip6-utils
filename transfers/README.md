# CMIP6 Transfer / downloading pipelines

This folder will contain pipelines transferring raw CMIP6 data to SNAP/UAF infrastructure. 

## Pipeline

This "pipeline" is being set up to transfer all desired CMIP6 data from ESGF nodes to the Arctic Climate Data Node. 

It is designed to be run from an Atlas compute node. 

1. follow steps 1-4 in the Globus CLI section below to set up your environment for using the CLI.
2. Use the `esgf_holdings.py` to create tables detailing availability on a particular ESGF node, which contains lists of the available filenames for all desired daily data, which is hard-coded in the python file. This may not need to be run as this file will likely be committed to version control. 


## Globus SDK

This codebase makes use of the Globus SDK for transferring CMIP6 data from ESGF nodes to SNAP's Arctic Climate Data Node. This simplifies things over using the Globus CLI, and functions within the various scripts of this transfer pipline will prompt you for the various Globus logins and such when required. 

This will require the following environment variables:

#### `CLIENT_ID`

Set the `CLIENT_ID` variable to the ID for the Globus client. Not yet sure whether this would be the same between users, which is why it is being specified as an env var. E.g., the client ID I (Kyle) created is:

```
teWjV6sgjo0gSDFQMyVfCfDijIDdoo
```



