# SNAP CMIP6 Utilities

This is SNAP's repo for working with CMIP6 data. Specific projects for working with these data may warrant their own repo, but for now, we can place all general CMIP6 pipelines in this repo. 

This repo contains the following pipelines:

* `transfers/`: for transferring data from ESGF to the ACDN via Globus
* `regridding/`: for getting the transferred CMIP6 data on a common grid
* `indicators/`: for computing indicators from regridded CMIP6 data

The other files currently in here are artifacts of preliminary efforts with CMIP6, which involved transferring monthly data via the ECMWF CDS API. 
