# CMIP6 Transfer / downloading pipelines

This folder is the "pipeline" for transferring raw CMIP6 model output data to SNAP/UAF infrastructure.

We can identify a potential dataset that we will call the "requested data" that we want to mirror on the Arctic Climate Data Node which are all model outputs contributed under the ScenarioMIP and CMIP activities that satisfy all possible combinations of a particular set of models, scenarios, variables, and sampling frequencies (i.e. time scales) identified by our collaborators as most useful for accessing. This will be for a recent historical to end-of-century time period (approximately 1950-2100), with the historical data coming from the CMIP activity and the projected data coming from the ScenarioMIP activity. We are only concerned with accessing a single variant for now, and so it will be desirable choose one such that it provides the maximal coverage amongst the aformentioned attributes we are basing the selection on. Of course, the target data will will be a subset of the requested data, as there are gaps in representation at each level. The requested data attributes are subject to change after more discussion with our stakeholders and gaining a better understanding of what is actually available.

This is kind of a tricky thing - one does not simply get all of the target data from the holdings by saying "I want these models, scenarios, and variables". The goals of this pipeline are thus:
1. Identify the target data by comparing our requested data with the current holdings
2. Mirror that target data
3. Provide some tools for auditing our own data to ensure we are successfully mirroring the target data, as well as other useful tasks, such as summarizing the data we have mirrored 

## Pipeline

### Structure

The pipeline is structured as follows:

1. Set up Globus using the "Globus setup" section below
2. Use the `esgf_holdings.py` to create tables detailing availability on a particular ESGF node. This contains lists of the available filenames for all "requested" data, which is hard-coded in the python file. This may not need to be run as this file will likely be committed to version control. Run it as a script like so:

```
python esgf_holdings.py --node llnl
```

3. Create "batch files" for transferring the files from the ESGF endpoint to the ACDN endpoint. 
4. Use the `transfer.py` script to run the transfer using the batch files. 

### Globus setup

#### Globus login

Before running the pipeline, you will need to be logged into Globus and have the correct permissions enabled. To do so, run `globus login` at the command line (logging in via a browser before hand may work as well)

#### Client ID

You will also need to set the `CLIENT_ID` variable to the ID for the Globus client. Not yet sure whether this would be the same between users, which is why it is being specified as an env var. To get this client ID, I followed the steps [in this tutorial](https://globus-sdk-python.readthedocs.io/en/stable/tutorial.html), where I created a "project" under my globus account (called it UAF SNAP CMIP6 ARDANO), and created an app under that called "UAF SNAP CMIP6 ESGF Transfers", which allowed me to get the client ID for that app. That is:

```
a316babe-5447-43b0-a82e-4fc86c91b71a
```

#### Endpoint authorization

Finally, for any external Globus endpoints you will be working with (e.g. the LLNL ESGF node), you will need to log in to get the right permissions. This may be possible via the command line / SDK, but it is not implemented here. You must navigate to the Globus endpoint in the web app, and log in from there. You need to have an account created for that ESGF node to do so prior. 
