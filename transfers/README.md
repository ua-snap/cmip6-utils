# CMIP6 Transfer / downloading pipelines

This folder is the "pipeline" for transferring raw CMIP6 model output data to SNAP/UAF infrastructure.

We have woreked with collaborators to identify a potential dataset that we will call the "requested data" that we want to mirror on the Arctic Climate Data Node, which are all outputs contributed under the ScenarioMIP and CMIP activities that satisfy all possible combinations of a particular set of models, scenarios, variables, and sampling frequencies (i.e. time scales) identified by our collaborators as most useful for accessing. This will be for a recent historical to end-of-century time period (approximately 1950-2100), with the historical data coming from the CMIP activity and the projected data coming from the ScenarioMIP activity. We are currently only concerned with accessing a single variant for each model, and so it will be desirable choose one such that it provides the maximal coverage amongst the aformentioned attributes we are basing the selection on. The target data, i.e. the files that are actually available, will be a subset of the requested data, as there are gaps in representation at each level. The requested data attributes are subject to change as we continue to communicate with collaborators about what is available and what is important.

## Strategy

This is kind of a tricky thing - one does not simply get all of the target data from the holdings by saying "I want these models, scenarios, and variables". The goals of this pipeline are thus:
1. Identify the target data by comparing our requested data with the current holdings
2. Mirror that target data
3. Provide some tools for auditing data on the ACDN to ensure we are successfully mirroring the target data, as well as other useful tasks, such as summarizing the data we have mirrored.

The ultimate goal of this pipeline is to mirror all target data using the native ESGF directory structure:

```
<root>/<activity>/<institution>/<model>/<scenario>/<variant>/<frequency>/<variable>/<grid type>/<version>/
```

For the ACDN, the `<root>` folder is `/CMIP6/`, which is found under the `/beegfs/CMIP6/arctic-cmip6`, and corresponds to the "UAF Arctic CMIP6" Collection in GLobus. On the LLNL ESGF node, `<root>` is `/css03_data/CMIP6/`.

## Pipeline

### Structure

Here is a description of the pipeline.

* `config.py`: sets some constant variables such as the main list of models, scenarios, and variables to mirror for our production mirror.
* `luts.py`: like `config.py`, but for lookups / dicts
* `esgf_holdings.py`: script to generate a reference table of CMIP6 holdings on a given ESGF node
* `generate_batch_files.py`: script to generate the batch files of \<source> \<destination> filepaths for transferring files
* `transfer.py`: script to execute the main transfer of all target files to be mirrored via globus
* `llnl_esgf_holdings.csv`: table of data audit results for LLNL ESGF node.  
* `batch_files/`: batch files with \<source> \<destination> filepaths for transferring files
* `tests/`: tests for verifying that the mirror is successful

### Running the pipeline

1. Set up Globus using the "Globus setup" section below
2. Use the `esgf_holdings.py` to create tables detailing availability on a particular ESGF node. This contains lists of the available filenames for all "requested" data, which is hard-coded in the python file. This may not need to be run as this file will likely be committed to version control. Run it as a script like so:

```
python esgf_holdings.py --node llnl
```

At the time of writing this, the relevant holdings at LLNL is summarized in the `llnl_esgf_holdings.csv` which has been committed to version control. 

3. Use the `generate_batch_files.py` for transferring the files from the ESGF endpoint to the ACDN endpoint. Run like so:

```
python generate_batch_files.py --node llnl
```

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
