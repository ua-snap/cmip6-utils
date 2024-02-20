# CMIP6 Transfer / downloading pipelines

This folder is the "pipeline" for transferring raw CMIP6 model output data to SNAP/UAF infrastructure.

We have worked with collaborators to identify a potential dataset that we will call the "requested data" that we want to mirror on the Arctic Climate Data Node, which are all outputs contributed under the ScenarioMIP and CMIP activities that satisfy all possible combinations of a particular set of models, scenarios, variables, and sampling frequencies (i.e. time scales) identified by our collaborators as most useful for accessing. This will be for a historical to end-of-century time period, with the historical data coming from the CMIP activity and the projected data coming from the ScenarioMIP activity. We are currently only concerned with accessing a single variant for each model, so a portion of this pipeline will be dedicated to determining the variant that provides the maximal coverage amongst the aformentioned attributes we are basing the selection on. The target data, i.e. the files that are actually available, will be a subset of the requested data, as there are gaps in representation at each level (i.e. missing model x scenario x variable x frequency combinations). The requested data attributes are subject to change as we continue to communicate with collaborators about what is available and what is important.

## Strategy

This is kind of a tricky thing - one does not simply get all of the target data from the holdings by saying "I want these models, scenarios, variables, and frequencies". The goals of this pipeline are thus:
1. Identify the target data by comparing our requested data with the current holdings in ESGF using an audit
2. Mirror that target data
3. Provide some tools for testing that we are successfully identifying and mirroring the target data

The ultimate goal of this pipeline is to mirror all target data using the native ESGF directory structure:

```
<root>/<activity>/<institution>/<model>/<scenario>/<variant>/<frequency>/<variable>/<grid type>/<version>/
```

For the ACDN, the `<root>` folder is `/beegfs/CMIP6/arctic-cmip6/CMIP6/`, which corresponds to the "UAF Arctic CMIP6" Collection in Globus. On the LLNL ESGF node, `<root>` is `/css03_data/CMIP6/`.

## Pipeline

### Structure

Here is a description of the pipeline.

* `batch_transfer.py`: script to submit transfer jobs for all existing batch files.
* `conda_init.sh`: a shell script for initializing conda in a blank shell that does not read the typical `.bashrc`, as is the case with new slurm jobs.
* `config.py`: sets some constant variables such as the **main list of models, scenarios, variables, and frequencies to mirror for our production data**.
* `esgf_holdings.py`: script to run an audit which will generate a table of CMIP6 holdings on a given ESGF node using models, scenarios, variables, and frequencies provided in `config.py`.
* `generate_batch_files.py`: script to generate the batch files of \<source> \<destination> filepaths for transferring files.
* `generate_manifest.py`: script for generating the manifest tables of files we wish to mirror on ARDANO.
* `holdings_summary_wrf`: notebook exposing summary tables of data availability based on the audit results tables (for WRF variables)
* `holdings_summary`: notebook exposing summary tables of data availability based on the audit results tables (for WRF variables)
* `llnl_esgf_holdings_wrf.csv`: table of data audit results for LLNL ESGF node produced by `esgf_holdings.py`, for the WRF variables / frequencies.
* `llnl_esgf_holdings.csv`: table of data audit results for LLNL ESGF node produced by `esgf_holdings.py`.
* `llnl_manifest_wrf.csv`: table of WRF-related files to mirror on ARDANO.
* `llnl_manifest.csv`: table of files to mirror on ARDANO.
* `quick_ls.py`: script to run an `ls` operation on a particular Globus path.
* `select_variants.ipynb`: notebook for exploring the data available for variants of each model to determine which one to mirror.
* `tests.slurm`: slurm script to run tests on mirrored data.
* `transfer.py`: original script for running transfers, not based on the Globus SDK like `batch_transfer.py` is, and allows user to supply variable name and frequency if running a subset is of interest. 
* `utils.py`: contains functions to help with multiple scripts in this pipeline.
* `batch_files/`: batch files with \<source> \<destination> filepaths for transferring files.
* `tests/`: tests for verifying that the mirror is successful.

### Running the pipeline

1. Set up Globus using the "Globus setup" section below

2. Copy the `transfers/conda_init.sh` script to your home directory. Note that this script assumes you have `miniconda3` installed in your home directory.

3. Set the environment variables. **NOTE**: these paths are passed as arguments to `slurm` functions, which do not recognize the tilde notation (`~/`) commonly used to alias a home directory. For this reason, the entire path must be explicitly defined e.g. `/home/kmredilla/path/to/dir` instead of `~/path/to/dir`.

##### `TEST_OUT_DIR`

This should be set to the path where you will write the test output files. Something like:

```sh
export TEST_OUT_DIR=/home/kmredilla/cmip6-test-outputs
```

##### `CONDA_INIT`

This should be set to the path of the `conda_init.sh` script copied to your home directory.

```sh
export CONDA_INIT=/home/kmredilla/conda_init.sh
```

##### `SLURM_EMAIL`

Email address to send failed slurm job notifications to. Honestly not sure if this is working on Chinook04 currently.

```sh
export SLURM_EMAIL=kmredilla@alaska.edu
```

4. Use the `esgf_holdings.py` to create tables detailing availability on a particular ESGF node. This contains lists of the available filenames for all "requested" data, which is hard-coded in the python file. (The relevant holdings at LLNL are summarized in the `llnl_esgf_holdings.csv` which has been committed to version control.) Run it as a script like so:

```
python esgf_holdings.py --node llnl
```

To do the same for the variables we want at non-standard freqeuncies for future WRF runs, add the `--wrf` flag:

```
python esgf_holdings.py --node llnl --wrf
```

This will create `llnl_esgf_holdings_wrf.csv`.

5. Use the `generate_manifest.py` script generate a complete manifest of all files to be mirrored on the ACDN. (The manifest has also been committed to version control for convenience.)

```
python generate_manifest.py --node llnl
```

To do the same for the variables we want at non-standard freqeuncies for future WRF runs, add the `--wrf` flag:

```
python generate_manifest.py --node llnl --wrf
```

This will create `llnl_manifest_wrf.csv`.

6. Use the `generate_batch_files.py` script to generate batch files from the manifest to run the transfer in batches.

```
python generate_batch_files.py --node llnl
```

To do the same for the variables we want at non-standard freqeuncies for future WRF runs, add the `--wrf` flag:

```
python generate_batch_files.py --node llnl --wrf
```

This will create additional batch files in the `batch_files/` folder for subdaily frequencies (table ID's).


7. Use the `batch_transfer.py` script to run the transfer from the ESGF endpoint to the ACDN endpoint using the batch files.

```
python batch_transfer.py --node llnl
```

Note - there may be multiple rounds of granting globus permissions/consents. For example, sometimes this error will pop up, maybe if it's the first time in a while?

```
The collection you are trying to access data on requires you to grant consent for the Globus CLI to access it.
message: Missing required data_access consent

Please run

  globus session consent 'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/7235217a-be50-46ba-be31-70bffe2b5bf4/data_access]'

to login with the required scopes
```

8. Run the tests in the `tests/` folder by submitting the `test.slurm` job script. This will request resoruces for a compute node and will make sure every file in the manifest is present on the ACDN and that it opens with `xarray`.

```
sbatch tests.slurm
```


**NOTE**: If you have not already loaded the `slurm` module on your Chinook account, you will need to add the following new line to your `~/.bashrc` or `~/.bash_profile`:

```
module load slurm
```




### Globus setup

#### Globus login

Before running the pipeline, you will need to be logged into Globus and have the correct permissions enabled. To do so, run `globus login` at the command line (logging in via a browser before hand may work as well)

#### Client ID

You will also need to set the `CLIENT_ID` variable to the ID for the Globus client. Not yet sure whether this would be the same between users, which is why it is being specified as an env var. To get this client ID, I followed the steps [in this tutorial](https://globus-sdk-python.readthedocs.io/en/stable/tutorial.html), where I created a "project" under my globus account (called it UAF SNAP CMIP6 ARDANO), and created an app under that called "UAF SNAP CMIP6 ESGF Transfers", which allowed me to get the client ID for that app. That is:

```
a316babe-5447-43b0-a82e-4fc86c91b71a
```
