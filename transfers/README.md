# CMIP6 Transfer & Download Pipeline

This folder is the "pipeline" for transferring raw [CMIP6](https://www.wcrp-climate.org/wgcm-cmip) model output data to SNAP/UAF infrastructure. The data are transferred from Lawrence Livermore National Laboratory's (LLNL) node on the decentralized database managed by the [Earth System Grid Federation](https://esgf.llnl.gov/) (ESGF). The data are transferred to SNAP's Arctic Climate Data Node (ACDN) via [Globus](https://www.globus.org/). The ACDN is mounted on the [Chinook HPC](https://uaf-rcs.gitbook.io/uaf-rcs-hpc-docs/hpc#chinook) at UAF, and so the user will need accounts on Chinook and Globus in order to initiate a transfer.

## Target CMIP6 Dataset

SNAP has worked with collaborators to identify a subset of CMIP6 data that we want to mirror on the ARDN. These "requested data" are all outputs contributed under the ScenarioMIP and CMIP activities that satisfy all possible combinations of a particular set of models, scenarios, variables, and sampling frequencies identified as most valuable by our collaborators. Requested combinations are stored in `config.py`.

These requested data cover an historical to end-of-century time period, with the historical data coming from the CMIP activity and the projected data coming from the ScenarioMIP activity. We are currently only concerned with accessing a single variant for each model, so a portion of this pipeline is dedicated to determining the variant that provides maximum coverage amongst the aformentioned combinations. The target data (i.e. the files that are actually available in current ESGF holdings) will be a subset of the requested data, as there are gaps in representation at each level (e.g. if certain model x scenario x variable x frequency combinations are missing from the holdings). The requested data combinations are subject to change as we continue to communicate with collaborators about what is available and what is important to their work.

## Strategy

The goals of this pipeline are to:
1. Identify the target data by comparing our requested data with the current holdings in ESGF using an audit.
2. Transfer that target data to the ARDN.
3. Testing that we have successfully identified and transferred the target data.

This pipeline will transfer all target data using the native ESGF directory structure:

```
<root>/<activity>/<institution>/<model>/<scenario>/<variant>/<frequency>/<variable>/<grid type>/<version>/
```

For the ACDN, the `<root>` folder is `/beegfs/CMIP6/arctic-cmip6/CMIP6/` directory on Chinook, which corresponds to the "UAF Arctic CMIP6" collection in Globus. On the LLNL's ESGF node, `<root>` is `/css03_data/CMIP6/`.

## Pipeline

### Components

Below is a list of the components in the pipeline and a short description of each.

* `batch_files/`: batch files with \<source> \<destination> filepaths for transferring files.
* `batch_transfer.py`: script to submit transfer jobs for all existing batch files, based on the Globus SDK.
* `conda_init.sh`: a shell script for initializing conda in a blank shell that does not read the typical `.bashrc`, as is the case with new slurm jobs.
* `config.py`: sets some constant variables such as the main list of models, scenarios, variables, and frequencies to transfer to ARDN.
* `esgf_holdings.py`: script to run an audit which will generate a table of CMIP6 holdings on a given ESGF node using models, scenarios, variables, and frequencies provided in `config.py`.
* `generate_batch_files.py`: script to generate the batch files of \<source> \<destination> filepaths for transferring files.
* `generate_manifest.py`: script for generating the manifest tables of files we wish to transfer to ARDN.
* `holdings_summary_wrf`: notebook exposing summary tables of data availability based on the audit results tables (for subdaily variables used in WRF only)
* `holdings_summary`: notebook exposing summary tables of data availability based on the audit results tables
* `llnl_esgf_holdings_wrf.csv`: table of data audit results for LLNL's ESGF node produced by `esgf_holdings.py`, (for subdaily variables / frequencies used in WRF only)
* `llnl_esgf_holdings.csv`: table of data audit results for LLNL's ESGF node produced by `esgf_holdings.py`.
* `llnl_manifest_wrf.csv`: table of WRF-related files to transfer to ARDN.
* `llnl_manifest.csv`: table of files to transfer to ARDN.
* `quick_ls.py`: script to run an `ls` operation on a particular Globus path.
* `select_variants.ipynb`: notebook for exploring the data available for variants of each model to determine which variant to transfer.
* `tests.slurm`: slurm script to run tests on transferred data.
* `transfer.py`: original script for running transfers, not based on the Globus SDK; allows user to supply variable name and frequency if transferring a subset of the data. 
* `utils.py`: contains functions to help with multiple scripts in this pipeline.
* `tests/`: tests for verifying that the transfer is successful.

### Running the pipeline

1. Set up Globus using the "Globus setup" section below.

2. Copy the `transfers/conda_init.sh` script to your home directory. Note that this script assumes you have `miniconda3` installed in your home directory. If not, download it from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).

3. Set the environment variables. **NOTE**: these paths are passed as arguments to `slurm` functions, which do not recognize the tilde notation (`~/`) commonly used to alias a home directory. For this reason, the entire path must be explicitly defined e.g. `/home/kmredilla/path/to/dir` instead of `~/path/to/dir`.

##### `TEST_OUT_DIR`

The path where you will write the test output files. Something like:

```sh
export TEST_OUT_DIR=/home/kmredilla/cmip6-test-outputs
```

##### `CONDA_INIT`

The path of the `conda_init.sh` script copied to your home directory.

```sh
export CONDA_INIT=/home/kmredilla/conda_init.sh
```

##### `SLURM_EMAIL`

Email address to send failed slurm job notifications to.

```sh
export SLURM_EMAIL=kmredilla@alaska.edu
```

4. Use the `esgf_holdings.py` to create tables detailing availability of requested data combinations on a particular ESGF node. Requested data combinations used in this script are sourced from `config.py`. This script creates lists of the available filenames for all requested data. To list available data on the LLNL node, run it as a script like so:

```
python esgf_holdings.py --node llnl
```

To do the same for the variables we want at non-standard freqeuncies for future WRF runs, add the `--wrf` flag:

```
python esgf_holdings.py --node llnl --wrf
```

These scripts will create `llnl_esgf_holdings.csv` and `llnl_esgf_holdings_wrf.csv`, which have been committed to version control.

5. Use the `generate_manifest.py` script generate a complete manifest of all files that will be transferred to the ACDN.

```
python generate_manifest.py --node llnl
```

To do the same for the variables we want at non-standard freqeuncies for future WRF runs, add the `--wrf` flag:

```
python generate_manifest.py --node llnl --wrf
```

These scripts will create `llnl_manifest.csv` and `llnl_manifest_wrf.csv`, which have been committed to version control.

6. Use the `generate_batch_files.py` script to generate batch files from the manifest to run the transfer in batches. Batch files will be saved to the `batch_files/` folder.

```
python generate_batch_files.py --node llnl
```

To do the same for the variables we want at non-standard freqeuncies for future WRF runs, add the `--wrf` flag:

```
python generate_batch_files.py --node llnl --wrf
```


7. Use the `batch_transfer.py` script to run the transfer from the ESGF endpoint to the ACDN endpoint using the batch files in the `batch_files/` folder.

```
python batch_transfer.py --node llnl
```

Note - while running the scripts above, there may be multiple rounds of granting Globus permissions. For example, sometimes this error will pop up:

```
The collection you are trying to access data on requires you to grant consent for the Globus CLI to access it.
message: Missing required data_access consent

Please run

  globus session consent 'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/7235217a-be50-46ba-be31-70bffe2b5bf4/data_access]'

to login with the required scopes
```

Or you may be directed to a URL to retrieve a code that you will need to enter in the command line, even if you have already logged in:

```
Please go to this URL and login:

https://auth.globus.org/v2/oauth2/authorize?client_id=a316babe-5447-43b0-a82e-4fc86c91b71a&redirect_uri=https%3A%2F%2Fauth.globus.org%2Fv2%2Fweb%2Fauth-code&scope=urn%3Aglobus%3Aauth%3Ascope%3Atransfer.api.globus.org%3Aall&state=_default&response_type=code&code_challenge=ljMPgQOa8vZ9ot5g_vR1gOuc3CBGPI1DvOEQ4QrX0yI&code_challenge_method=S256&access_type=online

Please enter the code here: 
```



8. Run the tests in the `tests/` folder by submitting the `test.slurm` job script. This will request resources for a compute node and will make sure every file in the manifest is present on the ACDN and that it opens with `xarray`.

```
sbatch tests.slurm
```


**NOTE**: If you have not already loaded the `slurm` module on your Chinook account, you will need to add the following new line to your `~/.bashrc` or `~/.bash_profile`:

```
module load slurm
```



### Globus setup

#### Globus login

Before running the pipeline, you will need to be logged into Globus and have the correct permissions enabled. Run the following at the command line, or log in via a browser at [Globus](https://www.globus.org/).

```sh
globus login
```

#### Client ID

You will also need to set the `CLIENT_ID` variable to the ID for the Globus client. 

```sh
export CLIENT_ID=a316babe-5447-43b0-a82e-4fc86c91b71a
```

This _should_ be the same between users. But if you experience any Globus client errors, follow the steps [in this tutorial](https://globus-sdk-python.readthedocs.io/en/stable/tutorial.html) to create a project in your Globus account (e.g. "UAF SNAP CMIP6 ARDN"), and create an app under that (e.g. "UAF SNAP CMIP6 ESGF Transfers"), which will provide you with a client ID for that app.

