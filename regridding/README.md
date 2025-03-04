# Regrid CMIP6 on ACDN

This directory is used for regridding the CMIP6 data mirrored on the Arctic Climate Data Node. This pipeline will regrid the specified set of models, scenarios, variables, and frequencies (e.g. temporal resolutions) to the grid of some specified file (referred to as the target grid file). This includes cropping the extent of regridded outputs to match the extent of the target file if it is of a larger extent.

Note - this pipeline was previously used for a single fixed grid - that used by the NCAR CESM2 model, which is also shared by a few of the other chosen models (TaiESM1, NorESM2-MM).

This pipeline also crops these datasets to a pan-arctic domain of 50N - 90N. 

## Regridding pipeline

### Structure

Here is a description of the pipeline.

* `conda_init.sh`: a shell script for initializing conda in a blank shell that does not read the typical `.bashrc`, as is the case with new slurm jobs.
* `config.py`: sets some constant variables.
* `explore_grids.ipynb`: notebook for initial exploration of the grids found in the mirrored data, look here for rationale on chosen common grid. 
* `explore_regrid.py`: notebook for exploring the regridded data using interactive viz tools. 
* `explore_regridding.ipynb`: notebook for exploring the process of regridding a file using `xesmf` package.
* `generate_batch_files.py`: script to generate the batch files in a single slurm job.
* `get_min_max.py`: script to extract CMIP6 variable min/max values across all source files for a particular variable.
* `get_min_max.sh`: script to call the corresponding python script for each variable listed. 
* `qc.ipynb`: quality control notebook for evaluating a sample of regridded files visually. 
* `regrid_cmip6.ipynb`: notebook for running the regridding of mirrored data.
* `regrid.py`: main worker script for regridding a set of files. 
* `slurm.py`: module with utilities for working with slurm.
* `tests.slurm`: main script for running the tests from within in a slurm job. 
* `write_get_min_max_all_variables_script.py`: helper script for updating the `get_min_max.sh` script, useful if there are many more variables added to the mirrored dataset. 

### Running the pipeline

1. Copy the `regridding/conda_init.sh` script to your home directory. Note that this script assumes you have `miniconda3` installed in your home directory. (If you have already performed the `transfers/` pipeline, you will already have this file and can skip this step.)

2. Set the environment variables. **NOTE**: these paths are passed as arguments to `slurm` functions, which do not recognize the tilde notation (`~/`) commonly used to alias a home directory. For this reason, the entire path must be explicitly defined e.g. `/home/kmredilla/path/to/dir` instead of `~/path/to/dir`.

##### `SCRATCH_DIR`

This should be set to the path where you will write the regridded files initially. Something like:

```sh
export SCRATCH_DIR=/center1/CMIP6/kmredilla/cmip6_regridding
```

##### `PROJECT_DIR`

This should be set to the path of the `cmip6-utils` repo on the system. This is done for referencing scripts and constants as an alternative to adding this path to the `PYTHONPATH` env var. E.g.:

```sh
export PROJECT_DIR=/home/kmredilla/repos/cmip6-utils
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

3. Generate batch files

Here we will generate text files containing batches of CMIP6 filepaths that have a common grid to be worked on in a given slurm job. 

Execute the `generate_batch_files.py` script to create these files. This could take a bit of time, as all of the grid information is being read in and grouped for each of the CMIP6 files. 

```sh
python generate_batch_files.py
```

4. Regrid the data

Next, use the `regrid_cmip6.ipynb` to orchestrate the slurm jobs which will regrid all CMIP6 files listed in the batch files created in the previous step. Follow the text in the notebook for instructions on running this step. **NOTE**: If you have not already loaded the `slurm` module on your Chinook account, you will need to add the following new line to your `~/.bashrc` or `~/.bash_profile`:

```
module load slurm
```

5. Use the `qc.ipynb` notebook to visually check a sample of the regridded data.

6. Run the `get_min_max.sh` script, then run the regridding test suite by executing `sbatch tests.slurm`. This will test all regridded files by variable and ensure that the regridded data falls within a tolerance of the minimums and maximums. The first script will extract the minimum and maximum values from the files to be regridded and write them in a `tmp/` directory for comparison. Note - if you have added new variables that are not yet in `get_min_max.sh`, you need to add those variables in the script or run the `write_get_min_max_all_variables_script.py` which will do so automatically.

7. Copy the regridded data off scratch space

Now, copy the regridded data off of scratch space to a permanent location. For now, this will be `/beegfs/CMIP6/arctic-cmip6/regrid`. It is recommended to do this in a `screen` session as it could take a while. **NOTE**: you may need to iterate by model or some other grouping factor in case of weird errors that can happen with `rsync` failing (contacted RCS about these but unresolved). Things seem to work better if you run `rsync` on something like each model's directory.

```sh
rsync -av $SCRATCH_DIR/regrid /beegfs/CMIP6/arctic-cmip6/
```

8. Ensure the permissions are set

Make sure that the permissions are set correctly for this. A sane strategy would be to set directories to 755 (drwxr-xr-x) and set the files to 644 (-rw-r--r--). This can be done with:

```sh
# directories
find /beegfs/CMIP6/arctic-cmip6/regrid -type d -exec chmod 755 {} \;

# files
find /beegfs/CMIP6/arctic-cmip6/regrid -type f -exec chmod 644 {} \;
```
