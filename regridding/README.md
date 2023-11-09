# Regrid CMIP6 on ACDN

This directory is used for regridding the CMIP6 data mirrored on the ACDN to a common grid. The target grid is the NCAR CESM2 grid, which is also shared by a number of the other chosen models.

This pipeline also crops these datasets to a pan-arctic domain of 50N - 90N. 

## Regridding pipeline

### Structure

Here is a description of the pipeline.

* `config.py`: sets some constant variables.
* `crop_non_regrid.py`: crops the files which were not regridded, and fixes the time axis to be consistent with the regridded schema.
* `explore_grids.ipyn;b`: notebook for initial exploration of the grids found in the mirrored data, look here for rationale on chosen common grid. 
* `explore_regrid.py`: notebook for exploring the regridded data using interactive viz tools. 
* `explore_regridding.ipynb`: notebook for exploring the process of regridding a file using `xesmf` package.
* `generate_batch_files.py`: script to generate the batch files of files to be regridded in a single slurm job.
* `get_min_max.py`: script to extract CMIP6 variable min/max values across all source files for a particular variable.
* `get_min_max.sh`: script to call the corresponding python script for each variable listed. 
* `qc.ipynb`: quality control notebook for evaluating a sample of regridded files visually. 
* `regrid_cmip6.ipynb`: notebook for running the regridding of mirrored data.
* `regrid.py`: main worker script for regridding a set of files. 
* `slurm.py`: module with utilities for working with slurm.
* `tests.slurm`: main script for running the tests from within in a slurm job. 
* `write_get_min_max_all_variables_script.py`: helper script for updating the `get_min_max.sh` script, useful if there are many more variables added to the mirrored dataset. 

### Running the pipelin

1. Set the environment variables

First, define the following environment variables (assuming you have activated the cmip6-utils conda environment):

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

This should be a shell script for initializing conda in a blank shell that does not read the typical `.bashrc`, as is the case with new slurm jobs.

It should look like this, with the variable `CONDA_PATH` below being the path to parent folder of your conda installation, e.g. `/home/UA/kmredilla/miniconda3`:

```sh
__conda_setup="$('$CONDA_PATH/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
        . "$CONDA_PATH/etc/profile.d/conda.sh"
    else
        export PATH="$CONDA_PATH/bin:$PATH"
    fi
fi
unset __conda_setup
```

##### `SLURM_EMAIL`

Email address to send failed slurm job notifications to. Honestly not sure if this is working on Chinook04 currently.

2. Generate batch files

Here we will generate text files containing batches of CMIP6 filepaths that have a common grid to be worked on in a given slurm job. 

Execute the `generate_batch_files.py` script to create these files:

```sh
python generate_batch_files.py
```

This could take a bit of time, as all of the grid information is being read in and grouped for each of the CMIP6 files. 

3. Regrid the data

Next, use the `regrid_cmip6.ipynb` to orchestrate the slurm jobs which will regrid all CMIP6 files listed in the batch files created in step 2. Follow the text in the notebook for instructions on running this step. 

4. Use the `qc.ipynb` notebook to visually check a sample of the regridded data.

5. Run the `get_min_max.sh` script, then run the regridding test suite by executing `sbatch tests.slurm`. This will test all regridded files by variable and ensure that the regridded data falls within a tolerance of the minimums and maximums. The first script will extract the minimum and maximum values from the files to be regridded and write them in a `tmp/` directory for comparison. Note - if you have added new variables that are not yet in `get_min_max.sh`, you need to add those variables in the script or run the `write_get_min_max_all_variables_script.py` which will do so automatically.

6. Copy the regridded data off scratch space

Now, copy the regridded data off of scratch space to a permanent location. For now, this will be `/beegfs/CMIP6/arctic-cmip6/regrid`. This can be achieved with:

```sh
rsync -av $SCRATCH_DIR/regrid /beegfs/CMIP6/arctic-cmip6/
```

It is recommended to do this in a `screen` session as it could take a while. 

Note - you may need to iterate by model or some other grouping factor in case of weird errors that can happen with rsync failing (contacted RCS about these but unresolved). Things seem to work better if you run `rsync` on something like each model's directory. 

7. Crop the files not slated for regridding

This step will crop the files which already have the correct spatial grid to a panarctic extent and adjust the time dimension as needed.
It might be best to make sure the output directory for this is clear (e.g. in case it is the same as the regridding output directory).

```sh
python crop_non_regrid.py
```

8. Run a test on the cropped data

This test just ensures that the cropped data really do have the target grid. 

```sh
python -m pytest tests/test_cropping.py
```

### 9. Copy cropped data

Again, something like:

```sh
rsync -av $SCRATCH_DIR/regrid /beegfs/CMIP6/arctic-cmip6/
```

10. Ensure the permissions are set

Make sure that the permissions are set correctly for this. A sane strategy would be to set directories to 755 (drwxr-xr-x) and set the files to 644 (-rw-r--r--). This can be done with:

```sh
# directories
find /beegfs/CMIP6/arctic-cmip6/regrid -type d -exec chmod 755 {} \;

# files
find /beegfs/CMIP6/arctic-cmip6/regrid -type f -exec chmod 644 {} \;
```
