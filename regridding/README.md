# Regrid CMIP6 on ACDN

This directory is used for regridding all of the CMIP6 data mirrored on the ACDN to a common grid. The target grid is the NCAR CESM2 grid, which is also shared by a number of the other chosen models. 


## Running the regridding pipeline

### 1. Set the environment variables

First, define the following environment variables (assuming you have activated the cmip6-utils conda environment):

##### `SCRATCH_DIR`

This should be set to the path where you will write the regridded files initially. Something like:

```
export SCRATCH_DIR=/center1/CMIP6/kmredilla/cmip6_regridding
```

##### `PROJECT_DIR`

This should be set to the path of the `cmip6-utils` repo on the system. This is done for referencing scripts and constants without adding this path to the `PYTHONPATH` env var. E.g.:

```
export PROJECT_DIR=/home/kmredilla/repos/cmip6-utils
``` 

##### `CONDA_INIT`

This should be a shell script for initializing conda in a blank shell that does not read the typical `.bashrc`, as is the case with new slurm jobs.

It should look like this, with the variable `CONDA_PATH` below being the path to parent folder of your conda installation, e.g. `/home/UA/kmredilla/miniconda3`:

```
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

Email address to send failed slurm job notifications to.

### 2. Generate batch files

Here we will generate text files containing batches of CMIP6 filepaths that have a common grid to be worked on in a given slurm job. 

Execute the `generate_batch_files.py` script to create these files:

```
python generate_batch_files.py
```

This could take a bit of time, as all of the grid information is being read in and grouped for each of the CMIP6 files. 

### 3. Regrid the data

Next, use the `regrid_cmip6.ipynb` to orchestrate the slurm jobs which will regrid all CMIP6 files listed in the batch files created in step 2. Follow the text in the notebook for instructions on running this step. 

### 4. Quality control on the regridded data

Use the `qc.ipynb` notebook to verify that the regridded files appear correct and consistent. 

### 5. Copy the regridded data off scratch space

Now, copy the regridded data off of scratch space to a permanent location. For now, this will be `/beegfs/CMIP6/arctic-cmip6/regrid`. This can be achieved with:

```
rsync -av $SCRATCH_DIR/regrid /beegfs/CMIP6/arctic-cmip6/
```

It is recommended to do this in a `screen` session as it could take a while. 

Make sure that the permissions are set correctly for this. A sane strategy would be to set directories to 755 (drwxr-xr-x) and set the files to 644 (-rw-r--r--). This can be done with:

```
# directories
find /beegfs/CMIP6/arctic-cmip6/regrid -type d -exec chmod 755 {} \;

# files
find /beegfs/CMIP6/arctic-cmip6/regrid -type f -exec chmod 644 {} \;
```

### 6. Symlink the non-regridded files

Some of the files where not re-gridded because they already had the correct grid.
