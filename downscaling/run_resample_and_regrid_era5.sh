#!/bin/bash

for year in {1965..2022}
do
    start_time=$(date +%s)
    echo "Processing year $year"
    
    # Run the python script with a timeout of 5 minutes (300 seconds)
    # dask seems to hang sometimes and retrying usually works
    cmd="python resample_and_regrid_era5.py --era5_dir /beegfs/CMIP6/wrf_era5/04km --output_dir /beegfs/CMIP6/kmredilla/daily_era5_4km_3338 --year $year --geo_file /beegfs/CMIP6/wrf_era5/geo_em.d02.nc --no_clobber"
    
    retries=5
    for ((i=1; i<=retries; i++)); do
        timeout 400 $cmd
        if [ $? -ne 124 ]; then
            break
        fi
        echo "Year $year processing timed out, retrying ($i/$retries)..."
    done
    
    if [ $? -eq 124 ]; then
        echo "Year $year processing failed after $retries retries."
    fi
    
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    echo "Year $year processed in $elapsed_time seconds"
done
