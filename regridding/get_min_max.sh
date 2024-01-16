#!/bin/sh

echo Gathering min and max values for all regrid variables
for var in clt evspsbl hfls hfss hus huss mrro mrsos pr prsn psl rlds rls rsds rss sfcWind sfcWindmax sithick snd snw ta tas tasmax tasmin tos ua uas va vas
do
  echo working on $var
  timeout 30m $PROJECT_DIR/regridding/get_min_max.py -v $var
  while [ $? == 124 ]; do
    timeout 30m $PROJECT_DIR/regridding/get_min_max.py -v $var
    echo ""
  done
done
