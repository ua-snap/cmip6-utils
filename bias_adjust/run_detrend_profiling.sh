for det_config in det1 det2 det3 det4 det5; do
    for region in Fairbanks MatSu Yakutat; do
        time python bias_adjust.py --train_path /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/trained/tasmax_GFDL-ESM4_dqm.zarr --sim_path /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/optimized_inputs/tasmax_day_GFDL-ESM4_historical.zarr --adj_path /beegfs/CMIP6/kmredilla/cmip6_4km_3338_adjusted_test/det_testing/tasmax_GFDL-ESM4_historical_${det_config}_${region}.zarr --det_config ${det_config} --region ${region}
        sleep 5
    done
done