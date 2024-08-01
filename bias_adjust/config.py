output_dir_name = "bias_adjust"

qc_dir_name = "qc"
doy_summary_dir_name = "doy_summaries"
doy_summary_tmp_fn = "{var_id}_{model}_{scenario}_{kind}.nc"

ref_tmp_fn = "era5_daily_regrid_{ref_var_id}_{year}.nc"

cmip6_tmp_fn = "{var_id}_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"

cmip6_adjusted_tmp_fn = "{var_id}_day_{model}_{scenario}_adjusted_{year}0101-{year}1231.nc"