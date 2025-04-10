output_dir_name = "bias_adjust"

qc_dir_name = "qc"
doy_summary_dir_name = "doy_summaries"
doy_summary_tmp_fn = "{var_id}_{model}_{scenario}_{kind}.nc"


ref_tmp_fn = "era5_{ref_var_id}_{year}_3338.nc"

cmip6_regrid_tmp_fn = "{var_id}_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"

cmip6_zarr_tmp_fn = "{var_id}_{model}_{scenario}.zarr"

biasadjust_sbatch_tmp_fn = "bias_adjust_{model}_{scenario}_{var_id}.sbatch"

trained_qm_tmp_fn = "trained_qdm_{var_id}_{model}.zarr"

cmip6_adjusted_tmp_fn = "{var_id}_{model}_{scenario}_adjusted.zarr"
