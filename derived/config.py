# this is what we will call the dir created in working_dir that will contain all outputs
output_dir_name = "dtr_processing"
# template file names for DTR, tasmax, and tasmin
dtr_tmp_dir_structure = "{model}/{scenario}/day/dtr"
dtr_tmp_fn = "dtr_day_{model}_{scenario}_regrid_{start_date}-{end_date}.nc"
tasmax_tmp_fn = "tasmax_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"
tasmin_tmp_fn = "tasmin_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"

dtr_sbatch_tmp_fn = "process_cmip6_dtr.slurm"
dtr_sbatch_config_tmp_fn = "process_cmip6_dtr_config.txt"

# min / max possible ranges
expected_value_ranges = {
    # this value was determined in exploration
    #  max temps for panarctic domain were ~45 C, min temps around -60 C
    #  within day differences of 100 C are extremely unlikely
    #  (units for data are kelvin but )
    "dtr": {"minimum": 0, "maximum": 100}
}
