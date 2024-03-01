# this is what we will call the dir created in working_dir that will contain all outputs
output_dir_name = "dtr_processing"
# template file names for DTR, tasmax, and tasmin
dtr_tmp_fn = "dtr_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"
tasmax_tmp_fn = "tasmax_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"
tasmin_tmp_fn = "tasmin_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"


# min / max possible ranges
expected_value_ranges = {
    # this value was determined in exploration
    #  max temps for panarctic domain were ~45 C, min temps around -60 C
    #  within day differences of 100 C are extremely unlikely
    #  (units for data are kelvin but )
    "dtr": {"minimum": 0, "maximum": 100}
}
