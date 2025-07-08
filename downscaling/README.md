# Downscaling

Scripts and notebooks to assist with downscaling CMIP6 data. 

The contents of this folder are for things specific to downscaling CMIP6 data. While regridding and bias-adjustment are both part of the downscaling process, the utilities for performing those steps are found in the `regridding/` and `bias_adjust/` folders, respectively. 

## `notebooks/`

The jupyter notebooks in this folder contain the exploratory data analysis performed to help set up a defensible downscaling pipeline (mostly on the bias-adjustment aspect) for CMIP6 data, as well as some post-hoc exploration of downscaled data as we approach a production dataset. It exposes some of the initial experimentation with regridding and bias adjustment of simulated GCM data using `xclim`, and it now serves as a record of how we chose the final parameters for developing our downscaled CMIP6 data.

With `xclim`, we have at our disposal a number of different methods of adjustment, as well as multiple knobs to turn to tweak those methods. There is more functionality available than we had capacity for evaluating and finding a "best" method, and this could be the subject of a future project.

Rather, we decided to use [Lavoie et al 2024](https://doi.org/10.1038/s41597-023-02855-z) as a template to mimic. The authors used detrended quantile mapping for the same variables we adjusted (at least, so far): daily maximum temperature ($T_{max}$), daily temperature range ($T_{\Delta}$; with the goal of generating a cohesive daily minimum temperature product via $T_{max} - T_{\Delta}$), and precipitation ($Pr$).  how we emulated their methods to bias-adjust CMIP6 data, and how we tuned these methods to ensure we are producing the best possible data within our constraints.

We explored detrended quantile mapping (with multiple different detrending configs) but found that the quantile delta mapping method seemed to perform similarly in most cases, and seemed to be better than the DQM methods for things like seasonal precipitation cycles. See the `discrete_comparison_qm_method.ipynb` notebook. 

## Which CMIP6 data?

The CMIP6 data we are working with (to be / have been adjusted) are daily fields of maximum temperature, daily temperature range, and that have been regridded to a 4km nominal grid in EPSG:3338. The models herein are part of the SNAP CMIP6 ensemble, a set of 13 models that was selected based on skill in Alaska and the arctic using the [GCMEval tool](https://gcmeval.met.no/) ([paper](https://doi.org/10.1016/j.cliser.2020.100167)).


## Other stuff here

Other stuff in the `downscaling/` folder, we have:
* `make_intermediate_target_grid_file.py` - this script is used for making an intermediate target grid for "cascade" regridding in the downscaling pipeline. 
* `round_negative_precip.py` - This is a post-processing script for the downscaled precipitatation data. This script is used to round negative precip values to the nearest whole number, assuming all negative precip values are small (< -0.5 mm). Will raise a warning if negative values remain after rounding - this would signal a possible issue with the downscaled data. It is not currently automated in the pipeline. 
* `tests/` - here you can run somes tests on the downscaled data corpus. The tests currently work on all downscaled data for the given variable being tested, and are limited to sanity checks only. You can run them separately or all together, see docstrings. Individual tests should take ~10 minutes on a chinook compute node. 
