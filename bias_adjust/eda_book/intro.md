# CMIP6 Bias Adjustment Exploration

This book presents the exploratory data analysis performed to help set up a defensible bias adjustment pipeline for CMIP6 data. It exposes some of the initial experimentation with bias adjustment of simulated GCM data using `xclim`, and it now serves as a record of how we chose the final parameters for developing our production bias-adjusted dataset.

With `xclim` we have at our disposal a number of different methods of adjustment, as well as multiple knobs to turn to tweak those methods. There is more functionality available than we had capacity for evaluating and finding a "best" method, and this could be the subject of a future project.

Rather, we decided to use [Lavoie et al 2024](https://doi.org/10.1038/s41597-023-02855-z) as a template to mimic. The authors used detrended quantile mapping for the same variables we adjusted (at least, so far): daily maximum temperature ($T_{max}$), daily temperature range ($T_{\Delta}$; with the goal of generating a cohesive daily minimum temperature product via $T_{max} - T_{\Delta}$), and precipitation ($Pr$). In this book, we demonstrate how we emulated their methods to adjust "common grid" CMIP6 data, and how we tuned these methods to ensure we are producing the best possible data within our constraints.

## Which CMIP6 data?

The CMIP6 data we are working with (to be / have been adjusted) are (again) daily fields of maximum temperature, daily temperature range, and that have been regridded to a common grid (**link here when available!**). The models herein are part of the SNAP CMIP6 ensemble, a set of 12 models that was selected based on skill in Alaska and the arctic using the [GCMEval tool](https://gcmeval.met.no/) ([paper](https://doi.org/10.1016/j.cliser.2020.100167)).

