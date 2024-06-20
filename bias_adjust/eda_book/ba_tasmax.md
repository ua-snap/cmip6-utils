---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: new_cmip6_utils
  language: python
  name: python3
---

# Bias adjustment of maximum air temperature

Here we will step through what we have learned in developing this workflow by performing a bias adjustment on maximum air temperature data using the exact method we plan to implement for all input data. (well, almost exact - there are some minor differences between adjusting $Pr$ and $T_{max}$ / $T_{\Delta}$ is that we will discuss in the precipitation adjustment section).

First, load the libs.

```{code-cell} python3
import xarray as xr
import matplotlib.pyplot as plt
from xclim import sdba
from dask.distributed import Client
from xclim.sdba.detrending import LoessDetrend
import dask

# we have to make some big chunks and this will silence a warning about that
dask.config.set(**{"array.slicing.split_large_chunks": False})


log_dir = "."
```

We have developed some functions that will help make file handling and other tasks easier. Their source can be viewed on **SOME PAGE**. They are saved in the `baeda` module:

```{code_cell} python3
from baeda import *
```

Start the dask client

```{code-cell} python3
client = Client()
```

We will be using ERA5 data as our historical reference for adjusting data. For now, we will just go with the most recent 30 years of available data that we have available (available via SNAP infra), i.e. 1993 - 2022.

```{code-cell} python3
# "tas" in ERA5 is t2m, so we have named the daily max version t2mmax
ref_var_id = "t2mmax"
ref_start_year = 1993
ref_end_year = 2022
ref_fps = get_era5_fps(ref_var_id, ref_start_year, ref_end_year)

var_id = "tasmax"
model = "GFDL-ESM4"
hist_start_year = 1993
hist_end_year = 2014
hist_fps = get_cmip6_fps(model, "historical", var_id, hist_start_year, hist_end_year)

scenario = "ssp585"
sim_ref_start_year = 2015
sim_ref_end_year = 2022
sim_ref_fps = get_cmip6_fps(
    model, scenario, var_id, sim_ref_start_year, sim_ref_end_year
)
```

Open the datasets from the yearly data files for each of ERA5 and CMIP6:

```{code-cell} python3
hist_ds = xr.open_mfdataset(hist_fps + sim_ref_fps)
# convert calendar to noleap to match CMIP6
ref_ds = xr.open_mfdataset(ref_fps).convert_calendar("noleap")
```

Here is a snapshot of what we are working with. Historical reference (ERA5) on the left, modeled historical GCM on the right, for the same time step (i.e. same day).

```{code-cell} python3
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

ref_ds["t2mmax"].isel(time=1).plot(ax=axes[0])
axes[0].set_title("Reference")
hist_ds["tasmax"].isel(time=1).plot(ax=axes[1])
axes[1].set_title("Historical simulated")
```

Next, we need to rechunk the time dimension into one chunk. The training functions in `xclim.sdba` will not work with datasets having multiple chunks along the adjustment dimension (time in this case). We split it up into chunks over the lat and lon dims for some added optimization (hopefully).

We will also initialize the bias adjustment here. We are performing a "detrended" quantile mapping grouped by day of the year (you will also see this spelled as DOY or doy). To quote Lavoie et al 2024:

>The procedure is univariate (applied to each variable individually), acts independently on the trends and the anomalies, and is applied iteratively on each day of the year as well as at each grid point.

Other parameters can be seen below. Not sure what they all mean, but we are using 50 quantiles and a window of 31 days. specifying `d=0` uses "local constancy", meaning local estimates are weighted averages.

The final adjusted output will be stored in `scen`, which is currently just a dask task graph until we call `.compute()` on it, or need the data in some way.

```{code-cell} python3
ref = get_rechunked_da(ref_ds, ref_var_id)
hist = get_rechunked_da(hist_ds, var_id)

dqm = sdba.DetrendedQuantileMapping.train(
    ref, hist, nquantiles=50, group="time.dayofyear", window=31, kind="+"
)
# Create the detrending object
det = LoessDetrend(group="time.dayofyear", d=0, niter=1, f=0.2, weights="tricube")
scen = dqm.adjust(hist, extrapolation="constant", interp="nearest", detrend=det)
```

Then run it. For demonstration purposes, we will only run this adjustment for a single pixel initially:


```{code-cell} python3
sel_di = {"lon": -147, "lat": 65}
scen = scen.sel(sel_di, method="nearest").compute()
```

Now have a look at the adjusted data which have been saved in `scen`. We will use this kind of plot often to help evaluate the adjustment performance.

```{code-cell} python3
plot_avg_ts(ref.sel(sel_di, method="nearest"), hist.sel(sel_di, method="nearest"), scen)
```

That is certainly an improvement. 

```{code-cell} python3
```
```{code-cell} python3
```
```{code-cell} python3
```
```{code-cell} python3
```
```{code-cell} python3
```
```{code-cell} python3
```
```{code-cell} python3
```
```{code-cell} python3
```
```{code-cell} python3
```
```{code-cell} python3
```
```{code-cell} python3
```
```{code-cell} python3
```