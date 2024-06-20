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

# Bias adjusting precipitation data

Following Lavoie et al 2024, the procedure for adjusting precipitation is similar to that for $T_{max}$ and $T_{\Delta}$, but with two additional considerations: 1) matching dry day frequency, and 2) removing (substituting) zero values. 

From their "Bias Adjustment of ESPO-G6-R2 v1.0.0" document included in the [configuration repository](https://github.com/Ouranosinc/ESPO-G/blob/fbfc55c6378f009877ec35e58805b847d300d99b/documentation/ESPO_G6_R2v100_adjustment.pdf):

> when the model has a higher dry-day frequency than the reference, the calibration step of the quantile mapping adjustment will incorrectly map all dry days to precipitation days, resulting in a wet bias. The frequency adaptation method finds the fraction of ”extra” dry days

So we will implement this same "frequency adaptation" method by supplying an additional argument in the training function. (See [this part](https://xclim.readthedocs.io/en/stable/notebooks/sdba.html#First-example-:-pr-and-frequency-adaptation) of the example notebook in the `xclim` docs for a little more info). 

Removal of any zero values will be done by replacing them with random "jitter" very close to zero (see that linked document above for more on this).

Perform the same setup steps as we did for the $T_{max}$:

```{code-cell} python3
import xarray as xr
from xclim import sdba
import matplotlib.pyplot as plt
from dask.distributed import Client
from xclim import units
from xclim.sdba.detrending import LoessDetrend
import dask

# we have to make some big chunks and this will silence a warning about that
dask.config.set(**{"array.slicing.split_large_chunks": False})

log_dir = "."
```

Import the same functions and constants used for the $T_{max}$ adjustment:

```{code-cell} python3
from baeda import *
```

Start the dask client

```{code-cell} python3
client = Client()
```

Again, we will be using ERA5 data as our historical reference, using 1993 - 2022.

```{code-cell} python3
ref_var_id = "tp"
ref_start_year = 1993
ref_end_year = 2022
ref_fps = get_era5_fps(ref_var_id, ref_start_year, ref_end_year)

var_id = "pr"
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
# for some reason the precip data for 2022 has an expver variable while the t2m data doesnt
# drop this as it might be causing problems
ref_ds = (
    xr.open_mfdataset(ref_fps)
    .convert_calendar("noleap")
    .sel(expver=1)
    .drop_vars("expver")
)

ref = get_rechunked_da(ref_ds, ref_var_id)
hist = get_rechunked_da(hist_ds, var_id)
```

I think we need to set the units to `m d-1` - instead of current `m` - so it is compatible with xclim's (`pint`'s) unit scheme (since these are at a daily time step):

```{code-cell} python3
ref.attrs["units"] = "m d-1"
```

To make sure there are no true 0's in the data, `xscen` uses `xclim`'s `sdba.processing.jitter_under_thresh`, using a threshold of 0.01 mm / day.

```{code-cell} python3
ref = sdba.processing.jitter_under_thresh(ref, thresh="0.01 mm d-1")
hist = sdba.processing.jitter_under_thresh(hist, thresh="0.01 mm d-1")
```

Have a look to see:

```{code-cell} python3
# select a subset of time range just for examination
before_jitter = ref_ds["tp"].isel(time=slice(100, 200)).compute()
after_jitter = ref.isel(time=slice(100, 200)).compute()

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

before_jitter.plot.hist(
    range=[-0.01, 0.01], alpha=0.75, bins=40, ax=ax, label="Before jitter"
)
after_jitter.plot.hist(
    range=[-0.01, 0.01], alpha=0.75, bins=40, ax=ax, label="After jitter"
)
plt.legend(loc="upper right")

plt.show()
```

You can see that there were some negative values in this model beforehand and those have been bounded at 0. You actually can't tell from this graph that those new values are indeed not 0, so here is proof:

```{code-cell} python3
print("AfBEforeter jitter minimum:", before_jitter.min().values)
print("After jitter minimum:", after_jitter.min().values)
```

Now set up the training object.

We will set the "frequency adaptation" threshold to 1 mm / d, again, following along with Lavoie et al:

```{code-cell} python3
dqm = sdba.DetrendedQuantileMapping.train(
    ref=ref,
    hist=hist,
    nquantiles=50,
    group="time.dayofyear",
    window=31,
    kind="*",
    adapt_freq_thresh="1 mm d-1",
)

# Create the detrending object
det = LoessDetrend(group="time.dayofyear", d=0, niter=1, f=0.2, weights="tricube")

# now the adjustment object / task graph
scen = dqm.adjust(hist, extrapolation="constant", interp="nearest", detrend=det)
```

Then run it. We can again run it for a single pixel to save some time:

```{code-cell} python3
sel_di = {"lon": -147, "lat": 65}
scen = scen.sel(sel_di, method="nearest").compute()
```
Do some plotting to evaluate:


```{code-cell} python3
# convert model data to meters / day to match ref and scen
hist_mpd = units.convert_units_to(hist.sel(sel_di, method="nearest"), "m d-1")
plot_avg_ts(ref.sel(sel_di, method="nearest"), hist_mpd, scen)
```
That's actually tough to make sense of, although it does appear to be an improvement, with the adjusted data seemignly closer to the reference. Maybe monthly averages would be better to visualize this:

```{code-cell} python3
plot_avg_ts(ref.sel(sel_di, method="nearest"), hist_mpd, scen, gb_str="time.month")
```

Yep, that's definitely an improvement!
