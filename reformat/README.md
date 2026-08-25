# Reformat

Reformats downscaled CMIP6 outputs (zarr) for public data release:

1. Reorders variables to CF conventions (`time, y, x, ensemble`)
2. Adds an `ensemble` dimension (`<model>_<variant>_<scenario>`, derived from the filename via `transfers/config.py`'s variant lookup)
3. Strips dataset attributes down to `contact`, `creation_date`, and `history`

## Components

- `reformat_for_data_release.py` / `.ipynb` — the reformatting step
- `zarr_qc.ipynb` — checks reformatted zarr stores for the files a valid store needs (`.zattrs`, `.zgroup`, `.zmetadata`, a directory per variable/dimension) before release; modeled on the [USGS GeoDataPortal Zarr QC tutorial](https://code.usgs.gov/wma/nhgf/geo-data-portal/gdp_data_processing/-/blob/main/workflows/zarr-data-review/tutorial.ipynb)

## Running

```sh
python reformat_for_data_release.py <input_dir> <output_dir>
```

Then run `zarr_qc.ipynb` against `<output_dir>` before release.
