"""Kelvin <-> Celsius conversion for temperature output variables.

Driven entirely by config.yaml's `units:` section: `temperature_unit`
picks "celsius" (the default) or "kelvin", and `temperature_vars` lists
which output variables this applies to.

`dtr` is a temperature *difference*, not an absolute temperature -- a 1 K
difference and a 1 degC difference are the same size, so its stored
values never change between unit systems. Only its reported `units` label
follows the configured system, for label consistency with the other
temperature variables. Everything else in `temperature_vars` (tmax, tmin,
tmean) is an absolute temperature and gets the K -> degC arithmetic shift
applied when celsius is selected.
"""

from __future__ import annotations

from config import Config

KELVIN_TO_CELSIUS_OFFSET = 273.15
DIFFERENCE_VARS = {"dtr"}


def is_temperature_var(output_var: str, config: Config) -> bool:
    return output_var in config.units["temperature_vars"]


def convert_value_if_needed(output_var: str, values, config: Config):
    """Apply the configured K<->degC shift to an absolute-temperature
    variable's values. `values` may be a numpy array or an xarray
    DataArray -- subtraction broadcasts over either. No-op for
    non-temperature variables and for difference variables (dtr)."""
    if not is_temperature_var(output_var, config):
        return values
    if output_var in DIFFERENCE_VARS:
        return values
    if config.units["temperature_unit"] == "celsius":
        return values - KELVIN_TO_CELSIUS_OFFSET
    return values


def units_label(output_var: str, base_units: str, config: Config) -> str:
    """Override a configured `units` string (e.g. "K") with "degC" when
    output_var is a temperature variable and celsius is selected."""
    if is_temperature_var(output_var, config) and config.units["temperature_unit"] == "celsius":
        return "degC"
    return base_units
