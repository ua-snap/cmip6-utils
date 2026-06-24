"""Kelvin -> Celsius conversion for the source temperature variables
(tasmax/tasmin) read by compute_fragment.py.

Driven entirely by config.yaml's `units:` section: `temperature_vars`
lists which source variables need the K -> degC shift before the
indicator thresholds (all defined in degC) are evaluated -- the source
zarrs are always Kelvin.
"""

from __future__ import annotations

from config import Config

KELVIN_TO_CELSIUS_OFFSET = 273.15


def is_temperature_var(name: str, config: Config) -> bool:
    return name in config.units["temperature_vars"]


def convert_value_if_needed(name: str, values, config: Config):
    """Apply the K -> degC shift to a source variable's values if it's
    listed in config.yaml's `units.temperature_vars`. `values` may be a
    numpy array or an xarray DataArray -- subtraction broadcasts over
    either. No-op otherwise."""
    if not is_temperature_var(name, config):
        return values
    return values - KELVIN_TO_CELSIUS_OFFSET
