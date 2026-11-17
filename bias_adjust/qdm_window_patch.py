"""Patch xclim 0.47's QuantileDeltaMapping.adjust so that `sim` is ranked against the
same N-day day-of-year window that the adjustment factors were trained with.

Background (ua-snap/cmip6-utils#125): `xclim.sdba._adjustment.qdm_adjust` hard-codes
`main_only=True` in the ranking step, so with `group="time.dayofyear", window=31` the
training uses a +/-15 day window (~1,550-2,666 values per day-of-year) while the
adjustment ranks each simulated value against the exact day-of-year only (~50-86
values). xsdba later exposed the windowed behaviour as `rank_window=True`; here we
reproduce it in place without modifying the conda environment.

Usage (before any call to `QuantileDeltaMapping.adjust`):

    import qdm_window_patch
    qdm_window_patch.apply(main_only=False)   # windowed ranking (the fix)
    qdm_window_patch.apply(main_only=True)    # legacy behaviour, for A/B tests

`apply` rebinds `xclim.sdba.adjustment.qdm_adjust` (the name looked up by
`QuantileDeltaMapping._adjust`) to a copy of the original function whose ranking call
uses the requested `main_only`. Everything else is byte-for-byte the xclim 0.47.0 code.
"""

import logging
import re
from pathlib import Path

import xarray as xr
import xclim
import xclim.sdba._adjustment as _xclim_adjustment
import xclim.sdba.adjustment as _xclim_adjustment_api
from xclim.sdba import utils as u
from xclim.sdba.base import Grouper, map_blocks

SUPPORTED_XCLIM_VERSIONS = ("0.47.0",)

# The exact ranking line we expect to find in the pinned xclim; if it changes, the
# behaviour we are patching around may have changed too, so refuse to run.
_EXPECTED_RANK_LINE = "sim_q = group.apply(u.rank, ds.sim, main_only=True, pct=True)"

_STATE = {"applied": None}


def _check_environment():
    """Make sure the installed xclim is the one this patch was written against."""
    if xclim.__version__ not in SUPPORTED_XCLIM_VERSIONS:
        raise RuntimeError(
            f"qdm_window_patch was written for xclim {SUPPORTED_XCLIM_VERSIONS}, "
            f"found {xclim.__version__}. Review the patch before using it."
        )
    src = Path(_xclim_adjustment.__file__).read_text()
    m = re.search(r"def qdm_adjust\(.*?\n(?:.*\n){0,15}", src)
    block = m.group(0) if m else ""
    if _EXPECTED_RANK_LINE not in block:
        raise RuntimeError(
            "Did not find the expected ranking line in xclim.sdba._adjustment.qdm_adjust; "
            "the installed xclim differs from the one this patch targets."
        )


def _make_qdm_adjust(main_only: bool):
    """Build a qdm_adjust clone with the requested ranking behaviour."""

    @map_blocks(reduces=[Grouper.PROP, "quantiles"], scen=[], sim_q=[])
    def qdm_adjust(ds, *, group, interp, extrapolation, kind) -> xr.Dataset:
        """QDM: Adjust process on one block (xclim 0.47.0 copy, ranking window patched).

        Dataset variables:
          af : Adjustment factors
          hist_q : Quantiles over the training data
          sim : Data to adjust.
        """
        sim_q = group.apply(u.rank, ds.sim, main_only=main_only, pct=True)
        af = u.interp_on_quantiles(
            sim_q,
            ds.quantiles,
            ds.af,
            group=group,
            method=interp,
            extrapolation=extrapolation,
        )
        scen = u.apply_correction(ds.sim, af, kind)
        return xr.Dataset(dict(scen=scen, sim_q=sim_q))

    return qdm_adjust


def apply(main_only: bool = False):
    """Rebind the QDM adjust kernel. `main_only=False` ranks over the training window."""
    _check_environment()
    _xclim_adjustment_api.qdm_adjust = _make_qdm_adjust(main_only)
    _STATE["applied"] = main_only
    logging.info(
        "qdm_window_patch applied: QDM ranking main_only=%s (%s)",
        main_only,
        "legacy exact-day-of-year ranking"
        if main_only
        else "ranking over the full training window",
    )
    return _xclim_adjustment_api.qdm_adjust


def current_main_only():
    """Return the `main_only` value currently applied, or None if the patch is not active."""
    return _STATE["applied"]
