"""Period membership and season-year labeling.

A "label year" is the calendar year a day's season-instance is attributed
to: for non-wrapping periods (single months, MAM/JJA/SON/AMJJAS) this is
just the day's own calendar year. For wrapping periods (DJF, ONDJFM) it's
the calendar year of the period's last month -- e.g. December days get
bumped forward one year to align with the Jan/Feb that completes their
winter.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def in_period_and_label_year(time: xr.DataArray, months: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (in_period, label_year) boolean/int arrays aligned to `time`.

    `months` must be in chronological order (not necessarily ascending --
    e.g. DJF is [12, 1, 2], not [1, 2, 12]). Works for both datetime64 and
    cftime-backed time indexes via xarray's `.dt` accessor.
    """
    cal_year = time.dt.year.values
    cal_month = time.dt.month.values
    wrap = months[0] > months[-1]

    if not wrap:
        in_period = np.isin(cal_month, months)
        label_year = cal_year.copy()
        return in_period, label_year

    tail = [m for m in months if m >= months[0]]  # from the previous label-year, e.g. DJF -> [12]
    head = [m for m in months if m <= months[-1]]  # already the label-year, e.g. DJF -> [1, 2]
    in_period = np.isin(cal_month, tail + head)
    label_year = np.where(np.isin(cal_month, tail), cal_year + 1, cal_year)
    return in_period, label_year
