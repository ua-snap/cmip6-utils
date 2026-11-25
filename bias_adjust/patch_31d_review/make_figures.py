"""Figures for the patch_31d review document (README.md in this directory).

Draws small multiples of new - previous (MIROC6 ssp585) for the six variables, from either
  * the comparison maps written by
    `compare_rank_modes.py summary --legacy <previous output> --windowed <new output>`
    (--figdata; used for 12 km), or
  * the per-scenario statistics written by downscaling/qc_report/qc_stats.py
    (--qcdata; per-cell means and quantiles of the new and published outputs; used for 4 km).

usage: python make_figures.py --figdata /path/to/review/figdata --out_dir figures
       python make_figures.py --qcdata /path/to/qc_reports/4km/data --res_label "4 km" \
           --suffix _4km --out_dir figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

VARS = [
    ("pr", "Precipitation", "mm d⁻¹"),
    ("tasmax", "Max. temperature", "K"),
    ("dtr", "Diurnal temp. range", "K"),
    ("hurs", "Relative humidity", "%"),
    ("hursmin", "Min. relative humidity", "%"),
    ("sfcWind", "Wind speed", "m s⁻¹"),
]
CASE = "MIROC6_ssp585"
# diverging: blue <-> neutral gray <-> red (two hues, gray midpoint)
CMAP = LinearSegmentedColormap.from_list(
    "div", ["#123f73", "#2a78d6", "#f0efec", "#e34948", "#7d1d1c"]
)
CMAP.set_bad("#ffffff")
INK, MUTED = "#1f1f1e", "#6b6a66"


def load_field(args, v, field):
    """new - previous for one variable; field is 'mean_diff' or 'q<level>_diff'."""
    if args.figdata:
        ds = xr.open_dataset(
            Path(args.figdata) / f"{v}_{CASE}" / "rank_mode_comparison_maps.nc"
        )
        return ds[field].transpose("y", "x")
    ds = xr.open_dataset(Path(args.qcdata) / f"{v}_{CASE}.nc")
    if field == "mean_diff":
        d = ds["mean_new"] - ds["mean_prev"]
    else:
        q = float(field[1:].replace("_diff", ""))
        d = ds["q_new"].sel(level=q, method="nearest") - ds["q_prev"].sel(
            level=q, method="nearest"
        )
    return d.transpose("y", "x")


def panel_grid(args, field, title, out_png):
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.2), facecolor="#ffffff")
    for ax, (v, name, units) in zip(axs.flat, VARS):
        d = load_field(args, v, field)
        vals = d.values[np.isfinite(d.values)]
        lim = float(np.percentile(np.abs(vals), 98)) or 1e-6
        im = ax.pcolormesh(
            d["x"], d["y"], d.values, cmap=CMAP, vmin=-lim, vmax=lim, shading="auto"
        )
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{name} ({v})", color=INK, fontsize=11, loc="left")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, extend="both", pad=0.02)
        cb.set_label(units, color=MUTED, fontsize=9)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors=MUTED, length=0)
        mean_abs = float(np.abs(vals).mean())
        ax.set_xlabel(
            f"domain mean |Δ| = {mean_abs:.3g} {units}", fontsize=8.5, color=MUTED
        )
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    fig.text(
        0.01,
        0.005,
        "New (31-day-window ranking) minus previous published output, MIROC6 ssp585 "
        f"2015-2100, {args.res_label}. Red = higher in the new output. Colour range = ±98th "
        "percentile of |Δ| per variable.",
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=1.5)
    fig.savefig(out_png, dpi=110, facecolor="#ffffff")
    plt.close(fig)
    print("wrote", out_png)


def domain_table(args):
    """Domain-average |new - previous| of the mean, median, 99th and 99.9th pct."""
    rows = []
    for v, _, units in VARS:
        cells = [
            np.nanmean(np.abs(load_field(args, v, f).values))
            for f in ("mean_diff", "q0.5_diff", "q0.99_diff", "q0.999_diff")
        ]
        rows.append(
            f"| {v} ({units}) | " + " | ".join(f"{c:.2g}" for c in cells) + " |"
        )
    print("\n".join(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--figdata")
    src.add_argument("--qcdata")
    p.add_argument("--out_dir", default="figures")
    p.add_argument("--res_label", default="12 km")
    p.add_argument("--suffix", default="", help="appended to the figure file names")
    p.add_argument("--table", action="store_true", help="print the README table rows")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel_grid(
        a, "mean_diff", "Change in long-term mean", out / f"change_mean{a.suffix}.png"
    )
    panel_grid(
        a,
        "q0.999_diff",
        "Change in the 99.9th percentile (≈ 1-in-1000-day value)",
        out / f"change_q999{a.suffix}.png",
    )
    if a.table:
        domain_table(a)
