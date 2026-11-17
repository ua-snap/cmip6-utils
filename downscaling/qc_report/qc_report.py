"""Render the QC PDF for one (resolution, variable, model), or the per-resolution summary PDF.

Reads only the small files written by qc_stats.py (no zarr access), so it is fast.
Section A compares the new (patched) outputs with the previous published outputs;
Section B repeats the standard QC from the production notebooks on the new outputs.

usage: python qc_report.py model   --res 12km --var pr --model MIROC6
       python qc_report.py summary --res 12km
"""

import argparse
import json
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from qc_common import (
    check_label,
    jobs_for,
    load_config,
    out_dir,
    scenarios_for,
)  # noqa: E402

PAGE = (11, 8.5)  # US letter, landscape
INK, MUTED, RULE = "#1f1f1e", "#6b6a66", "#d6d4ce"
DIV = LinearSegmentedColormap.from_list(
    "div", ["#123f73", "#2a78d6", "#f0efec", "#e34948", "#7d1d1c"]
)
DIV.set_bad("#ffffff")
SEQ = LinearSegmentedColormap.from_list("seq", ["#8fbbe8", "#2a78d6", "#123f73"])
GREY = plt.get_cmap("Greys")
# Same colour logic as the notebooks: black = ERA5, light = un-adjusted, dark = downscaled,
# blue = historical, green = scenario.
SRC = {
    "era5": ("ERA5", "#1f1f1e"),
    "hist_sim": ("{model} historical (un-adjusted)", "#9cc3e6"),
    "sim": ("{model} {scen} (un-adjusted)", "#9fd39a"),
    "hist_new": ("{model} historical (downscaled)", "#2a78d6"),
    "new": ("{model} {scen} (downscaled)", "#2b8a3e"),
}
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.titlecolor": INK,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
)


def fmt_units(u):
    return {"degC": "°C", "mm d-1": "mm d⁻¹", "m s-1": "m s⁻¹"}.get(u, u)


def g(v, n=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{n}g}" if abs(v) < 10**n else f"{v:,.0f}"


def show_map(ax, a, x, y, **kw):
    a = np.asarray(a)
    dx, dy = abs(x[1] - x[0]) / 2, abs(y[1] - y[0]) / 2
    origin = "upper" if y[0] > y[-1] else "lower"
    ext = (x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy)
    im = ax.imshow(a, extent=ext, origin=origin, interpolation="nearest", **kw)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def background(ax, a, x, y):
    a = np.asarray(a)
    f = a[np.isfinite(a)]
    lo, hi = np.percentile(f, [2, 98]) if f.size else (0, 1)
    return show_map(
        ax, a, x, y, cmap=GREY, vmin=lo, vmax=hi + (hi - lo) * 1.5, alpha=0.7
    )


def lonlat(x, y):
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:3338", "EPSG:4326", always_xy=True)
    return t.transform(x, y)


def ecdf(ax, v, color, label, lw=1.4, z=2):
    v = np.sort(v[np.isfinite(v)])
    if v.size:
        ax.step(
            v,
            np.arange(1, v.size + 1) / v.size,
            where="post",
            color=color,
            lw=lw,
            label=label,
            zorder=z,
        )


def doy_window(doy, center, half=15):
    d = np.abs(doy.astype(int) - int(center))
    return np.minimum(d, 365 - d) <= half


def doy_mean(v, doy):
    ok = np.isfinite(v)
    s = np.bincount(doy[ok], weights=v[ok], minlength=367)
    n = np.bincount(doy[ok], minlength=367)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / n
    return np.arange(1, 367), m[1:]


class Doc:
    """Pages are planned first (so the contents page has page numbers), then drawn."""

    def __init__(self, path, header, png_dir=None):
        self.path, self.header, self.png_dir = path, header, png_dir
        self.pages, self.toc = [], []

    def section(self, title, level=0):
        self.toc.append((level, title, len(self.pages) + 1))

    def add(self, fn):
        self.pages.append(fn)

    def write(self, metadata):
        n = len(self.pages)
        tmp = self.path.with_suffix(".tmp.pdf")
        with PdfPages(tmp, metadata=metadata) as pdf:
            for i, fn in enumerate(self.pages, 1):
                fig = fn()
                fig.text(0.02, 0.985, self.header, fontsize=7.5, color=MUTED, va="top")
                fig.text(
                    0.98, 0.012, f"{i} / {n}", fontsize=7.5, color=MUTED, ha="right"
                )
                pdf.savefig(fig)
                if self.png_dir:
                    fig.savefig(
                        Path(self.png_dir) / f"{self.path.stem}_p{i:03d}.png", dpi=60
                    )
                plt.close(fig)
        tmp.replace(self.path)
        return n


def page(title=None, subtitle=None):
    fig = plt.figure(figsize=PAGE)
    if title:
        fig.text(0.04, 0.935, title, fontsize=15, weight="bold", va="top")
    if subtitle:
        paragraph(fig, 0.04, 0.895, subtitle, size=9, color=MUTED)
    return fig


def paragraph(fig, x, y, text, width=None, size=9.5, color=INK, gap=None):
    width = width or int((1 - 2 * x) * PAGE[0] * 72 / (0.56 * size))
    gap = gap or size * 1.45 / 72 / PAGE[1]
    for line in textwrap.wrap(text, width):
        fig.text(x, y, line, fontsize=size, color=color, va="top")
        y -= gap
    return y


def table(fig, rect, header, rows, col_w=None, size=8.5, align=None):
    ax = fig.add_axes(rect)
    ax.axis("off")
    if not rows:
        return ax
    t = ax.table(
        cellText=rows,
        colLabels=header,
        loc="upper left",
        cellLoc="right",
        colWidths=col_w,
    )
    t.auto_set_font_size(False)
    t.set_fontsize(size)
    t.scale(1, 1.6)
    hdr_lines = max(str(h).count("\n") + 1 for h in header)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_height(cell.get_height() * hdr_lines)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK)
            cell.set_facecolor("#f4f3ef")
        if align and align[c] == "left":
            cell.set_text_props(ha="left")
            cell._loc = "left"
    return ax


# =============================================================================== model report


class ModelReport:
    def __init__(self, cfg, res, var, model):
        self.cfg, self.res, self.var, self.model = cfg, res, var, model
        self.v = cfg["variables"][var]
        self.u = fmt_units(self.v["units"])
        self.data = out_dir(cfg, res) / "data"
        self.scens = [
            s
            for s in scenarios_for(cfg, res, var, model)
            if (self.data / f"{var}_{model}_{s}.nc").exists()
        ]
        if not self.scens:
            raise SystemExit(f"no qc_stats output for {var} {model} {res}")
        self.ds = {
            s: xr.open_dataset(self.data / f"{var}_{model}_{s}.nc").load()
            for s in self.scens
        }
        self.js = {
            s: json.loads((self.data / f"{var}_{model}_{s}.json").read_text())
            for s in self.scens
        }
        self.era5 = xr.open_dataset(self.data / f"era5_{var}.nc").load()
        self.levels = np.asarray(cfg["comparison_quantiles"])
        self.has_prev = [s for s in self.scens if "q_prev" in self.ds[s]]
        self.chk = check_label(self.v["check"], self.u)
        ref = self.ds[self.scens[0]]
        self.x, self.y = ref["x"].values, ref["y"].values
        self.lab = {
            k: (lab.format(model=model, scen="{scen}"), col)
            for k, (lab, col) in SRC.items()
        }

    # ------------------------------------------------------------------ helpers
    def label(self, k, scen):
        lab, col = self.lab[k]
        return lab.format(scen=scen), col

    def series(self, scen, kind, i):
        """kind: fixed|top|td. Returns {source: (values, doy, month)}."""
        d, h = self.ds[scen], self.ds.get("historical")
        out = {}
        if kind == "fixed":
            out["era5"] = (
                self.era5["era5_fixed"].values[i],
                self.era5["doy_era5"].values,
                self.era5["month_era5"].values,
            )
            for k, var in (("new", "fixed_new"), ("sim", "fixed_sim")):
                out[k] = (d[var].values[i], d["doy"].values, d["month"].values)
            if scen != "historical" and h is not None:
                for k, var in (("hist_new", "fixed_new"), ("hist_sim", "fixed_sim")):
                    out[k] = (h[var].values[i], h["doy"].values, h["month"].values)
        else:
            out["era5"] = (
                d[f"{kind}_era5"].values[i],
                d["doy_era5"].values,
                d["month_era5"].values,
            )
            for k in ("new", "sim"):
                if f"{kind}_{k}" in d:
                    out[k] = (
                        d[f"{kind}_{k}"].values[i],
                        d["doy"].values,
                        d["month"].values,
                    )
            for k in ("hist_new", "hist_sim"):
                if f"{kind}_{k}" in d:
                    out[k] = (
                        d[f"{kind}_{k}"].values[i],
                        d["doy_hist"].values,
                        d["month_hist"].values,
                    )
        return {k: v for k, v in out.items() if np.isfinite(v[0]).any()}

    def extreme_doy(self, v, doy):
        if not np.isfinite(v).any():
            return None
        i = np.nanargmin(v) if self.v["extreme"] == "min" else np.nanargmax(v)
        return int(doy[i])

    def ecdf_panel(self, ax, scen, src, mode, center_from="new"):
        """mode: window31 | fire | full."""
        note = ""
        if mode == "window31":
            base = src.get(center_from) or src.get("new")
            c = self.extreme_doy(*base[:2]) if base else None
            if c is None:
                return ""
            note = f"DOY {c} ± 15"
        for k in ("hist_sim", "sim", "hist_new", "new", "era5"):
            if k not in src:
                continue
            v, doy, mon = src[k]
            if mode == "window31":
                v = v[doy_window(doy, c)]
            elif mode == "fire":
                v = v[np.isin(mon, self.cfg["fire_season_months"])]
            lab, col = self.label(k, scen)
            ecdf(
                ax,
                v,
                col,
                lab,
                lw=1.6 if k in ("new", "era5") else 1.2,
                z=5 if k == "era5" else 2,
            )
        ax.set_xlabel(f"{self.var} ({self.u})")
        ax.grid(alpha=0.25, lw=0.5)
        return note

    def legend_fig(self, fig, scen):
        handles = []
        for k in ("era5", "hist_sim", "sim", "hist_new", "new"):
            if scen == "historical" and k.startswith("hist"):
                continue
            lab, col = self.label(k, scen)
            handles.append(plt.Line2D([], [], color=col, lw=2, label=lab))
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=len(handles),
            fontsize=8.5,
            bbox_to_anchor=(0.5, 0.025),
        )

    # ------------------------------------------------------------------ section A
    def comparison_rows(self):
        tq = self.cfg["tail_quantile"]
        iq, i50 = int(np.argmin(abs(self.levels - tq))), int(
            np.argmin(abs(self.levels - 0.5))
        )
        rows = []
        for s in self.has_prev:
            d, t = self.ds[s], self.js[s]["totals"]
            dm = np.abs(d["mean_new"] - d["mean_prev"]).values
            dq = np.abs(d["q_new"] - d["q_prev"]).values
            pairs = max(t["pairs"], 1)
            rows.append(
                {
                    "scen": s,
                    "mean": float(np.nanmean(d["mean_new"])),
                    "dmean": float(np.nanmean(dm)),
                    "q50": float(np.nanmean(d["q_new"][i50])),
                    "dq50": float(np.nanmean(dq[i50])),
                    "qt": float(np.nanmean(d["q_new"][iq])),
                    "dqt": float(np.nanmean(dq[iq])),
                    "dqt_max": float(np.nanmax(dq[iq])),
                    "mad": t["sum_abs_diff"] / pairs,
                    "maxad": t["max_abs_diff"],
                    "equal": 100 * t["equal"] / pairs,
                    "nanmm": t["nan_mismatch"],
                }
            )
        return rows

    def plan_comparison(self, doc):
        doc.section("A. New vs previous outputs")
        if not self.has_prev:
            doc.add(lambda: self._no_prev())
            return
        rows = self.comparison_rows()
        doc.add(lambda: self._comparison_table(rows))
        doc.section("Quantile differences", 1)
        doc.add(self._quantile_profiles)
        doc.section("Maps of the change", 1)
        doc.add(
            lambda: self._delta_maps(
                "mean", "Change in the long-term mean (new − previous)"
            )
        )
        doc.add(
            lambda: self._delta_maps(
                "tail",
                f"Change in the {100 * self.cfg['tail_quantile']:g}th percentile (new − previous)",
            )
        )

    def _no_prev(self):
        fig = page("A. New vs previous outputs")
        paragraph(
            fig,
            0.04,
            0.85,
            "No previous output was found for this variable/model, so there is nothing to compare.",
        )
        return fig

    def _comparison_table(self, rows):
        tq = 100 * self.cfg["tail_quantile"]
        fig = page("A. New vs previous outputs")
        y = paragraph(
            fig, 0.04, 0.89, self.cfg["text"]["comparison_intro"], color=MUTED, size=9
        )
        u = self.u
        header = [
            "Scenario",
            f"Mean\n({u})",
            "|Δ mean|",
            f"Median\n({u})",
            "|Δ median|",
            f"{tq:g}th pct\n({u})",
            f"|Δ {tq:g}th|",
            f"max |Δ {tq:g}th|\n(any cell)",
            "mean |Δ|\ndaily",
            "max |Δ|\ndaily",
            "Identical\ncell-days",
            "NaN-mask\ndiffs",
        ]
        body = [
            [
                r["scen"],
                g(r["mean"]),
                g(r["dmean"]),
                g(r["q50"]),
                g(r["dq50"]),
                g(r["qt"]),
                g(r["dqt"]),
                g(r["dqt_max"]),
                g(r["mad"]),
                g(r["maxad"]),
                f"{r['equal']:.1f} %",
                f"{r['nanmm']:,}",
            ]
            for r in rows
        ]
        h = 0.033 * (len(body) + 1) + 0.035
        table(
            fig,
            [0.04, y - 0.02 - h, 0.92, h],
            header,
            body,
            align=["left"] + ["right"] * 11,
            size=8,
        )
        y = y - 0.05 - h
        fig.text(
            0.04,
            y,
            "Columns: domain averages over grid cells of the per-cell statistic (all days of the scenario). "
            "|Δ| = |new − previous|. Identical days = share of cell-days with exactly the same value.",
            fontsize=8,
            color=MUTED,
            va="top",
        )
        # plain-language reading of the numbers
        r_m = max(r["dmean"] for r in rows)
        r_50 = max(r["dq50"] for r in rows)
        r_t = max(r["dqt"] for r in rows)
        ratio = r_t / max(r_50, 1e-12)
        msg = (
            f"Reading: across scenarios the long-term mean changes by at most {g(r_m)} {u} and the median by at most "
            f"{g(r_50)} {u} on average per cell, whereas the {tq:g}th percentile changes by up to {g(r_t)} {u} "
            f"(≈{ratio:.0f}× the median change). "
            + (
                "No cell has a different NaN pattern between the two versions."
                if all(r["nanmm"] == 0 for r in rows)
                else "Some cells have a different NaN pattern between the two versions (see the NaN-mask column)."
            )
        )
        paragraph(fig, 0.04, y - 0.06, msg, size=10.5)
        return fig

    def _quantile_profiles(self):
        n = len(self.has_prev)
        fig = page(
            "Quantile differences: new − previous, per grid cell",
            "Top: median over grid cells (line) and 5–95 % of cells (band). Bottom: mean |Δ| over cells, "
            "log scale. Both are close to zero through the bulk of the distribution and grow in the tails.",
        )
        gs = fig.add_gridspec(
            2, n, left=0.08, right=0.98, top=0.83, bottom=0.12, hspace=0.12, wspace=0.08
        )
        lv = self.levels
        ticks = [
            t
            for t in (0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999)
            if lv.min() <= t <= lv.max()
        ]
        top_axes, bot_axes = [], []
        for k, s in enumerate(self.has_prev):
            d = self.ds[s]
            dq = (d["q_new"] - d["q_prev"]).values.reshape(len(lv), -1)
            with np.errstate(all="ignore"):
                med = np.nanmedian(dq, 1)
                lo, hi = np.nanpercentile(dq, 5, 1), np.nanpercentile(dq, 95, 1)
                mabs = np.nanmean(np.abs(dq), 1)
            a1 = fig.add_subplot(gs[0, k], sharey=top_axes[0] if top_axes else None)
            a2 = fig.add_subplot(
                gs[1, k], sharex=a1, sharey=bot_axes[0] if bot_axes else None
            )
            top_axes.append(a1), bot_axes.append(a2)
            a1.fill_between(lv, lo, hi, color="#2a78d6", alpha=0.25, lw=0)
            a1.plot(lv, med, color="#123f73", lw=1.8, marker="o", ms=3)
            a1.axhline(0, color=INK, lw=0.6)
            a1.set_title(s, loc="left")
            a2.plot(
                lv,
                np.where(mabs > 0, mabs, np.nan),
                color="#123f73",
                lw=1.8,
                marker="o",
                ms=3,
            )
            a2.set_yscale("log")
            for a in (a1, a2):
                a.set_xscale("logit")
                a.set_xticks(ticks)
                a.minorticks_off()
                a.grid(alpha=0.25, lw=0.5)
            a2.set_xticklabels([f"{t:g}" for t in ticks], rotation=45)
            plt.setp(a1.get_xticklabels(), visible=False)
            a2.set_xlabel("quantile level q")
            if k:
                plt.setp(a1.get_yticklabels(), visible=False)
                plt.setp(a2.get_yticklabels(), visible=False)
        ymax = max(np.nanmax(a.lines[0].get_ydata()) for a in bot_axes)
        if np.isfinite(ymax) and ymax > 0:
            bot_axes[0].set_ylim(ymax * 1e-4, ymax * 2)
        top_axes[0].set_ylabel(f"Δ quantile ({self.u})")
        bot_axes[0].set_ylabel(f"mean |Δ quantile| ({self.u})")
        return fig

    def _delta_maps(self, which, title):
        tq = self.cfg["tail_quantile"]
        iq = int(np.argmin(abs(self.levels - tq)))
        fields = {}
        for s in self.has_prev:
            d = self.ds[s]
            fields[s] = (
                (d["mean_new"] - d["mean_prev"]).values
                if which == "mean"
                else (d["q_new"][iq] - d["q_prev"][iq]).values
            )
        allv = np.concatenate([f[np.isfinite(f)] for f in fields.values()])
        lim = float(np.percentile(np.abs(allv), 98)) if allv.size else 1.0
        lim = lim or 1e-6
        fig = page(
            title,
            f"Colour range ±{g(lim)} {self.u} (98th percentile of |Δ|, shared by all panels). "
            "Red = higher in the new output.",
        )
        n = len(fields)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        gs = fig.add_gridspec(
            nrow,
            ncol,
            left=0.03,
            right=0.9,
            top=0.84,
            bottom=0.05,
            wspace=0.05,
            hspace=0.18,
        )
        im = None
        for k, (s, f) in enumerate(fields.items()):
            ax = fig.add_subplot(gs[k // ncol, k % ncol])
            im = show_map(ax, f, self.x, self.y, cmap=DIV, vmin=-lim, vmax=lim)
            ax.set_aspect("equal")
            ax.set_title(
                f"{s}   (mean |Δ| = {g(float(np.nanmean(np.abs(f))))} {self.u})",
                loc="left",
                fontsize=9,
            )
        cax = fig.add_axes([0.92, 0.25, 0.012, 0.45])
        cb = fig.colorbar(im, cax=cax, extend="both")
        cb.set_label(self.u)
        cb.outline.set_visible(False)
        return fig

    # ------------------------------------------------------------------ section B
    def plan_qc(self, doc):
        doc.section("B. Standard QC of the new outputs")
        for s in self.scens:
            doc.section(f"{s}", 1)
            doc.add(lambda s=s: self._threshold_page(s))
            d = self.ds[s]
            ssp = s != "historical"
            if self.var == "dtr":
                for stat in ("min", "mean", "max"):
                    doc.add(lambda s=s, stat=stat: self._dtr_stage_maps(s, stat))
                if "td_value" in d:
                    doc.add(lambda s=s: self._dtr_top_days(s))
                continue
            if "top_count" in d:
                doc.add(lambda s=s: self._top6_page(s))
                if ssp and self.v["ecdf"] == "window31":
                    doc.add(
                        lambda s=s: self._ecdf_grid(
                            s,
                            "top",
                            list(range(6)),
                            "window31",
                            f"ECDF at the 6 pixels with the most days {self.chk}",
                        )
                    )
                    doc.add(
                        lambda s=s: self._doy_grid(
                            s,
                            "top",
                            list(range(6)),
                            "Day-of-year means at the same 6 pixels",
                        )
                    )
            if not ssp:
                continue
            names = list(self.ds[s]["fixed_name"].values)
            for k, pset in enumerate(self.cfg["place_sets"]):
                idx = [names.index(p) for p in pset]
                if self.v["ecdf"] == "season":
                    doc.add(
                        lambda s=s, idx=idx, pset=pset: self._ecdf_grid(
                            s,
                            "fixed",
                            idx,
                            "fire",
                            "Hand-picked places, fire season (May–Sep)",
                            pset,
                        )
                    )
                    doc.add(
                        lambda s=s, idx=idx, pset=pset: self._ecdf_grid(
                            s,
                            "fixed",
                            idx,
                            "full",
                            "Hand-picked places, full year",
                            pset,
                        )
                    )
                else:
                    doc.add(
                        lambda s=s, idx=idx, pset=pset: self._ecdf_grid(
                            s, "fixed", idx, "window31", "Hand-picked places", pset
                        )
                    )
            if self.v["ecdf"] == "season":
                idx = [names.index(p) for p in self.cfg["place_sets"][0]]
                doc.add(
                    lambda s=s, idx=idx: self._doy_grid(
                        s,
                        "fixed",
                        idx,
                        "Day-of-year means at hand-picked places",
                        self.cfg["place_sets"][0],
                    )
                )
            ridx = [i for i, n in enumerate(names) if n.startswith("random_")]
            modes = ["fire", "full"] if self.v["ecdf"] == "season" else ["window31"]
            for m in modes:
                doc.add(lambda s=s, ridx=ridx, m=m: self._random_page(s, ridx, m))

    def _threshold_page(self, s):
        d, js = self.ds[s], self.js[s]
        t = js["totals"]
        fig = page(
            f"{s}: values {self.chk}",
            "Map: long-term mean (grey) with the number of days beyond the threshold "
            "(blue). Right: histogram of those daily values.",
        )
        ax1 = fig.add_axes([0.04, 0.32, 0.45, 0.52])
        background(ax1, d["mean_new"].values, self.x, self.y)
        cnt = d["exceed_count"].values.astype(float)
        cnt[cnt == 0] = np.nan
        if np.isfinite(cnt).any():
            im = show_map(ax1, cnt, self.x, self.y, cmap=SEQ)
            cb = fig.colorbar(im, ax=ax1, shrink=0.7, pad=0.01)
            cb.set_label(f"days {self.chk}")
            cb.outline.set_visible(False)
        ax1.set_aspect("equal")
        ax2 = fig.add_axes([0.58, 0.36, 0.38, 0.45])
        hv = d["hist_values"].values
        if hv.size:
            ax2.hist(hv, bins=30, color="#2a78d6", alpha=0.85)
            ax2.set_xlabel(f"daily {self.var} ({self.u})")
            ax2.set_ylabel("cell-days")
            ax2.set_title(
                f"{t['hist_values_total']:,} cell-days {self.chk} ({g(100 * t['hist_values_total'] / max(t['valid_new'], 1), 2)} %)",
                loc="left",
            )
        else:
            ax2.axis("off")
            ax2.text(
                0.5,
                0.5,
                f"No values {self.chk}.",
                ha="center",
                va="center",
                fontsize=12,
            )
        rows = [["ERA5 (1965–2014)", f"{js['era5_exceed_total']:,}"]]
        if s != "historical" and "historical" in self.js:
            th = self.js["historical"]["totals"]
            rows += [
                [f"{self.model} historical, un-adjusted", f"{th['exceed_sim']:,}"],
                [f"{self.model} historical, downscaled", f"{th['exceed_new']:,}"],
            ]
        rows += [
            [f"{self.model} {s}, un-adjusted", f"{t['exceed_sim']:,}"],
            [f"{self.model} {s}, downscaled", f"{t['exceed_new']:,}"],
        ]
        table(
            fig,
            [0.58, 0.05, 0.38, 0.045 * (len(rows) + 1)],
            ["Source", f"cell-days {self.chk}"],
            rows,
            col_w=[0.7, 0.3],
            align=["left", "right"],
        )
        return fig

    def _top6_page(self, s):
        d = self.ds[s]
        iy, ix = d["top_iy"].values, d["top_ix"].values
        xs, ys = self.x[ix], self.y[iy]
        lon, lat = lonlat(xs, ys)
        fig = page(f"{s}: the 6 pixels with the most days {self.chk}")
        ax = fig.add_axes([0.04, 0.08, 0.5, 0.75])
        background(ax, d["mean_new"].values, self.x, self.y)
        ax.scatter(
            xs, ys, s=40, facecolor="#2a78d6", edgecolor="white", lw=1.2, zorder=3
        )
        for k in range(len(xs)):
            ax.annotate(
                str(k + 1),
                (xs[k], ys[k]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8.5,
                weight="bold",
            )
        ax.set_aspect("equal")
        ax.set_title("Long-term mean (grey) and pixel locations", loc="left")
        rows = [
            [
                str(k + 1),
                f"{xs[k]:.0f}",
                f"{ys[k]:.0f}",
                f"{lat[k]:.2f}",
                f"{lon[k]:.2f}",
                f"{int(d['top_count'].values[k]):,}",
            ]
            for k in range(len(xs))
        ]
        table(
            fig,
            [0.58, 0.45, 0.39, 0.32],
            ["#", "x (m)", "y (m)", "lat", "lon", f"days\n{self.chk}"],
            rows,
            col_w=[0.07, 0.17, 0.17, 0.14, 0.15, 0.3],
        )
        return fig

    def _ecdf_grid(self, s, kind, idx, mode, title, names=None):
        n = len(idx)
        ncol = int(np.ceil(n / 2)) if n > 3 else n
        nrow = int(np.ceil(n / ncol))
        sub = {
            "window31": "Days within ±15 days of the day-of-year of each pixel's most extreme downscaled value, all years.",
            "fire": "Fire-season days (May–Sep), all years.",
            "full": "All days, all years.",
        }[mode]
        fig = page(f"{s}: {title}", sub)
        axs = fig.subplots(nrow, ncol, squeeze=False, sharey=True)
        fig.subplots_adjust(
            left=0.06, right=0.98, top=0.83, bottom=0.15, hspace=0.45, wspace=0.08
        )
        for k, ax in enumerate(axs.flat):
            if k >= n:
                ax.axis("off")
                continue
            i = idx[k]
            src = self.series(s, kind, i)
            note = self.ecdf_panel(ax, s, src, mode)
            nm = names[k] if names else self._pixel_name(s, kind, i)
            ax.set_title(
                f"{nm}" + (f"   ({note})" if note else ""), loc="left", fontsize=9
            )
            if k % ncol == 0:
                ax.set_ylabel("ECDF")
        self.legend_fig(fig, s)
        return fig

    def _pixel_name(self, s, kind, i):
        d = self.ds[s]
        if kind == "top":
            return f"Pixel {i + 1} (x={self.x[d['top_ix'].values[i]]:.0f}, y={self.y[d['top_iy'].values[i]]:.0f})"
        return str(d["fixed_name"].values[i])

    def _doy_grid(self, s, kind, idx, title, names=None):
        fig = page(f"{s}: {title}", "Mean over all years for each day of year.")
        axs = fig.subplots(2, 3, squeeze=False, sharex=True)
        fig.subplots_adjust(
            left=0.06, right=0.98, top=0.83, bottom=0.15, hspace=0.35, wspace=0.18
        )
        for k, ax in enumerate(axs.flat):
            if k >= len(idx):
                ax.axis("off")
                continue
            src = self.series(s, kind, idx[k])
            for key in ("era5", "hist_sim", "sim", "hist_new", "new"):
                if key in src:
                    v, doy, _ = src[key]
                    xx, m = doy_mean(v, doy.astype(int))
                    lab, col = self.label(key, s)
                    ax.plot(
                        xx,
                        m,
                        color=col,
                        lw=1.3 if key in ("new", "era5") else 1.0,
                        label=lab,
                    )
            ax.set_title(
                names[k] if names else self._pixel_name(s, kind, idx[k]),
                loc="left",
                fontsize=9,
            )
            if k % 3 == 0:
                ax.set_ylabel(f"{self.var} ({self.u})")
            if k >= 3:
                ax.set_xlabel("day of year")
            ax.grid(alpha=0.25, lw=0.5)
        self.legend_fig(fig, s)
        return fig

    def _random_page(self, s, ridx, mode):
        d = self.ds[s]
        sub = {
            "window31": "ECDF over ±15 days around the day-of-year of the most extreme historical downscaled value.",
            "fire": "ECDF over fire-season days (May–Sep).",
            "full": "ECDF over all days.",
        }[mode]
        fig = page(f"{s}: random locations", sub)
        gs = fig.add_gridspec(
            2, 3, left=0.05, right=0.98, top=0.83, bottom=0.15, hspace=0.45, wspace=0.2
        )
        ax = fig.add_subplot(gs[0, 0])
        background(ax, d["mean_new"].values, self.x, self.y)
        iy, ix = d["fixed_iy"].values[ridx], d["fixed_ix"].values[ridx]
        ax.scatter(
            self.x[ix],
            self.y[iy],
            s=30,
            facecolor="#2a78d6",
            edgecolor="white",
            lw=1,
            zorder=3,
        )
        for k in range(len(ridx)):
            ax.annotate(
                str(k + 1),
                (self.x[ix[k]], self.y[iy[k]]),
                xytext=(3, 3),
                textcoords="offset points",
                weight="bold",
                fontsize=8,
            )
        ax.set_aspect("equal")
        ax.set_title("Locations", loc="left", fontsize=9)
        cells = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for k, i in enumerate(ridx[: len(cells)]):
            a = fig.add_subplot(gs[cells[k]])
            note = self.ecdf_panel(
                a, s, self.series(s, "fixed", i), mode, center_from="hist_new"
            )
            a.set_title(
                f"Location {k + 1}" + (f"  ({note})" if note else ""),
                loc="left",
                fontsize=9,
            )
            if cells[k] == (1, 0):
                a.set_ylabel("ECDF")
        self.legend_fig(fig, s)
        return fig

    # ------------------------------------------------------------------ dtr extras
    def _dtr_stage_maps(self, s, stat):
        d = self.ds[s]
        stages = [
            (f"{st}_{stat}", f"{st}_lon", f"{st}_lat", t)
            for st, t in (
                ("source", "Source"),
                ("first_regrid", "First regrid"),
                ("second_regrid", "Second regrid"),
            )
        ]
        panels = [
            (d[v].values, d[lo].values, d[la].values, t, True)
            for v, lo, la, t in stages
            if v in d
        ]
        panels.append(
            (
                d[f"{stat}_sim"].values,
                self.x,
                self.y,
                "Final regrid (un-adjusted)",
                False,
            )
        )
        panels.append(
            (d[f"{stat}_new"].values, self.x, self.y, "Downscaled (new)", False)
        )
        panels.append((self.era5[f"era5_{stat}"].values, self.x, self.y, "ERA5", False))
        vals = np.concatenate([p[0][np.isfinite(p[0])].ravel() for p in panels])
        vmin, vmax = np.percentile(vals, [1, 99])
        fig = page(
            f"{s}: {dict(min='Minimum', mean='Mean', max='Maximum')[stat]} dtr at each processing stage",
            f"Shared colour range (1st–99th percentile of all panels), {self.u}.",
        )
        n = len(panels)
        gs = fig.add_gridspec(
            2, 3, left=0.03, right=0.9, top=0.84, bottom=0.05, wspace=0.06, hspace=0.2
        )
        im = None
        for k, (a, xx, yy, t, ll) in enumerate(panels[:6]):
            ax = fig.add_subplot(gs[k // 3, k % 3])
            if ll:
                im = ax.pcolormesh(
                    xx, yy, a, cmap="coolwarm", vmin=vmin, vmax=vmax, shading="auto"
                )
                ax.set_xticks([]), ax.set_yticks([])
            else:
                im = show_map(ax, a, xx, yy, cmap="coolwarm", vmin=vmin, vmax=vmax)
                ax.set_aspect("equal")
            ax.set_title(t, loc="left", fontsize=9)
        cax = fig.add_axes([0.92, 0.25, 0.012, 0.45])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(self.u)
        cb.outline.set_visible(False)
        return fig

    def _dtr_top_days(self, s):
        d = self.ds[s]
        k = d["td_value"].size
        it, iy, ix = d["td_it"].values, d["td_iy"].values, d["td_ix"].values
        lon, lat = lonlat(self.x[ix], self.y[iy])
        ymd = d["ymd"].values[it]
        fig = page(
            f"{s}: the {k} most extreme dtr pixel-days ({self.js[s]['totals']['exceed_new']:,} cell-days {self.chk})",
            "With the downscaled tasmax and derived tasmin on the same day. ECDFs: all days of the scenario "
            "(ERA5: 1965–2014) at each pixel.",
        )
        rows = [
            [
                f"{ymd[j] // 10000}-{ymd[j] // 100 % 100:02d}-{ymd[j] % 100:02d}",
                f"{lat[j]:.2f}",
                f"{lon[j]:.2f}",
                g(float(d["td_value"][j]), 4),
                g(float(d["td_tasmax"][j]), 5),
                g(float(d["td_tasmin"][j]), 5),
            ]
            for j in range(k)
        ]
        h = 0.05 * (k + 1)
        table(
            fig,
            [0.03, 0.8 - h, 0.42, h],
            ["Date", "lat", "lon", "dtr (K)", "tasmax (K)", "tasmin (K)"],
            rows,
            size=8,
        )
        gs = fig.add_gridspec(
            2, 5, left=0.48, right=0.98, top=0.83, bottom=0.15, hspace=0.5, wspace=0.12
        )
        for j in range(min(k, 10)):
            ax = fig.add_subplot(gs[j // 5, j % 5])
            src = self.series(s, "td", j)
            for key in ("sim", "new", "era5"):
                if key in src:
                    lab, col = self.label(key, s)
                    ecdf(ax, src[key][0], col, lab, lw=1.2, z=5 if key == "era5" else 2)
            ax.set_title(f"#{j + 1}", loc="left", fontsize=8.5)
            if j >= 5:
                ax.set_xlabel("dtr (K)", fontsize=8)
            ax.tick_params(labelsize=7)
            if j % 5:
                ax.set_yticklabels([])
        handles = [
            plt.Line2D(
                [], [], color=self.label(k2, s)[1], lw=2, label=self.label(k2, s)[0]
            )
            for k2 in ("era5", "sim", "new")
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=3,
            fontsize=8.5,
            bbox_to_anchor=(0.5, 0.025),
        )
        return fig

    # ------------------------------------------------------------------ title page & build
    def title_page(self, doc):
        T = self.cfg["text"]
        res_label = self.cfg["resolutions"][self.res]["label"]
        title = T["title"].format(
            var=self.var, long_name=self.v["long_name"], model=self.model, res=res_label
        )

        def draw():
            fig = page(title, T["subtitle"])
            y = paragraph(fig, 0.04, 0.85, T["intro"], size=10)
            js0 = self.js[self.scens[0]]
            facts = [
                ("Scenarios", ", ".join(self.scens)),
                ("QC check", f"daily values {self.chk}"),
                ("New outputs", str(Path(js0["paths"]["new"]).parent)),
                (
                    "Previous outputs",
                    str(Path(js0["paths"]["previous"]).parent)
                    if js0["paths"]["previous"]
                    else "none",
                ),
                ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ]
            for k, v in js0.get("new_attrs", {}).items():
                facts.append((k, str(v)))
            y -= 0.02
            for k, v in facts[:12]:
                fig.text(0.04, y, k, fontsize=8.5, color=MUTED, va="top")
                fig.text(0.2, y, textwrap.shorten(v, 140), fontsize=8.5, va="top")
                y -= 0.026
            y -= 0.02
            fig.text(0.04, y, "Contents", fontsize=12, weight="bold", va="top")
            y -= 0.035
            for level, name, pg in doc.toc:
                if level > 1:
                    continue
                fig.text(
                    0.06 + 0.03 * level,
                    y,
                    name,
                    fontsize=9.5 if level == 0 else 9,
                    va="top",
                    weight="bold" if level == 0 else "normal",
                )
                fig.text(0.6, y, str(pg), fontsize=9, va="top", ha="right", color=MUTED)
                y -= 0.024
                if y < 0.05:
                    break
            return fig

        return draw

    def build(self, png_dir=None):
        pdf_dir = out_dir(self.cfg, self.res) / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        path = pdf_dir / f"qc_{self.var}_{self.model}_{self.res}.pdf"
        res_label = self.cfg["resolutions"][self.res]["label"]
        doc = Doc(
            path,
            f"{self.var} · {self.model} · {res_label} · downscaled CMIP6 QC",
            png_dir,
        )
        doc.add(None)  # title page placeholder
        self.plan_comparison(doc)
        self.plan_qc(doc)
        doc.pages[0] = self.title_page(doc)
        n = doc.write(
            {
                "Title": f"QC {self.var} {self.model} {res_label}",
                "Subject": "Downscaled CMIP6 QC",
            }
        )
        print(f"wrote {path} ({n} pages)")
        return path


# =============================================================================== summary


def summary(cfg, res, png_dir=None):
    data = out_dir(cfg, res) / "data"
    res_label = cfg["resolutions"][res]["label"]
    tq = cfg["tail_quantile"]
    levels = np.asarray(cfg["comparison_quantiles"])
    iq, i50 = int(np.argmin(abs(levels - tq))), int(np.argmin(abs(levels - 0.5)))
    rows, profiles, missing = [], {}, []
    for var, model in jobs_for(cfg, res):
        for s in scenarios_for(cfg, res, var, model):
            f = data / f"{var}_{model}_{s}.nc"
            if not f.exists():
                missing.append(f"{var} {model} {s}")
                continue
            js = json.loads(f.with_suffix(".json").read_text())
            t = js["totals"]
            with xr.open_dataset(f) as d:
                r = {
                    "var": var,
                    "model": model,
                    "scen": s,
                    "exceed_new": t["exceed_new"],
                    "exceed_sim": t["exceed_sim"],
                    "units": fmt_units(js["units"]),
                    "check": check_label(js["check"], fmt_units(js["units"])),
                }
                if "q_prev" in d:
                    dq = (d["q_new"] - d["q_prev"]).values.reshape(len(levels), -1)
                    pairs = max(t["pairs"], 1)
                    r.update(
                        dmean=float(np.nanmean(np.abs(d["mean_new"] - d["mean_prev"]))),
                        dq50=float(np.nanmean(np.abs(dq[i50]))),
                        dqt=float(np.nanmean(np.abs(dq[iq]))),
                        maxad=t["max_abs_diff"],
                        equal=100 * t["equal"] / pairs,
                        nanmm=t["nan_mismatch"],
                    )
                    with np.errstate(all="ignore"):
                        profiles.setdefault(var, []).append(np.nanmean(np.abs(dq), 1))
                rows.append(r)
    pdf_dir = out_dir(cfg, res) / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    path = pdf_dir / f"qc_summary_{res}.pdf"
    doc = Doc(path, f"{res_label} · downscaled CMIP6 QC summary", png_dir)
    doc.add(None)
    doc.section(
        "A. New vs previous: mean |Δ| by quantile level, all models and scenarios"
    )
    doc.add(lambda: _summary_profiles(cfg, profiles, levels, rows))
    variables = list(dict.fromkeys(r["var"] for r in rows))
    per_page = 22
    for v in variables:
        vr = [r for r in rows if r["var"] == v]
        doc.section(f"{v}: per model and scenario")
        for p0 in range(0, len(vr), per_page):
            doc.add(lambda vr=vr[p0 : p0 + per_page], v=v: _summary_table(cfg, v, vr))

    def title():
        fig = page(
            cfg["text"]["summary_title"].format(res=res_label), cfg["text"]["subtitle"]
        )
        y = paragraph(
            fig,
            0.04,
            0.85,
            f"{len(rows)} outputs summarised"
            + (
                f"; {len(missing)} without QC statistics yet: "
                + ", ".join(missing[:20])
                if missing
                else ""
            )
            + ".",
        )
        y = paragraph(fig, 0.04, y - 0.01, cfg["text"]["comparison_intro"], color=MUTED)
        y -= 0.03
        fig.text(0.04, y, "Contents", fontsize=12, weight="bold", va="top")
        y -= 0.035
        for _, name, pg in doc.toc:
            fig.text(0.06, y, name, fontsize=9.5, va="top")
            fig.text(0.7, y, str(pg), fontsize=9, va="top", ha="right", color=MUTED)
            y -= 0.025
        return fig

    doc.pages[0] = title
    n = doc.write({"Title": f"QC summary {res_label}"})
    print(f"wrote {path} ({n} pages)")


def _summary_profiles(cfg, profiles, levels, rows):
    fig = page(
        "New vs previous: mean |Δ quantile| per grid cell, by quantile level",
        "One line per model × scenario. Small through the bulk of the distribution, larger in the tails.",
    )
    vs = list(profiles)
    n = max(len(vs), 1)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    axs = fig.subplots(nrow, ncol, squeeze=False)
    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.82, bottom=0.1, hspace=0.45, wspace=0.3
    )
    for k, ax in enumerate(axs.flat):
        if k >= len(vs):
            ax.axis("off")
            continue
        v = vs[k]
        for p in profiles[v]:
            ax.plot(levels, p, color="#2a78d6", alpha=0.35, lw=0.8)
        ax.set_xscale("logit")
        ax.set_yscale("log")
        ax.set_xticks([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])
        ax.set_xticklabels(
            ["0.001", "0.01", "0.1", "0.5", "0.9", "0.99", "0.999"],
            rotation=45,
            fontsize=7,
        )
        ax.minorticks_off()
        u = next(r["units"] for r in rows if r["var"] == v)
        ax.set_title(f"{v} ({u}), n={len(profiles[v])}", loc="left")
        ax.set_ylabel(f"mean |Δ| ({u})")
        ax.grid(alpha=0.25, lw=0.5)
    return fig


def _summary_table(cfg, v, vr):
    u, chk = vr[0]["units"], vr[0]["check"]
    tq = 100 * cfg["tail_quantile"]
    fig = page(
        f"{v}: new vs previous, and QC threshold counts",
        f"Units {u}. Threshold check: daily values {chk}.",
    )
    header = [
        "Model",
        "Scenario",
        "|Δ mean|",
        "|Δ median|",
        f"|Δ {tq:g}th|",
        "max |Δ daily|",
        "Identical days",
        "NaN-mask diffs",
        "beyond thr. (new)",
        "beyond thr. (un-adj.)",
    ]
    body = [
        [
            r["model"],
            r["scen"],
            g(r.get("dmean")),
            g(r.get("dq50")),
            g(r.get("dqt")),
            g(r.get("maxad")),
            f"{r['equal']:.1f} %" if "equal" in r else "–",
            f"{r['nanmm']:,}" if "nanmm" in r else "–",
            f"{r['exceed_new']:,}",
            f"{r['exceed_sim']:,}",
        ]
        for r in vr
    ]
    table(
        fig,
        [0.03, 0.86 - 0.033 * (len(body) + 1) - 0.02, 0.94, 0.033 * (len(body) + 1)],
        header,
        body,
        align=["left", "left"] + ["right"] * 8,
        size=7.5,
    )
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model")
    m.add_argument("--res", required=True)
    m.add_argument("--var", required=True)
    m.add_argument("--model", required=True)
    m.add_argument("--config")
    m.add_argument("--png_dir", help="also save each page as a PNG (for previewing)")
    s = sub.add_parser("summary")
    s.add_argument("--res", required=True)
    s.add_argument("--config")
    s.add_argument("--png_dir", help="also save each page as a PNG (for previewing)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "model":
        ModelReport(cfg, a.res, a.var, a.model).build(a.png_dir)
    else:
        summary(cfg, a.res, a.png_dir)
