"""
ert_plot_style.py
Figure style extracted from Pyjimli_ERT_Pangea.ipynb (GN forward/inversion cell).
Import this module at the top of any plotting script:  from ert_plot_style import *
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # figures saved to disk, not shown
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# ── Global rcParams ─────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 18, "font.weight": "bold",
    "axes.labelsize": 24, "axes.labelweight": "bold",
    "axes.titlesize": 26, "axes.titleweight": "bold",
    "axes.linewidth": 3.0,
    "xtick.labelsize": 22, "ytick.labelsize": 22,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 9, "ytick.major.size": 9,
    "xtick.major.width": 2.4, "ytick.major.width": 2.4,
    "legend.fontsize": 22, "legend.framealpha": 0.95,
    "legend.edgecolor": "black", "legend.fancybox": False,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "lines.linewidth": 2.8,
})

# ── Font sizes (pt) ─────────────────────────────────────────────
FS_PANEL, FS_SUBPLOT, FS_SINGLE = 22, 22, 26
FS_AXLABEL, FS_TICK = 24, 22
FS_CBAR, FS_CTICK = 24, 22
FS_LEGEND, FS_STAT = 22, 18

# ── Colours ─────────────────────────────────────────────────────
CMAP_RHO  = plt.cm.rainbow          # resistivity sections
CMAP_MISS = "RdBu_r"                # misfit / difference (symmetric about 0)
CMAP_COND = plt.cm.viridis_r        # conductivity sections (mS/m = 1000/rho)
C_OBS, C_FWD = "steelblue", "tomato"
C_RELIABLE, C_UNRELIABLE = "#2E7D32", "#FF9800"
C_TABLE_HEAD, C_TABLE_ALT = "#4472C4", "#F2F2F2"
C_SIREN_IN, C_SIREN_HID, C_SIREN_OUT = "#4CAF50", "#2196F3", "#F44336"

TRAPEZOID_TAPER = 0.55
SHOW_TITLES = True                  # set False for journal versions (captions go in manuscript)


# ── Text helpers ────────────────────────────────────────────────
def apply_tnr(fig):
    for obj in fig.findobj(matplotlib.text.Text):
        try:
            obj.set_fontfamily("serif"); obj.set_fontweight("bold")
        except Exception:
            pass

def set_xlabel(ax, txt, fontsize=FS_AXLABEL):
    ax.set_xlabel(txt, fontsize=fontsize, fontweight="bold", fontfamily="serif")

def set_ylabel(ax, txt, fontsize=FS_AXLABEL):
    ax.set_ylabel(txt, fontsize=fontsize, fontweight="bold", fontfamily="serif")

def set_title(ax, txt, fontsize=FS_SINGLE, pad=16):
    if SHOW_TITLES and txt:
        ax.set_title(txt, fontsize=fontsize, fontweight="bold", fontfamily="serif", pad=pad)

def panel_title(ax, idx, txt, fontsize=FS_SUBPLOT, pad=14):
    """Subplot title with panel label: '(a) Title'. Panel label kept even if SHOW_TITLES=False."""
    label = f"({chr(97 + idx)})"
    ax.set_title(f"{label} {txt}" if SHOW_TITLES else label, fontsize=fontsize,
                 fontweight="bold", fontfamily="serif", pad=pad)


# ── Axes / colourbar / legend ───────────────────────────────────
def decorate_ax(ax, lw=3.0, ticksize=FS_TICK, grid_alpha=0.15):
    for sp in ax.spines.values():
        sp.set_linewidth(lw); sp.set_visible(True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="both", which="major", labelsize=ticksize, width=2.4, length=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("serif"); lbl.set_fontweight("bold"); lbl.set_fontsize(ticksize)
    ax.grid(True, alpha=grid_alpha, ls="--", lw=0.9)

def add_colorbar(fig_or_plt, im, ax, label):
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.030)
    cb.set_label(label, fontsize=FS_CBAR, fontweight="bold", fontfamily="serif")
    cb.ax.tick_params(labelsize=FS_CTICK, width=2.4, length=9)
    for t in cb.ax.get_yticklabels():
        t.set_fontfamily("serif"); t.set_fontweight("bold"); t.set_fontsize(FS_CTICK)
    cb.outline.set_linewidth(2.4)
    return cb

def style_legend(ax, fontsize=FS_LEGEND, **kwargs):
    leg = ax.legend(fontsize=fontsize, **kwargs)
    if leg:
        leg.get_frame().set_linewidth(2.4)
        for t in leg.get_texts():
            t.set_fontfamily("serif"); t.set_fontweight("bold"); t.set_fontsize(fontsize)
    return leg

def stats_box(ax, txt, loc="lower right", fs=FS_STAT):
    xp = 0.97 if "right" in loc else 0.03
    yp = 0.05 if "lower" in loc else 0.95
    ax.text(xp, yp, txt, transform=ax.transAxes,
            ha="right" if "right" in loc else "left",
            va="bottom" if "lower" in loc else "top",
            fontsize=fs, fontfamily="serif", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.92,
                      edgecolor="black", linewidth=2.2))


# ── Colour limits ───────────────────────────────────────────────
def norm_percentile(values, lo=2, hi=98):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    return Normalize(np.percentile(v, lo), np.percentile(v, hi))

def norm_symmetric(values, pct=97, floor=0.1):
    v = np.abs(np.asarray(values, float)); v = v[np.isfinite(v)]
    lim = max(np.percentile(v, pct), floor)
    return Normalize(-lim, lim)


# ── Trapezoid-masked ERT section ────────────────────────────────
def _trap_mask(xi, zi, x_min, x_max, z_max, taper, pad=0.75):
    X, Z = np.meshgrid(xi, zi)
    center = 0.5 * (x_min + x_max)
    hw = (0.5 * (x_max - x_min) + pad) * (1.0 - taper * np.clip(Z / z_max, 0, 1))
    return (np.abs(X - center) <= hw) & (Z >= 0) & (Z <= z_max)

def draw_trapezoid_outline(ax, x_min, x_max, z_max, taper=TRAPEZOID_TAPER):
    c = 0.5 * (x_min + x_max); top = 0.5 * (x_max - x_min); bot = top * (1 - taper)
    xs = [c - top, c + top, c + bot, c - bot, c - top]
    zs = [0, 0, z_max, z_max, 0]
    ax.plot(xs, zs, "-", lw=2.5, color="k", alpha=0.9, zorder=6)

def plot_trap_section(ax, grid, xi, zi, x_min, x_max, z_max, norm, cmap,
                      elec_x=None, meas_x=None, meas_z=None, taper=TRAPEZOID_TAPER,
                      elec_offset=-0.25, title=None):
    d = np.asarray(grid, float).copy()
    d[~_trap_mask(xi, zi, x_min, x_max, z_max, taper)] = np.nan
    im = ax.imshow(d, extent=[xi.min(), xi.max(), zi.max(), zi.min()], aspect="auto",
                   origin="upper", norm=norm, cmap=cmap, interpolation="bilinear", zorder=2)
    draw_trapezoid_outline(ax, x_min, x_max, z_max, taper)
    if meas_x is not None:
        ax.scatter(meas_x, meas_z, c="k", s=4, marker=".", alpha=0.55, zorder=4)
    if elec_x is not None:
        ax.scatter(elec_x, np.full_like(elec_x, elec_offset, dtype=float), c="k", s=60,
                   marker="v", zorder=7, edgecolors="white", linewidths=0.9, label="Electrodes")
    ax.set_xlim(x_min - 1.5, x_max + 1.5)
    ax.set_ylim(z_max * 1.05, elec_offset - 0.3)   # depth increases downward
    set_title(ax, title)
    return im


# ── Saving (PNG + PDF, atomic write) ────────────────────────────
def savefig(fig, name, out_dir, dpi=300, exts=(".png", ".pdf")):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    apply_tnr(fig); fig.tight_layout()
    for ext in exts:
        tmp, final = out / f"__tmp_{name}{ext}", out / f"{name}{ext}"
        fig.savefig(tmp, dpi=dpi, bbox_inches="tight")
        if final.exists():
            final.unlink()
        tmp.rename(final)
    plt.close(fig)
