"""Journal style for the manuscript figures (Fig. 1-10): ert_plot_style in journal mode plus the
rules shared by every figure. Plotting only - imported lazily by synth_common.plot_section and by
the figure scripts; no computational step imports it."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # ert_plot_style.py lives next to this file
import ert_plot_style as PS
from ert_plot_style import (decorate_ax, add_colorbar, style_legend, panel_title, set_xlabel, set_ylabel,
                            CMAP_RHO, FS_LEGEND)
import matplotlib.pyplot as plt, matplotlib.colors as mcol
from matplotlib.ticker import LogFormatterMathtext, NullLocator
from matplotlib.transforms import ScaledTranslation

PS.SHOW_TITLES = False                     # journal mode: no suptitles, panel titles only "(a)", "(b)" ...
FIG_W = 20.0                               # every figure 20 in wide (as Fig. 1) -> same scale at 7 in print
MAX_H = 22.4                               # tight-cropped h/w <= 1.143 -> printed height <= 8 in at 7 in

# same quantity = same colour map and limits everywhere
RHO = dict(cmap=CMAP_RHO, norm=lambda: mcol.LogNorm(2, 4000), ticks=[10, 100, 1000])
CHG = dict(cmap='viridis', norm=lambda: mcol.Normalize(0, 14), ticks=list(range(0, 15, 2)))
DIV = 'RdBu_r'                             # ratios, distortions (and signed misfit), symmetric limits
OUTLINE = dict(colors='white', linewidths=2.0, linestyles='--')   # true bodies, every figure
ELEC = dict(c='k', s=220, marker='v', edgecolors='white', linewidths=1.2, zorder=7)  # as Fig. 1
PSEUDO_S = 260                             # pseudosection squares, as Fig. 1
LBL_N = 'Separation factor n'


def figure(nrows, ncols=1, height=None, **kw):
    h = height or min(MAX_H, 4.6 * nrows)
    assert h <= MAX_H + 1e-9, f'figure height {h} > {MAX_H}'
    return plt.subplots(nrows, ncols, figsize=(FIG_W, h), **kw)

def colorbar(fig, mappable, ax, label, ticks=None, log=False):
    cb = add_colorbar(fig, mappable, ax, label)
    if ticks is not None:
        cb.set_ticks(ticks)
        if log:
            cb.ax.yaxis.set_major_formatter(LogFormatterMathtext()); cb.ax.yaxis.set_minor_locator(NullLocator())
        for t in cb.ax.get_yticklabels():
            t.set_fontfamily('serif'); t.set_fontweight('bold'); t.set_fontsize(PS.FS_CTICK)
    return cb

def blank_colorbar(fig, ax):
    """Invisible colourbar so panels without one keep the same width as the others (as Fig. 1)."""
    cb = plt.colorbar(plt.cm.ScalarMappable(), ax=ax, pad=0.02, fraction=0.030); cb.ax.set_visible(False)

def corner_label(ax, txt):
    """Short in-panel label (<= 3 words): same font and corner (upper right) in every figure."""
    if txt:
        ax.text(0.99, 0.95, txt, transform=ax.transAxes, ha='right', va='top', fontsize=FS_LEGEND,
                fontweight='bold', fontfamily='serif', zorder=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.92, edgecolor='black', linewidth=2.0))

def electrodes(ax, x, z, above=False):
    """Electrodes as in Fig. 1. above=True sits the triangles on top of the surface (sections),
    so they do not hide the shallow model."""
    tr = ax.transData + ScaledTranslation(0, 7 / 72, ax.figure.dpi_scale_trans) if above else ax.transData
    return ax.scatter(x, z, transform=tr, clip_on=not above, label='Electrodes', **ELEC)

def finish(fig, axes, xlabel='Distance (m)', shared=True):
    """Panel labels (a), (b) ...; spines/ticks/grid; x label only on the bottom panel of stacked figures."""
    axes = list(np.atleast_1d(axes).ravel())
    for i, ax in enumerate(axes):
        panel_title(ax, i, '')
        decorate_ax(ax)
        if shared and ax is not axes[-1]:
            ax.set_xlabel('')
    if xlabel:
        set_xlabel(axes[-1], xlabel)

def pseudosection(fig, ax, xmid, nvals, vals, kind, label):
    """Pseudosection panel exactly as Fig. 1 (squares at the quadrupole midpoints, n downward)."""
    if kind == 'rho':
        sc = ax.scatter(xmid, nvals, c=vals, cmap=RHO['cmap'], norm=RHO['norm'](), s=PSEUDO_S, marker='s', edgecolors='none')
        colorbar(fig, sc, ax, label, RHO['ticks'], log=True)
    elif kind == 'chg':
        sc = ax.scatter(xmid, nvals, c=vals, cmap=CHG['cmap'], norm=CHG['norm'](), s=PSEUDO_S, marker='s', edgecolors='none')
        colorbar(fig, sc, ax, label, CHG['ticks'])
    else:                                  # ('div', limit): diverging, symmetric limits
        lim = kind[1]
        sc = ax.scatter(xmid, nvals, c=vals, cmap=DIV, norm=mcol.Normalize(-lim, lim), s=PSEUDO_S, marker='s', edgecolors='none')
        colorbar(fig, sc, ax, label)
    ax.set_ylim(5.5, 0.5); ax.set_yticks(range(1, 6)); set_ylabel(ax, LBL_N)
    return sc

def save(fig, name, out_dir='.'):
    """PNG (300 dpi, white, not transparent) + PDF via ert_plot_style.savefig (atomic)."""
    fig.patch.set_facecolor('white'); fig.patch.set_alpha(1.0)
    PS.savefig(fig, name, out_dir)
