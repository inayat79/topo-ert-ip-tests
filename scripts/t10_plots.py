import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import setup_geom as G
from synth_common import *
from synth_common import _outline
import journal_style as J                                    # manuscript Figs. 2, 7, 8, 9 (journal style)
mt = G.make_mesh(True); mf = G.make_mesh(False); x, z = ele_xz(mt)

# ---------- F01 validation / convergence (numbers from t02 / t04 logs) — manuscript Fig. 2 ----------
fig, ax = J.figure(1, 2, height=8)
dxs = [5, 4, 2.5]; n1 = [2.31, 1.49, 0.57]; mean = [0.75, 0.47, 0.17]
ax[0].plot(dxs, n1, 'o-', color=J.PS.C_OBS, ms=8, mec='k', mew=0.8, label='max |error| (n = 1)')
ax[0].plot(dxs, mean, 's-', color=J.PS.C_FWD, ms=8, mec='k', mew=0.8, label='mean |error|')
ax[0].axhline(0, color='k', lw=.5); ax[0].invert_xaxis(); J.set_xlabel(ax[0], 'Horizontal cell size (m)')
J.set_ylabel(ax[0], r'$\rho_a$ error vs analytic (%)'); J.style_legend(ax[0])
lab = ['2.5 m, damp 60', '2.5 m, damp 30', '2.5 m, undamped', '1.25 m, damp 60']
ax[1].bar(lab, [0, 0.34, 0.41, 0.60], color='gray', edgecolor='k', linewidth=1.2)
J.set_ylabel(ax[1], r'max |$\Delta\rho_a$| vs baseline (%)')
J.finish(fig, ax, xlabel=None, shared=False)
ax[1].tick_params(axis='x', rotation=30)
for t in ax[1].get_xticklabels(): t.set_ha('right')
J.save(fig, 'F01_validation')

# ---------- F04 noise robustness — manuscript Fig. 7 ----------
fig, ax = J.figure(2, height=10, sharex=True)
for i, (tag, hang, lab) in enumerate([('N5_flat', True, 'Flat-earth'),
                                      ('N5_topo', False, 'Topography-corrected')]):
    r = np.load(f'res_{tag}.npz')
    plot_section(ax[i], mf if hang else mt, r['rho_e'], lab, hang=hang, kind='rho')
J.finish(fig, ax); J.save(fig, 'F04_noise5')

# ---------- F05 IP offset test — manuscript Fig. 9 ----------
fig, ax = J.figure(4, height=19, sharex=True)
for i, sc in enumerate(('coincident', 'offset')):
    plot_section(ax[2*i], mt, true_charg(x, z, sc), f'True, {sc}', kind='chg', cblabel='m (%)', charg_outline=sc,
                 outline=False, bodies=[(sc, 12.0, 1.0)])
    me = np.load(f'ip_A_topo_L1_z05_{sc}.npy')
    plot_section(ax[2*i+1], mt, me, f'Recovered, {sc}', kind='chg', cblabel='m (%)', charg_outline=sc, outline=False)
for a in ax: _outline(a, 'C')                                 # conductor only (resistive block is not part of the IP test)
from matplotlib.lines import Line2D; import matplotlib.patheffects as pe
_st = [pe.withStroke(linewidth=4.5, foreground='k')]          # white outline made visible in the white legend box
J.style_legend(ax[0], loc='lower left', handles=[
    Line2D([], [], color='white', lw=2.0, ls=ls, path_effects=_st, label=lab)
    for ls, lab in (('--', 'Conductor'), (':', 'Chargeable body'))])   # panel (a) only
J.finish(fig, ax); J.save(fig, 'F05_ip_offset_test')

# ---------- F06 pseudosection offset for coincident bodies — manuscript Fig. 8 ----------
fig, ax = J.figure(2, height=10, sharex=True)
ra = np.load('rhoa_clean.npy'); ma = np.load('ma_clean_coincident.npy')
J.pseudosection(fig, ax[0], G.xmid, G.nvals, ra, 'rho', r'$\rho_a$ (Ω·m)')
J.pseudosection(fig, ax[1], G.xmid, G.nvals, ma, 'chg', r'$m_a$ (%)')
i, j = np.argmin(ra), np.argmax(ma)
for a in ax:
    a.plot(G.xmid[i], G.nvals[i], 'wx', ms=18, mew=3.5); a.plot(G.xmid[j], G.nvals[j], 'r+', ms=20, mew=3.5)
    a.set_xlim(-20, 920)
J.finish(fig, ax); J.save(fig, 'F06_pseudo_offset_coincident')
print('plots done')
