import numpy as np, time, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import setup_geom as G
from ert25d import Forward
t=time.time()
mflat = G.make_mesh(topo=False); mtopo = G.make_mesh(topo=True, damp_depth=60.0)
ff = Forward(mflat, G.elec_x, G.abmn, G.A); ft = Forward(mtopo, G.elec_x, G.abmn, G.A)
corr = ff.response(np.full(len(mflat.tri), 1.0))        # numerical K correction
np.save('corr.npy', corr)
r_flat = ff.response(np.full(len(mflat.tri), 100.0)) / corr
r_topo = ft.response(np.full(len(mtopo.tri), 100.0)) / corr
e = 100 * (r_topo / 100 - 1)
print(f'flat-earth check after correction: max|err| {np.abs(100*(r_flat/100-1)).max():.4f}%')
print(f'TOPOGRAPHIC EFFECT on homogeneous 100 ohm.m: min {r_topo.min():.1f}  max {r_topo.max():.1f} ohm.m')
print(f'   distortion range {e.min():+.1f}% to {e.max():+.1f}%   mean |dist| {np.abs(e).mean():.1f}%')
for n in range(1,6):
    s=G.nvals==n; print(f'   n={n}: {e[s].min():+6.1f}% .. {e[s].max():+6.1f}%')
np.save('topo_effect.npy', r_topo)
import journal_style as J                                   # plotting only (manuscript Fig. 3)
fig, ax = J.figure(2, height=9, sharex=True, gridspec_kw=dict(height_ratios=[1, 0.6]))
J.pseudosection(fig, ax[0], G.xmid, G.nvals, e, ('div', 60), r'$\rho_a$ distortion (%)')
xs = np.linspace(-20, 920, 500); ax[1].plot(xs, G.surf_topo(xs), 'k')
J.electrodes(ax[1], G.elec_x, G.surf_topo(G.elec_x)); J.style_legend(ax[1], loc='lower right')
J.set_ylabel(ax[1], 'Elevation (m)'); J.blank_colorbar(fig, ax[1])
for a in ax: a.set_xlim(-20, 920)
J.finish(fig, ax); J.save(fig, 'F02_topographic_effect')
print(f'time {time.time()-t:.1f}s')
