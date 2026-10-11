import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import setup_geom as G
from synth_common import *
from synth_common import _outline
import journal_style as J                                    # manuscript Fig. 5 (journal style)
mt = G.make_mesh(True); mf = G.make_mesh(False); x, z = ele_xz(mt)
fig, ax = J.figure(3, height=14, sharex=True)
plot_section(ax[0], mt, true_model(x, z, 'B'), 'True model', outline=False, kind='rho',
             bodies=[('CB', 5.0, 800.0), ('R', 3000.0, 800.0)])
for i, (tag, hang, lab) in enumerate([('B_flat', True, 'Flat-earth'), ('B_topo', False, 'Topography-corrected')], 1):
    r = np.load(f'res_{tag}.npz')
    plot_section(ax[i], mf if hang else mt, r['rho_e'], lab, hang=hang, outline=False, kind='rho')
for a in ax: _outline(a, 'CB'); _outline(a, 'R')
J.finish(fig, ax); J.save(fig, 'F08_geometryB'); print('ok')
