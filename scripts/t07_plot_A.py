import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import setup_geom as G
from synth_common import *
import journal_style as J                                    # manuscript Fig. 4 (journal style)
mt = G.make_mesh(True); mf = G.make_mesh(False)
x, z = ele_xz(mt)
fig, ax = J.figure(5, height=J.MAX_H, sharex=True)
plot_section(ax[0], mt, true_model(x, z), 'True model', kind='rho', bodies=[('C', 5.0, 800.0), ('R', 3000.0, 800.0)])
for i, (tag, hang, lab) in enumerate([('A_flat_L2', True, 'Flat-earth, L2'),
                                      ('A_flat_L1', True, 'Flat-earth, L1'),
                                      ('A_topo_L2', False, 'Topography-corrected, L2'),
                                      ('A_topo_L1', False, 'Topography-corrected, L1')], 1):
    r = np.load(f'res_{tag}.npz')
    plot_section(ax[i], mf if hang else mt, r['rho_e'], lab, hang=hang, kind='rho')
J.finish(fig, ax); J.save(fig, 'F03_testA_sections')
