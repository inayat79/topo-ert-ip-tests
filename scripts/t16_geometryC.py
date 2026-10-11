"""Generalisation test 2: conductor STRADDLING the steepest slope break (x 260-280 m). Same settings as models A and B."""
import numpy as np, json, os, time, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import setup_geom as G
from synth_common import *
from synth_common import _outline
from ert25d import Forward, ParaGrid, invert
t = time.time()
if not os.path.exists('corr.npy'):
    mf = G.make_mesh(False); np.save('corr.npy', Forward(mf, G.elec_x, G.abmn, G.A).response(np.ones(len(mf.tri))))
if not os.path.exists('rhoa_C_noise3.npy'):
    mt = fine_mesh(True); mf = fine_mesh(False)
    corr_f = Forward(mf, G.elec_x, G.abmn, G.A).response(np.ones(len(mf.tri)))
    x, z = ele_xz(mt)
    clean = Forward(mt, G.elec_x, G.abmn, G.A).response(true_model(x, z, 'C')) / corr_f
    rng = np.random.default_rng(11)
    np.save('rhoa_C_noise3.npy', clean * (1 + 0.03 * rng.standard_normal(len(clean))))
    print(f'geometry C data: {clean.min():.1f}-{clean.max():.1f} ohm.m', flush=True)
corr = np.load('corr.npy'); dobs = np.load('rhoa_C_noise3.npy')
for topo in (False, True):
    tag = 'C_topo' if topo else 'C_flat'
    if os.path.exists(f'met_{tag}.json'): continue
    mesh = G.make_mesh(topo=topo)
    class F(Forward):
        def response(self, rho_e, return_sols=False):
            o = super().response(rho_e, return_sols)
            return (o[0] / corr, o[1], o[2]) if return_sols else o / corr
    fwd = F(mesh, G.elec_x, G.abmn, G.A); grid = ParaGrid(mesh, GRID_X, GRID_D)
    m, rhoa, hist, J = invert(fwd, grid, dobs, np.full(len(dobs), 0.03), lam0=60, lam_min=0.3, maxit=14,
                              zweight=0.5, robust=True, verbose=False)
    rho_e = np.exp(grid.to_ele(m)); met = metrics(mesh, rho_e, which='C', hang_under_topo=not topo)
    print(f'[{tag}] chi2={hist[-1,2]:.2f} log-misfit={met["rms_log"]:.3f} IoU={met["iou"]:.2f} '
          f'dx={met["dx"]:+.1f} m dz={met["dz"]:+.1f} m  ({time.time()-t:.0f}s)', flush=True)
    np.savez(f'res_{tag}.npz', rho_e=rho_e, topo=topo)
    json.dump({k: (float(v) if np.isscalar(v) else [float(a) for a in v]) for k, v in met.items()} | {'chi2': float(hist[-1, 2])},
              open(f'met_{tag}.json', 'w'))
# figure (manuscript Fig. 6, journal style)
import journal_style as J
mt = G.make_mesh(True); mf = G.make_mesh(False); x, z = ele_xz(mt)
fig, ax = J.figure(3, height=14, sharex=True)
plot_section(ax[0], mt, true_model(x, z, 'C'), 'True model', outline=False, kind='rho',
             bodies=[('CC', 5.0, 800.0), ('R', 3000.0, 800.0)])
for i, (tag, hang, lab) in enumerate([('C_flat', True, 'Flat-earth'), ('C_topo', False, 'Topography-corrected')], 1):
    r = np.load(f'res_{tag}.npz')
    plot_section(ax[i], mf if hang else mt, r['rho_e'], lab, hang=hang, outline=False, kind='rho')
for a in ax: _outline(a, 'CC'); _outline(a, 'R')
J.finish(fig, ax); J.save(fig, 'F09_geometryC'); print('figure saved')
