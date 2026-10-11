"""Generalisation test: repeat Test 3 with the conductive body under the VALLEY instead of the ridge."""
import numpy as np, json, sys, time
import setup_geom as G
from synth_common import *
from ert25d import Forward, ParaGrid, invert
t = time.time()
if not __import__('os').path.exists('rhoa_B_noise3.npy'):
    mt = fine_mesh(True); mf = fine_mesh(False)
    corr_f = Forward(mf, G.elec_x, G.abmn, G.A).response(np.ones(len(mf.tri)))
    x, z = ele_xz(mt)
    clean = Forward(mt, G.elec_x, G.abmn, G.A).response(true_model(x, z, 'B')) / corr_f
    rng = np.random.default_rng(7)
    np.save('rhoa_B_noise3.npy', clean * (1 + 0.03 * rng.standard_normal(len(clean))))
    print(f'geometry B data: {clean.min():.1f}-{clean.max():.1f} ohm.m', flush=True)
corr = np.load('corr.npy'); dobs = np.load('rhoa_B_noise3.npy')
for topo in (False, True):
    tag = 'B_topo' if topo else 'B_flat'
    mesh = G.make_mesh(topo=topo)
    class F(Forward):
        def response(self, rho_e, return_sols=False):
            o = super().response(rho_e, return_sols)
            return (o[0] / corr, o[1], o[2]) if return_sols else o / corr
    fwd = F(mesh, G.elec_x, G.abmn, G.A); grid = ParaGrid(mesh, GRID_X, GRID_D)
    m, rhoa, hist, J = invert(fwd, grid, dobs, np.full(len(dobs), 0.03), lam0=60, lam_min=0.3, maxit=14,
                              zweight=0.5, robust=True, verbose=False)
    rho_e = np.exp(grid.to_ele(m)); met = metrics(mesh, rho_e, which='B', hang_under_topo=not topo)
    print(f'[{tag}] chi2={hist[-1,2]:.2f} log-misfit={met["rms_log"]:.3f} IoU={met["iou"]:.2f} '
          f'dx={met["dx"]:+.1f} m dz={met["dz"]:+.1f} m  ({time.time()-t:.0f}s)', flush=True)
    np.savez(f'res_{tag}.npz', rho_e=rho_e, topo=topo)
    json.dump({k: (float(v) if np.isscalar(v) else [float(a) for a in v]) for k, v in met.items()} | {'chi2': float(hist[-1, 2])},
              open(f'met_{tag}.json', 'w'))
