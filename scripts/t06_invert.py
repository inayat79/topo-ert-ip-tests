import numpy as np, time, sys, json
import setup_geom as G
from synth_common import *
from ert25d import Forward, ParaGrid, invert
tag, datafile, topo, robust, zw, noise = sys.argv[1], sys.argv[2], sys.argv[3] == '1', sys.argv[4] == '1', float(sys.argv[5]), float(sys.argv[6])
t = time.time()
corr = np.load('corr.npy')
mesh = G.make_mesh(topo=topo, damp_depth=None)
class F(Forward):
    def response(self, rho_e, return_sols=False):
        out = super().response(rho_e, return_sols)
        if return_sols: return out[0] / corr, out[1], out[2]
        return out / corr
fwd = F(mesh, G.elec_x, G.abmn, G.A)
grid = ParaGrid(mesh, GRID_X, GRID_D)
dobs = np.load(datafile)
print(f'[{tag}] topo={topo} robust={robust} zweight={zw} noise={noise} npar={grid.npar}', flush=True)
m, rhoa, hist, J = invert(fwd, grid, dobs, np.full(len(dobs), noise), lam0=60.0, lam_min=0.3,
                          maxit=14, zweight=zw, robust=robust)
rho_e = np.exp(grid.to_ele(m))
met = metrics(mesh, rho_e, hang_under_topo=not topo)
print(f'[{tag}] METRICS rms_log={met["rms_log"]:.3f} IoU={met["iou"]:.2f} '
      f'centroid shift dx={met["dx"]:+.1f} m dz={met["dz"]:+.1f} m  min rho in body={met["min_rho_in_body"]:.1f}  '
      f'time {time.time()-t:.0f}s', flush=True)
np.savez(f'res_{tag}.npz', m=m, rhoa=rhoa, hist=hist, J=J, rho_e=rho_e, topo=topo)
json.dump({k: (float(v) if np.isscalar(v) else [float(a) for a in v]) for k, v in met.items()} |
          {'chi2': float(hist[-1, 2]), 'rms': float(hist[-1, 3]), 'iters': int(len(hist) - 1)},
          open(f'met_{tag}.json', 'w'))
