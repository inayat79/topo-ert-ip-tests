import numpy as np, json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import setup_geom as G
from synth_common import *
from ert25d import ParaGrid, invert_ip
res_tag = sys.argv[1]           # resistivity inversion whose Jacobian is used
r = np.load(f'res_{res_tag}.npz'); topo = bool(r['topo'])
mesh = G.make_mesh(topo=topo); grid = ParaGrid(mesh, GRID_X, GRID_D); J = r['J']
x, z = ele_xz(mesh)
if not topo: z = G.surf_topo(x) - mesh.cdepth           # hang flat result under topography
dep = G.surf_topo(x) - z; msk = sens_mask_xyd(x, dep); w = mesh.area
out = {}
for sc in ('coincident', 'offset'):
    ma = np.load(f'ma_noise_{sc}.npy'); err = 0.2
    # discrepancy principle: bisection on log(lambda) for chi2 = 1
    lo, hi = -3, 4
    for _ in range(18):
        mid = 0.5 * (lo + hi); mp, pred, chi2 = invert_ip(J, ma, err, grid, lam=10 ** mid)
        if chi2 > 1.0: hi = mid
        else: lo = mid
    mp, pred, chi2 = invert_ip(J, ma, err, grid, lam=10 ** lo)
    me = grid.to_ele(mp)
    tru = true_charg(x, z, sc)
    rec_b = msk & (me > 4.0); tru_b = msk & (tru > 4.0)
    cx = lambda sel, zz: (np.sum(w[sel] * x[sel]) / w[sel].sum(), np.sum(w[sel] * zz[sel]) / w[sel].sum())
    rc, tc = cx(rec_b, z), cx(tru_b, z)
    rho = r['rho_e']; con = msk & (rho < 30); cc = cx(con, z)
    iou = np.sum(w[rec_b & tru_b]) / np.sum(w[rec_b | tru_b])
    out[sc] = dict(lam=10 ** lo, chi2=chi2, iou=iou, charg_centroid=rc, true_charg_centroid=tc,
                   conductive_centroid=cc, recovered_offset=(rc[0] - cc[0], rc[1] - cc[1]),
                   max_m=float(me[msk].max()))
    print(f'[{res_tag}] {sc:10s} lam={10**lo:7.3f} chi2={chi2:.2f} IoU={iou:.2f} max m={me[msk].max():5.1f}%  '
          f'chargeable centroid ({rc[0]:.0f},{rc[1]:.0f}) true ({tc[0]:.0f},{tc[1]:.0f}) | '
          f'offset charg-vs-conductive: dx={rc[0]-cc[0]:+.0f} m dz={rc[1]-cc[1]:+.0f} m')
    np.save(f'ip_{res_tag}_{sc}.npy', me)
json.dump(out, open(f'ipmet_{res_tag}.json', 'w'), default=float)
