import numpy as np, time
import setup_geom as G
from ert25d import Mesh, Forward
def run(dxf, h0, grow):
    fine = np.arange(-40, 940 + 1e-6, dxf)
    pad = dxf * np.cumprod(np.full(16, 1.45)); pad = pad[np.cumsum(pad) < 3000]
    xn = np.r_[fine[0] - np.cumsum(pad)[::-1], fine, fine[-1] + np.cumsum(pad)]
    d = [0.0]; h = h0
    while d[-1] < 120: d.append(d[-1] + h); h = min(h * grow, 8.0)
    dn = np.r_[d, d[-1] + np.cumsum(8.0 * np.cumprod(np.full(12, 1.45)))]
    mesh = Mesh(xn, dn, G.surf_flat, zref=G.tz.mean())
    t = time.time()
    fwd = Forward(mesh, G.elec_x, G.abmn, G.A, nk=8)
    rho = fwd.response(np.full(len(mesh.tri), 100.0)); e = 100 * (rho / 100 - 1)
    print(f'dx={dxf:4.1f} h0={h0:4.2f} nodes={len(mesh.nodes):6d} | mean {e.mean():+.2f}% max {np.abs(e).max():.2f}% '
          + ' '.join(f'n{n}:{e[G.nvals==n].mean():+.2f}' for n in range(1, 6)) + f' | {time.time()-t:.1f}s')
for args in [(5, 1.0, 1.12), (5, 0.5, 1.15), (4, 0.5, 1.15), (2.5, 0.5, 1.15)]:
    run(*args)
