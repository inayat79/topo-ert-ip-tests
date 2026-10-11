import numpy as np, time
import setup_geom as G
from ert25d import Mesh, Forward
corr25 = np.load('corr.npy')
def mk(dxf, damp):
    fine = np.arange(-40, 940 + 1e-6, dxf)
    pad = dxf * np.cumprod(np.full(20, 1.45)); pad = pad[np.cumsum(pad) < 3000]
    xn = np.r_[fine[0] - np.cumsum(pad)[::-1], fine, fine[-1] + np.cumsum(pad)]
    return Mesh(xn, G.depths(), G.surf_topo, damp_depth=damp, zref=G.tz.mean()), \
           Mesh(xn, G.depths(), G.surf_flat, zref=G.tz.mean())
ref = None
for dxf, damp in [(2.5, 60.0), (2.5, 30.0), (2.5, None), (1.25, 60.0)]:
    t = time.time(); mt, mf = mk(dxf, damp)
    c = Forward(mf, G.elec_x, G.abmn, G.A).response(np.ones(len(mf.tri)))
    r = Forward(mt, G.elec_x, G.abmn, G.A).response(np.full(len(mt.tri), 100.0)) / c
    if ref is None: ref = r
    d = 100 * (r / ref - 1)
    print(f'dx={dxf:5.2f} damp={str(damp):5s} nodes={len(mt.nodes):6d}: range {r.min():6.1f}-{r.max():6.1f}  '
          f'vs baseline: mean|diff| {np.abs(d).mean():.2f}%  max|diff| {np.abs(d).max():.2f}%  ({time.time()-t:.0f}s)')
