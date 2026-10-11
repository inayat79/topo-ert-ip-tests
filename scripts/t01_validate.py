import numpy as np, time
from setup_geom import *
from ert25d import Forward
t = time.time()
mesh = make_mesh(topo=False)
print(f'mesh: {len(mesh.nodes)} nodes, {len(mesh.tri)} triangles, domain x {mesh.xn.min():.0f}..{mesh.xn.max():.0f}, depth to {mesh.dn.max():.0f} m')
for nk in (6, 8, 10):
    fwd = Forward(mesh, elec_x, abmn, A, nk=nk)
    rho = fwd.response(np.full(len(mesh.tri), 100.0))
    err = 100 * (rho / 100 - 1)
    print(f'nk={nk:2d}: quadrature err {100*fwd.qerr:.3f}% | rho_a error mean {err.mean():+.2f}%  '
          f'max|err| {np.abs(err).max():.2f}%  per n: ' +
          ' '.join(f'n{n}:{err[nvals==n].mean():+.2f}' for n in range(1, 6)))
print(f'time {time.time()-t:.1f}s')
