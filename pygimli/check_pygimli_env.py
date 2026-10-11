"""Run FIRST. Tests every PyGIMLi call used by the other scripts on a tiny model (~20 s)."""
import os, sys, traceback
# same DLL-path / backend setup as pg_common.py (lets the env python run without 'conda activate')
os.environ['PATH'] = os.pathsep.join([os.path.join(sys.prefix, *p) for p in
    [(), ('Library', 'mingw-w64', 'bin'), ('Library', 'usr', 'bin'), ('Library', 'bin'), ('Scripts',)]]
    + [os.environ.get('PATH', '')])
import matplotlib; matplotlib.use('Agg')
def step(msg, fn):
    try: out = fn(); print(f'  OK   {msg}'); return out
    except Exception as e:
        print(f'  FAIL {msg}\n       {type(e).__name__}: {e}'); traceback.print_exc(limit=1); sys.exit(1)
import pygimli as pg; print('pyGIMLi version', pg.__version__)
from pygimli.physics import ert; import pygimli.meshtools as mt, numpy as np
d = pg.DataContainerERT()
for x in range(0, 200, 10): d.createSensor(pg.Pos(float(x), 5.0 * np.sin(x / 40)))
for i in range(15): d.createFourPointData(d.size(), i, i + 1, i + 3, i + 4)
step('DataContainerERT + createFourPointData', lambda: d.size())
plc = step('createParaMeshPLC', lambda: mt.createParaMeshPLC(d, paraDepth=30, paraDX=0.25, paraMaxCellSize=10, boundary=4))
mesh = step('createMesh + createH2', lambda: mt.createMesh(plc, quality=33.5).createH2())
d['k'] = step('geometricFactors (analytic)', lambda: ert.geometricFactors(d))
sim = step('ert.simulate with noise', lambda: ert.simulate(mesh, scheme=d, res=np.full(mesh.cellCount(), 100.0),
                                                           noiseLevel=1, noiseAbs=0, seed=1))
sim['err'] = step('estimateError', lambda: ert.estimateError(sim, relativeError=0.03))
mgr = ert.ERTManager(sim)
step('invert (blocky, robustData, zWeight)', lambda: mgr.invert(paraDepth=30, paraDX=0.25, zWeight=0.5,
                                                                blockyModel=True, robustData=True, lam=20, maxIter=3, verbose=False))
step('chi2 / relrms', lambda: (mgr.inv.chi2(), mgr.inv.relrms()))
step('paraDomain cell centres', lambda: [p.x() for p in mgr.paraDomain.cellCenters()])
step('jacobian -> numpy', lambda: pg.utils.gmat2numpy(mgr.fop.jacobian()).shape)
step('constraints -> scipy', lambda: pg.utils.sparseMatrix2coo(mgr.fop.constraints()).shape)
import tempfile
step('showResult + savefig (plotting)', lambda: mgr.showResult()[0].figure.savefig(
    os.path.join(tempfile.gettempdir(), 'pg_check.png')))
print('\nAll PyGIMLi calls work on this installation. Run pygimli_reproduce_synthetic.py next.')
