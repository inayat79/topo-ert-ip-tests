"""Shared geometry for the PyGIMLi scripts (same test-profile geometry as the NumPy code, read from ../data)."""
import os, re, sys
# Allow running the env's python.exe WITHOUT 'conda activate': put the env DLL folders on PATH,
# otherwise matplotlib's delay-loaded DLLs resolve elsewhere and savefig crashes (0xc06d007f).
os.environ['PATH'] = os.pathsep.join([os.path.join(sys.prefix, *p) for p in
    [(), ('Library', 'mingw-w64', 'bin'), ('Library', 'usr', 'bin'), ('Library', 'bin'), ('Scripts',)]]
    + [os.environ.get('PATH', '')])
import matplotlib; matplotlib.use('Agg')            # batch run: write PNGs, no GUI windows
import numpy as np, pandas as pd
import pygimli as pg
if os.environ.get("PG_THREADS"):                    # e.g. PG_THREADS=1 (does NOT make runs bit-identical)
    pg.setThreadCount(int(os.environ['PG_THREADS']))
import pygimli.meshtools as mt
from pygimli.physics import ert

HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(HERE, '..', 'data')
_read = lambda name: pd.read_csv(os.path.join(_DATA, name))

A = 20.0                                                    # unit electrode spacing a (m)
CONF = _read('geometry_configurations.csv')                 # C1 C2 P1 P2 n, same order as setup_geom.abmn
topo = _read('geometry_topography.csv').rename(columns={'x_m': 'x', 'elevation_m': 'H'}).sort_values('x')
ex = _read('geometry_electrodes.csv').sort_values('electrode').x_m.values.astype(float)   # 0..900 m
ez = np.interp(ex, topo.x.values, topo.H.values)            # held flat beyond last station
surf = lambda x: np.interp(x, ex, ez)

def make_scheme(z):
    d = pg.DataContainerERT()
    for x, zz in zip(ex, z):
        d.createSensor(pg.Pos(float(x), float(zz)))
    for c1, c2, p1, p2 in CONF[['C1', 'C2', 'P1', 'P2']].values.astype(int):
        d.createFourPointData(d.size(), int(c1), int(c2), int(p1), int(p2))   # C1 C2 P1 P2
    return d

def forward_mesh(scheme):
    """Fine mesh only inside the survey region; coarse padding outside; quadratic (H2) for accuracy."""
    plc = mt.createParaMeshPLC(scheme, paraDepth=80, paraDX=0.25, paraMaxCellSize=10, boundary=4)
    return mt.createMesh(plc, quality=33.5).createH2()

RESULTS = os.path.join(HERE, '..', 'RESULTS_pygimli.txt')

def save_results(section, lines):
    """Print `lines` and store them under '[section]' in RESULTS_pygimli.txt (other sections kept)."""
    for l in lines: print(l)
    secs, cur = {}, None
    if os.path.isfile(RESULTS):
        for l in open(RESULTS, encoding='utf-8').read().splitlines():
            if re.fullmatch(r'\[\d+ [^\]]*\]', l): cur = l[1:-1]; secs[cur] = []   # e.g. [1 synthetic]
            elif cur and l: secs[cur].append(l)
    secs[section] = list(lines)
    with open(RESULTS, 'w', encoding='utf-8') as f:
        for k in sorted(secs): f.write(f'[{k}]\n' + '\n'.join(secs[k]) + '\n\n')

def sens_mask(x, dep):
    """Resolved zone, identical to the NumPy code (synth_common.sens_mask_xyd)."""
    return (x > 60) & (x < 840) & (dep < 45)

def cell_xz(mesh, flat_elev=None):
    """Cell centres (x, elevation), depth below surface and area. A flat-earth result is hung
    under the real topography: depth below its own flat surface, elevation = surf(x) - depth."""
    c = np.array([[p.x(), p.y()] for p in mesh.cellCenters()]); x, z = c[:, 0], c[:, 1]
    a = np.array([cl.size() for cl in mesh.cells()])
    if flat_elev is not None:
        dep = flat_elev - z; z = surf(x) - dep
    else:
        dep = surf(x) - z
    return x, z, dep, a

def centroid(x, z, a, sel):
    return np.sum(a[sel] * x[sel]) / a[sel].sum(), np.sum(a[sel] * z[sel]) / a[sel].sum()

def centroid_error(mgr, truth, thresh=30.0, flat_elev=None):
    """As the NumPy metric: area-weighted centroid of cells with rho < thresh INSIDE sens_mask,
    minus the (equally masked) true body centroid. Returns (dx, dz, cx, cz) or None."""
    x, z, dep, a = cell_xz(mgr.paraDomain, flat_elev)
    s = sens_mask(x, dep) & (np.asarray(mgr.model) < thresh)
    if not s.any(): return None
    cx, cz = centroid(x, z, a, s)
    return cx - truth[0], cz - truth[1], cx, cz
