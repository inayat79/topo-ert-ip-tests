"""Survey geometry of the test profile: electrodes, four-electrode configurations and topography.
Read from ../data/geometry_*.csv (geometry only - no measured values are needed or included)."""
import os
import numpy as np, pandas as pd
from ert25d import Mesh
from scipy.interpolate import interp1d

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
_read = lambda name: pd.read_csv(os.path.join(_DATA, name))

A = 20.0                                          # unit electrode spacing a (m)
elec_x = _read('geometry_electrodes.csv').sort_values('electrode').x_m.values.astype(float)   # 46 electrodes, 0..900 m
_conf = _read('geometry_configurations.csv')
abmn = _conf[['C1', 'C2', 'P1', 'P2']].values.astype(int)   # dipole-dipole C1 C2 P1 P2 (electrode indices), n = 1..5
nvals = _conf.n.values.astype(int)
xmid = elec_x[abmn].mean(1)

# topography (39 stations) extended flat beyond the data
topo = _read('geometry_topography.csv').rename(columns={'x_m': 'x', 'elevation_m': 'H'}).sort_values('x')
tx = np.r_[-2000, topo.x.values, 2000 + 900]
tz = np.r_[topo.H.values[0], topo.H.values, topo.H.values[-1]]
# electrodes beyond the last station: hold its elevation
surf_topo = interp1d(tx, tz, kind='linear', bounds_error=False, fill_value=(tz[0], tz[-1]))
surf_flat = lambda x: np.full_like(np.asarray(x, float), tz.mean())

def x_nodes():
    fine = np.arange(-40, 940 + 1e-6, 2.5)              # 2.5 m in survey zone (a/8)
    pad = 2.5 * np.cumprod(np.full(16, 1.45)); pad = pad[np.cumsum(pad) < 3000]
    left = fine[0] - np.cumsum(pad)[::-1]; right = fine[-1] + np.cumsum(pad)
    return np.r_[left, fine, right]

def depths():
    d = [0.0]; h = 0.5
    while d[-1] < 120:
        d.append(d[-1] + h); h = min(h * 1.15, 8.0)
    pad = d[-1] + np.cumsum(8.0 * np.cumprod(np.full(12, 1.45)))
    return np.r_[d, pad]

def make_mesh(topo=True, damp_depth=None):
    s = surf_topo if topo else surf_flat
    return Mesh(x_nodes(), depths(), s, damp_depth=damp_depth, zref=tz.mean())
