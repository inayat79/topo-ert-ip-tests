"""
PyGIMLi reproduction of Tests 2 and 3 (topographic effect; flat vs topography-corrected inversion).
Run check_pygimli_env.py first.  Expected (NumPy reference code):
   Test 2: homogeneous 100 ohm.m under topography -> rho_a ~44-149 ohm.m
   Test 3: flat-earth centroid error dx ~ -20 m ; topography-corrected dx ~ -7 to -10 m
Small differences (a few m / a few %) are expected between codes; the flat >> topo pattern is what matters.
"""
from pg_common import *
scheme_topo = make_scheme(ez)
scheme_flat = make_scheme(np.full_like(ez, ez.mean()))
k_flat = ert.geometricFactors(scheme_flat)          # analytic flat-earth K (what an instrument reports)
mesh_fwd = forward_mesh(scheme_topo)
print('forward mesh cells:', mesh_fwd.cellCount())
cc = np.array([[p.x(), p.y()] for p in mesh_fwd.cellCenters()]); cx, cz = cc[:, 0], cc[:, 1]
dep = surf(cx) - cz

# ---- TEST 2 --------------------------------------------------------------------------
scheme_topo['k'] = k_flat
d_h = ert.simulate(mesh_fwd, scheme=scheme_topo, res=np.full(mesh_fwd.cellCount(), 100.0), noiseLevel=0, noiseAbs=0)
r = np.asarray(d_h['rhoa'])
print(f'TEST 2  topographic effect: rho_a {r.min():.1f}-{r.max():.1f} ohm.m '
      f'({100*(r.min()/100-1):+.1f}% .. {100*(r.max()/100-1):+.1f}%)')

# ---- TEST 3 --------------------------------------------------------------------------
rho = np.full(mesh_fwd.cellCount(), 800.0)
rho[dep < 6] = 250.0
rho[(cx > 260) & (cx < 380) & (cz > 1597) & (cz < 1632) & (dep > 8)] = 5.0
rho[(cx > 640) & (cx < 780) & (dep > 10) & (dep < 45)] = 3000.0
data = ert.simulate(mesh_fwd, scheme=scheme_topo, res=rho, noiseLevel=3, noiseAbs=0, seed=42)
data['err'] = ert.estimateError(data, relativeError=0.03)
data.save('synthetic_topo_3pct.dat')

LAMS, WIN, MAXBIS = (20, 10, 5, 2, 1), (0.95, 1.05), 8   # discrepancy principle: chi2 in 0.95-1.05
def invert_discrepancy(d, label):
    """Same procedure for flat and topo: coarse lam list until chi2 = 1 is bracketed (or chi2 is in
    the window), then bisect log10(lam) between the last lam with chi2 > 1 and the first with chi2 < 1
    until chi2 is in WIN. If not reached, the lam with chi2 closest to 1 is kept and flagged."""
    tried = []
    def run(lam):
        mgr = ert.ERTManager(d)
        mgr.invert(paraDepth=70, paraDX=0.25, zWeight=0.5, blockyModel=True, lam=lam, verbose=False)
        c2 = mgr.inv.chi2(); tried.append((lam, c2, mgr))
        print(f'  [{label}] lam {lam:7.3f}: chi2 {c2:.3f}', flush=True)
        return c2
    inwin = lambda c: WIN[0] <= c <= WIN[1]
    above = below = None
    for lam in LAMS:
        c2 = run(lam)
        if inwin(c2): break
        if c2 > 1: above = lam
        else: below = lam; break
    if not inwin(c2) and above is not None and below is not None:
        for _ in range(MAXBIS):
            lam = 10 ** ((np.log10(above) + np.log10(below)) / 2); c2 = run(lam)
            if inwin(c2): break
            if c2 > 1: above = lam
            else: below = lam
    lam, c2, mgr = min(tried, key=lambda t: abs(t[1] - 1))
    return mgr, lam, c2, inwin(c2), tried

# true body centroid, masked exactly like the NumPy metric (60 < x < 840, depth < 45 m), forward mesh
_, _, _, area = cell_xz(mesh_fwd)
true_c = centroid(cx, cz, area, (rho == 5.0) & sens_mask(cx, dep))
print(f'true body centroid (masked, forward mesh): x {true_c[0]:.1f} m  z {true_c[1]:.1f} m')

res = {}
mgr_t, *res['topo'] = invert_discrepancy(data, 'topo')
res['topo'].append(centroid_error(mgr_t, true_c))

data_flat = pg.DataContainerERT(data)
for i in range(data_flat.sensorCount()):
    data_flat.setSensorPosition(i, pg.Pos(data_flat.sensorPosition(i).x(), float(ez.mean())))
mgr_f, *res['flat'] = invert_discrepancy(data_flat, 'flat')
res['flat'].append(centroid_error(mgr_f, true_c, flat_elev=float(ez.mean())))

# keep the final models so metrics can be recomputed without re-inverting
_sv = {}
for k, m_, fe in (('topo', mgr_t, None), ('flat', mgr_f, float(ez.mean()))):
    x_, z_, d_, a_ = cell_xz(m_.paraDomain, fe)
    _sv.update({f'{k}_x': x_, f'{k}_z': z_, f'{k}_dep': d_, f'{k}_area': a_,
                f'{k}_rho': np.asarray(m_.model), f'{k}_lam': res[k][0], f'{k}_chi2': res[k][1]})
np.savez('PG_test3_models.npz', true_centroid=np.array(true_c), **_sv)

print('\nTEST 3 RESULTS (compare with report Table, Section 3.2)')
out = [f'TEST 2  rho_a {r.min():.1f}-{r.max():.1f} ohm.m  (reference 44.4-149.3)',
       f'TEST 3  true body centroid (mask 60<x<840 m, depth<45 m, forward mesh): '
       f'x {true_c[0]:.1f} m  z {true_c[1]:.1f} m  (expected ~308.0, 1617.8)']
ref = {'flat': '(NumPy, masked: dx -20.0 dz -9.1)', 'topo': '(NumPy, masked: dx -6.7 dz -5.6)'}
for k in ('flat', 'topo'):
    lam, chi2, ok, tried, ce = res[k]
    tr = ', '.join(f'{l:.3g}:{c:.3f}' for l, c, _ in tried)
    out.append(f'TEST 3  {k}: lam = {lam:.3g}  chi2 = {chi2:.3f}{"" if ok else " (NOT in 0.95-1.05)"}  ' +
               (f'centroid x {ce[2]:.1f} z {ce[3]:.1f} -> dx = {ce[0]:+.1f} m  dz = {ce[1]:+.1f} m'
                if ce else 'no body < 30 ohm.m') + f'  {ref[k]}   [tried lam:chi2 {tr}]')
save_results('1 synthetic', out)
fig, ax = pg.plt.subplots(2, 1, figsize=(11, 7))
mgr_f.showResult(ax=ax[0], cMin=2, cMax=4000, logScale=True); ax[0].set_title('Flat-earth inversion')
mgr_t.showResult(ax=ax[1], cMin=2, cMax=4000, logScale=True); ax[1].set_title('Topography-corrected inversion')
fig.savefig('PG_test3_flat_vs_topo.png', dpi=200); print('saved PG_test3_flat_vs_topo.png')
