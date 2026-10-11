import numpy as np, time
import setup_geom as G
from synth_common import *
t = time.time()
mt = fine_mesh(True); mf = fine_mesh(False)
corr_f = Forward(mf, G.elec_x, G.abmn, G.A).response(np.ones(len(mf.tri)))
fwd = Forward(mt, G.elec_x, G.abmn, G.A)
x, z = ele_xz(mt); rho_true = true_model(x, z)
rhoa_clean, sols, V = fwd.response(rho_true, True); rhoa_clean /= corr_f
# Jacobian on the fine mesh for IP synthetic data (Seigel linear relation)
Je = fwd.jacobian(rho_true, sols, V)
np.save('rhoa_clean.npy', rhoa_clean)
rng = np.random.default_rng(42)
for lvl in (0.03, 0.05):
    np.save(f'rhoa_noise{int(lvl*100)}.npy', rhoa_clean * (1 + lvl * rng.standard_normal(len(rhoa_clean))))
for sc in ('coincident', 'offset'):
    ma = Je @ true_charg(x, z, sc)
    np.save(f'ma_clean_{sc}.npy', ma)
    np.save(f'ma_noise_{sc}.npy', ma + 0.2 * rng.standard_normal(len(ma)))   # 0.2 % abs noise
print(f'synthetic rho_a: {rhoa_clean.min():.2f} .. {rhoa_clean.max():.1f} ohm.m')
for sc in ('coincident','offset'):
    ma=np.load(f'ma_clean_{sc}.npy'); i=np.argmax(ma); j=np.argmin(rhoa_clean)
    print(f'  {sc:10s}: ma {ma.min():.2f}..{ma.max():.2f}% ; ma max at xmid={G.xmid[i]:.0f} n={G.nvals[i]} ; '
          f'rho_a min at xmid={G.xmid[j]:.0f} n={G.nvals[j]}')
print(f'fine mesh {len(mt.nodes)} nodes; time {time.time()-t:.0f}s')
