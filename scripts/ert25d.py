"""
ert25d.py — 2.5D DC resistivity + IP forward modelling and Gauss-Newton inversion
with topography. Pure NumPy/SciPy (P1 triangular finite elements).

Physics
-------
3D Poisson  -div(sigma grad phi) = I delta(r - r_s)
Cosine-transform along strike (y):
    -div(sigma grad Phi) + k^2 sigma Phi = (I/2) delta(x - x_s) delta(z - z_s)
    phi(x,z) = sum_i w_i Phi(x, k_i, z)      (weights w_i absorb the 2/pi factor)
Boundary: Neumann (free surface) on top, mixed (Robin) on sides and bottom:
    dPhi/dn + k K1(kr)/K0(kr) cos(theta) Phi = 0

Inversion
---------
Model m = log(rho) on a terrain-following parameter grid.
Gauss-Newton with first-derivative smoothness:  Phi = ||Wd(d - f(m))||^2 + lam ||C m||^2
Sensitivities by the adjoint (reciprocity) method from pole solutions.
IP: Seigel's linear relation  ma_i = sum_j J_ij m_j   (J = dlog rho_a / dlog rho_j)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import k0, k1, k0e, k1e
from scipy.optimize import least_squares

# ----------------------------------------------------------------- quadrature
def wavenumber_quadrature(rmin, rmax, nk=8):
    """Optimise k_i, w_i so sum w_i K0(k_i r) = 1/r on [rmin, rmax] (Xu et al. 2000)."""
    r = np.geomspace(rmin, rmax, 200)
    def resid(p):
        k = np.exp(p[:nk]); w = np.exp(p[nk:])
        return (np.sum(w[:, None] * k0(np.outer(k, r)), 0) * r) - 1.0
    k_init = np.geomspace(0.1 / rmax, 3.0 / rmin, nk)
    w_init = np.full(nk, 1.0 / nk) * k_init
    sol = least_squares(resid, np.r_[np.log(k_init), np.log(w_init)], xtol=1e-14, ftol=1e-14,
                        max_nfev=20000)
    k = np.exp(sol.x[:nk]); w = np.exp(sol.x[nk:])
    return k, w, np.max(np.abs(resid(sol.x)))

# ----------------------------------------------------------------- mesh
class Mesh:
    """Terrain-following triangular mesh.
    x_nodes: horizontal node coordinates; depths: node depths below surface (0 first).
    surf(x): surface elevation function. damp_depth: depth over which topo distortion decays
    (0 -> flat mesh). Elevation of node = surf(x)*f + zref*(1-f) - depth, f = exp(-depth/D)
    """
    def __init__(self, x_nodes, depths, surf, damp_depth=None, zref=None):
        self.xn = np.asarray(x_nodes, float); self.dn = np.asarray(depths, float)
        nx, nz = len(self.xn), len(self.dn)
        s = surf(self.xn)
        zref = np.mean(s) if zref is None else zref
        X = np.repeat(self.xn[None, :], nz, 0)
        if damp_depth is None:           # undamped: every layer parallel to surface
            f = np.ones(nz)
        else:
            f = np.exp(-self.dn / damp_depth)
        Z = (s[None, :] * f[:, None] + zref * (1 - f[:, None])) - self.dn[:, None]
        self.X, self.Z = X, Z
        self.nodes = np.c_[X.ravel(), Z.ravel()]
        idx = np.arange(nx * nz).reshape(nz, nx)
        tris = []
        for j in range(nz - 1):
            for i in range(nx - 1):
                a, b, c, d = idx[j, i], idx[j, i + 1], idx[j + 1, i], idx[j + 1, i + 1]
                tris += [[a, b, d], [a, d, c]]
        self.tri = np.array(tris)
        self.nx, self.nz = nx, nz
        # element centroid in (x, depth) coordinates -> used for parameter mapping
        cx = self.nodes[self.tri, 0].mean(1)
        dep = np.repeat(self.dn[None, :], nx, 0).T.ravel()
        self.cdepth = dep[self.tri].mean(1)
        self.cx = cx
        self._geom()
        # boundary edges (left, right, bottom) for Robin condition
        be = []
        for j in range(nz - 1):
            be.append((idx[j, 0], idx[j + 1, 0])); be.append((idx[j, -1], idx[j + 1, -1]))
        for i in range(nx - 1):
            be.append((idx[-1, i], idx[-1, i + 1]))
        self.bedges = np.array(be)
        self.surf_idx = idx[0]

    def _geom(self):
        P = self.nodes[self.tri]                   # (ne,3,2)
        x, z = P[..., 0], P[..., 1]
        b = np.c_[z[:, 1] - z[:, 2], z[:, 2] - z[:, 0], z[:, 0] - z[:, 1]]
        c = np.c_[x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]]
        area2 = x[:, 0] * b[:, 0] + x[:, 1] * b[:, 1] + x[:, 2] * b[:, 2]
        self.area = 0.5 * np.abs(area2)
        sgn = np.sign(area2)[:, None]
        self.gb = b / (np.abs(area2)[:, None]) * sgn   # d(N_i)/dx
        self.gc = c / (np.abs(area2)[:, None]) * sgn   # d(N_i)/dz
        ne = len(self.tri)
        Ke = (self.gb[:, :, None] * self.gb[:, None, :] + self.gc[:, :, None] * self.gc[:, None, :]) \
             * self.area[:, None, None]
        Me = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0 * self.area[:, None, None]
        self.Ke, self.Me = Ke, Me
        self.I = np.repeat(self.tri, 3, 1).ravel(); self.J = np.tile(self.tri, (1, 3)).ravel()

    def node_at(self, x):
        """surface node index closest to horizontal position x"""
        return self.surf_idx[np.argmin(np.abs(self.xn - x))]

# ----------------------------------------------------------------- forward
class Forward:
    def __init__(self, mesh, elec_x, abmn, a, nk=8):
        self.m = mesh
        self.ex = np.asarray(elec_x, float)
        self.enode = np.array([mesh.node_at(x) for x in self.ex])
        self.abmn = np.asarray(abmn, int)          # indices into elec_x
        rmin = 0.5 * a; rmax = 1.5 * (self.ex.max() - self.ex.min())
        self.k, self.w, self.qerr = wavenumber_quadrature(rmin, rmax, nk)
        self.centre = np.array([self.ex.mean(), mesh.nodes[self.enode, 1].mean()])
        # analytic flat-earth geometric factor
        xa, xb, xm, xn = (self.ex[self.abmn[:, i]] for i in range(4))
        g = (1 / np.abs(xa - xm) - 1 / np.abs(xb - xm) - 1 / np.abs(xa - xn) + 1 / np.abs(xb - xn))
        self.K = 2 * np.pi / g

    def _robin(self, sigma_node_edge, k):
        m = self.m
        p1, p2 = m.nodes[m.bedges[:, 0]], m.nodes[m.bedges[:, 1]]
        mid = 0.5 * (p1 + p2); L = np.linalg.norm(p2 - p1, axis=1)
        t = (p2 - p1) / L[:, None]; nvec = np.c_[t[:, 1], -t[:, 0]]
        rv = mid - self.centre; r = np.linalg.norm(rv, axis=1)
        nvec *= np.sign(np.sum(nvec * rv, 1))[:, None]           # outward
        cos = np.sum(nvec * rv, 1) / r
        alpha = k * k1e(k * r) / k0e(k * r) * cos
        c = sigma_node_edge * alpha * L / 6.0
        I = np.r_[m.bedges[:, 0], m.bedges[:, 1], m.bedges[:, 0], m.bedges[:, 1]]
        J = np.r_[m.bedges[:, 0], m.bedges[:, 1], m.bedges[:, 1], m.bedges[:, 0]]
        V = np.r_[2 * c, 2 * c, c, c]
        n = len(m.nodes)
        return sp.coo_matrix((V, (I, J)), shape=(n, n))

    def pole_solutions(self, rho_e):
        """Unit-current pole potentials for every electrode and every k: (nk, nelec, nnode)."""
        m = self.m; sig = 1.0 / rho_e; n = len(m.nodes)
        # boundary sigma: from element adjacent -> approximate with global mean of boundary elems
        sig_b = np.exp(np.mean(np.log(sig)))
        sols = np.zeros((len(self.k), len(self.ex), n))
        rhs = np.zeros((n, len(self.ex)))
        rhs[self.enode, np.arange(len(self.ex))] = 0.5
        for ik, k in enumerate(self.k):
            V = (sig[:, None, None] * (m.Ke + k * k * m.Me)).ravel()
            A = sp.coo_matrix((V, (m.I, m.J)), shape=(n, n)) + self._robin(sig_b, k)
            lu = spla.splu(A.tocsc())
            sols[ik] = lu.solve(rhs).T
        return sols

    def response(self, rho_e, return_sols=False):
        sols = self.pole_solutions(rho_e)
        phi = np.tensordot(self.w, sols, 1)          # (nelec, nnode)
        pe = phi[:, self.enode]                      # potentials at electrodes
        A, B, M, N = self.abmn.T
        V = pe[A, M] - pe[A, N] - pe[B, M] + pe[B, N]
        rhoa = self.K * V
        return (rhoa, sols, V) if return_sols else rhoa

    def jacobian(self, rho_e, sols, V):
        """d log rho_a / d log rho_e  (ndata, nele) via adjoint method."""
        m = self.m; tri = m.tri; sig = 1.0 / rho_e
        A, B, M, N = self.abmn.T
        Jac = np.zeros((len(A), len(tri)))
        for ik, k in enumerate(self.k):
            S = sols[ik][:, tri]                         # (nelec, ne, 3)
            gx = np.einsum('pei,ei->pe', S, m.gb); gz = np.einsum('pei,ei->pe', S, m.gc)
            ssum = S.sum(2)
            for d in range(len(A)):
                sAB = S[A[d]] - S[B[d]]; sMN = S[M[d]] - S[N[d]]
                gxa, gza = gx[A[d]] - gx[B[d]], gz[A[d]] - gz[B[d]]
                gxm, gzm = gx[M[d]] - gx[N[d]], gz[M[d]] - gz[N[d]]
                stiff = (gxa * gxm + gza * gzm) * m.area
                mass = m.area / 12.0 * (np.sum(sAB * sMN, 1) + sAB.sum(1) * sMN.sum(1))
                # dV/dsigma_e = -2 * sum_k w_k [ ... ]  (factor 2: adjoint RHS = 1 vs 0.5)
                Jac[d] += -2.0 * self.w[ik] * (stiff + k * k * mass)
        # dlog rhoa/dlog rho_e = -(sigma_e / V) dV/dsigma_e
        return -(Jac * sig[None, :]) / V[:, None]

# ----------------------------------------------------------------- parameterisation
class ParaGrid:
    """Rectangular (x, depth-below-surface) parameter cells. Elements outside the grid are
    assigned to the nearest cell (padding absorbed into edge cells)."""
    def __init__(self, mesh, xedges, dedges):
        self.xe, self.de = np.asarray(xedges), np.asarray(dedges)
        ix = np.clip(np.searchsorted(self.xe, mesh.cx) - 1, 0, len(self.xe) - 2)
        iz = np.clip(np.searchsorted(self.de, mesh.cdepth) - 1, 0, len(self.de) - 2)
        self.nx, self.nz = len(self.xe) - 1, len(self.de) - 1
        self.cell = iz * self.nx + ix
        self.npar = self.nx * self.nz
        self.P = sp.csr_matrix((np.ones(len(self.cell)), (np.arange(len(self.cell)), self.cell)),
                               shape=(len(self.cell), self.npar))
        # smoothness operator
        rows = []; nx, nz = self.nx, self.nz
        dx = np.diff(self.xe); dz = np.diff(self.de)
        for j in range(nz):
            for i in range(nx - 1):
                rows.append((j * nx + i, j * nx + i + 1, 1.0))
        nh = len(rows)
        for j in range(nz - 1):
            for i in range(nx):
                rows.append((j * nx + i, (j + 1) * nx + i, 1.0))
        C = sp.lil_matrix((len(rows), self.npar))
        for r, (p, q, wgt) in enumerate(rows):
            C[r, p] = -wgt; C[r, q] = wgt
        self.C = C.tocsr()
        self.nh = nh
        self.xc = 0.5 * (self.xe[1:] + self.xe[:-1]); self.dc = 0.5 * (self.de[1:] + self.de[:-1])

    def to_ele(self, mpar):
        return mpar[self.cell]

# ----------------------------------------------------------------- inversion
def invert(fwd, grid, dobs, err, lam0=20.0, lam_min=0.5, maxit=12, m0=None, zweight=1.0,
           robust=False, verbose=True, target_chi2=1.0, robust_data=False, conv=None):
    """Gauss-Newton on log(rho). err: relative data error (fraction). Discrepancy principle:
    lambda reduced by factor 0.6 each iteration until chi2 <= target."""
    d = np.log(dobs); Wd = 1.0 / err
    if m0 is None:
        m0 = np.full(grid.npar, np.log(np.median(dobs)))
    m = m0.copy()
    C = grid.C.copy().tolil()
    C[grid.nh:] *= zweight
    C = C.tocsr()
    lam = lam0; hist = []
    rhoa, sols, V = fwd.response(np.exp(grid.to_ele(m)), True)
    for it in range(maxit):
        r = d - np.log(rhoa)
        chi2 = np.mean((r * Wd) ** 2)
        rms = 100 * np.sqrt(np.mean((rhoa - dobs) ** 2 / dobs ** 2))
        hist.append((it, lam, chi2, rms))
        if verbose: print(f'   it {it:2d}  lam {lam:7.2f}  chi2 {chi2:8.3f}  RMS {rms:6.2f}%')
        if chi2 <= target_chi2 * 1.05: break
        if conv and it > 2 and (hist[-2][2] - chi2) / hist[-2][2] < conv: break
        Je = fwd.jacobian(np.exp(grid.to_ele(m)), sols, V)
        Jp = Je @ grid.P                                # (ndata, npar)
        Wm = C
        if robust:   # IRLS L1 on model roughness (Farquharson & Oldenburg 1998)
            g = C @ m; Rm = 1.0 / np.sqrt(g ** 2 + (0.05) ** 2)
            Wm = sp.diags(np.sqrt(Rm)) @ C
        Wde = Wd * np.ones_like(r)
        if robust_data:   # IRLS L1 data norm (Claerbout & Muir 1973)
            rn = np.abs(r * Wd); Wde = Wde * np.sqrt(1.0 / np.maximum(rn, 1.0))
        WdJ = Jp * Wde[:, None]
        H = WdJ.T @ WdJ + lam * (Wm.T @ Wm).toarray()
        g_ = WdJ.T @ (r * Wde) - lam * (Wm.T @ (Wm @ m))
        dm = np.linalg.solve(H, g_)
        # line search
        best = None
        for step in (1.0, 0.5, 0.25):
            mt = m + step * dm
            rt, st, Vt = fwd.response(np.exp(grid.to_ele(mt)), True)
            phi_d = np.mean(((d - np.log(rt)) * Wd) ** 2)
            if best is None or phi_d < best[0]:
                best = (phi_d, mt, rt, st, Vt)
            if phi_d < chi2: break
        _, m, rhoa, sols, V = best
        lam = max(lam * 0.6, lam_min)
    r = d - np.log(rhoa); chi2 = np.mean((r * Wd) ** 2)
    rms = 100 * np.sqrt(np.mean((rhoa - dobs) ** 2 / dobs ** 2))
    hist.append((len(hist), lam, chi2, rms))
    if verbose: print(f'   final     chi2 {chi2:8.3f}  RMS {rms:6.2f}%')
    J_final = fwd.jacobian(np.exp(grid.to_ele(m)), sols, V) @ grid.P
    return m, rhoa, np.array(hist), J_final

def invert_ip(J, ma_obs, err_abs, grid, lam=5.0, zweight=1.0, bounds=(0.0, 50.0)):
    """Linear IP inversion ma = J m (Seigel), smoothness regularised, bounded least squares."""
    from scipy.optimize import lsq_linear
    C = grid.C.copy().tolil(); C[grid.nh:] *= zweight; C = C.tocsr().toarray()
    err_abs = np.broadcast_to(np.asarray(err_abs, float), ma_obs.shape)
    Aaug = np.vstack([J / err_abs[:, None], np.sqrt(lam) * C])
    baug = np.r_[ma_obs / err_abs, np.zeros(C.shape[0])]
    res = lsq_linear(Aaug, baug, bounds=bounds, lsmr_tol='auto', max_iter=5000)
    m = res.x
    pred = J @ m
    chi2 = np.mean(((pred - ma_obs) / err_abs) ** 2)
    return m, pred, chi2
