import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.colors as mcol
from matplotlib.tri import Triangulation
import setup_geom as G
from ert25d import Mesh, Forward, ParaGrid

def fine_mesh(topo=True, dxf=1.25):
    fine = np.arange(-40, 940 + 1e-6, dxf)
    pad = dxf * np.cumprod(np.full(22, 1.45)); pad = pad[np.cumsum(pad) < 3000]
    xn = np.r_[fine[0] - np.cumsum(pad)[::-1], fine, fine[-1] + np.cumsum(pad)]
    return Mesh(xn, G.depths(), G.surf_topo if topo else G.surf_flat, damp_depth=None, zref=G.tz.mean())

def ele_xz(mesh):
    P = mesh.nodes[mesh.tri]; return P[..., 0].mean(1), P[..., 1].mean(1)

def true_model(x, z, which='A'):
    """True resistivity at (x, elevation). Geology defined in real space."""
    s = G.surf_topo(x); dep = s - z
    rho = np.full_like(x, 800.0)                                   # host rock
    rho[dep < 6] = 250.0                                           # weathered cover
    if which == 'C':                                               # geometry C: body STRADDLING the slope break (260-280 m)
        C = (x > 220) & (x < 320) & (dep > 10) & (dep < 32)
        rho[C] = 5.0; R = (x > 640) & (x < 780) & (dep > 10) & (dep < 45); rho[R] = 3000.0
        return rho
    if which == 'B':                                               # geometry B: body under the VALLEY floor
        C = (x > 170) & (x < 270) & (dep > 10) & (dep < 32)
        rho[C] = 5.0; R = (x > 640) & (x < 780) & (dep > 10) & (dep < 45); rho[R] = 3000.0
        return rho
    C = (x > 260) & (x < 380) & (z > 1597) & (z < 1632) & (dep > 8) # conductive body
    rho[C] = 5.0
    R = (x > 640) & (x < 780) & (dep > 10) & (dep < 45)            # resistive block
    rho[R] = 3000.0
    return rho

def true_charg(x, z, scenario):
    """Chargeability (%) at (x, elevation)."""
    s = G.surf_topo(x); dep = s - z
    m = np.full_like(x, 1.0)
    if scenario == 'coincident':          # chargeable = conductive body
        B = (x > 260) & (x < 380) & (z > 1597) & (z < 1632) & (dep > 8)
    elif scenario == 'offset':            # chargeable body deeper and 60 m to the right
        B = (x > 340) & (x < 440) & (dep > 12) & (dep < 35)   # 80 m right, within DOI
    m[B] = 12.0
    return m

GRID_X = np.arange(-20, 920 + 1e-6, 10.0)
GRID_D = np.array([0, 2, 4.5, 7.5, 11, 15, 20, 26, 33, 41, 50, 60, 72])

def sens_mask_xyd(x, dep):
    return (x > 60) & (x < 840) & (dep < 45)

def metrics(mesh, rho_e, which='A', hang_under_topo=False):
    """Compare recovered element model with truth. For a flat inversion the result is
    'hung' under the real topography (elevation = surface - depth), as practitioners do."""
    x, z = ele_xz(mesh)
    if hang_under_topo:
        dep = mesh.cdepth; z = G.surf_topo(x) - dep
    else:
        dep = G.surf_topo(x) - z
    tru = true_model(x, z, which)
    msk = sens_mask_xyd(x, dep); w = mesh.area[msk]
    lr = np.log10(rho_e[msk] / tru[msk])
    rms_log = np.sqrt(np.sum(w * lr ** 2) / w.sum())
    # conductive body recovery: rho < 30 (true 5), compared with truth region
    rec_c = rho_e[msk] < 30; tru_c = tru[msk] < 30
    iou = np.sum(w[rec_c & tru_c]) / max(np.sum(w[rec_c | tru_c]), 1e-9)
    xc = np.sum(w[rec_c] * x[msk][rec_c]) / max(w[rec_c].sum(), 1e-9)
    zc = np.sum(w[rec_c] * z[msk][rec_c]) / max(w[rec_c].sum(), 1e-9)
    xt = np.sum(w[tru_c] * x[msk][tru_c]) / w[tru_c].sum(); zt = np.sum(w[tru_c] * z[msk][tru_c]) / w[tru_c].sum()
    minrho = rho_e[msk][tru_c].min() if tru_c.any() else np.nan
    return dict(rms_log=rms_log, iou=iou, dx=xc - xt, dz=zc - zt, min_rho_in_body=minrho,
                centroid=(xc, zc), true_centroid=(xt, zt))

def plot_section(ax, mesh, vals, title, hang=False, cmap='jet', log=True, vmin=2, vmax=4000, outline=True,
                 cblabel='Resistivity (Ω·m)', charg_outline=None, kind=None, cbticks=None, bodies=None):
    """Section plot in the manuscript journal style (journal_style.py).
    title  : short in-panel label (<= 3 words), upper-right corner; panel letters are set by J.finish.
    kind   : 'rho' -> rainbow, log 2-4000 ohm.m, ticks 10^1..10^3 ; 'chg' -> viridis 0-14 %, ticks 0..14.
             None  -> cmap / log / vmin / vmax as given (e.g. the diverging ratio panel), ticks = cbticks.
    bodies : true-model panels only, [(name, body value, host value), ...]: triangles carrying the body value
             are drawn in the host value and the body is drawn on top as a filled polygon from its exact
             definition (no saw-tooth edges). Display only - the model values are not changed."""
    import journal_style as J                      # plotting only; computational scripts never get here
    x, z = mesh.nodes[:, 0], mesh.nodes[:, 1]
    if hang:
        dep = np.repeat(mesh.dn[None, :], mesh.nx, 0).T.ravel(); z = G.surf_topo(x) - dep
    tri = Triangulation(x, z, mesh.tri)
    xc = x[mesh.tri].mean(1); dep_c = (G.surf_topo(xc) - z[mesh.tri].mean(1))
    tri.set_mask(~((xc > -20) & (xc < 920) & (dep_c < 60)))
    cblog = False
    if kind == 'rho':
        cmap, norm, cbticks, cblog = J.RHO['cmap'], J.RHO['norm'](), J.RHO['ticks'], True
        if cblabel == 'Resistivity (Ω·m)': cblabel = r'$\rho$ (Ω·m)'
    elif kind == 'chg':
        cmap, norm, cbticks = J.CHG['cmap'], J.CHG['norm'](), J.CHG['ticks']
    else:
        norm = mcol.LogNorm(vmin, vmax) if log else mcol.Normalize(vmin, vmax)
    shown = np.asarray(vals, float).copy()
    for _, val, host in (bodies or []):
        shown[np.isclose(shown, val)] = host
    pc = ax.tripcolor(tri, facecolors=shown, cmap=cmap, norm=norm, edgecolors='none')
    for name, val, _ in (bodies or []):
        XX, ZZ, M = _body_grid(name)
        ax.contourf(XX, ZZ, M.astype(float), [0.5, 1.5], colors=[pc.cmap(pc.norm(val))], zorder=1.5)
    xs = np.linspace(-20, 920, 400); ax.plot(xs, G.surf_topo(xs), 'k')
    J.electrodes(ax, G.elec_x, G.surf_topo(G.elec_x), above=True)
    if outline:   # truth outlines
        _outline(ax, 'C'); _outline(ax, 'R')
    if charg_outline: _outline(ax, charg_outline, ls=':')   # chargeable body dotted, conductor dashed
    ax.set_xlim(-20, 920); ax.set_ylim(1560, 1695)
    J.set_ylabel(ax, 'Elevation (m)')
    cb = J.colorbar(ax.figure, pc, ax, cblabel, cbticks, log=cblog)
    if cbticks is not None and kind is None:       # plain numbers (e.g. ratio 0.2 ... 5)
        cb.set_ticks(cbticks, labels=[f'{t:g}' for t in cbticks])
        cb.ax.yaxis.set_minor_locator(J.NullLocator())     # no automatic 3x10^-1 ... minor labels
        for t in cb.ax.get_yticklabels():
            t.set_fontfamily('serif'); t.set_fontweight('bold'); t.set_fontsize(J.PS.FS_CTICK)
    J.corner_label(ax, title)

def _outline(ax, which, color=None, ls='--'):
    """True body outline: same colour and line width in every figure (journal_style.OUTLINE);
    dashed for resistivity bodies, ls=':' (dotted) for chargeable bodies."""
    import journal_style as J
    XX, ZZ, M = _body_grid(which)
    ax.contour(XX, ZZ, M.astype(float), [0.5], **(J.OUTLINE | dict(linestyles=ls)))

def _body_grid(which):
    """Exact body geometry on a fine grid (same definitions as true_model / true_charg)."""
    xs = np.linspace(-20, 920, 1800); zs = np.linspace(1560, 1695, 700)
    XX, ZZ = np.meshgrid(xs, zs); S = G.surf_topo(XX); D = S - ZZ
    if which == 'C': M = (XX > 260) & (XX < 380) & (ZZ > 1597) & (ZZ < 1632) & (D > 8)
    elif which == 'CC': M = (XX > 220) & (XX < 320) & (D > 10) & (D < 32)
    elif which == 'CB': M = (XX > 170) & (XX < 270) & (D > 10) & (D < 32)
    elif which == 'R': M = (XX > 640) & (XX < 780) & (D > 10) & (D < 45)
    elif which == 'offset': M = (XX > 340) & (XX < 440) & (D > 12) & (D < 35)
    elif which == 'coincident': M = (XX > 260) & (XX < 380) & (ZZ > 1597) & (ZZ < 1632) & (D > 8)
    return XX, ZZ, M
