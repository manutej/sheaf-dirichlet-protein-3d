"""Build an interactive 3D Plotly view of the sheaf-Dirichlet picture on 1CRN."""
import os
import numpy as np
import plotly.graph_objects as go
from jax_sheaf_fold3d import (
    fetch_pdb, load_ca_pdb, anm_sheaf_hessian, harmonic_spectrum,
    flexibility_from_modes, oseens_velocity,
)

def main():
    path = "1CRN.pdb"
    if not os.path.exists(path):
        fetch_pdb("1CRN", path)
    X, seq = load_ca_pdb(path)
    H, ii, jj, D = anm_sheaf_hessian(X)
    evals, evecs = harmonic_spectrum(H)
    flex = flexibility_from_modes(evals, evecs, len(X))
    n = len(X)
    mode = evecs[:, 6].reshape(n, 3)
    scale = 8.0

    backbone = go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode="lines+markers+text",
        line=dict(color="#88aacc", width=6),
        marker=dict(size=6, color=flex, colorscale="Viridis",
                    colorbar=dict(title="flex"), cmin=0, cmax=1),
        text=[f"{s}{k+1}" for k, s in enumerate(seq)],
        textposition="top center",
        name="Ca + sequence",
    )
    xe, ye, ze = [], [], []
    for a, b in zip(ii, jj):
        if abs(int(a) - int(b)) == 1 or D[a, b] < 8.0:
            xe += [X[a, 0], X[b, 0], None]
            ye += [X[a, 1], X[b, 1], None]
            ze += [X[a, 2], X[b, 2], None]
    contacts = go.Scatter3d(
        x=xe, y=ye, z=ze, mode="lines",
        line=dict(color="rgba(180,200,220,0.25)", width=2),
        name="sheaf edges",
        hoverinfo="skip",
    )
    xa, ya, za = [], [], []
    for k in range(n):
        p, q = X[k], X[k] + scale * mode[k]
        xa += [p[0], q[0], None]
        ya += [p[1], q[1], None]
        za += [p[2], q[2], None]
    arrows = go.Scatter3d(
        x=xa, y=ya, z=za, mode="lines",
        line=dict(color="#ff7a3d", width=5),
        name="softest non-rigid mode",
        hoverinfo="skip",
    )

    mn, mx = X.min(0) - 6, X.max(0) + 6
    grid = np.array(np.meshgrid(
        np.linspace(mn[0], mx[0], 8),
        np.linspace(mn[1], mx[1], 8),
        np.linspace(mn[2], mx[2], 6),
        indexing="ij",
    )).reshape(3, -1).T
    keep = []
    for p in grid:
        if np.linalg.norm(X - p, axis=1).min() >= 3.5:
            keep.append(p)
    keep = np.asarray(keep)
    V = oseens_velocity(keep, X, mode)
    nrm = np.linalg.norm(V, axis=1)
    V = V / (np.percentile(nrm[nrm > 0], 80) + 1e-12) * 2.5
    xs, ys, zs = [], [], []
    for p, v in zip(keep, V):
        q = p + v
        xs += [p[0], q[0], None]
        ys += [p[1], q[1], None]
        zs += [p[2], q[2], None]
    stokes = go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color="rgba(80,220,180,0.55)", width=3),
        name="Stokes/Oseen field",
        hoverinfo="skip",
    )

    fig = go.Figure(data=[contacts, backbone, arrows, stokes])
    fig.update_layout(
        title="1CRN — sheaf Dirichlet energy in 3D (flex / soft mode / Stokes)",
        scene=dict(aspectmode="data", bgcolor="#0b1020",
                   xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)"),
        paper_bgcolor="#070b14", font=dict(color="#dce6f5"),
        width=1100, height=800,
    )
    fig.write_html("crambin_sheaf_3d.html", include_plotlyjs="cdn")
    print("wrote crambin_sheaf_3d.html")
    print("zero modes", int(np.sum(np.abs(evals) < 1e-8)), "soft", evals[6:10])

if __name__ == "__main__":
    main()
