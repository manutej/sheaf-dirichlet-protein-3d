# jax_sheaf_fold3d.py
# Sheaf-Dirichlet 3D protein prototype.
# pip install jax jaxlib numpy
#
# Physics contract:
#   Discrete Dirichlet energy on an anisotropic cellular sheaf = ANM Hessian.
#   Automatic sequence = softmax logits on residue stalks + contact energy.
#   Protein-scale "NS" is Stokes: R_dot = M_RPY(R) * (-grad E).
#   No closed-form 3D NS solution exists; NS is not a folding oracle.

from __future__ import annotations

import json
from typing import Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, value_and_grad
    HAVE_JAX = True
except Exception:
    jax = None
    jnp = np
    HAVE_JAX = False

    def jit(fn):
        return fn

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def load_ca_pdb(path: str) -> Tuple[np.ndarray, str]:
    coords, seq = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                seq.append(AA3TO1.get(line[17:20].strip(), "X"))
    return np.asarray(coords, dtype=np.float64), "".join(seq)


def fetch_pdb(pdb_id: str, dest: str) -> str:
    import urllib.request
    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", dest)
    return dest


def contact_pairs(X, cutoff=13.0):
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    i, j = np.where(np.triu(d <= cutoff, k=1))
    return i, j, d


def anm_sheaf_hessian(X, cutoff=13.0, gamma=1.0):
    """Anisotropic cellular sheaf Laplacian L_F = ANM Hessian."""
    n = len(X)
    i, j, d = contact_pairs(X, cutoff)
    H = np.zeros((3 * n, 3 * n), dtype=np.float64)
    for a, b in zip(i, j):
        rij = X[b] - X[a]
        r0 = float(np.linalg.norm(rij))
        u = rij / r0
        K = gamma * np.outer(u, u)
        ia, ib = 3 * int(a), 3 * int(b)
        H[ia:ia+3, ia:ia+3] += K
        H[ib:ib+3, ib:ib+3] += K
        H[ia:ia+3, ib:ib+3] -= K
        H[ib:ib+3, ia:ia+3] -= K
    return H, i, j, d


def dirichlet_energy(H, x_flat):
    return 0.5 * float(x_flat @ H @ x_flat)


def harmonic_spectrum(H):
    return np.linalg.eigh(H)


def flexibility_from_modes(evals, evecs, n, n_modes=42):
    flex = np.zeros(n)
    for k in range(6, min(len(evals), 6 + n_modes)):
        if evals[k] < 1e-10:
            continue
        v = evecs[:, k].reshape(n, 3)
        flex += np.sum(v * v, axis=1) / evals[k]
    m = float(flex.max())
    return flex / m if m > 0 else flex


def sequence_energy(logits, R, cutoff=8.0):
    p = jax.nn.softmax(logits, axis=-1)
    hydro_idx = jnp.array([AA20.index(a) for a in "AILMFVWY"])
    hydro = jnp.sum(p[:, hydro_idx], axis=-1)
    diff = R[:, None, :] - R[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-12)
    n = R.shape[0]
    neigh = ((dist < cutoff) & (jnp.eye(n) == 0)).astype(logits.dtype)
    burial = neigh.sum(axis=1)
    burial = burial / (jnp.max(burial) + 1e-6)
    e_burial = jnp.mean((hydro - burial) ** 2)
    compat = (hydro[:, None] - hydro[None, :]) ** 2
    e_contact = jnp.sum(neigh * compat) / (jnp.sum(neigh) + 1e-6)
    entropy = -jnp.mean(jnp.sum(p * jnp.log(p + 1e-8), axis=-1))
    return e_burial + 0.3 * e_contact + 0.05 * entropy


def optimize_sequence(R, steps=200, lr=0.2, seed=0):
    if not HAVE_JAX:
        raise RuntimeError("Install JAX to run automatic sequence.")
    key = jax.random.PRNGKey(seed)
    logits = 0.1 * jax.random.normal(key, (R.shape[0], 20))

    def loss_fn(lg):
        return sequence_energy(lg, jnp.asarray(R))

    @jit
    def step(lg):
        val, g = value_and_grad(loss_fn)(lg)
        return lg - lr * g, val

    hist = []
    for _ in range(steps):
        logits, val = step(logits)
        hist.append(float(val))
    p = jax.nn.softmax(logits, axis=-1)
    seq = "".join(AA20[int(i)] for i in np.array(jnp.argmax(p, axis=-1)))
    return seq, np.array(p), hist


def oseens_velocity(points, sources, forces, mu=1.0):
    V = np.zeros_like(points, dtype=np.float64)
    for p_idx, p in enumerate(points):
        vel = np.zeros(3)
        for r, f in zip(sources, forces):
            d = p - r
            s = float(np.linalg.norm(d))
            if s < 1e-8:
                continue
            rhat = d / s
            vel += (f + rhat * np.dot(rhat, f)) / (8.0 * np.pi * mu * s)
        V[p_idx] = vel
    return V


def rpy_self_mobility(radius=2.0, eta=1.0):
    return np.eye(3) / (6.0 * np.pi * eta * radius)


def demo_1crn(pdb_path="1CRN.pdb"):
    X, seq = load_ca_pdb(pdb_path)
    H, i, j, d = anm_sheaf_hessian(X)
    evals, evecs = harmonic_spectrum(H)
    flex = flexibility_from_modes(evals, evecs, len(X))
    rigid = np.tile([1.0, 0.0, 0.0], len(X))
    e_rigid = dirichlet_energy(H, rigid)
    rng = np.random.default_rng(0)
    rnd = rng.normal(size=3 * len(X))
    rnd /= np.linalg.norm(rnd)
    report = {
        "n_residues": int(len(X)),
        "native_sequence": seq,
        "n_contacts": int(len(i)),
        "n_zero_modes": int(np.sum(np.abs(evals) < 1e-8)),
        "soft_evals": [float(v) for v in evals[6:10]],
        "E_rigid": e_rigid,
        "E_random": dirichlet_energy(H, rnd),
        "have_jax": HAVE_JAX,
    }
    print(json.dumps(report, indent=2))
    return X, seq, H, evals, evecs, flex, report


if __name__ == "__main__":
    import os
    path = "1CRN.pdb"
    if not os.path.exists(path):
        print("fetching 1CRN from RCSB...")
        fetch_pdb("1CRN", path)
    demo_1crn(path)
