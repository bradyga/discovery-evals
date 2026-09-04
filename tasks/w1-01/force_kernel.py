"""w1-01 — coarse-grained force law with coordination-dependent oscillator stiffness.

Implements the model specified in ``spec.md`` (section ``## Method``):

    U(r, xi) = U_pair(r) + U_stiff(r, xi)

    U_pair  = sum_{i<j} 4 eps_ab [ (sig_ab/r)^12 - (sig_ab/r)^6 ] S(r; rc_on, rc_off)
    n_i     = sum_{j != i, s_j = s_i} S(r_ij; rn_on, rn_off)
    omega(n)= omega0 [ 1/2 - (1/pi) arctan( alpha (n - n0) ) ]
    U_stiff = (1/2) sum_i m_{s_i} omega(n_i)^2 sum_g xi[i,g]^2

with ``xi`` held fixed and F_k = -dU/dr_k.

The model is a free cluster in vacuum: no box, no periodic images, no
minimum-image convention.  Every parameter is read from ``params``; N and l are
read from the argument shapes.  No file, argv or network access; import has no
side effects.

Analytic gradients used below (u_ij = (r_i - r_j) / r_ij):

    pair        F_k = -sum_{j != k} phi'(r_kj) u_kj

    with C_i = dU_stiff/dn_i = m_{s_i} (sum_g xi[i,g]^2) omega(n_i) omega'(n_i)
    and  W_kj = S'(r_kj; rn_on, rn_off) for s_k == s_j, k != j, else 0 (W is symmetric),

    coord_self  F_k = -C_k       sum_j W_kj u_kj        (from k's own n_k)
    coord_cross F_k = -sum_j C_j W_kj u_kj              (from every other n_j)

The two coordination half-terms sum to -sum_j (C_k + C_j) W_kj u_kj, which is
pairwise antisymmetric and therefore conserves momentum; neither half does so
alone, and their net forces cancel exactly against each other.
"""

from __future__ import annotations

import numpy as np

__all__ = ["energy_and_forces", "force_terms"]


# ---------------------------------------------------------------------------
# switching function
# ---------------------------------------------------------------------------
def _switch(r, a, b):
    """C^2 quintic switch S(r; a, b) and its radial derivative dS/dr.

    S = 1 for r <= a, S = 1 - 10x^3 + 15x^4 - 6x^5 for a < r < b, S = 0 for
    r >= b, with x = (r - a) / (b - a).  Clipping x to [0, 1] reproduces the
    three branches exactly and branch-free: the quintic is 1 at x = 0 and 0 at
    x = 1, and its first two derivatives vanish at both endpoints, so the
    clipped form is C^2 across r = a and r = b as well.
    """
    width = b - a
    if not width > 0.0:
        raise ValueError(f"switching window requires r_on < r_off, got r_on={a}, r_off={b}")
    x = np.clip((r - a) / width, 0.0, 1.0)
    x2 = x * x
    x3 = x2 * x
    s = 1.0 + x3 * (-10.0 + x * (15.0 - 6.0 * x))
    ds = (-30.0 * x2 * (1.0 - x) ** 2) / width
    return s, ds


# ---------------------------------------------------------------------------
# shared kernel
# ---------------------------------------------------------------------------
def _kernel(positions, species, xi, params):
    """Compute the energy and the three named force terms in one pass."""
    pos = np.asarray(positions, dtype=np.float64)
    sp = np.asarray(species, dtype=np.int64)
    x = np.asarray(xi, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")
    n_part = pos.shape[0]
    if sp.shape != (n_part,):
        raise ValueError(f"species must have shape ({n_part},), got {sp.shape}")
    if x.ndim != 2 or x.shape[0] != n_part:
        raise ValueError(f"xi must have shape ({n_part}, l), got {x.shape}")

    # --- parameters, all read from `params` -------------------------------
    eps_mat = np.asarray(params["epsilon"], dtype=np.float64)
    sig_mat = np.asarray(params["sigma"], dtype=np.float64)
    rc_on = float(params["lj_switch"]["r_on"])
    rc_off = float(params["lj_switch"]["r_off"])
    rn_on = float(params["coord_switch"]["r_on"])
    rn_off = float(params["coord_switch"]["r_off"])
    omega0 = float(params["omega0"])
    alpha = float(params["alpha"])
    n0 = float(params["n0"])
    m_osc = np.asarray(params["m_osc"], dtype=np.float64)

    # --- geometry: free cluster, no images ---------------------------------
    diff = pos[:, None, :] - pos[None, :, :]          # diff[i, j] = r_i - r_j
    r2 = np.einsum("ijk,ijk->ij", diff, diff)
    off = ~np.eye(n_part, dtype=bool)
    r = np.sqrt(np.where(off, r2, 1.0))               # diagonal parked at 1, masked out below
    unit = np.where(off[:, :, None], diff / r[:, :, None], 0.0)   # unit[i, j] = u_ij

    eps = eps_mat[sp[:, None], sp[None, :]]
    sig = sig_mat[sp[:, None], sp[None, :]]

    # --- pair term ---------------------------------------------------------
    sr6 = (sig / r) ** 6
    sr12 = sr6 * sr6
    lj = 4.0 * eps * (sr12 - sr6)
    dlj = 4.0 * eps * (-12.0 * sr12 + 6.0 * sr6) / r
    s_lj, ds_lj = _switch(r, rc_on, rc_off)

    phi = np.where(off, lj * s_lj, 0.0)
    dphi = np.where(off, dlj * s_lj + lj * ds_lj, 0.0)

    e_pair = 0.5 * float(phi.sum())                   # 0.5 * sum_{i != j} == sum_{i<j}
    f_pair = -np.einsum("ij,ijk->ik", dphi, unit)

    # --- homo-coordination and the stiffness term --------------------------
    same = off & (sp[:, None] == sp[None, :])
    s_n, ds_n = _switch(r, rn_on, rn_off)
    n_coord = np.where(same, s_n, 0.0).sum(axis=1)

    z = alpha * (n_coord - n0)
    omega = omega0 * (0.5 - np.arctan(z) / np.pi)
    d_omega = -omega0 * alpha / (np.pi * (1.0 + z * z))   # d omega / d n

    xi2 = (x * x).sum(axis=1)                          # sum_g xi[i, g]^2
    mass = m_osc[sp]

    e_stiff = 0.5 * float((mass * omega**2 * xi2).sum())

    # C_i = dU_stiff / dn_i
    c_coeff = mass * xi2 * omega * d_omega

    w = np.where(same, ds_n, 0.0)                      # symmetric: W_kj = S'(r_kj)
    f_self = -c_coeff[:, None] * np.einsum("kj,kjm->km", w, unit)
    f_cross = -np.einsum("j,kj,kjm->km", c_coeff, w, unit)

    energy = e_pair + e_stiff
    return energy, {"pair": f_pair, "coord_self": f_self, "coord_cross": f_cross}


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------
def energy_and_forces(positions, species, xi, params):
    """Total energy U(r, xi) and forces F_k = -dU/dr_k at fixed xi.

    positions : float64 ndarray, shape (N, 3)
    species   : int64 ndarray, shape (N,), 0 = A, 1 = B
    xi        : float64 ndarray, shape (N, l)
    params    : dict, exactly as parsed from config/params.json

    Returns (energy, forces) with energy a float and forces float64 (N, 3).
    Inputs are never modified in place.
    """
    energy, terms = _kernel(positions, species, xi, params)
    forces = terms["pair"] + terms["coord_self"] + terms["coord_cross"]
    return energy, forces


def force_terms(positions, species, xi, params):
    """The three named force terms, summing to the force from energy_and_forces().

    Keys: "pair", "coord_self", "coord_cross", each float64 (N, 3).
    """
    _, terms = _kernel(positions, species, xi, params)
    return terms
