"""w1-02 — two coordination-dependent frequency laws, behind one interface.

Implements the model of ``spec.md`` (section ``## Method``) for *both* named
frequency laws:

    U(r, xi) = U_pair(r) + U_stiff(r, xi)

    U_pair  = sum_{i<j} 4 eps_ab [ (sig_ab/r)^12 - (sig_ab/r)^6 ] S(r; rc_on, rc_off)
    n_i     = sum_{j != i, s_j = s_i} S(r_ij; rn_on, rn_off)
    U_stiff = (1/2) sum_i m_{s_i} omega(n_i)^2 sum_g xi[i,g]^2

    form "arctan"    omega_A(n) = omega0 [ 1/2 - (1/pi) arctan( alpha (n - n0) ) ]
    form "rational"  omega_R(n) = omega0 / [ 1 + (n/n0)^p ],   p = 4 n0 alpha / pi

with ``xi`` held fixed and F_k = -dU/dr_k.  The system is a free cluster in
vacuum: no box, no periodic images, no minimum-image convention.

The calibration ``p = 4 n0 alpha / pi`` is applied on every call: it is derived
from ``params["alpha"]`` and ``params["n0"]``, never stored and never pinned.
It puts the two laws on a common footing by making them agree in value
(omega0/2) and in first derivative (-omega0 alpha / pi) at n = n0.

Analytic gradients (u_ij = (r_i - r_j) / r_ij):

    pair        F_k = -sum_{j != k} phi'(r_kj) u_kj

    with C_i = dU_stiff/dn_i = m_{s_i} (sum_g xi[i,g]^2) omega(n_i) omega'(n_i)
    and  W_kj = S'(r_kj; rn_on, rn_off) for s_k == s_j, k != j, else 0,

    coord_self  F_k = -C_k       sum_j W_kj u_kj      (from k's own n_k)
    coord_cross F_k = -sum_j C_j W_kj u_kj            (from every other n_j)

Only ``omega`` and ``omega'`` differ between the two forms; every other part of
the kernel is shared, so any difference the task measures is a difference of
frequency laws and of nothing else.

Every parameter is read from ``params``; N and l are read from the argument
shapes.  No file, argv or network access; import has no side effects and does
not depend on the working directory.

Conventions adopted where ``spec.md`` leaves a choice open are marked
``[UNDET-k]`` below and are listed in ``report.md``, section
``## Underdetermined in the specification``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "FORMS",
    "calibrate",
    "omega",
    "coordination",
    "energy_and_forces",
    "force_terms",
    "compare",
]

FORMS = ("arctan", "rational")

_TERM_KEYS = ("pair", "coord_self", "coord_cross")
_DS_KEYS = ("lo", "mid", "hi")


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
# parameters
# ---------------------------------------------------------------------------
def _check_form(form):
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS!r}, got {form!r}")
    return form


def _freq_params(params):
    """The three pinned frequency parameters, read from `params` every time."""
    omega0 = float(params["omega0"])
    alpha = float(params["alpha"])
    n0 = float(params["n0"])
    if not omega0 > 0.0:
        raise ValueError(f"omega0 must be positive, got {omega0}")
    if not n0 > 0.0:
        raise ValueError(f"n0 must be positive, got {n0}")
    return omega0, alpha, n0


def calibrate(params):
    """The common footing: the rational form's exponent, derived from alpha and n0.

    Both laws are pinned to agree at n = n0 in value and in first derivative,

        omega(n0) = omega0 / 2        omega'(n0) = - omega0 alpha / pi

    The arctan law satisfies both identically for every alpha.  For the
    rational law omega_R(n) = omega0 / [1 + (n/n0)^p] the value condition holds
    for every p, and

        omega_R'(n0) = - omega0 p / (4 n0)

    so the slope condition fixes  p = 4 n0 alpha / pi.

    `p` is derived here on every call and is never cached: `params` is the only
    source for `alpha` and `n0`.

    Returns a dict carrying "p" (required by the interface contract) together
    with the parameters it was derived from and the calibrated value and slope,
    so that a caller can check the footing without re-deriving it.
    """
    omega0, alpha, n0 = _freq_params(params)
    p = 4.0 * n0 * alpha / math.pi
    return {
        "p": p,
        "alpha": alpha,
        "n0": n0,
        "omega0": omega0,
        "omega_at_n0": 0.5 * omega0,
        "domega_at_n0": -omega0 * alpha / math.pi,
    }


# ---------------------------------------------------------------------------
# the two frequency laws
# ---------------------------------------------------------------------------
def _omega_and_deriv(n, form, params):
    """omega(n) and d omega / d n for the named form, as float64 arrays."""
    _check_form(form)
    n = np.asarray(n, dtype=np.float64)
    omega0, alpha, n0 = _freq_params(params)

    if form == "arctan":
        z = alpha * (n - n0)
        w = omega0 * (0.5 - np.arctan(z) / np.pi)
        dw = -omega0 * alpha / (np.pi * (1.0 + z * z))
        dw = np.broadcast_to(dw, n.shape).astype(np.float64, copy=True)
        return w, dw

    # form == "rational", in the closed expression omega0 / [1 + (n/n0)^p].
    p = float(calibrate(params)["p"])
    u = n / n0

    if p == 0.0:
        # [UNDET-1] alpha = 0 gives p = 0, where (n/n0)^p is 0^0 at n = 0.
        # spec.md states the intended consequence -- "both forms reduce to the
        # constant omega0 / 2" -- so the whole p = 0 law is taken as that
        # constant, i.e. 0^0 = 1.  This is the only reading that makes the law
        # constant, and it is discontinuous in p at n = 0: omega_R(0) = omega0
        # for every p > 0.
        w = np.full(n.shape, 0.5 * omega0, dtype=np.float64)
        dw = np.zeros(n.shape, dtype=np.float64)
        return w, dw

    # 0^p = 0 for p > 0, which numpy already gives; n < 0 is outside the law's
    # domain and is rejected rather than silently returning a nan.
    if np.any(n < 0.0):
        raise ValueError("the rational form is defined for n >= 0 only")
    t = u**p
    w = omega0 / (1.0 + t)
    # d omega / d n = -(omega^2 / omega0) * p * u^(p-1) / n0
    with np.errstate(divide="ignore", invalid="ignore"):
        du = p * u ** (p - 1.0) / n0
    dw = -(w * w / omega0) * du
    dw = np.where(np.isfinite(dw), dw, 0.0)
    return w.astype(np.float64, copy=False), dw.astype(np.float64, copy=False)


def omega(n, form, params):
    """The named frequency law evaluated at n.

    n      : float64 ndarray of any shape, or a real scalar; n >= 0
    form   : one of FORMS
    params : dict, exactly as parsed from config/params.json

    Returns omega(n), float64, of the same shape as n.  Inputs are not modified.
    """
    w, _ = _omega_and_deriv(n, form, params)
    return np.array(w, dtype=np.float64, copy=True)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _pair_geometry(positions):
    """Separations, unit vectors and the off-diagonal mask for a free cluster."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")
    n_part = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]           # diff[i, j] = r_i - r_j
    r2 = np.einsum("ijk,ijk->ij", diff, diff)
    off = ~np.eye(n_part, dtype=bool)
    r = np.sqrt(np.where(off, r2, 1.0))                # diagonal parked at 1, masked out
    unit = np.where(off[:, :, None], diff / r[:, :, None], 0.0)
    return pos, n_part, r, unit, off


def coordination(positions, species, params):
    """Smooth homo-coordination n_i: same-species neighbours only.

    positions : float64 ndarray, shape (N, 3)
    species   : int64 ndarray, shape (N,), 0 = A, 1 = B
    params    : dict

    Returns n_i, float64 ndarray, shape (N,).  Inputs are not modified.
    """
    pos, n_part, r, _, off = _pair_geometry(positions)
    sp = np.asarray(species, dtype=np.int64)
    if sp.shape != (n_part,):
        raise ValueError(f"species must have shape ({n_part},), got {sp.shape}")
    rn_on = float(params["coord_switch"]["r_on"])
    rn_off = float(params["coord_switch"]["r_off"])
    same = off & (sp[:, None] == sp[None, :])
    s_n, _ = _switch(r, rn_on, rn_off)
    return np.where(same, s_n, 0.0).sum(axis=1)


# ---------------------------------------------------------------------------
# energy and forces
# ---------------------------------------------------------------------------
def _kernel(positions, species, xi, params, form):
    """Total energy and the three named force terms, in one pass, for one form."""
    _check_form(form)
    pos, n_part, r, unit, off = _pair_geometry(positions)
    sp = np.asarray(species, dtype=np.int64)
    x = np.asarray(xi, dtype=np.float64)
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
    m_osc = np.asarray(params["m_osc"], dtype=np.float64)

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

    e_pair = 0.5 * float(phi.sum())                    # 0.5 * sum_{i != j} == sum_{i<j}
    f_pair = -np.einsum("ij,ijk->ik", dphi, unit)

    # --- homo-coordination and the stiffness term --------------------------
    same = off & (sp[:, None] == sp[None, :])
    s_n, ds_n = _switch(r, rn_on, rn_off)
    n_coord = np.where(same, s_n, 0.0).sum(axis=1)

    w, dw = _omega_and_deriv(n_coord, form, params)

    xi2 = (x * x).sum(axis=1)                          # sum_g xi[i, g]^2
    mass = m_osc[sp]

    e_stiff = 0.5 * float((mass * w**2 * xi2).sum())

    # C_i = dU_stiff / dn_i
    c_coeff = mass * xi2 * w * dw

    wmat = np.where(same, ds_n, 0.0)                   # symmetric: W_kj = S'(r_kj)
    f_self = -c_coeff[:, None] * np.einsum("kj,kjm->km", wmat, unit)
    f_cross = -np.einsum("j,kj,kjm->km", c_coeff, wmat, unit)

    energy = e_pair + e_stiff
    return energy, {"pair": f_pair, "coord_self": f_self, "coord_cross": f_cross}


def energy_and_forces(positions, species, xi, params, form):
    """Total energy U(r, xi) and forces F_k = -dU/dr_k at fixed xi, for one form.

    Returns (energy, forces): a float and a float64 (N, 3) array.  Inputs are
    never modified in place.
    """
    energy, terms = _kernel(positions, species, xi, params, form)
    forces = terms["pair"] + terms["coord_self"] + terms["coord_cross"]
    return energy, forces


def force_terms(positions, species, xi, params, form):
    """The three named force terms, summing to the force from energy_and_forces().

    Keys: "pair", "coord_self", "coord_cross", each float64 (N, 3).
    """
    _, terms = _kernel(positions, species, xi, params, form)
    return terms


# ---------------------------------------------------------------------------
# the criterion
# ---------------------------------------------------------------------------
def _params_at_alpha(params, alpha):
    """A shallow copy of `params` with `alpha` replaced.

    Shallow is enough and is deliberate: nothing downstream mutates the nested
    blocks, and the caller's dict is left untouched.
    """
    q = dict(params)
    q["alpha"] = float(alpha)
    return q


def delta_s(n_map, n_pinned, form, params):
    """Oscillator entropy change on demixing, in units of l k_B.

        dS = - sum_i [ ln omega(n_i^m) - ln omega(n_i^0) ]

    Not part of the interface contract; exposed because it is the quantity the
    criterion is built from and is worth being able to call directly.
    """
    w_m = omega(n_map, form, params)
    w_0 = omega(n_pinned, form, params)
    if np.any(w_m <= 0.0) or np.any(w_0 <= 0.0):
        raise ValueError("omega() is non-positive; the oscillator entropy is undefined there")
    return -float((np.log(w_m) - np.log(w_0)).sum())


def compare(positions, species, demix_maps, params):
    """Evaluate the criterion of spec.md, section '## The criterion'.

    For each demixing map m and each form, dS is evaluated at alpha_lo, alpha
    and alpha_hi, with `p` recalibrated at each alpha and with n0, omega0, the
    positions, the maps and the switching windows held fixed.  The decision
    statistic is

        R = | dS(alpha) | / | dS(alpha_hi) - dS(alpha_lo) |

    -- the demixing signal divided by its sensitivity to the model's own tuning
    parameter -- and on each map the larger R is preferred.  The choice is the
    form preferred on both maps with a relative margin of at least
    params["criterion"]["margin"] on both; otherwise "undecided".

    Returns {"delta_s": {map: {form: {"lo", "mid", "hi"}}},
             "R": {map: {form: float}},
             "choice": "arctan" | "rational" | "undecided"}
    """
    crit = params["criterion"]
    alpha = float(params["alpha"])
    settings = {
        "lo": float(crit["alpha_scale_lo"]) * alpha,
        "mid": alpha,
        "hi": float(crit["alpha_scale_hi"]) * alpha,
    }
    margin_threshold = float(crit["margin"])

    # Coordination depends on positions, species and coord_switch only -- not on
    # alpha -- so it is computed once per labelling and reused at all three
    # alpha settings.  Same positions, same switching window, throughout.
    n_pinned = coordination(positions, species, params)
    n_by_map = {m: coordination(positions, demix_maps[m], params) for m in demix_maps}

    ds_out, r_out = {}, {}
    for m in demix_maps:
        ds_out[m], r_out[m] = {}, {}
        for form in FORMS:
            vals = {
                key: delta_s(n_by_map[m], n_pinned, form, _params_at_alpha(params, a))
                for key, a in settings.items()
            }
            ds_out[m][form] = vals
            spread = abs(vals["hi"] - vals["lo"])
            if not spread > 0.0:
                raise ValueError(
                    f"dS for form {form!r} on map {m!r} does not respond to alpha; "
                    "R is undefined"
                )
            r_out[m][form] = abs(vals["mid"]) / spread

    choice, _, _ = _apply_rule(r_out, sorted(demix_maps), margin_threshold)
    return {"delta_s": ds_out, "R": r_out, "choice": choice}


def _apply_rule(R, maps, margin_threshold):
    """The rule of spec.md -> (choice, preference per map, margin per map).

    [UNDET-2] spec.md says "the form with the larger R is preferred" and does
    not say what an exact tie does.  A tie has margin 0, which is below any
    positive threshold, so the verdict is "undecided" whichever way the tie is
    broken; the arbitrary break below is therefore never load-bearing.
    [UNDET-3] spec.md's margin has min(R_arctan, R_rational) in the
    denominator and does not say what happens if that minimum is zero.  A zero
    R means a form shows no demixing signal at all while the other does, which
    is a decisive separation, so it is taken as an infinite margin.
    """
    prefs, margins = {}, {}
    for m in maps:
        ra, rb = float(R[m][FORMS[0]]), float(R[m][FORMS[1]])
        prefs[m] = FORMS[0] if ra > rb else FORMS[1]
        lo = min(ra, rb)
        margins[m] = (abs(ra - rb) / lo) if lo > 0.0 else math.inf
    winners = set(prefs.values())
    if len(winners) != 1:
        return "undecided", prefs, margins
    if any(margins[m] < margin_threshold for m in maps):
        return "undecided", prefs, margins
    return winners.pop(), prefs, margins
