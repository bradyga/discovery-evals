#!/usr/bin/env python3
"""w1-01 — checks beyond the four in verify.py.

verify.py certifies that the *returned* force is the gradient of the *returned*
energy and that the returned energy has the right symmetries and degenerate
limits.  Everything here is aimed at what that leaves open, and each check is
labelled with what it does and does not rule out.

  A  independent re-implementation      naive O(N^2) Python loops, written from
                                        spec.md by a different route (explicit
                                        branches in S, definitional coordination
                                        derivative), compared term by term
  B  complex-step gradient              d/dx of the NAIVE energy at ~1e-16,
                                        against the vectorised kernel's forces
  C  per-term gradient attribution      the coord_self / coord_cross split
                                        against numerical dn_i/dr_k
  D  scale covariance                   r, sigma and both switch windows scaled
                                        by lambda: U invariant, F -> F/lambda
  E  strict locality                    a particle beyond both cutoffs changes
                                        nothing, and feels no force
  F  parity                             r -> -r: U invariant, F -> -F
  G  particle-order covariance          relabelling permutes F and leaves U alone
  H  omega(n) range and monotonicity    the stated property of omega, on the
                                        coordination numbers this config produces

Run:  python extra_checks.py
"""

from __future__ import annotations

import cmath
import json
import math
from pathlib import Path

import numpy as np

import force_kernel as fk

HERE = Path(__file__).resolve().parent
TERMS = ("pair", "coord_self", "coord_cross")


# ==========================================================================
# An independent, naive implementation, written straight from spec.md.
# Explicit loops, explicit switch branches, and the coordination derivative
# taken from the definition of n_i rather than from the pairwise identity the
# vectorised kernel uses.  `dtype` lets the same code run in complex arithmetic
# for the complex-step check, so the energy it returns is holomorphic in the
# positions.
# ==========================================================================
def _atan(z):
    """arctan, valid for real and complex argument.

    The complex branch is CPython's own cmath.atan (libm-backed), not a
    hand-written derivative: the log-form identity
    atan(z) = (log(1+iz) - log(1-iz)) / 2i is mathematically equivalent but
    numerically useless for a complex step, because it folds the O(1e-30)
    imaginary part into an O(1) real part where it is rounded away.
    """
    if isinstance(z, complex):
        return cmath.atan(z)
    return np.arctan(z)


def _S_naive(r, a, b):
    """S and dS/dr, with the three branches written out."""
    rr = r.real if isinstance(r, complex) else r
    if rr <= a:
        return 1.0 + 0.0 * r, 0.0 * r
    if rr >= b:
        return 0.0 * r, 0.0 * r
    x = (r - a) / (b - a)
    s = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
    ds = (-30.0 * x**2 + 60.0 * x**3 - 30.0 * x**4) / (b - a)
    return s, ds


def naive(pos, sp, xi, p, want_forces=True):
    """Return (energy, terms) computed with explicit loops. terms is None if
    want_forces is False (the path used for the complex-step check)."""
    n = len(sp)
    eps = p["epsilon"]
    sig = p["sigma"]
    rc_on, rc_off = p["lj_switch"]["r_on"], p["lj_switch"]["r_off"]
    rn_on, rn_off = p["coord_switch"]["r_on"], p["coord_switch"]["r_off"]
    w0, alpha, n0 = p["omega0"], p["alpha"], p["n0"]
    m_osc = p["m_osc"]

    def dist(i, j):
        d = [pos[i][k] - pos[j][k] for k in range(3)]
        r = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5
        return d, r

    # --- pair energy, sum over i<j -------------------------------------
    e_pair = 0.0 * pos[0][0]
    for i in range(n):
        for j in range(i + 1, n):
            _, r = dist(i, j)
            s, _ = _S_naive(r, rc_on, rc_off)
            e = eps[sp[i]][sp[j]]
            g = sig[sp[i]][sp[j]]
            e_pair = e_pair + 4.0 * e * ((g / r) ** 12 - (g / r) ** 6) * s

    # --- homo-coordination ----------------------------------------------
    ncoord = []
    for i in range(n):
        acc = 0.0 * pos[0][0]
        for j in range(n):
            if j == i or sp[j] != sp[i]:
                continue
            _, r = dist(i, j)
            s, _ = _S_naive(r, rn_on, rn_off)
            acc = acc + s
        ncoord.append(acc)

    omega = [w0 * (0.5 - _atan(alpha * (ncoord[i] - n0)) / math.pi) for i in range(n)]
    xi2 = [sum(v * v for v in xi[i]) for i in range(n)]
    e_stiff = 0.0 * pos[0][0]
    for i in range(n):
        e_stiff = e_stiff + 0.5 * m_osc[sp[i]] * omega[i] ** 2 * xi2[i]

    energy = e_pair + e_stiff
    if not want_forces:
        return energy, None

    # --- pair force ------------------------------------------------------
    f_pair = np.zeros((n, 3))
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            d, r = dist(i, j)
            s, ds = _S_naive(r, rc_on, rc_off)
            e = eps[sp[i]][sp[j]]
            g = sig[sp[i]][sp[j]]
            lj = 4.0 * e * ((g / r) ** 12 - (g / r) ** 6)
            dlj = 4.0 * e * (-12.0 * g**12 / r**13 + 6.0 * g**6 / r**7)
            dphi = dlj * s + lj * ds
            for k in range(3):
                f_pair[i][k] -= dphi * d[k] / r

    # --- C_i = dU_stiff/dn_i --------------------------------------------
    dom = [-w0 * alpha / (math.pi * (1.0 + (alpha * (ncoord[i] - n0)) ** 2)) for i in range(n)]
    C = [m_osc[sp[i]] * xi2[i] * omega[i] * dom[i] for i in range(n)]

    # --- coordination forces, straight from the definition ---------------
    # coord_self_k  = -C_k sum_{j in nbrs(k)} dS(r_kj)/dr_k
    # coord_cross_k = -sum_{i != k} C_i dn_i/dr_k, i.e. for every i whose
    #                 neighbour list contains k, the derivative of that one term.
    f_self = np.zeros((n, 3))
    f_cross = np.zeros((n, 3))
    for k in range(n):
        for j in range(n):
            if j == k or sp[j] != sp[k]:
                continue
            d, r = dist(k, j)
            _, ds = _S_naive(r, rn_on, rn_off)
            for a in range(3):
                f_self[k][a] -= C[k] * ds * d[a] / r
    for i in range(n):
        for k in range(n):
            if k == i or sp[k] != sp[i]:
                continue
            # n_i contains S(r_ik); its derivative w.r.t. r_k is S' (r_k-r_i)/r_ik
            d, r = dist(k, i)
            _, ds = _S_naive(r, rn_on, rn_off)
            for a in range(3):
                f_cross[k][a] -= C[i] * ds * d[a] / r

    return energy, {"pair": f_pair, "coord_self": f_self, "coord_cross": f_cross}


# ==========================================================================
def report(name, value, ok, note=""):
    mark = "ok  " if ok else "FAIL"
    v = f"{value:.3e}" if isinstance(value, float) else str(value)
    print(f"  [{mark}] {name:<52s} {v}  {note}")
    return bool(ok)


def main() -> int:
    cfg = json.loads((HERE / "config" / "config.json").read_text())
    params = json.loads((HERE / "config" / "params.json").read_text())
    pos = np.asarray(cfg["positions"], dtype=np.float64)
    sp = np.asarray(cfg["species"], dtype=np.int64)
    xi = np.asarray(cfg["xi"], dtype=np.float64)

    E, F = fk.energy_and_forces(pos, sp, xi, params)
    T = fk.force_terms(pos, sp, xi, params)
    fs = float(np.linalg.norm(F, axis=1).mean())
    print(f"w1-01 extra checks   (E = {E:.9e},  F_scale = {fs:.6e})\n")
    ok = True

    # ---- A: independent naive re-implementation -----------------------
    print("A  independent naive re-implementation (loops, definitional coord derivative)")
    posl = pos.tolist()
    En, Tn = naive(posl, sp.tolist(), xi.tolist(), params)
    ok &= report("|E_naive - E| / |E|", abs(En - E) / abs(E), abs(En - E) / abs(E) < 1e-12)
    for k in TERMS:
        d = float(np.abs(Tn[k] - T[k]).max()) / fs
        ok &= report(f"max |{k}_naive - {k}| / F_scale", d, d < 1e-12)
    print()

    # ---- B: complex-step gradient of the naive energy -----------------
    print("B  complex-step d(E_naive)/dx  vs  the vectorised kernel's force")
    rng = np.random.default_rng(11)
    h = 1e-30
    worst = 0.0
    for _ in range(6):
        i, a = int(rng.integers(len(sp))), int(rng.integers(3))
        pc = [[complex(v) for v in row] for row in posl]
        pc[i][a] += 1j * h
        Ec, _ = naive(pc, sp.tolist(), xi.tolist(), params, want_forces=False)
        fnum = -Ec.imag / h
        worst = max(worst, abs(fnum - F[i, a]) / fs)
    ok &= report("worst |F_cs - F| / F_scale  (6 samples)", worst, worst < 1e-13)
    print()

    # ---- C: attribution of the coordination split ---------------------
    print("C  coord_self / coord_cross split vs numerical dn_i/dr_k")
    def ncoord_of(p):
        d = p[:, None, :] - p[None, :, :]
        r = np.sqrt((d * d).sum(-1) + np.eye(len(p)))
        s, _ = fk._switch(r, params["coord_switch"]["r_on"], params["coord_switch"]["r_off"])
        m = (~np.eye(len(p), dtype=bool)) & (sp[:, None] == sp[None, :])
        return np.where(m, s, 0.0).sum(axis=1)

    # C_i recovered from the model definition, independently of the kernel
    n_i = ncoord_of(pos)
    w0, al, n0 = params["omega0"], params["alpha"], params["n0"]
    z = al * (n_i - n0)
    om = w0 * (0.5 - np.arctan(z) / np.pi)
    dom = -w0 * al / (np.pi * (1 + z * z))
    Ci = np.asarray(params["m_osc"])[sp] * (xi * xi).sum(1) * om * dom

    step = 1e-6
    ws, wc = 0.0, 0.0
    for k in [3, 41, 97, 150, 188]:
        for a in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[k, a] += step
            pm[k, a] -= step
            dn = (ncoord_of(pp) - ncoord_of(pm)) / (2 * step)   # dn_i/dr_ka for all i
            self_pred = -Ci[k] * dn[k]
            cross_pred = -(Ci * dn).sum() + Ci[k] * dn[k]
            ws = max(ws, abs(self_pred - T["coord_self"][k, a]) / fs)
            wc = max(wc, abs(cross_pred - T["coord_cross"][k, a]) / fs)
    ok &= report("worst |coord_self - (-C_k dn_k/dr_k)| / F_scale", ws, ws < 1e-9)
    ok &= report("worst |coord_cross - (-sum_i!=k C_i dn_i/dr_k)| / F_scale", wc, wc < 1e-9)
    print()

    # ---- D: scale covariance -------------------------------------------
    print("D  scale covariance: r, sigma, both switch windows x lambda")
    lam = 1.7
    q = json.loads(json.dumps(params))
    q["sigma"] = [[lam * v for v in row] for row in params["sigma"]]
    q["lj_switch"] = {k: lam * v for k, v in params["lj_switch"].items()}
    q["coord_switch"] = {k: lam * v for k, v in params["coord_switch"].items()}
    Es, Fs = fk.energy_and_forces(pos * lam, sp, xi, q)
    dE = abs(Es - E) / abs(E)
    dF = float(np.abs(Fs - F / lam).max()) / fs
    ok &= report("|U(lambda) - U| / |U|", dE, dE < 1e-12)
    ok &= report("max |F(lambda) - F/lambda| / F_scale", dF, dF < 1e-12)
    print()

    # ---- E: strict locality --------------------------------------------
    print("E  strict locality: an extra particle beyond both cutoffs")
    far = pos.max() + params["lj_switch"]["r_off"] + 10.0
    pos2 = np.vstack([pos, [[far, far, far]]])
    sp2 = np.concatenate([sp, [0]])
    xi2 = np.vstack([xi, xi[:1]])
    E2, F2 = fk.energy_and_forces(pos2, sp2, xi2, params)
    T2 = fk.force_terms(pos2, sp2, xi2, params)
    dloc = float(np.abs(F2[:-1] - F).max()) / fs
    dfar = float(np.abs(F2[-1]).max()) / fs
    ok &= report("max |F(with distant particle) - F| / F_scale", dloc, dloc == 0.0)
    ok &= report("max |F on the distant particle| / F_scale", dfar, dfar == 0.0)
    for k in TERMS:
        d = float(np.abs(T2[k][:-1] - T[k]).max()) / fs
        ok &= report(f"  {k}: unchanged", d, d == 0.0)
    print()

    # ---- F: parity ------------------------------------------------------
    print("F  parity r -> -r (an improper transformation; verify.py tests only proper ones)")
    Ep, Fp = fk.energy_and_forces(-pos, sp, xi, params)
    Tp = fk.force_terms(-pos, sp, xi, params)
    dE = abs(Ep - E) / abs(E)
    dF = float(np.abs(Fp + F).max()) / fs
    ok &= report("|U(-r) - U| / |U|", dE, dE < 1e-14)
    ok &= report("max |F(-r) + F| / F_scale", dF, dF < 1e-14)
    for k in TERMS:
        d = float(np.abs(Tp[k] + T[k]).max()) / fs
        ok &= report(f"  {k}: max |F(-r) + F| / F_scale", d, d < 1e-14)
    print()

    # ---- G: particle-order covariance -----------------------------------
    print("G  particle-order covariance")
    perm = np.random.default_rng(5).permutation(len(sp))
    Eq, Fq = fk.energy_and_forces(pos[perm], sp[perm], xi[perm], params)
    Tq = fk.force_terms(pos[perm], sp[perm], xi[perm], params)
    dE = abs(Eq - E) / abs(E)
    dF = float(np.abs(Fq - F[perm]).max()) / fs
    ok &= report("|U(perm) - U| / |U|", dE, dE < 1e-13)
    ok &= report("max |F(perm) - F[perm]| / F_scale", dF, dF < 1e-13)
    for k in TERMS:
        d = float(np.abs(Tq[k] - T[k][perm]).max()) / fs
        ok &= report(f"  {k}: permutes", d, d < 1e-13)
    print()

    # ---- H: omega(n) recovered from the kernel's own energy ---------------
    print("H  omega(n) recovered from the kernel by isolating each particle's xi")
    # U(xi_iso_i) - U(0) = (1/2) m_i omega(n_i)^2 sum_g xi[i,g]^2 : the pair energy
    # is identical in both and cancels, so this reads omega_i out of the kernel
    # without re-transcribing the omega(n) formula.
    E0, _ = fk.energy_and_forces(pos, sp, np.zeros_like(xi), params)
    mass = np.asarray(params["m_osc"], dtype=np.float64)[sp]
    xi2v = (xi * xi).sum(axis=1)
    om_meas = np.empty(len(sp))
    for i in range(len(sp)):
        iso = np.zeros_like(xi)
        iso[i] = xi[i]
        Ei, _ = fk.energy_and_forces(pos, sp, iso, params)
        om_meas[i] = math.sqrt(max(2.0 * (Ei - E0) / (mass[i] * xi2v[i]), 0.0))
    print(f"       n_i (from spec): min {n_i.min():.4f}  max {n_i.max():.4f}  mean {n_i.mean():.4f}")
    print(f"       omega_i (from kernel): min {om_meas.min():.4f}  max {om_meas.max():.4f}")
    w0f = float(params["omega0"])
    in_range = bool(np.all((om_meas > 0.0) & (om_meas < w0f)))
    ok &= report("omega_i in (0, omega0) for every particle", in_range, in_range)
    order = np.argsort(n_i)
    dn = np.diff(n_i[order])
    dw = np.diff(om_meas[order])
    # strictly decreasing in n wherever n actually differs; single-valued where it does not
    viol = int(np.sum((dn > 1e-9) & (dw >= 0.0)))
    ok &= report("omega strictly decreasing in n (violations)", viol, viol == 0)
    tie = float(np.abs(dw[dn <= 1e-9]).max()) if np.any(dn <= 1e-9) else 0.0
    ok &= report("omega single-valued in n at ties", tie, tie < 1e-9)
    spread = float(om_meas.max() - om_meas.min())
    ok &= report("omega spread across the configuration", spread, spread > 1e-3,
                 "(non-degenerate: the arctan is actually exercised)")
    print()

    print("EXTRA CHECKS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
