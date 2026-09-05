"""w1-02 — the measurements report.md runs on.

Not a deliverable and not part of the interface contract: every number quoted
in report.md is produced here, from forms.py, so that the report can be
re-derived rather than trusted.  It imports forms.py and config/ and writes
nothing.

    python analysis.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import forms

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "config" / "params.json").read_text())
CFG = json.loads((HERE / "config" / "config.json").read_text())

POS = np.asarray(CFG["positions"], dtype=np.float64)
SPECIES = np.asarray(CFG["species"], dtype=np.int64)
XI = np.asarray(CFG["xi"], dtype=np.float64)
DEMIX = {k: np.asarray(v, dtype=np.int64) for k, v in CFG["demix_maps"].items()}
MAPS = sorted(DEMIX)
W0 = float(PARAMS["omega0"])
ALPHA = float(PARAMS["alpha"])
N0 = float(PARAMS["n0"])


def head(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def at(alpha=None, n0=None):
    q = dict(PARAMS)
    if alpha is not None:
        q["alpha"] = float(alpha)
    if n0 is not None:
        q["n0"] = float(n0)
    return q


# ===========================================================================
head("A.  the configuration the criterion sees")
# ===========================================================================
n_pin = forms.coordination(POS, SPECIES, PARAMS)
n_map = {m: forms.coordination(POS, DEMIX[m], PARAMS) for m in MAPS}
print(f"  N = {POS.shape[0]}, l = {XI.shape[1]},  n0 = {N0},  alpha = {ALPHA}, "
      f"omega0 = {W0}")
print(f"  p (calibrated)                = {forms.calibrate(PARAMS)['p']:.6f}")
for label, n in [("pinned", n_pin)] + [(m, n_map[m]) for m in MAPS]:
    print(f"  n_i [{label:>8s}]  min {n.min():6.3f}  mean {n.mean():6.3f}  "
          f"max {n.max():6.3f}   frac(n_i > n0) {np.mean(n > N0):.3f}")
print(f"  ln(n_i/n0) [pinned] range     = [{np.log(np.maximum(n_pin,1e-300)/N0).min():.3f}, "
      f"{np.log(n_pin.max()/N0):.3f}]   zeros: {int((n_pin==0).sum())}")

# ===========================================================================
head("B.  non-equivalence")
# ===========================================================================
print("  B1. the two exact identities, at the pinned parameters and on a grid")
GRID = [(ALPHA, N0), (0.4, 2.0), (0.4, 4.5), (0.8, 3.0), (1.5, 2.0), (1.5, 4.5)]
FRACS = (0.1, 0.25, 0.4, 0.6, 0.8)
TS = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def refl(form, q):
    nn0 = float(q["n0"])
    d = np.array([f * nn0 for f in FRACS])
    return float(np.abs(forms.omega(nn0 + d, form, q) + forms.omega(nn0 - d, form, q)
                        - float(q["omega0"])).max()) / float(q["omega0"])


def inv(form, q):
    nn0 = float(q["n0"])
    t = np.array(TS)
    return float(np.abs(forms.omega(nn0 * np.exp(t), form, q)
                        + forms.omega(nn0 * np.exp(-t), form, q)
                        - float(q["omega0"])).max()) / float(q["omega0"])


print(f"      {'(alpha, n0)':>14s} {'refl arctan':>13s} {'refl rational':>14s} "
      f"{'inv rational':>14s} {'inv arctan':>13s}")
for a, nn in GRID:
    q = at(a, nn)
    print(f"      {'(%.2f, %.2f)' % (a, nn):>14s} {refl('arctan', q):13.3e} "
          f"{refl('rational', q):14.3e} {inv('rational', q):14.3e} {inv('arctan', q):13.3e}")

print("\n  B2. the asymptotic decay exponent  -d ln omega / d ln n  at large n")
for nn in (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0):
    h = 1e-4 * nn
    out = []
    for form in forms.FORMS:
        wp = float(forms.omega(np.array([nn + h]), form, PARAMS)[0])
        wm = float(forms.omega(np.array([nn - h]), form, PARAMS)[0])
        w = float(forms.omega(np.array([nn]), form, PARAMS)[0])
        out.append(-(wp - wm) / (2 * h) * nn / w)
    print(f"      n = {nn:8.1f}   arctan {out[0]:8.5f}   rational {out[1]:8.5f}")

print("\n  B4. the second step of the analytic argument, measured.  If p = 1 the")
print("      rational law is Moebius in n, so 1/omega is exactly affine and")
print("      d^2(1/omega)/dn^2 vanishes identically.  It does not for arctan.")


def d2_inv_omega(form, q, n):
    h = 1e-3 * max(n, 1.0)
    f = [1.0 / float(forms.omega(np.array([n + k * h]), form, q)[0]) for k in (-1, 0, 1)]
    return (f[0] - 2 * f[1] + f[2]) / (h * h)


q_p1 = at(math.pi / (4 * N0))                     # the alpha for which p = 1 exactly
print(f"      (p = 1 is reached at alpha = pi/(4 n0) = {math.pi / (4 * N0):.6f}; "
      f"p there = {forms.calibrate(q_p1)['p']:.10f})")
print(f"      {'n':>8s} {'arctan @pinned':>16s} {'arctan @p=1 alpha':>19s} {'rational @p=1':>15s}")
for n in (4.0, 6.0, 10.0, 30.0, 100.0):
    print(f"      {n:8.1f} {d2_inv_omega('arctan', PARAMS, n):16.6e} "
          f"{d2_inv_omega('arctan', q_p1, n):19.6e} "
          f"{d2_inv_omega('rational', q_p1, n):15.6e}")
print("      over the (alpha, n0) grid, smallest |d^2(1/omega)/dn^2| for arctan at n = 2 n0:")
worst = min(abs(d2_inv_omega("arctan", at(a, nn), 2 * nn)) for a, nn in GRID)
print(f"        {worst:.6e}   (rational, any p = 1 setting: 0 identically)")

print("\n  B5. the asymptotic alpha-sensitivity, closed form vs measured")
print("      arctan:   s_A(n) -> 1                as n -> inf, for every (omega0, alpha, n0)")
print("      rational: s_R(n) -> p ln(n/n0)       as n -> inf, unbounded")
pp = forms.calibrate(PARAMS)["p"]
for n in (6.0, 10.0, 30.0, 100.0):
    h = 1e-6 * ALPHA
    meas = []
    for form in forms.FORMS:
        wp = float(forms.omega(np.array([n]), form, at(ALPHA + h))[0])
        wm = float(forms.omega(np.array([n]), form, at(ALPHA - h))[0])
        meas.append(-(math.log(wp) - math.log(wm)) / (2 * h) * ALPHA)
    print(f"      n = {n:6.1f}:  s_A = {meas[0]:8.5f} (limit 1)   s_R = {meas[1]:8.5f} "
          f"(p ln(n/n0) = {pp * math.log(n / N0):8.5f})")

print("\n  B3. best-fit collapse: the smallest max|omega_A - omega_R|/omega0 that ANY")
print("      (omega0', p, n0') can achieve against the pinned arctan curve,")
print("      on n in [0, 10] (the configuration's n_i lie in [0, 9.6]).")
ngrid = np.linspace(0.0, 10.0, 2001)
target = forms.omega(ngrid, "arctan", PARAMS)


def rat_raw(n, w0r, p, n0r):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(n > 0, (n / n0r) ** p, 0.0)
    return w0r / (1.0 + t)


def loss(v):
    w0r, p, n0r = math.exp(v[0]), math.exp(v[1]), math.exp(v[2])
    return float(np.abs(rat_raw(ngrid, w0r, p, n0r) - target).max()) / W0


best = None
for lw in np.linspace(math.log(0.3), math.log(3.0), 25):
    for lp in np.linspace(math.log(0.2), math.log(12.0), 30):
        for ln in np.linspace(math.log(0.5), math.log(12.0), 30):
            v = np.array([lw, lp, ln])
            f = loss(v)
            if best is None or f < best[0]:
                best = (f, v)
step = 0.25
cur = best
while step > 1e-6:                                   # coordinate descent
    improved = False
    for k in range(3):
        for s in (+1, -1):
            v = cur[1].copy()
            v[k] += s * step
            f = loss(v)
            if f < cur[0]:
                cur, improved = (f, v), True
    if not improved:
        step *= 0.5
print(f"      unconstrained 3-parameter best fit:  max residual / omega0 = {cur[0]:.4e}")
print(f"        at omega0' = {math.exp(cur[1][0]):.4f}, p = {math.exp(cur[1][1]):.4f}, "
      f"n0' = {math.exp(cur[1][2]):.4f}")
sep = float(np.abs(forms.omega(n_pin, "arctan", PARAMS)
                   - forms.omega(n_pin, "rational", PARAMS)).max()) / W0
print(f"      calibrated forms, on the configuration's own n_i: max diff / omega0 = {sep:.4e}")

# ===========================================================================
head("C.  calibration — what it does and does not remove")
# ===========================================================================
print("  C1. value and slope at n0, both forms, at three alpha settings")
for a in (0.75 * ALPHA, ALPHA, 1.25 * ALPHA):
    q = at(a)
    p = forms.calibrate(q)["p"]
    row = [f"alpha={a:.3f}  p={p:.4f}"]
    for form in forms.FORMS:
        h = 1e-6
        w = float(forms.omega(np.array([N0]), form, q)[0])
        d = (float(forms.omega(np.array([N0 + h]), form, q)[0])
             - float(forms.omega(np.array([N0 - h]), form, q)[0])) / (2 * h)
        row.append(f"{form}: omega(n0)/omega0 = {w / W0:.10f}  omega'(n0) = {d:+.8f}")
    print("      " + "   ".join(row))
print(f"      calibration target: omega(n0)/omega0 = 0.5, omega'(n0) = "
      f"{-W0 * ALPHA / math.pi:+.8f} at the pinned alpha")

print("\n  C2. what an UNcalibrated comparison would have shown: R with p free")
print("      (rational at fixed p, alpha varied only through the arctan form's")
print("       band -- i.e. the rational law not responding to alpha at all)")
for p_fixed in (1.0, 2.0, 3.055775, 5.0):
    q = dict(PARAMS)
    w_pin = W0 / (1.0 + np.where(n_pin > 0, (n_pin / N0) ** p_fixed, 0.0))
    line = [f"p fixed at {p_fixed:.4f}:"]
    for m in MAPS:
        w_m = W0 / (1.0 + np.where(n_map[m] > 0, (n_map[m] / N0) ** p_fixed, 0.0))
        ds = -float((np.log(w_m) - np.log(w_pin)).sum())
        line.append(f"{m}: dS = {ds:+9.3f} (identical at every alpha -> R = inf)")
    print("      " + "  ".join(line))

print("\n  C3. how far the common footing reaches: |omega_A - omega_R| / omega0")
print("      as a function of distance from n0")
for d in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 6.0):
    n = np.array([N0 - d, N0 + d]) if d <= N0 else np.array([0.0, N0 + d])
    wa = forms.omega(n, "arctan", PARAMS)
    wr = forms.omega(n, "rational", PARAMS)
    print(f"      n = {n[0]:5.2f} / {n[1]:5.2f}:  |dW|/omega0 = "
          f"{abs(wa[0] - wr[0]) / W0:.4e} / {abs(wa[1] - wr[1]) / W0:.4e}")

# ===========================================================================
head("D.  the criterion, measured")
# ===========================================================================
res = forms.compare(POS, SPECIES, DEMIX, PARAMS)
print("      dS (units of l k_B), per map, form and alpha setting:")
print(f"      {'map':>8s} {'form':>9s} {'alpha_lo':>10s} {'alpha':>10s} {'alpha_hi':>10s}"
      f" {'spread':>10s} {'R':>9s}")
for m in MAPS:
    for form in forms.FORMS:
        d = res["delta_s"][m][form]
        spread = abs(d["hi"] - d["lo"])
        print(f"      {m:>8s} {form:>9s} {d['lo']:10.4f} {d['mid']:10.4f} {d['hi']:10.4f}"
              f" {spread:10.4f} {res['R'][m][form]:9.5f}")
for m in MAPS:
    ra, rr = res["R"][m]["arctan"], res["R"][m]["rational"]
    print(f"      {m}: R_arctan = {ra:.5f}, R_rational = {rr:.5f}, "
          f"preferred = {'arctan' if ra > rr else 'rational'}, "
          f"margin = {abs(ra - rr) / min(ra, rr):.4f}")
print(f"      choice = {res['choice']}   (threshold {PARAMS['criterion']['margin']})")

print("\n  D2. where R comes from.  R = signal / sensitivity, so")
print("      R_arctan / R_rational = (signal ratio)^-1 x (sensitivity ratio),")
print("      both ratios taken rational-over-arctan.")
print(f"      {'map':>8s} {'signal rat/arc':>15s} {'spread rat/arc':>15s} "
      f"{'R arc/rat':>11s} {'margin':>8s}")
for m in MAPS:
    da, dr = res["delta_s"][m]["arctan"], res["delta_s"][m]["rational"]
    sig = abs(dr["mid"]) / abs(da["mid"])
    spr = abs(dr["hi"] - dr["lo"]) / abs(da["hi"] - da["lo"])
    ra, rr = res["R"][m]["arctan"], res["R"][m]["rational"]
    print(f"      {m:>8s} {sig:15.5f} {spr:15.5f} {ra / rr:11.5f} "
          f"{abs(ra - rr) / min(ra, rr):8.4f}")
print("      (signal ratio > 1 favours rational; sensitivity ratio > 1 favours arctan;")
print("       the second is roughly twice the first, and that is the whole margin.)")

print("\n  D3. the per-particle alpha-sensitivity of -ln omega, as a function of n")
print("      s(n) = -d ln omega / d ln alpha   at the pinned alpha")
print(f"      {'n':>7s} {'n/n0':>7s} {'arctan':>10s} {'rational':>10s}")
for n in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
    h = 1e-6 * ALPHA
    row = []
    for form in forms.FORMS:
        wp = float(forms.omega(np.array([n]), form, at(ALPHA + h))[0])
        wm = float(forms.omega(np.array([n]), form, at(ALPHA - h))[0])
        row.append(-(math.log(wp) - math.log(wm)) / (2 * h) * ALPHA)
    print(f"      {n:7.2f} {n / N0:7.3f} {row[0]:10.5f} {row[1]:10.5f}")

print("\n  D4. how much of each spread the far tail carries.")
print("      Restrict BOTH the signal and the sensitivity to particles with")
print("      |ln(n_i/n0)| <= L, keeping everything else fixed.")


def restricted(mask, form, m, alpha):
    q = at(alpha)
    w_m = forms.omega(n_map[m][mask], form, q)
    w_0 = forms.omega(n_pin[mask], form, q)
    return -float((np.log(w_m) - np.log(w_0)).sum())


def R_masked(mask, form, m):
    lo = restricted(mask, form, m, 0.75 * ALPHA)
    mid = restricted(mask, form, m, ALPHA)
    hi = restricted(mask, form, m, 1.25 * ALPHA)
    sp = abs(hi - lo)
    return abs(mid) / sp if sp > 0 else math.inf


print(f"      {'L':>6s} {'kept':>6s} " + " ".join(f"{m + ':' + f:>18s}"
                                                  for m in MAPS for f in forms.FORMS)
      + "   verdict")
for L in (0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 10.0):
    lp = np.abs(np.log(np.maximum(n_pin, 1e-12) / N0))
    lm = {m: np.abs(np.log(np.maximum(n_map[m], 1e-12) / N0)) for m in MAPS}
    mask = np.ones(POS.shape[0], dtype=bool)
    for m in MAPS:
        mask &= (lp <= L) & (lm[m] <= L)
    if mask.sum() < 5:
        continue
    Rm = {m: {f: R_masked(mask, f, m) for f in forms.FORMS} for m in MAPS}
    v, prefs, marg = forms._apply_rule(Rm, MAPS, float(PARAMS["criterion"]["margin"]))
    cells = " ".join(f"{Rm[m][f]:18.5f}" for m in MAPS for f in forms.FORMS)
    print(f"      {L:6.2f} {int(mask.sum()):6d} {cells}   {v} "
          f"(margins {', '.join('%.3f' % marg[m] for m in MAPS)})")

# ===========================================================================
head("E.  sensitivity analysis this session designed")
# ===========================================================================
print("  E1. the alpha band width.  The criterion pins +-25%; sweep it.")
print(f"      {'band':>8s} " + " ".join(f"{m + ':' + f:>16s}" for m in MAPS
                                        for f in forms.FORMS) + "   verdict")


def R_band(delta, form, m, alpha=ALPHA, n0=N0):
    vals = {}
    for key, a in (("lo", (1 - delta) * alpha), ("mid", alpha), ("hi", (1 + delta) * alpha)):
        q = at(a, n0)
        vals[key] = -float((np.log(forms.omega(n_map[m], form, q))
                            - np.log(forms.omega(n_pin, form, q))).sum())
    sp = abs(vals["hi"] - vals["lo"])
    return abs(vals["mid"]) / sp if sp > 0 else math.inf


for delta in (0.01, 0.02, 0.05, 0.10, 0.25, 0.40, 0.60, 0.80):
    Rb = {m: {f: R_band(delta, f, m) for f in forms.FORMS} for m in MAPS}
    v, prefs, marg = forms._apply_rule(Rb, MAPS, float(PARAMS["criterion"]["margin"]))
    cells = " ".join(f"{Rb[m][f]:16.5f}" for m in MAPS for f in forms.FORMS)
    print(f"      +-{delta:5.0%} {cells}   {v} (margins "
          f"{', '.join('%.3f' % marg[m] for m in MAPS)})")

print("\n      band-free limit: delta * R -> |dS| / (2 |d dS/d alpha| alpha), so the")
print("      ratio R_arctan/R_rational converges as delta -> 0:")
for delta in (0.60, 0.25, 0.05, 0.01, 0.002):
    rr = [R_band(delta, "arctan", m) / R_band(delta, "rational", m) for m in MAPS]
    print(f"      delta = {delta:6.3f}   " + "   ".join(f"{m}: {v:.5f}"
                                                        for m, v in zip(MAPS, rr)))

print("\n  E2. the pinned (alpha, n0) is not special: sweep both, verdict each time")
print(f"      {'alpha':>7s} {'n0':>6s} {'p':>7s} " + " ".join(f"{m + ':' + f:>13s}"
                                                              for m in MAPS
                                                              for f in forms.FORMS)
      + "   verdict")
for a in (0.2, 0.4, 0.8, 1.2, 2.0, 4.0):
    for nn in (1.5, 3.0, 5.0):
        Rg = {m: {f: R_band(0.25, f, m, alpha=a, n0=nn) for f in forms.FORMS} for m in MAPS}
        v, prefs, marg = forms._apply_rule(Rg, MAPS, float(PARAMS["criterion"]["margin"]))
        cells = " ".join(f"{Rg[m][f]:13.5f}" for m in MAPS for f in forms.FORMS)
        frac = np.mean(np.concatenate([n_map[m] for m in MAPS]) > nn)
        print(f"      {a:7.2f} {nn:6.2f} {4 * nn * a / math.pi:7.3f} {cells}   "
              f"{v:>9s}  frac(n^m > n0) = {frac:.3f}")

print("\n  E3. maps this task did not pin: three further position sorts and a")
print("      random relabelling, scored the same way")
extra = {}
order = np.argsort(POS[:, 1], kind="stable")
lab = np.zeros(POS.shape[0], dtype=np.int64)
lab[order[POS.shape[0] // 2:]] = 1
extra["plane_y"] = lab
order = np.argsort(POS[:, 2], kind="stable")
lab = np.zeros(POS.shape[0], dtype=np.int64)
lab[order[POS.shape[0] // 2:]] = 1
extra["plane_z"] = lab
order = np.argsort(POS @ np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0), kind="stable")
lab = np.zeros(POS.shape[0], dtype=np.int64)
lab[order[POS.shape[0] // 2:]] = 1
extra["plane_111"] = lab
rng = np.random.default_rng(11)
lab = np.zeros(POS.shape[0], dtype=np.int64)
lab[POS.shape[0] // 2:] = 1
rng.shuffle(lab)
extra["random"] = lab
extra["antiradial"] = 1 - DEMIX["radial"]
for name, lb in extra.items():
    nm = forms.coordination(POS, lb, PARAMS)
    rs = {}
    for form in forms.FORMS:
        vals = {}
        for key, a in (("lo", 0.75 * ALPHA), ("mid", ALPHA), ("hi", 1.25 * ALPHA)):
            q = at(a)
            vals[key] = -float((np.log(forms.omega(nm, form, q))
                                - np.log(forms.omega(n_pin, form, q))).sum())
        sp = abs(vals["hi"] - vals["lo"])
        rs[form] = abs(vals["mid"]) / sp if sp > 0 else math.inf
    ra, rr = rs["arctan"], rs["rational"]
    print(f"      {name:>11s}:  R_arctan = {ra:9.5f}  R_rational = {rr:9.5f}  "
          f"preferred {'arctan' if ra > rr else 'rational'}  margin {abs(ra - rr) / min(ra, rr):.4f}")

print("\n  E4. position jitter: 200 draws, Gaussian, three widths; verdict each time")
rng = np.random.default_rng(20260905)
for width in (0.02, 0.05, 0.10):
    counts, margins_lo = {}, []
    for _ in range(200):
        p2 = POS + rng.normal(0.0, width, size=POS.shape)
        npin2 = forms.coordination(p2, SPECIES, PARAMS)
        nmap2 = {m: forms.coordination(p2, DEMIX[m], PARAMS) for m in MAPS}
        Rj = {}
        for m in MAPS:
            Rj[m] = {}
            for form in forms.FORMS:
                vals = {}
                for key, a in (("lo", 0.75 * ALPHA), ("mid", ALPHA), ("hi", 1.25 * ALPHA)):
                    q = at(a)
                    vals[key] = -float((np.log(forms.omega(nmap2[m], form, q))
                                        - np.log(forms.omega(npin2, form, q))).sum())
                sp = abs(vals["hi"] - vals["lo"])
                Rj[m][form] = abs(vals["mid"]) / sp if sp > 0 else math.inf
        v, prefs, marg = forms._apply_rule(Rj, MAPS, float(PARAMS["criterion"]["margin"]))
        counts[v] = counts.get(v, 0) + 1
        margins_lo.append(min(marg[m] for m in MAPS))
    print(f"      width {width:.2f} sigma_AA:  verdicts {counts}   "
          f"min margin over draws = {min(margins_lo):.4f}")

print("\n  E5. the coordination window is not pinned by the criterion: move it")
print(f"      {'r_on':>6s} {'r_off':>6s} {'mean n':>8s} " +
      " ".join(f"{m + ':' + f:>13s}" for m in MAPS for f in forms.FORMS) + "   verdict")
for r_on, r_off in ((1.0, 1.4), (1.2, 1.6), (1.4, 1.8), (1.6, 2.2), (2.0, 2.6)):
    q = json.loads(json.dumps(PARAMS))
    q["coord_switch"] = {"r_on": r_on, "r_off": r_off}
    np2 = forms.coordination(POS, SPECIES, q)
    nm2 = {m: forms.coordination(POS, DEMIX[m], q) for m in MAPS}
    Rw = {}
    for m in MAPS:
        Rw[m] = {}
        for form in forms.FORMS:
            vals = {}
            for key, a in (("lo", 0.75 * ALPHA), ("mid", ALPHA), ("hi", 1.25 * ALPHA)):
                qq = dict(q)
                qq["alpha"] = a
                vals[key] = -float((np.log(forms.omega(nm2[m], form, qq))
                                    - np.log(forms.omega(np2, form, qq))).sum())
            sp = abs(vals["hi"] - vals["lo"])
            Rw[m][form] = abs(vals["mid"]) / sp if sp > 0 else math.inf
    v, prefs, marg = forms._apply_rule(Rw, MAPS, float(PARAMS["criterion"]["margin"]))
    cells = " ".join(f"{Rw[m][f]:13.5f}" for m in MAPS for f in forms.FORMS)
    print(f"      {r_on:6.2f} {r_off:6.2f} {np2.mean():8.4f} {cells}   {v}")

# ===========================================================================
head("F.  what would overturn it — variants of the statistic, run")
# ===========================================================================
print("  F1. perturb n0 instead of alpha: R' = |dS| / |dS(n0 x 1.25) - dS(n0 x 0.75)|")
print(f"      {'map':>8s} {'R_arctan':>11s} {'R_rational':>11s}  preferred   margin")
for m in MAPS:
    rs = {}
    for form in forms.FORMS:
        vals = {}
        for key, nn in (("lo", 0.75 * N0), ("mid", N0), ("hi", 1.25 * N0)):
            q = at(ALPHA, nn)
            vals[key] = -float((np.log(forms.omega(n_map[m], form, q))
                                - np.log(forms.omega(n_pin, form, q))).sum())
        sp = abs(vals["hi"] - vals["lo"])
        rs[form] = abs(vals["mid"]) / sp if sp > 0 else math.inf
    ra, rr = rs["arctan"], rs["rational"]
    print(f"      {m:>8s} {ra:11.5f} {rr:11.5f}  {'arctan' if ra > rr else 'rational':>9s}"
          f"   {abs(ra - rr) / min(ra, rr):.4f}")

print("\n  F2. perturb omega0 instead: R'' = |dS| / |dS(omega0 x 1.25) - dS(omega0 x 0.75)|")
for m in MAPS:
    rs = {}
    for form in forms.FORMS:
        vals = {}
        for key, w in (("lo", 0.75 * W0), ("mid", W0), ("hi", 1.25 * W0)):
            q = dict(PARAMS)
            q["omega0"] = w
            vals[key] = -float((np.log(forms.omega(n_map[m], form, q))
                                - np.log(forms.omega(n_pin, form, q))).sum())
        sp = abs(vals["hi"] - vals["lo"])
        rs[form] = abs(vals["mid"]) / sp if sp > 0 else math.inf
    print(f"      {m:>8s} arctan {rs['arctan']:.6g}   rational {rs['rational']:.6g}")

print("\n  F3. a configuration whose n_i sit close to n0: synthetic n drawn from")
print("      N(n0, sigma) for a ladder of sigma, demixed by shifting the mean")
rng = np.random.default_rng(7)
print(f"      {'sigma_n':>8s} {'shift':>6s} {'R_arctan':>11s} {'R_rational':>11s}  preferred  margin")
for sigma_n in (0.15, 0.3, 0.6, 1.0, 1.6):
    base = np.abs(rng.normal(N0, sigma_n, size=240))
    shifted = np.abs(base + 0.5 * sigma_n)
    rs = {}
    for form in forms.FORMS:
        vals = {}
        for key, a in (("lo", 0.75 * ALPHA), ("mid", ALPHA), ("hi", 1.25 * ALPHA)):
            q = at(a)
            vals[key] = -float((np.log(forms.omega(shifted, form, q))
                                - np.log(forms.omega(base, form, q))).sum())
        sp = abs(vals["hi"] - vals["lo"])
        rs[form] = abs(vals["mid"]) / sp if sp > 0 else math.inf
    ra, rr = rs["arctan"], rs["rational"]
    print(f"      {sigma_n:8.2f} {0.5 * sigma_n:6.2f} {ra:11.5f} {rr:11.5f}  "
          f"{'arctan' if ra > rr else 'rational':>8s}  {abs(ra - rr) / min(ra, rr):.4f}")


head("G.  which particles carry the signal and which carry the sensitivity")
print("      cumulative share of |dS| and of the spread, by |ln(n_i/n0)| decile")
for m in MAPS:
    key = np.maximum(np.abs(np.log(np.maximum(n_map[m], 1e-300) / N0)),
                     np.abs(np.log(np.maximum(n_pin, 1e-300) / N0)))
    order = np.argsort(key)
    print(f"      --- {m} ---")
    print(f"      {'top q by |ln(n/n0)|':>21s} {'arc |dS|':>10s} {'arc spread':>11s} "
          f"{'rat |dS|':>10s} {'rat spread':>11s}")
    per = {}
    for form in forms.FORMS:
        cols = {}
        for k, a in (("lo", 0.75 * ALPHA), ("mid", ALPHA), ("hi", 1.25 * ALPHA)):
            q = at(a)
            cols[k] = -(np.log(forms.omega(n_map[m], form, q))
                        - np.log(forms.omega(n_pin, form, q)))
        per[form] = (cols["mid"], cols["hi"] - cols["lo"])
    for frac in (0.1, 0.25, 0.5, 1.0):
        take = order[int(round((1 - frac) * len(order))):]
        cells = []
        for form in forms.FORMS:
            sig, spr = per[form]
            cells += [abs(sig[take].sum()) / abs(sig.sum()),
                      abs(spr[take].sum()) / abs(spr.sum())]
        print(f"      {'top %2d%%' % round(frac * 100):>21s} "
              f"{cells[0]:10.4f} {cells[1]:11.4f} {cells[2]:10.4f} {cells[3]:11.4f}")


# ===========================================================================
head("H.  does the mechanism explain the overturns, or only survive them?")
# ===========================================================================
def s_alpha(n, form, a=ALPHA, n0=N0):
    """-d ln omega / d ln alpha, with p recalibrated."""
    h = 1e-6 * a
    return -(np.log(forms.omega(n, form, at(a + h, n0)))
             - np.log(forms.omega(n, form, at(a - h, n0)))) / (2 * h) * a


def s_n0(n, form, a=ALPHA, n0=N0):
    """-d ln omega / d ln n0, with p recalibrated."""
    h = 1e-6 * n0
    return -(np.log(forms.omega(n, form, at(a, n0 + h)))
             - np.log(forms.omega(n, form, at(a, n0 - h)))) / (2 * h) * n0


print("  H1. the pointwise claim '|s_R| > |s_A| iff n > n0' is FALSE: there are")
print("      three crossings, and the operative one is near n0 exp(1/p).")
print(f"      {'alpha':>6s} {'n0':>5s} {'p':>7s} {'n0 exp(1/p)':>12s} {'measured last crossing':>23s}"
      f" {'claim holds on':>15s}")
for a, nn in ((0.05, 3.0), (0.1, 3.0), (0.2, 3.0), (0.4, 3.0), (0.8, 3.0),
              (1.5, 3.0), (0.8, 5.0), (0.8, 7.0), (0.1, 6.0)):
    p = 4 * nn * a / math.pi
    g = np.linspace(1e-6, 40 * nn, 20001)
    big = np.abs(s_alpha(g, "rational", a, nn)) > np.abs(s_alpha(g, "arctan", a, nn))
    idx = np.where(np.diff(big.astype(int)) != 0)[0]
    last = g[idx[-1]] if len(idx) else float("nan")
    print(f"      {a:6.2f} {nn:5.2f} {p:7.3f} {nn * math.exp(1 / p):12.3f} {last:23.3f}"
          f" {np.mean(big == (g > nn)):15.2%}")

print("\n  H2. near n0 the contrast is SECOND order: s_R - s_A = -alpha eps^2/(pi n0)")
print("      + O(eps^3).  This is why restricting to a neighbourhood of n0 must")
print("      drive both ratios to 1, and it is derived, not fitted.")
print(f"      {'eps':>8s} {'measured s_R - s_A':>20s} {'predicted':>13s} {'ratio':>8s}")
for e in (0.4, 0.2, 0.1, 0.05, 0.025):
    nn = np.array([N0 + e])
    d = float(s_alpha(nn, "rational")[0] - s_alpha(nn, "arctan")[0])
    pr = -ALPHA * e * e / (math.pi * N0)
    print(f"      {e:8.3f} {d:20.6e} {pr:13.4e} {d / pr:8.4f}")

print("\n  H3. narrow-band decomposition, applied to every overturn.")
print("      R_A/R_R = (sensitivity ratio) / (signal ratio), both rational-over-arctan.")


def decompose(n0v, nmv, a=ALPHA, n0=N0, sens=s_alpha):
    out = {}
    for form in forms.FORMS:
        w0 = np.log(forms.omega(n0v, form, at(a, n0)))
        wm = np.log(forms.omega(nmv, form, at(a, n0)))
        out[form] = (float((w0 - wm).sum()),
                     float((sens(nmv, form, a, n0) - sens(n0v, form, a, n0)).sum()))
    sig = abs(out["rational"][0]) / abs(out["arctan"][0])
    sen = abs(out["rational"][1]) / abs(out["arctan"][1])
    return sig, sen, sen / sig


def drow(label, n0v, nmv, a=ALPHA, n0=N0, sens=s_alpha):
    sig, sen, rr = decompose(n0v, nmv, a, n0, sens)
    print(f"      {label:<40s} signal {sig:7.4f}  sens {sen:7.4f}  R_A/R_R {rr:8.4f}"
          f" -> {'arctan' if rr > 1 else 'rational'}")


print("      (a) pinned, alpha-band                             [observed: arctan]")
for m in MAPS:
    drow(f"  {m}", n_pin, n_map[m])
print("      (b) n0 = 7.0, alpha-band                           [observed: rational]")
for m in MAPS:
    drow(f"  {m}", n_pin, n_map[m], n0=7.0)
print("      (b') alpha = 0.1, n0 = 6.0                         [observed: rational]")
for m in MAPS:
    drow(f"  {m}", n_pin, n_map[m], a=0.1, n0=6.0)
print("      (c) restricted to |ln(n_i/n0)| <= L                [observed: undecided, L <= 0.35]")
for L in (0.25, 0.35, 0.5, 1.0):
    lp = np.abs(np.log(np.maximum(n_pin, 1e-12) / N0))
    mask = np.ones(POS.shape[0], dtype=bool)
    for m in MAPS:
        mask &= (lp <= L) & (np.abs(np.log(np.maximum(n_map[m], 1e-12) / N0)) <= L)
    for m in MAPS:
        drow(f"  L={L:.2f} ({int(mask.sum()):3d} kept)  {m}", n_pin[mask], n_map[m][mask])
print("      (d) n0-band instead of alpha-band                  [observed: maps split]")
for m in MAPS:
    drow(f"  {m}", n_pin, n_map[m], sens=s_n0)

print("\n  H4. why (d) is NOT explained by the mechanism.  The n0-sensitivity t(n)")
print("      has no saturate-vs-diverge structure: t_R changes sign, and |t_R| > |t_A|")
print("      flips repeatedly.  Compare with s(n), which does not.")
print(f"      {'n':>7s} {'t_A':>9s} {'t_R':>9s} {'|t_R|>|t_A|':>12s} {'s_A':>9s} {'s_R':>9s}"
      f" {'|s_R|>|s_A|':>12s}")
for nn in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 30.0):
    v = np.array([nn])
    ta, tr = float(s_n0(v, "arctan")[0]), float(s_n0(v, "rational")[0])
    sa, sr = float(s_alpha(v, "arctan")[0]), float(s_alpha(v, "rational")[0])
    print(f"      {nn:7.2f} {ta:9.4f} {tr:9.4f} {str(abs(tr) > abs(ta)):>12s}"
          f" {sa:9.4f} {sr:9.4f} {str(abs(sr) > abs(sa)):>12s}")

print("\n  H5. and (d)'s numbers are a near-cancellation, not a contrast:")
print("      how much of sum_i |delta| survives in |sum_i delta|")
for m in MAPS:
    for sens, tag in ((s_n0, "n0 "), (s_alpha, "alp")):
        for form in forms.FORMS:
            d = sens(n_map[m], form) - sens(n_pin, form)
            print(f"      {m:>8s} {tag} {form:>9s}: sum = {d.sum():+10.4f}  "
                  f"sum|.| = {np.abs(d).sum():9.4f}  cancellation = "
                  f"{1 - abs(d.sum()) / np.abs(d).sum():6.1%}")

# ===========================================================================
head("I.  are the two demixing maps independent?")
# ===========================================================================
l1, l2 = DEMIX["plane_x"], DEMIX["radial"]
print(f"      label agreement, up to the global A/B swap (n is swap-invariant): "
      f"{max(np.mean(l1 == l2), np.mean(l1 != l2)):.4f}   (chance = 0.5)")
print(f"      corr(n^plane_x, n^radial) over the 240 particles              : "
      f"{np.corrcoef(n_map['plane_x'], n_map['radial'])[0, 1]:.4f}")
for form in forms.FORMS:
    c = [np.log(forms.omega(n_pin, form, PARAMS)) - np.log(forms.omega(n_map[m], form, PARAMS))
         for m in MAPS]
    d = [s_alpha(n_map[m], form) - s_alpha(n_pin, form) for m in MAPS]
    print(f"      {form:>9s}: corr(per-particle signal) {np.corrcoef(*c)[0, 1]:.4f}   "
          f"corr(per-particle sensitivity) {np.corrcoef(*d)[0, 1]:.4f}")


def score(lab, a=ALPHA, n0=N0):
    nm = forms.coordination(POS, lab, PARAMS)
    out = {}
    for form in forms.FORMS:
        v = {}
        for k, aa in (("lo", 0.75 * a), ("mid", a), ("hi", 1.25 * a)):
            q = at(aa, n0)
            v[k] = -float((np.log(forms.omega(nm, form, q))
                           - np.log(forms.omega(n_pin, form, q))).sum())
        sp = abs(v["hi"] - v["lo"])
        out[form] = abs(v["mid"]) / sp if sp > 0 else math.inf
    return out, nm.mean()


rng = np.random.default_rng(3)
ra, rr, mn = [], [], []
for _ in range(200):
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    o = np.argsort(POS @ u, kind="stable")
    lab = np.zeros(POS.shape[0], dtype=np.int64)
    lab[o[POS.shape[0] // 2:]] = 1
    r, mm = score(lab)
    ra.append(r["arctan"]); rr.append(r["rational"]); mn.append(mm)
ra, rr, mn = np.array(ra), np.array(rr), np.array(mn)
print(f"\n      200 random-direction half-space cuts (the family plane_x belongs to):")
print(f"        R_arctan   {ra.mean():.4f} +- {ra.std():.4f}  (relative spread {ra.std() / ra.mean():.3%})")
print(f"        R_rational {rr.mean():.4f} +- {rr.std():.4f}  (relative spread {rr.std() / rr.mean():.3%})")
print(f"        arctan preferred in {int((ra > rr).sum())}/200;  mean n^m = {mn.mean():.4f} +- {mn.std():.4f}")
for m in MAPS:
    r, _ = score(DEMIX[m])
    print(f"        {m:>8s}: R_arctan {r['arctan']:.5f} ({(r['arctan'] - ra.mean()) / ra.std():+.2f} sigma)"
          f"   R_rational {r['rational']:.5f} ({(r['rational'] - rr.mean()) / rr.std():+.2f} sigma)")

ra2, rr2, mn2 = [], [], []
for _ in range(400):
    lab = np.zeros(POS.shape[0], dtype=np.int64)
    lab[POS.shape[0] // 2:] = 1
    rng.shuffle(lab)
    r, mm = score(lab)
    ra2.append(r["arctan"]); rr2.append(r["rational"]); mn2.append(mm)
ra2, rr2, mn2 = np.array(ra2), np.array(rr2), np.array(mn2)
flip = rr2 > ra2
marg = np.abs(ra2 - rr2) / np.minimum(ra2, rr2)
print(f"\n      400 NON-spatial random relabellings (a different family):")
print(f"        R_arctan {ra2.mean():.3f} +- {ra2.std():.3f}   "
      f"R_rational {rr2.mean():.3f} +- {rr2.std():.3f}")
print(f"        rational preferred in {int(flip.sum())}/400, of which "
      f"{int((marg[flip] >= 0.10).sum())} clear the 0.10 margin")

print(f"\n      can any relabelling of these positions demix DOWNWARD?")
lab = SPECIES.copy()
nb = forms.coordination(POS, lab, PARAMS).mean()
for _ in range(60):
    improved = False
    for i in rng.permutation(POS.shape[0]):
        lab[i] = 1 - lab[i]
        v = forms.coordination(POS, lab, PARAMS).mean()
        if v < nb - 1e-12:
            nb, improved = v, True
        else:
            lab[i] = 1 - lab[i]
    if not improved:
        break
print(f"        pinned labelling mean n      = {n_pin.mean():.4f}")
print(f"        hill-climbed minimum mean n  = {nb:.4f}  ({int(lab.sum())}/{POS.shape[0]} species-1)")
print(f"        400 random relabellings      : [{mn2.min():.4f}, {mn2.max():.4f}]")
print(f"        200 half-space cuts          : [{mn.min():.4f}, {mn.max():.4f}]")


# ===========================================================================
head("J.  the decisive experiment: make the reachable range straddle the crossover")
# ===========================================================================
print("  No relabelling of the PINNED positions can demix downward (§I), so the")
print("  criterion cannot be tested on the other side of the crossover by changing")
print("  the maps.  It can be tested by changing the configuration.  Below: the")
print("  same sampling procedure as config/make_config.py -- same N, same MIN_SEP,")
print("  same map construction, same params.json, same criterion -- with the")
print("  cluster RADIUS swept, so the demixed coordination runs from well above")
print("  the crossover to well below it.  Nothing in config/ is read or written.")
print("  (Rejection sampling at the dense end is slow; this section dominates the")
print("   runtime of analysis.py.)")

_N_J, _MIN_SEP_J = 240, 0.95

g_j = np.linspace(1e-6, 40 * N0, 40001)
_big = np.abs(s_alpha(g_j, "rational")) > np.abs(s_alpha(g_j, "arctan"))
XOVER = float(g_j[np.where(np.diff(_big.astype(int)) != 0)[0][-1]])
print(f"\n      crossover at the pinned (alpha={ALPHA}, n0={N0}, "
      f"p={forms.calibrate(PARAMS)['p']:.4f}):  n = {XOVER:.4f}")


def _sample_cluster(radius, seed):
    """config/make_config.py's procedure, with the radius as a knob."""
    rng = np.random.default_rng(seed)
    pos = np.empty((_N_J, 3))
    k = 0
    tries = 0
    while k < _N_J:
        tries += 1
        if tries > 8_000_000:
            raise RuntimeError(f"placement failed at radius {radius}")
        c = rng.uniform(-radius, radius, 3)
        if c @ c > radius ** 2:
            continue
        if k and np.linalg.norm(pos[:k] - c, axis=1).min() < _MIN_SEP_J:
            continue
        pos[k] = c
        k += 1
    pos -= pos.mean(0)
    sp = np.zeros(_N_J, dtype=np.int64)
    sp[_N_J // 2:] = 1
    rng.shuffle(sp)

    def split(order):
        l = np.zeros(_N_J, dtype=np.int64)
        l[order[_N_J // 2:]] = 1
        return l

    return pos, sp, {"plane_x": split(np.argsort(pos[:, 0], kind="stable")),
                     "radial": split(np.argsort(np.linalg.norm(pos, axis=1), kind="stable"))}


def _score_cluster(pos, sp, maps):
    npin = forms.coordination(pos, sp, PARAMS)
    R, means = {}, {}
    for m in sorted(maps):
        nm = forms.coordination(pos, maps[m], PARAMS)
        means[m] = nm
        R[m] = {}
        for form in forms.FORMS:
            v = {}
            for k, a in (("lo", 0.75 * ALPHA), ("mid", ALPHA), ("hi", 1.25 * ALPHA)):
                q = at(a)
                v[k] = -float((np.log(forms.omega(nm, form, q))
                               - np.log(forms.omega(npin, form, q))).sum())
            s = abs(v["hi"] - v["lo"])
            R[m][form] = abs(v["mid"]) / s if s > 0 else math.inf
    return npin, means, R


def _monotone(pos, sp, maps):
    """dS(alpha) strictly monotone across [0.5 alpha, 1.5 alpha] for every map and form?"""
    npin = forms.coordination(pos, sp, PARAMS)
    grid = np.linspace(0.5 * ALPHA, 1.5 * ALPHA, 31)
    for m in sorted(maps):
        nm = forms.coordination(pos, maps[m], PARAMS)
        for form in forms.FORMS:
            ds = np.array([-float((np.log(forms.omega(nm, form, at(a)))
                                   - np.log(forms.omega(npin, form, at(a)))).sum())
                           for a in grid])
            d = np.diff(ds)
            if not (np.all(d > 0) or np.all(d < 0)):
                return False
    return True


print(f"\n      {'radius':>7s} {'mean n^0':>9s} {'mean n^m':>9s} {'frac n^m > xo':>14s} "
      f"{'R_A(px)':>9s} {'R_R(px)':>9s} {'R_A(rad)':>9s} {'R_R(rad)':>9s} {'guard':>6s}"
      f"   verdict (margins)")
for radius in (3.9, 4.1, 4.4, 4.8, 5.2, 5.6, 6.0, 6.5, 7.2):
    pos, sp, maps = _sample_cluster(radius, 20260905)
    npin, means, R = _score_cluster(pos, sp, maps)
    v, prefs, marg = forms._apply_rule(R, sorted(maps), float(PARAMS["criterion"]["margin"]))
    allm = np.concatenate([means[m] for m in sorted(maps)])
    ok = _monotone(pos, sp, maps)
    print(f"      {radius:7.2f} {npin.mean():9.4f} {allm.mean():9.4f} "
          f"{np.mean(allm > XOVER):14.3f} "
          f"{R['plane_x']['arctan']:9.4f} {R['plane_x']['rational']:9.4f} "
          f"{R['radial']['arctan']:9.4f} {R['radial']['rational']:9.4f} "
          f"{'pass' if ok else 'FLAG':>6s}   {v} "
          f"({marg['plane_x']:.3f}, {marg['radial']:.3f})")

print("\n      three seeds per radius through the flip, so it is not one draw:")
for radius in (4.8, 5.2, 5.6, 6.0):
    cells = []
    for seed in (20260905, 11, 4242):
        pos, sp, maps = _sample_cluster(radius, seed)
        _, _, R = _score_cluster(pos, sp, maps)
        v, prefs, marg = forms._apply_rule(R, sorted(maps), float(PARAMS["criterion"]["margin"]))
        cells.append(f"{v}({min(marg.values()):.2f})")
    print(f"        radius {radius:.1f}:  " + "   ".join(cells))
