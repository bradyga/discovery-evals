#!/usr/bin/env python3
"""w1-02 verifier — two functional forms, non-equivalence, and a stated criterion.

This verifier imports the implementation named in the interface contract of
spec.md and runs every check itself.  It contains no reference energy, no
reference force, no reference entropy, no reference R, and **no record of which
form the criterion prefers**: it applies the rule stated in spec.md to the
numbers the implementation returns, and to its own recomputation of them from
the implementation's own omega() and coordination().

Every criterion is an identity, an invariance, a limit, a structural
requirement, or the stated rule recomputed.

Usage (from the task directory):

    python verify.py

Options are for the build-time dry-run only; the graded session needs none:

    --impl PATH        implementation module (default: ./forms.py)
    --config-dir PATH  configuration directory (default: ./config)
    --report PATH      report (default: report.md beside --impl)

Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Tolerances.  All are dimensionless ratios: against omega0, against the
# configuration's own force scale F_scale = mean_k |F_k|, or against the
# compared quantity's own magnitude.  None is an absolute number in reduced
# units.
# --------------------------------------------------------------------------
TOL_EXACT = 1.0e-12         # identities that hold exactly for one form
MIN_VIOLATION = 1.0e-3      # by how much the OTHER form must violate them
TOL_SLOPE = 1.0e-7          # calibrated slope, relative
TOL_SUM_ZERO = 1.0e-10      # |sum_k F_k| / sum_k |F_k| for a conserving total
TOL_TERM_CANCEL = 1.0e-8    # cancellation of the two coordination half-terms
MIN_TERM_SUM = 1.0e-3       # |sum_k F^t_k| / sum_k |F^t_k| a half-term must EXCEED
MIN_TERM_MAG = 1.0e-6       # sum_k |F^t_k| / sum_k |F_k| a half-term must EXCEED
TOL_RIGID = 1.0e-9          # rigid-motion invariance, relative
TOL_FD = 1.0e-7             # best finite-difference residual, relative to F_scale
MIN_FD_ORDER = 1.5          # observed convergence order over the coarse steps
TOL_TERMS_SUM = 1.0e-12     # sum of named terms vs. the returned total force
TOL_LIMIT = 1.0e-12         # limiting-case identities, relative
TOL_RECOMPUTE = 1.0e-9      # returned criterion values vs. recomputation
MIN_RESPONSE = 1.0e-6       # by how much a quantity must move under perturbation
MIN_SWITCH_ORDER_D1 = 1.5   # observed order of |n'| at the switching-window edge
MIN_SWITCH_ORDER_D2 = 0.7   # observed order of |n''| at the switching-window edge
TOL_SWITCH_PLATEAU = 1.0e-12

FD_STEPS = (1.0e-2, 10.0**-2.5, 1.0e-3, 10.0**-3.5, 1.0e-4, 10.0**-4.5)
FD_SAMPLES = 12
RNG_SEED = 20260905         # pinned: the verifier's own draws are reproducible

FORMS = ("arctan", "rational")
TERM_KEYS = ("pair", "coord_self", "coord_cross")
COORD_TERMS = ("coord_self", "coord_cross")
DS_KEYS = ("lo", "mid", "hi")

REPORT_SECTIONS = (
    "## Implementation",
    "## Non-equivalence",
    "## Calibration",
    "## The criterion, measured",
    "## The choice, defended",
    "## What would overturn it",
    "## Underdetermined in the specification",
)
CHOICE_RE = re.compile(r"^CHOICE:\s*(arctan|rational|undecided)\s*$")

# Grid over (alpha, n0) on which neither form may satisfy the other's identity.
REPARAM_GRID = ((0.4, 2.0), (0.4, 4.5), (0.8, 3.0), (1.5, 2.0), (1.5, 4.5))
# Reflection offsets, as fractions of n0, so that n0 - d stays non-negative.
REFLECT_FRACS = (0.1, 0.25, 0.4, 0.6, 0.8)
# Inversion half-widths in log n.
INVERT_TS = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


class ContractError(Exception):
    """The implementation does not meet the interface contract in spec.md."""


# --------------------------------------------------------------------------
# result plumbing
# --------------------------------------------------------------------------
class Check:
    def __init__(self, number: str, title: str):
        self.number = number
        self.title = title
        self.lines: list[str] = []
        self.failures: list[str] = []

    def record(self, label: str, value, ok: bool | None = None, fmt: str = "{: .6e}"):
        shown = fmt.format(value) if isinstance(value, float) else str(value)
        mark = "" if ok is None else ("  ok" if ok else "  FAIL")
        self.lines.append(f"      {label:<52s} {shown}{mark}")

    def require(self, ok: bool, message: str):
        if not ok:
            self.failures.append(message)

    @property
    def passed(self) -> bool:
        return not self.failures

    def emit(self):
        status = "PASS" if self.passed else "FAIL"
        print(f"  [{status}] check {self.number} — {self.title}")
        for line in self.lines:
            print(line)
        for f in self.failures:
            print(f"      -> {f}")
        print()


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_impl(path: Path):
    """Import the implementation with the working directory set elsewhere.

    The contract forbids reading files and depending on the current directory;
    importing from a scratch directory is what makes that requirement bite
    rather than merely being stated.
    """
    if not path.is_file():
        raise ContractError(
            f"no implementation at {path}.\n"
            "  spec.md, section 'Interface contract', requires a module named "
            "forms.py in the task directory."
        )
    spec = importlib.util.spec_from_file_location("w1_02_impl", path)
    if spec is None or spec.loader is None:
        raise ContractError(f"{path} is not importable as a Python module")
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        try:
            os.chdir(scratch)
            spec.loader.exec_module(module)
        except Exception:
            raise ContractError(
                f"importing {path} (with the working directory elsewhere) raised:\n"
                f"{traceback.format_exc()}"
            ) from None
        finally:
            os.chdir(cwd)

    forms = getattr(module, "FORMS", None)
    if forms is None:
        raise ContractError(f"{path} does not define FORMS. See spec.md, 'Interface contract'.")
    if tuple(forms) != FORMS:
        raise ContractError(f"{path}: FORMS is {tuple(forms)!r}, expected {FORMS!r}")

    for name in ("calibrate", "omega", "coordination", "energy_and_forces",
                 "force_terms", "compare"):
        if not hasattr(module, name):
            raise ContractError(
                f"{path} does not define {name}(). See spec.md, 'Interface contract'."
            )
        if not callable(getattr(module, name)):
            raise ContractError(f"{path}: {name} is not callable")
    return module


def load_config(config_dir: Path):
    for name in ("config.json", "params.json"):
        if not (config_dir / name).is_file():
            raise ContractError(f"missing {config_dir / name}")
    cfg = json.loads((config_dir / "config.json").read_text())
    params = json.loads((config_dir / "params.json").read_text())
    positions = np.asarray(cfg["positions"], dtype=np.float64)
    species = np.asarray(cfg["species"], dtype=np.int64)
    xi = np.asarray(cfg["xi"], dtype=np.float64)
    demix = {k: np.asarray(v, dtype=np.int64) for k, v in cfg["demix_maps"].items()}
    return cfg, params, positions, species, xi, demix


def clone(params: dict) -> dict:
    return json.loads(json.dumps(params))


# --------------------------------------------------------------------------
# contract-checked calls
# --------------------------------------------------------------------------
def call_ef(module, positions, species, xi, params, form, where: str = ""):
    try:
        out = module.energy_and_forces(positions, species, xi, params, form)
    except Exception:
        raise ContractError(
            f"energy_and_forces(..., form={form!r}) raised{' ' + where if where else ''}:\n"
            f"{traceback.format_exc()}"
        ) from None
    if not (isinstance(out, tuple) and len(out) == 2):
        raise ContractError("energy_and_forces() must return a 2-tuple (energy, forces)")
    energy, forces = out
    try:
        energy = float(energy)
    except (TypeError, ValueError):
        raise ContractError("energy_and_forces(): energy is not a real scalar") from None
    if not math.isfinite(energy):
        raise ContractError(f"energy_and_forces(): energy is not finite ({energy})")
    forces = np.asarray(forces, dtype=np.float64)
    if forces.shape != positions.shape:
        raise ContractError(
            f"energy_and_forces(): forces have shape {forces.shape}, expected {positions.shape}"
        )
    if not np.all(np.isfinite(forces)):
        raise ContractError("energy_and_forces(): forces contain non-finite entries")
    return energy, forces


def call_terms(module, positions, species, xi, params, form, where: str = ""):
    try:
        terms = module.force_terms(positions, species, xi, params, form)
    except Exception:
        raise ContractError(
            f"force_terms(..., form={form!r}) raised{' ' + where if where else ''}:\n"
            f"{traceback.format_exc()}"
        ) from None
    if not isinstance(terms, dict):
        raise ContractError("force_terms() must return a dict")
    if set(terms) != set(TERM_KEYS):
        raise ContractError(
            f"force_terms() returned keys {sorted(terms)}, expected {sorted(TERM_KEYS)}"
        )
    out = {}
    for k in TERM_KEYS:
        arr = np.asarray(terms[k], dtype=np.float64)
        if arr.shape != positions.shape:
            raise ContractError(
                f"force_terms()['{k}'] has shape {arr.shape}, expected {positions.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ContractError(f"force_terms()['{k}'] contains non-finite entries")
        out[k] = arr
    return out


def call_omega(module, n, form, params, where: str = ""):
    n = np.asarray(n, dtype=np.float64)
    try:
        out = module.omega(n, form, params)
    except Exception:
        raise ContractError(
            f"omega(..., form={form!r}) raised{' ' + where if where else ''}:\n"
            f"{traceback.format_exc()}"
        ) from None
    out = np.asarray(out, dtype=np.float64)
    if out.shape != n.shape:
        raise ContractError(f"omega(): returned shape {out.shape}, expected {n.shape}")
    if not np.all(np.isfinite(out)):
        raise ContractError(f"omega(..., form={form!r}): returned non-finite values{' ' + where}")
    return out


def call_coord(module, positions, species, params, where: str = ""):
    try:
        out = module.coordination(positions, species, params)
    except Exception:
        raise ContractError(
            f"coordination() raised{' ' + where if where else ''}:\n{traceback.format_exc()}"
        ) from None
    out = np.asarray(out, dtype=np.float64)
    if out.shape != (positions.shape[0],):
        raise ContractError(
            f"coordination(): returned shape {out.shape}, expected {(positions.shape[0],)}"
        )
    if not np.all(np.isfinite(out)):
        raise ContractError("coordination(): returned non-finite values")
    return out


def call_calibrate(module, params, where: str = ""):
    try:
        out = module.calibrate(params)
    except Exception:
        raise ContractError(
            f"calibrate() raised{' ' + where if where else ''}:\n{traceback.format_exc()}"
        ) from None
    if not isinstance(out, dict) or "p" not in out:
        raise ContractError("calibrate() must return a dict containing the key 'p'")
    try:
        p = float(out["p"])
    except (TypeError, ValueError):
        raise ContractError("calibrate()['p'] is not a real scalar") from None
    if not math.isfinite(p):
        raise ContractError(f"calibrate()['p'] is not finite ({p})")
    return p


def call_compare(module, positions, species, demix, params, where: str = ""):
    try:
        out = module.compare(positions, species, demix, params)
    except Exception:
        raise ContractError(
            f"compare() raised{' ' + where if where else ''}:\n{traceback.format_exc()}"
        ) from None
    if not isinstance(out, dict):
        raise ContractError("compare() must return a dict")
    for key in ("delta_s", "R", "choice"):
        if key not in out:
            raise ContractError(f"compare() returned no {key!r} key")
    if out["choice"] not in ("arctan", "rational", "undecided"):
        raise ContractError(
            f"compare()['choice'] is {out['choice']!r}; expected one of "
            "'arctan', 'rational', 'undecided'"
        )
    for key in ("delta_s", "R"):
        block = out[key]
        if not isinstance(block, dict) or set(block) != set(demix):
            raise ContractError(
                f"compare()['{key}'] keys are {sorted(block) if isinstance(block, dict) else block!r}, "
                f"expected {sorted(demix)}"
            )
        for m in demix:
            inner = block[m]
            if not isinstance(inner, dict) or set(inner) != set(FORMS):
                raise ContractError(
                    f"compare()['{key}']['{m}'] keys are "
                    f"{sorted(inner) if isinstance(inner, dict) else inner!r}, "
                    f"expected {sorted(FORMS)}"
                )
    for m in demix:
        for f in FORMS:
            inner = out["delta_s"][m][f]
            if not isinstance(inner, dict) or set(inner) != set(DS_KEYS):
                raise ContractError(
                    f"compare()['delta_s']['{m}']['{f}'] keys are "
                    f"{sorted(inner) if isinstance(inner, dict) else inner!r}, "
                    f"expected {sorted(DS_KEYS)}"
                )
            for k in DS_KEYS:
                v = float(inner[k])
                if not math.isfinite(v):
                    raise ContractError(f"compare()['delta_s']['{m}']['{f}']['{k}'] is not finite")
            v = float(out["R"][m][f])
            if not math.isfinite(v):
                raise ContractError(
                    f"compare()['R']['{m}']['{f}'] is not finite; if dS does not respond "
                    "to alpha the criterion is undefined, and that is a failure of the "
                    "implementation, not a verdict"
                )
    return out


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def net(f):
    return float(np.linalg.norm(f.sum(axis=0)))


def total(f):
    return float(np.linalg.norm(f, axis=1).sum())


def rel_net(f, floor: float = 0.0):
    """|sum_k f_k| / sum_k |f_k|, with a guarded denominator.

    An identically zero term reports 0 rather than an undefined ratio: a term
    with no force at all has no net force either, and must fail a "this term
    must not conserve momentum on its own" requirement rather than pass it
    vacuously.
    """
    t = max(total(f), floor)
    return net(f) / t if t > 0.0 else 0.0


def reldiff(x: float, y: float) -> float:
    d = max(abs(x), abs(y), 1.0e-300)
    return abs(x - y) / d


def apply_rule(R: dict, maps, margin_threshold: float):
    """The rule from spec.md, section 'The criterion' -> (choice, prefs, margins).

    This function encodes the *rule*, which spec.md states in full.  It encodes
    no expectation about which form the rule selects.
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


def recompute_criterion(module, positions, species, demix, params):
    """dS and R, computed from the implementation's own omega() and coordination()."""
    crit = params["criterion"]
    alpha = float(params["alpha"])
    settings = {
        "lo": float(crit["alpha_scale_lo"]) * alpha,
        "mid": alpha,
        "hi": float(crit["alpha_scale_hi"]) * alpha,
    }
    n_pinned = call_coord(module, positions, species, params, where="for the pinned labelling")
    n_map = {
        m: call_coord(module, positions, demix[m], params, where=f"for demix map {m!r}")
        for m in demix
    }
    delta_s, R = {}, {}
    for m in demix:
        delta_s[m], R[m] = {}, {}
        for f in FORMS:
            vals = {}
            for key, a in settings.items():
                q = clone(params)
                q["alpha"] = a
                w_m = call_omega(module, n_map[m], f, q, where=f"at alpha={a:g}")
                w_0 = call_omega(module, n_pinned, f, q, where=f"at alpha={a:g}")
                if np.any(w_m <= 0.0) or np.any(w_0 <= 0.0):
                    raise ContractError(
                        f"omega(..., form={f!r}) returned a non-positive value at alpha={a:g}; "
                        "the entropy of a harmonic oscillator is not defined there"
                    )
                vals[key] = -float((np.log(w_m) - np.log(w_0)).sum())
            delta_s[m][f] = vals
            spread = abs(vals["hi"] - vals["lo"])
            if spread <= 0.0:
                raise ContractError(
                    f"dS for form {f!r} on map {m!r} is identical at alpha_lo and alpha_hi, "
                    "so R is undefined and the criterion cannot be evaluated: the demixing "
                    "entropy change does not respond to alpha at all"
                )
            R[m][f] = abs(vals["mid"]) / spread
    return delta_s, R


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def check_0_contract(module, positions, species, xi, params):
    c = Check("0", "interface contract")
    pos_before, xi_before, sp_before = positions.copy(), xi.copy(), species.copy()

    ef, terms = {}, {}
    for form in FORMS:
        energy, forces = call_ef(module, positions, species, xi, params, form)
        t = call_terms(module, positions, species, xi, params, form)
        ef[form] = (energy, forces)
        terms[form] = t
        f_scale = float(np.linalg.norm(forces, axis=1).mean())
        c.record(f"{form}: energy", energy)
        c.record(f"{form}: mean |F_k|", f_scale, ok=f_scale > 0.0)
        c.require(f_scale > 0.0, f"the {form} implementation returns identically zero forces")

        stacked = t["pair"] + t["coord_self"] + t["coord_cross"]
        scale = float(np.abs(forces).max())
        if scale > 0.0:
            dev = float(np.abs(stacked - forces).max()) / scale
            c.record(f"{form}: max |sum(terms) - F| / max|F|", dev, ok=dev <= TOL_TERMS_SUM)
            c.require(
                dev <= TOL_TERMS_SUM,
                f"{form}: force_terms() does not sum to the force from energy_and_forces()",
            )

        energy2, forces2 = call_ef(module, positions, species, xi, params, form)
        same = energy2 == energy and np.array_equal(forces2, forces)
        c.record(f"{form}: repeat call identical", same, ok=same, fmt="{}")
        c.require(same, f"{form}: energy_and_forces() is not deterministic for fixed inputs")

    unmutated = (
        np.array_equal(positions, pos_before)
        and np.array_equal(xi, xi_before)
        and np.array_equal(species, sp_before)
    )
    c.record("inputs left unmutated", unmutated, ok=unmutated, fmt="{}")
    c.require(unmutated, "the implementation mutated its input arrays")

    # The two forms must not be the same object dressed up: they must differ on
    # the returned energy at the pinned parameters.
    de = reldiff(ef[FORMS[0]][0], ef[FORMS[1]][0])
    c.record("relative |E(arctan) - E(rational)|", de, ok=de >= MIN_RESPONSE)
    c.require(de >= MIN_RESPONSE, "the two forms return the same energy; 'form' is being ignored")

    # Parameters must be read from `params`, not hard-coded.  Each parameter
    # that carries a scale is perturbed on its own, and m_osc differentially,
    # so that an implementation which has collapsed the two species is caught.
    for form in FORMS:
        base = ef[form][0]
        for key, mutate in (
            ("epsilon", lambda q: [[1.5 * v for v in row] for row in q["epsilon"]]),
            ("omega0", lambda q: 1.5 * float(q["omega0"])),
            ("m_osc", lambda q: [1.5 * float(q["m_osc"][0]), 0.5 * float(q["m_osc"][1])]),
            ("alpha", lambda q: 1.4 * float(q["alpha"])),
        ):
            scaled = clone(params)
            scaled[key] = mutate(params)
            e_scaled, _ = call_ef(
                module, positions, species, xi, scaled, form,
                where=f"with params['{key}'] perturbed",
            )
            responds = abs(e_scaled - base) > 1e-12 * max(1.0, abs(base))
            c.record(f"{form}: energy responds to params['{key}']", responds, ok=responds, fmt="{}")
            c.require(
                responds,
                f"{form}: the implementation ignores params['{key}']: it appears to be hard-coded",
            )

    # calibrate() must derive p from alpha and n0 on every call.
    p_pinned = call_calibrate(module, params)
    c.record("calibrate()['p'] at the pinned parameters", p_pinned)
    for key, factor in (("alpha", 1.4), ("n0", 1.3)):
        q = clone(params)
        q[key] = factor * float(params[key])
        p_q = call_calibrate(module, q, where=f"with params['{key}'] perturbed")
        moved = reldiff(p_q, p_pinned) >= MIN_RESPONSE
        c.record(f"p responds to params['{key}']", moved, ok=moved, fmt="{}")
        c.require(moved, f"calibrate() ignores params['{key}']: p appears to be hard-coded")
    q0 = clone(params)
    q0["alpha"] = 0.0
    p_zero = call_calibrate(module, q0, where="at alpha = 0")
    c.record("calibrate()['p'] at alpha = 0", p_zero, ok=abs(p_zero) <= TOL_EXACT)
    c.require(
        abs(p_zero) <= TOL_EXACT,
        "calibrate() does not return p = 0 at alpha = 0; the calibration "
        "p = 4 n0 alpha / pi is not being applied",
    )
    return c, ef, terms


def check_1_nonequivalence(module, params, n_config):
    c = Check("1", "the two forms are not equivalent")
    w0 = float(params["omega0"])
    alpha = float(params["alpha"])
    n0 = float(params["n0"])

    # (a) calibration: both forms agree in value and first derivative at n0.
    for form in FORMS:
        v = float(call_omega(module, np.array([n0]), form, params)[0])
        d = abs(v - 0.5 * w0) / w0
        c.record(f"{form}: |omega(n0) - omega0/2| / omega0", d, ok=d <= TOL_EXACT)
        c.require(d <= TOL_EXACT, f"{form}: omega(n0) is not omega0/2; the calibration is not met")

        h = 1.0e-5
        wp = float(call_omega(module, np.array([n0 + h]), form, params)[0])
        wm = float(call_omega(module, np.array([n0 - h]), form, params)[0])
        slope = (wp - wm) / (2.0 * h)
        want = -w0 * alpha / math.pi
        d = abs(slope - want) / abs(want)
        c.record(f"{form}: omega'(n0) vs -omega0 alpha / pi, relative", d, ok=d <= TOL_SLOPE)
        c.require(d <= TOL_SLOPE, f"{form}: omega'(n0) does not meet the calibrated slope")

    # (b, c) the two exact identities, each satisfied by one form and violated
    #        by the other.  The lower bounds are as binding as the upper ones.
    def reflection_worst(form, q):
        nn0 = float(q["n0"])
        ds = np.array([f * nn0 for f in REFLECT_FRACS])
        wp = call_omega(module, nn0 + ds, form, q)
        wm = call_omega(module, nn0 - ds, form, q)
        return float(np.abs(wp + wm - float(q["omega0"])).max()) / float(q["omega0"])

    def inversion_worst(form, q):
        nn0 = float(q["n0"])
        ts = np.array(INVERT_TS)
        wp = call_omega(module, nn0 * np.exp(ts), form, q)
        wm = call_omega(module, nn0 * np.exp(-ts), form, q)
        return float(np.abs(wp + wm - float(q["omega0"])).max()) / float(q["omega0"])

    r_arc = reflection_worst("arctan", params)
    r_rat = reflection_worst("rational", params)
    i_rat = inversion_worst("rational", params)
    i_arc = inversion_worst("arctan", params)
    c.record("reflection residual, arctan   (must be <= 1e-12)", r_arc, ok=r_arc <= TOL_EXACT)
    c.record("reflection residual, rational (must be >= 1e-3)", r_rat, ok=r_rat >= MIN_VIOLATION)
    c.record("inversion  residual, rational (must be <= 1e-12)", i_rat, ok=i_rat <= TOL_EXACT)
    c.record("inversion  residual, arctan   (must be >= 1e-3)", i_arc, ok=i_arc >= MIN_VIOLATION)
    c.require(r_arc <= TOL_EXACT, "the arctan form is not symmetric under reflection of n about n0")
    c.require(
        r_rat >= MIN_VIOLATION,
        "the rational form satisfies the arctan form's reflection identity; the two "
        "forms are not distinct as implemented",
    )
    c.require(i_rat <= TOL_EXACT, "the rational form is not symmetric under inversion of n about n0")
    c.require(
        i_arc >= MIN_VIOLATION,
        "the arctan form satisfies the rational form's inversion identity; the two "
        "forms are not distinct as implemented",
    )

    # (d) no reparameterization reconciles them: the violations persist over a
    #     grid in (alpha, n0), which is what makes this non-equivalence rather
    #     than disagreement at one parameter setting.
    worst_r, worst_i = math.inf, math.inf
    for a, nn0 in REPARAM_GRID:
        q = clone(params)
        q["alpha"], q["n0"] = a, nn0
        worst_r = min(worst_r, reflection_worst("rational", q))
        worst_i = min(worst_i, inversion_worst("arctan", q))
    c.record(
        f"over {len(REPARAM_GRID)} (alpha, n0) settings: weakest rational reflection violation",
        worst_r, ok=worst_r >= MIN_VIOLATION,
    )
    c.record(
        f"over {len(REPARAM_GRID)} (alpha, n0) settings: weakest arctan inversion violation",
        worst_i, ok=worst_i >= MIN_VIOLATION,
    )
    c.require(worst_r >= MIN_VIOLATION, "at some (alpha, n0) the rational form becomes reflection-symmetric")
    c.require(worst_i >= MIN_VIOLATION, "at some (alpha, n0) the arctan form becomes inversion-symmetric")

    # (e) shape: bounded and non-increasing, with a non-trivial total drop.
    n_max = float(max(n_config.max(), 2.0 * n0))
    grid = np.linspace(0.0, n_max, 401)
    for form in FORMS:
        w = call_omega(module, grid, form, params)
        in_range = bool(np.all(w > 0.0) and np.all(w <= w0 * (1.0 + 1e-12)))
        c.record(f"{form}: omega in (0, omega0] on [0, n_max]", in_range, ok=in_range, fmt="{}")
        c.require(in_range, f"{form}: omega leaves (0, omega0] on the sampled range")
        worst_rise = float(np.diff(w).max()) / w0
        c.record(f"{form}: largest rise in omega / omega0 (must be <= 0)", worst_rise,
                 ok=worst_rise <= 1e-14)
        c.require(worst_rise <= 1e-14, f"{form}: omega is not non-increasing in n")
        drop = float(w[0] - w[-1]) / w0
        c.record(f"{form}: total drop over [0, n_max] / omega0", drop, ok=drop >= 0.1)
        c.require(drop >= 0.1, f"{form}: omega barely varies over the sampled range")

    # (f) and they differ where it matters: on the configuration's own n_i.
    wa = call_omega(module, n_config, "arctan", params)
    wr = call_omega(module, n_config, "rational", params)
    sep = float(np.abs(wa - wr).max()) / w0
    c.record("max |omega_arctan - omega_rational| / omega0 on the config's n_i", sep,
             ok=sep >= MIN_VIOLATION)
    c.require(
        sep >= MIN_VIOLATION,
        "the two forms are numerically indistinguishable on the pinned configuration",
    )
    return c


def check_2_coordination(module, positions, species, params):
    c = Check("2", "the coordination is homo-coordination")
    n = positions.shape[0]
    n_pinned = call_coord(module, positions, species, params)
    c.record("mean n_i (pinned labelling)", float(n_pinned.mean()))

    # (a) relabelling everything to one species cannot lose homo-neighbours,
    #     and must gain a substantial number.  A count over ALL neighbours is
    #     invariant here and fails the strict-increase row; a hetero count
    #     collapses to zero and fails the positivity row.
    ones = np.zeros_like(species)
    n_one = call_coord(module, positions, ones, params, where="with every particle one species")
    worst_drop = float((n_one - n_pinned).min())
    c.record("min (n_one_species - n_pinned)  (must be >= 0)", worst_drop, ok=worst_drop >= -1e-12)
    c.require(worst_drop >= -1e-12,
              "relabelling every particle to one species decreased a coordination number; "
              "n_i is not counting same-species neighbours")
    gained = int(np.count_nonzero((n_one - n_pinned) > 1e-6))
    c.record(f"particles gaining coordination (must be >= {n // 4})", gained, ok=gained >= n // 4,
             fmt="{}")
    c.require(gained >= n // 4,
              "relabelling every particle to one species changed almost nothing; n_i appears "
              "to count all neighbours regardless of species")
    max_one = float(n_one.max())
    c.record("max n_i with every particle one species (must be > 0)", max_one, ok=max_one > 0.0)
    c.require(max_one > 0.0,
              "with every particle the same species every coordination number is zero; "
              "n_i appears to count hetero-species neighbours")

    # (b) homo-coordination is invariant under a global A/B swap.
    n_swap = call_coord(module, positions, 1 - species, params, where="under a global A/B swap")
    dev = float(np.abs(n_swap - n_pinned).max()) / max(float(n_pinned.max()), 1.0)
    c.record("max |n(swapped) - n| / max n  under s -> 1 - s", dev, ok=dev <= TOL_EXACT)
    c.require(dev <= TOL_EXACT,
              "n_i changes under a global A/B relabelling; it is not symmetric in the species")

    # (c) flipping one particle's species is a strictly local change.
    r_off = float(params["coord_switch"]["r_off"])
    k = int(np.argmax(n_pinned))
    flipped = species.copy()
    flipped[k] = 1 - flipped[k]
    n_flip = call_coord(module, positions, flipped, params, where="with one particle relabelled")
    changed = np.abs(n_flip - n_pinned) > 1e-12
    d_k = np.linalg.norm(positions - positions[k], axis=1)
    allowed = (d_k < r_off) | (np.arange(n) == k)
    outside = int(np.count_nonzero(changed & ~allowed))
    c.record("particles beyond r_off whose n changed (must be 0)", outside, ok=outside == 0, fmt="{}")
    c.require(outside == 0,
              "relabelling one particle changed coordination numbers beyond r_off of it; "
              "n_i is not a local, cut-off sum")
    own = float(abs(n_flip[k] - n_pinned[k]))
    c.record("change in the relabelled particle's own n (must be > 0)", own, ok=own > 1e-6)
    c.require(own > 1e-6,
              "relabelling a particle did not change its own coordination number")

    # (d) two-particle probes pin the switch's plateaux with no reference value.
    r_on = float(params["coord_switch"]["r_on"])
    two_sp = np.zeros(2, dtype=np.int64)

    def probe(r):
        pos2 = np.array([[0.0, 0.0, 0.0], [float(r), 0.0, 0.0]])
        return call_coord(module, pos2, two_sp, params, where=f"on a two-particle probe at r={r:g}")

    far = probe(r_off + 0.5)
    near = probe(max(r_on - 0.2, 1.0e-3))
    c.record("two-particle probe, r > r_off: max n (must be 0)", float(np.abs(far).max()),
             ok=float(np.abs(far).max()) <= TOL_SWITCH_PLATEAU)
    c.require(float(np.abs(far).max()) <= TOL_SWITCH_PLATEAU,
              "a particle beyond r_off still contributes to the coordination number")
    d_near = float(np.abs(near - 1.0).max())
    c.record("two-particle probe, r < r_on: |n - 1| (must be 0)", d_near,
             ok=d_near <= TOL_SWITCH_PLATEAU)
    c.require(d_near <= TOL_SWITCH_PLATEAU,
              "a same-species particle inside r_on does not contribute exactly 1")

    # (e) the switch is C^2 at the window edges: the observed order with which
    #     |n'| and |n''| vanish there.  A C^1 switch (a raised cosine, say) has
    #     |n'| vanishing at order 1 and |n''| not vanishing at all.
    width = r_off - r_on
    h = 1.0e-4 * width

    def derivs(r):
        f0 = float(probe(r)[0])
        fp = float(probe(r + h)[0])
        fm = float(probe(r - h)[0])
        return abs((fp - fm) / (2.0 * h)), abs((fp - 2.0 * f0 + fm) / (h * h))

    for label, edge, sign in (("r_on", r_on, +1.0), ("r_off", r_off, -1.0)):
        e1, e2 = 0.02 * width, 0.01 * width
        d1a, d2a = derivs(edge + sign * e1)
        d1b, d2b = derivs(edge + sign * e2)
        o1 = math.log(d1a / d1b) / math.log(2.0) if d1b > 0 else math.inf
        o2 = math.log(d2a / d2b) / math.log(2.0) if d2b > 0 else math.inf
        c.record(f"{label}: observed order of |n'| (must be >= {MIN_SWITCH_ORDER_D1})", o1,
                 ok=o1 >= MIN_SWITCH_ORDER_D1)
        c.record(f"{label}: observed order of |n''| (must be >= {MIN_SWITCH_ORDER_D2})", o2,
                 ok=o2 >= MIN_SWITCH_ORDER_D2)
        c.require(o1 >= MIN_SWITCH_ORDER_D1,
                  f"the switching function's first derivative does not vanish to order 2 at {label}")
        c.require(o2 >= MIN_SWITCH_ORDER_D2,
                  f"the switching function's second derivative does not vanish at {label}; "
                  "the switch is not C^2")
    return c, n_pinned


def check_3_battery(module, positions, species, xi, params, ef, terms, rng):
    """The w1-01 certification battery, run per form.  A gate, not the decision."""
    c = Check("3", "each form passes the w1-01 battery (a gate, not the decision)")
    zeros = np.zeros_like(xi)
    xi2 = (xi * xi).sum(axis=1)

    for form in FORMS:
        energy, forces = ef[form]
        t = terms[form]
        f_scale = float(np.linalg.norm(forces, axis=1).mean())
        c.record(f"--- {form} ---", "", fmt="{}")
        c.record(f"{form}: F_scale = mean_k |F_k|", f_scale)

        # translation, both directions
        r_tot = rel_net(forces)
        c.record(f"{form}: |sum_k F_k| / sum_k |F_k|", r_tot, ok=r_tot <= TOL_SUM_ZERO)
        c.require(r_tot <= TOL_SUM_ZERO, f"{form}: the total force does not sum to zero")
        r_pair = rel_net(t["pair"])
        c.record(f"{form}: pair: |sum F| / sum |F|", r_pair, ok=r_pair <= TOL_SUM_ZERO)
        c.require(r_pair <= TOL_SUM_ZERO, f"{form}: the pair term alone does not sum to zero")

        scale_all = total(forces)
        for key in COORD_TERMS:
            mag = total(t[key]) / scale_all
            ok_mag = mag >= MIN_TERM_MAG
            c.record(f"{form}: {key}: sum |F| / sum |F_total|  (>= {MIN_TERM_MAG:.0e})", mag,
                     ok=ok_mag)
            c.require(ok_mag, f"{form}: the {key} term is absent or negligible")
            r = rel_net(t[key], floor=1.0e-12 * scale_all)
            ok = r >= MIN_TERM_SUM
            c.record(f"{form}: {key}: |sum F| / sum |F|  (>= {MIN_TERM_SUM:.0e})", r, ok=ok)
            c.require(ok, f"{form}: the {key} term conserves momentum on its own; the two "
                          "halves of the coordination force must each carry a net force")
        s_self = t["coord_self"].sum(axis=0)
        s_cross = t["coord_cross"].sum(axis=0)
        denom = float(max(np.linalg.norm(s_self), np.linalg.norm(s_cross)))
        if denom > 0.0:
            cancel = float(np.linalg.norm(s_self + s_cross)) / denom
            c.record(f"{form}: coordination halves: cancellation", cancel, ok=cancel <= TOL_TERM_CANCEL)
            c.require(cancel <= TOL_TERM_CANCEL,
                      f"{form}: the two coordination half-terms do not cancel")

        d = rng.normal(size=3) * 2.0
        e_t, f_t = call_ef(module, positions + d, species, xi, params, form,
                           where="under rigid translation")
        de = abs(e_t - energy) / max(abs(energy), 1.0)
        df = float(np.abs(f_t - forces).max()) / f_scale
        c.record(f"{form}: |dE| / |E| under translation", de, ok=de <= TOL_RIGID)
        c.record(f"{form}: max |dF| / F_scale under translation", df, ok=df <= TOL_RIGID)
        c.require(de <= TOL_RIGID, f"{form}: the energy is not invariant under a rigid translation")
        c.require(df <= TOL_RIGID, f"{form}: the forces are not invariant under a rigid translation")

        # force-energy consistency, with the observed order
        n = positions.shape[0]
        picks = [(int(rng.integers(n)), int(rng.integers(3))) for _ in range(FD_SAMPLES)]
        worst_best, worst_order = 0.0, math.inf
        for i, ax in picks:
            errs = []
            for hstep in FD_STEPS:
                plus, minus = positions.copy(), positions.copy()
                plus[i, ax] += hstep
                minus[i, ax] -= hstep
                e_p, _ = call_ef(module, plus, species, xi, params, form,
                                 where="at a displaced position")
                e_m, _ = call_ef(module, minus, species, xi, params, form,
                                 where="at a displaced position")
                num = -(e_p - e_m) / (2.0 * hstep)
                errs.append(abs(forces[i, ax] - num) / f_scale)
            worst_best = max(worst_best, min(errs))
            worst_order = min(worst_order,
                              math.log(errs[0] / errs[2]) / math.log(FD_STEPS[0] / FD_STEPS[2]))
        c.record(f"{form}: worst best-step FD residual / F_scale", worst_best, ok=worst_best <= TOL_FD)
        c.record(f"{form}: worst observed convergence order", worst_order,
                 ok=worst_order >= MIN_FD_ORDER)
        c.require(worst_best <= TOL_FD,
                  f"{form}: the analytic force is not the gradient of the returned energy")
        c.require(worst_order >= MIN_FD_ORDER,
                  f"{form}: the finite-difference residual does not converge at the expected "
                  "order; the force is not the gradient of the returned energy")

        # rotation
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1.0
        rotated = positions @ q.T
        e_r, f_r = call_ef(module, rotated, species, xi, params, form, where="under rigid rotation")
        de = abs(e_r - energy) / max(abs(energy), 1.0)
        df = float(np.abs(f_r - forces @ q.T).max()) / f_scale
        c.record(f"{form}: |dE| / |E| under rotation", de, ok=de <= TOL_RIGID)
        c.record(f"{form}: max |F(Qr) - Q F(r)| / F_scale", df, ok=df <= TOL_RIGID)
        c.require(de <= TOL_RIGID, f"{form}: the energy is not invariant under a rigid rotation")
        c.require(df <= TOL_RIGID, f"{form}: the forces are not covariant under a rigid rotation")
        terms_r = call_terms(module, rotated, species, xi, params, form, where="under rigid rotation")
        for key in TERM_KEYS:
            dv = float(np.abs(terms_r[key] - t[key] @ q.T).max()) / f_scale
            c.record(f"{form}: {key}: max |F(Qr) - Q F(r)| / F_scale", dv, ok=dv <= TOL_RIGID)
            c.require(dv <= TOL_RIGID, f"{form}: the {key} term is not rotationally covariant")

        # limits
        e_x0, f_x0 = call_ef(module, positions, species, zeros, params, form, where="at xi = 0")
        t_x0 = call_terms(module, positions, species, zeros, params, form, where="at xi = 0")
        scale_x0 = max(float(np.linalg.norm(f_x0, axis=1).mean()), 1.0e-300)
        for key in COORD_TERMS:
            dv = float(np.abs(t_x0[key]).max()) / scale_x0
            c.record(f"{form}: xi=0: max |{key}| / F_scale", dv, ok=dv <= TOL_LIMIT)
            c.require(dv <= TOL_LIMIT, f"{form}: at xi = 0 the {key} term does not vanish")
        dv = float(np.abs(f_x0 - t_x0["pair"]).max()) / scale_x0
        c.record(f"{form}: xi=0: max |F - F_pair| / F_scale", dv, ok=dv <= TOL_LIMIT)
        c.require(dv <= TOL_LIMIT, f"{form}: at xi = 0 the total force is not the pair force")

        # alpha = 0.  Because p = 4 n0 alpha / pi, BOTH forms degenerate to the
        # constant omega0 / 2 here, so the same closed form applies to each.
        flat = clone(params)
        flat["alpha"] = 0.0
        e_a0, f_a0 = call_ef(module, positions, species, xi, flat, form, where="at alpha = 0")
        t_a0 = call_terms(module, positions, species, xi, flat, form, where="at alpha = 0")
        scale_a0 = max(float(np.linalg.norm(f_a0, axis=1).mean()), 1.0e-300)
        for key in COORD_TERMS:
            dv = float(np.abs(t_a0[key]).max()) / scale_a0
            c.record(f"{form}: alpha=0: max |{key}| / F_scale", dv, ok=dv <= TOL_LIMIT)
            c.require(dv <= TOL_LIMIT, f"{form}: at alpha = 0 the {key} term does not vanish")
        e_x0a0, _ = call_ef(module, positions, species, zeros, flat, form,
                            where="at xi = 0, alpha = 0")
        dv = abs(e_x0a0 - e_x0) / max(abs(e_x0), 1.0)
        c.record(f"{form}: alpha independence at xi = 0", dv, ok=dv <= TOL_LIMIT)
        c.require(dv <= TOL_LIMIT, f"{form}: at xi = 0 the energy still depends on alpha")

        # The closed form must hold across several (omega0, m_osc) settings, not
        # one: at the pinned values alone any implementation that hard-codes
        # them satisfies it for free.
        pinned_w0 = float(params["omega0"])
        pinned_m = [float(v) for v in params["m_osc"]]
        for label, w0v, m in (
            ("pinned", pinned_w0, pinned_m),
            ("omega0 x 1.5", 1.5 * pinned_w0, pinned_m),
            ("m_osc perturbed", pinned_w0, [1.5 * pinned_m[0], 0.5 * pinned_m[1]]),
        ):
            q2 = clone(params)
            q2["alpha"], q2["omega0"], q2["m_osc"] = 0.0, w0v, m
            e_q, _ = call_ef(module, positions, species, xi, q2, form,
                             where=f"at alpha = 0, {label}")
            e_q0, _ = call_ef(module, positions, species, zeros, q2, form,
                              where=f"at alpha = 0, xi = 0, {label}")
            m_arr = np.asarray(m, dtype=np.float64)
            expected = 0.5 * float((m_arr[species] * xi2).sum()) * (0.5 * w0v) ** 2
            got = e_q - e_q0
            dv = abs(got - expected) / max(abs(expected), 1.0)
            c.record(f"{form}: alpha=0, {label}: stiffness energy, relative deviation", dv,
                     ok=dv <= TOL_LIMIT)
            c.require(dv <= TOL_LIMIT,
                      f"{form}: at alpha = 0 with {label} the stiffness energy does not equal "
                      "(1/2) (omega0/2)^2 sum_i m_i sum_g xi_ig^2")

        non_degenerate = abs(e_a0 - energy) > 1e-9 * max(abs(energy), 1.0)
        c.record(f"{form}: alpha = 0 differs from the pinned alpha", non_degenerate,
                 ok=non_degenerate, fmt="{}")
        c.require(non_degenerate, f"{form}: the energy does not depend on alpha at all")
    return c


def check_4_criterion(module, positions, species, demix, params):
    c = Check("4", "the criterion, recomputed")
    margin_threshold = float(params["criterion"]["margin"])
    maps = sorted(demix)

    got = call_compare(module, positions, species, demix, params)
    ds_ref, r_ref = recompute_criterion(module, positions, species, demix, params)

    worst_ds, worst_r = 0.0, 0.0
    for m in maps:
        for f in FORMS:
            for k in DS_KEYS:
                d = reldiff(float(got["delta_s"][m][f][k]), ds_ref[m][f][k])
                worst_ds = max(worst_ds, d)
            c.record(f"dS[{m}][{f}] (lo, mid, hi), recomputed",
                     "  ".join(f"{ds_ref[m][f][k]:+.6f}" for k in DS_KEYS), fmt="{}")
            worst_r = max(worst_r, reldiff(float(got["R"][m][f]), r_ref[m][f]))
            c.record(f"R[{m}][{f}], recomputed", r_ref[m][f])
    c.record("worst relative deviation, returned dS vs recomputed", worst_ds,
             ok=worst_ds <= TOL_RECOMPUTE)
    c.record("worst relative deviation, returned R vs recomputed", worst_r,
             ok=worst_r <= TOL_RECOMPUTE)
    c.require(worst_ds <= TOL_RECOMPUTE,
              "compare()['delta_s'] does not match the entropy change recomputed from the "
              "implementation's own omega() and coordination()")
    c.require(worst_r <= TOL_RECOMPUTE,
              "compare()['R'] does not match the statistic recomputed from the "
              "implementation's own numbers")

    verdict, prefs, margins = apply_rule(r_ref, maps, margin_threshold)
    for m in maps:
        c.record(f"{m}: preferred by the rule / margin (threshold {margin_threshold:.2f})",
                 f"{prefs[m]} / {margins[m]:.4f}", fmt="{}")
    c.record("rule applied to the recomputed R", verdict, fmt="{}")
    c.record("compare()['choice']", got["choice"], fmt="{}")
    c.require(got["choice"] == verdict,
              f"compare()['choice'] is {got['choice']!r}, but the rule in spec.md applied to the "
              f"implementation's own numbers gives {verdict!r}")

    # Second half: an implementation that returns a hard-coded verdict with
    # plausible numbers is internally consistent, and is caught only by
    # requiring the numbers to respond to the parameters.
    for key, factor in (("alpha", 1.3), ("n0", 1.2)):
        q = clone(params)
        q[key] = factor * float(params[key])
        alt = call_compare(module, positions, species, demix, q,
                           where=f"with params['{key}'] perturbed")
        moved = 0.0
        for m in maps:
            for f in FORMS:
                for k in DS_KEYS:
                    moved = max(moved, reldiff(float(alt["delta_s"][m][f][k]),
                                               float(got["delta_s"][m][f][k])))
        c.record(f"dS moves when params['{key}'] is perturbed", moved, ok=moved >= MIN_RESPONSE)
        c.require(moved >= MIN_RESPONSE,
                  f"compare() returns the same dS when params['{key}'] changes: the criterion "
                  "is not being evaluated from the parameters")
    return c, got["choice"]


def check_5_report(report_path: Path, choice: str | None):
    c = Check("5", "the report")
    exists = report_path.is_file()
    c.record("report path", str(report_path), fmt="{}")
    c.record("report present", exists, ok=exists, fmt="{}")
    c.require(exists, f"no report at {report_path}")
    if not exists:
        return c
    text = report_path.read_text()
    lines = text.splitlines()
    for heading in REPORT_SECTIONS:
        present = any(line.strip() == heading for line in lines)
        c.record(f"section {heading!r}", present, ok=present, fmt="{}")
        c.require(present, f"the report has no {heading!r} section")

    found = [m.group(1) for m in (CHOICE_RE.match(ln.strip()) for ln in lines) if m]
    c.record("CHOICE: lines found", len(found), ok=len(found) == 1, fmt="{}")
    c.require(len(found) == 1,
              "the report must carry exactly one line of the form "
              "'CHOICE: arctan' (or rational, or undecided); found "
              f"{len(found)}")
    if len(found) == 1 and choice is not None:
        c.record("report CHOICE vs compare()['choice']", f"{found[0]} / {choice}", fmt="{}")
        c.require(found[0] == choice,
                  f"the report states CHOICE: {found[0]}, but compare() returned {choice!r}")
    c.lines.append("      note: this verifier checks that the report exists, is structured, and")
    c.lines.append("      states a verdict consistent with the implementation's own numbers;")
    c.lines.append("      the defence is judged against the disqualifiers in spec.md, not here.")
    return c


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", type=Path, default=HERE / "forms.py")
    ap.add_argument("--config-dir", type=Path, default=HERE / "config")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()
    impl_path = args.impl.resolve()
    report_path = (args.report or impl_path.parent / "report.md").resolve()

    print("w1-02 — non-equivalence and a stated criterion")
    print(f"  implementation : {impl_path}")
    print(f"  configuration  : {args.config_dir.resolve()}")
    print()

    try:
        cfg, params, positions, species, xi, demix = load_config(args.config_dir.resolve())
        module = load_impl(impl_path)
    except ContractError as exc:
        print(f"  [FAIL] interface contract\n      -> {exc}\n")
        print("RESULT: FAIL (contract)")
        return 2

    print(
        f"  configuration: N={cfg['n_particles']}, n_osc={cfg['n_osc']}, "
        f"boundary={cfg['boundary']}, seed={cfg['seed']}, "
        f"demix maps={sorted(demix)}"
    )
    print()

    rng = np.random.default_rng(RNG_SEED)
    checks: list[Check] = []
    choice = None
    try:
        c0, ef, terms = check_0_contract(module, positions, species, xi, params)
        checks.append(c0)
        if c0.passed:
            c2, n_pinned = check_2_coordination(module, positions, species, params)
            checks.append(check_1_nonequivalence(module, params, n_pinned))
            checks.append(c2)
            checks.append(check_3_battery(module, positions, species, xi, params, ef, terms, rng))
            if c2.passed:
                c4, choice = check_4_criterion(module, positions, species, demix, params)
                checks.append(c4)
            else:
                # The criterion is computed from coordination().  Evaluating it on a
                # coordination that failed check 2 would report a verdict about the
                # wrong quantity, so it is recorded as not run rather than skipped.
                c4 = Check("4", "the criterion, recomputed")
                c4.require(
                    False,
                    "not run: coordination() did not meet check 2, so the criterion would "
                    "be evaluated on the wrong quantity",
                )
                checks.append(c4)
    except ContractError as exc:
        for c in checks:
            c.emit()
        print(f"  [FAIL] interface contract\n      -> {exc}\n")
        print("RESULT: FAIL (contract)")
        return 2

    checks.append(check_5_report(report_path, choice))
    checks.sort(key=lambda c: c.number)

    for c in checks:
        c.emit()

    failed = [c for c in checks if not c.passed]
    if failed:
        print("RESULT: FAIL — " + ", ".join(f"check {c.number}" for c in failed))
        return 1
    print("RESULT: PASS — all checks satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
