#!/usr/bin/env python3
"""w1-01 verifier — certification battery for a from-the-paper force implementation.

This verifier imports the implementation named in the interface contract of
spec.md and runs every check itself.  It contains no reference values and no
answer key: every criterion is an identity, an invariance, or a limit of the
model as specified.  A reported pass on a check the implementation does not
actually satisfy therefore cannot survive.

Usage (from the task directory):

    python verify.py

Options are for the build-time dry-run only; the graded session needs none:

    --impl PATH        implementation module (default: ./force_kernel.py)
    --config-dir PATH  configuration directory (default: ./config)
    --report PATH      certification report (default: report.md beside --impl)

Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Tolerances.  All are dimensionless ratios, taken against the configuration's
# own force scale F_scale = mean_k |F_k| or against the quantity's own
# magnitude.  No tolerance is an absolute number in reduced units.
# --------------------------------------------------------------------------
TOL_SUM_ZERO = 1.0e-10      # |sum_k F_k| / sum_k |F_k| for a conserving total
TOL_TERM_CANCEL = 1.0e-8    # cancellation of the two coordination half-terms
MIN_TERM_SUM = 1.0e-3       # |sum_k F^t_k| / sum_k |F^t_k| that a half-term must EXCEED
MIN_TERM_MAG = 1.0e-6       # sum_k |F^t_k| / sum_k |F_k| that a half-term must EXCEED
TOL_RIGID = 1.0e-9          # rigid-motion invariance, relative
TOL_FD = 1.0e-7             # best finite-difference residual, relative to F_scale
MIN_FD_ORDER = 1.5          # observed convergence order over the coarse steps
TOL_TERMS_SUM = 1.0e-12     # sum of named terms vs. the returned total force
TOL_LIMIT = 1.0e-12         # limiting-case identities, relative

FD_STEPS = (1.0e-2, 10.0**-2.5, 1.0e-3, 10.0**-3.5, 1.0e-4, 10.0**-4.5)
FD_SAMPLES = 12
RNG_SEED = 20260904         # pinned: the verifier's own draws are reproducible

TERM_KEYS = ("pair", "coord_self", "coord_cross")
COORD_TERMS = ("coord_self", "coord_cross")

REPORT_SECTIONS = (
    "## Implementation",
    "## Certification battery",
    "## Error-class mapping",
    "## What no check catches",
    "## Underdetermined in the specification",
)


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
        self.lines.append(f"      {label:<46s} {shown}{mark}")

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
    if not path.is_file():
        raise ContractError(
            f"no implementation at {path}.\n"
            "  spec.md, section 'Interface contract', requires a module named "
            "force_kernel.py in the task directory."
        )
    spec = importlib.util.spec_from_file_location("w1_01_impl", path)
    if spec is None or spec.loader is None:
        raise ContractError(f"{path} is not importable as a Python module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        raise ContractError(f"importing {path} raised:\n{traceback.format_exc()}") from None
    for name in ("energy_and_forces", "force_terms"):
        if not hasattr(module, name):
            raise ContractError(
                f"{path} does not define {name}(). "
                "See spec.md, section 'Interface contract'."
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
    return cfg, params, positions, species, xi


# --------------------------------------------------------------------------
# contract-checked calls
# --------------------------------------------------------------------------
def call_ef(module, positions, species, xi, params, where: str = ""):
    try:
        out = module.energy_and_forces(positions, species, xi, params)
    except Exception:
        raise ContractError(
            f"energy_and_forces() raised{' ' + where if where else ''}:\n"
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


def call_terms(module, positions, species, xi, params, where: str = ""):
    try:
        terms = module.force_terms(positions, species, xi, params)
    except Exception:
        raise ContractError(
            f"force_terms() raised{' ' + where if where else ''}:\n{traceback.format_exc()}"
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


def net(f):
    return float(np.linalg.norm(f.sum(axis=0)))


def total(f):
    return float(np.linalg.norm(f, axis=1).sum())


def rel_net(f, floor: float = 0.0):
    """|sum_k f_k| / sum_k |f_k|.

    `floor` guards the denominator so that an identically zero term reports 0
    rather than an undefined ratio: a term with no force at all has no net
    force either, and must fail a "this term must not conserve momentum on its
    own" requirement rather than pass it vacuously.
    """
    t = max(total(f), floor)
    return net(f) / t if t > 0.0 else 0.0


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def check_0_contract(module, positions, species, xi, params):
    c = Check("0", "interface contract")
    pos_before = positions.copy()
    xi_before = xi.copy()

    energy, forces = call_ef(module, positions, species, xi, params)
    terms = call_terms(module, positions, species, xi, params)

    f_scale = float(np.linalg.norm(forces, axis=1).mean())
    c.record("energy", energy)
    c.record("mean |F_k|", f_scale, ok=f_scale > 0.0)
    c.require(f_scale > 0.0, "the implementation returns identically zero forces")

    # inputs must not be mutated
    unmutated = np.array_equal(positions, pos_before) and np.array_equal(xi, xi_before)
    c.record("inputs left unmutated", unmutated, ok=unmutated, fmt="{}")
    c.require(unmutated, "the implementation mutated its input arrays")

    # calls must be deterministic
    energy2, forces2 = call_ef(module, positions, species, xi, params)
    same = energy2 == energy and np.array_equal(forces2, forces)
    c.record("repeat call identical", same, ok=same, fmt="{}")
    c.require(same, "energy_and_forces() is not deterministic for fixed inputs")

    # the named terms must sum to the returned total force
    stacked = terms["pair"] + terms["coord_self"] + terms["coord_cross"]
    scale = float(np.abs(forces).max())
    if scale > 0.0:
        dev = float(np.abs(stacked - forces).max()) / scale
        c.record("max |sum(terms) - F| / max|F|", dev, ok=dev <= TOL_TERMS_SUM)
        c.require(
            dev <= TOL_TERMS_SUM,
            "force_terms() does not sum to the force from energy_and_forces()",
        )
    else:
        c.record("max |sum(terms) - F| / max|F|", "n/a — the forces are zero", fmt="{}")

    # parameters must be read from `params`, not hard-coded
    scaled = json.loads(json.dumps(params))
    scaled["epsilon"] = [[1.5 * v for v in row] for row in params["epsilon"]]
    e_scaled, _ = call_ef(module, positions, species, xi, scaled, where="with epsilon scaled")
    responds = abs(e_scaled - energy) > 1e-12 * max(1.0, abs(energy))
    c.record("energy responds to params['epsilon']", responds, ok=responds, fmt="{}")
    c.require(responds, "the implementation ignores params: parameters appear to be hard-coded")

    return c, energy, forces, terms


def check_1_translation(module, positions, species, xi, params, energy, forces, terms, rng):
    c = Check("1", "translational invariance and the term sum rule")
    f_scale = float(np.linalg.norm(forces, axis=1).mean())
    c.record("F_scale = mean_k |F_k|", f_scale)

    r_total = rel_net(forces)
    c.record("|sum_k F_k| / sum_k |F_k|", r_total, ok=r_total <= TOL_SUM_ZERO)
    c.require(r_total <= TOL_SUM_ZERO, "the total force does not sum to zero")

    r_pair = rel_net(terms["pair"])
    c.record("pair: |sum F| / sum |F|", r_pair, ok=r_pair <= TOL_SUM_ZERO)
    c.require(r_pair <= TOL_SUM_ZERO, "the pair term alone does not sum to zero")

    scale_all = total(forces)
    for key in COORD_TERMS:
        mag = total(terms[key]) / scale_all
        ok_mag = mag >= MIN_TERM_MAG
        label = f"{key}: sum |F| / sum |F_total|  (must exceed {MIN_TERM_MAG:.0e})"
        c.record(label, mag, ok=ok_mag)
        c.require(ok_mag, f"the {key} term is absent or negligible")
        r = rel_net(terms[key], floor=1.0e-12 * scale_all)
        ok = r >= MIN_TERM_SUM
        c.record(f"{key}: |sum F| / sum |F|  (must exceed {MIN_TERM_SUM:.0e})", r, ok=ok)
        c.require(
            ok,
            f"the {key} term conserves momentum on its own; the two halves of the "
            "coordination force must each carry a non-vanishing net force",
        )

    s_self = terms["coord_self"].sum(axis=0)
    s_cross = terms["coord_cross"].sum(axis=0)
    denom = float(max(np.linalg.norm(s_self), np.linalg.norm(s_cross)))
    if denom > 0.0:
        cancel = float(np.linalg.norm(s_self + s_cross)) / denom
        c.record("coordination halves: cancellation", cancel, ok=cancel <= TOL_TERM_CANCEL)
        c.require(cancel <= TOL_TERM_CANCEL, "the two coordination half-terms do not cancel")
    else:
        c.record(
            "coordination halves: cancellation",
            "n/a — neither half carries a net force",
            fmt="{}",
        )

    # explicit rigid translation
    d = rng.normal(size=3) * 2.0
    e_t, f_t = call_ef(module, positions + d, species, xi, params, where="under rigid translation")
    de = abs(e_t - energy) / max(abs(energy), 1.0)
    df = float(np.abs(f_t - forces).max()) / f_scale
    c.record("|dE| / |E| under translation", de, ok=de <= TOL_RIGID)
    c.record("max |dF| / F_scale under translation", df, ok=df <= TOL_RIGID)
    c.require(de <= TOL_RIGID, "the energy is not invariant under a rigid translation")
    c.require(df <= TOL_RIGID, "the forces are not invariant under a rigid translation")
    return c


def check_2_gradient(module, positions, species, xi, params, forces, rng):
    c = Check("2", "force-energy consistency by finite difference")
    f_scale = float(np.linalg.norm(forces, axis=1).mean())
    n = positions.shape[0]
    picks = [(int(rng.integers(n)), int(rng.integers(3))) for _ in range(FD_SAMPLES)]

    worst_best = 0.0
    worst_order = math.inf
    for idx, (i, ax) in enumerate(picks):
        errs = []
        for h in FD_STEPS:
            plus = positions.copy()
            minus = positions.copy()
            plus[i, ax] += h
            minus[i, ax] -= h
            e_p, _ = call_ef(module, plus, species, xi, params, where="at a displaced position")
            e_m, _ = call_ef(module, minus, species, xi, params, where="at a displaced position")
            num = -(e_p - e_m) / (2.0 * h)          # -dE/dx = force component
            errs.append(abs(forces[i, ax] - num) / f_scale)
        best = min(errs)
        order = math.log(errs[0] / errs[2]) / math.log(FD_STEPS[0] / FD_STEPS[2])
        worst_best = max(worst_best, best)
        worst_order = min(worst_order, order)
        if idx < 3:
            ladder = "  ".join(f"{e:.2e}" for e in errs)
            c.record(f"particle {i:3d} axis {ax}: err(h) ladder", ladder, fmt="{}")

    c.record("steps h (sigma_AA)", "  ".join(f"{h:.2e}" for h in FD_STEPS), fmt="{}")
    c.record("worst best-step residual / F_scale", worst_best, ok=worst_best <= TOL_FD)
    c.record("worst observed convergence order", worst_order, ok=worst_order >= MIN_FD_ORDER)
    c.require(worst_best <= TOL_FD, "the analytic force is not the gradient of the returned energy")
    c.require(
        worst_order >= MIN_FD_ORDER,
        "the finite-difference residual does not converge at the expected order; "
        "the force is not the gradient of the returned energy",
    )
    return c


def check_3_rotation(module, positions, species, xi, params, energy, forces, terms, rng):
    c = Check("3", "rotational invariance and covariance")
    f_scale = float(np.linalg.norm(forces, axis=1).mean())
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    rotated = positions @ q.T

    e_r, f_r = call_ef(module, rotated, species, xi, params, where="under rigid rotation")
    de = abs(e_r - energy) / max(abs(energy), 1.0)
    df = float(np.abs(f_r - forces @ q.T).max()) / f_scale
    c.record("det(Q)", float(np.linalg.det(q)))
    c.record("|dE| / |E| under rotation", de, ok=de <= TOL_RIGID)
    c.record("max |F(Qr) - Q F(r)| / F_scale", df, ok=df <= TOL_RIGID)
    c.require(de <= TOL_RIGID, "the energy is not invariant under a rigid rotation")
    c.require(df <= TOL_RIGID, "the forces are not covariant under a rigid rotation")

    terms_r = call_terms(module, rotated, species, xi, params, where="under rigid rotation")
    for key in TERM_KEYS:
        d = float(np.abs(terms_r[key] - terms[key] @ q.T).max()) / f_scale
        c.record(f"{key}: max |F(Qr) - Q F(r)| / F_scale", d, ok=d <= TOL_RIGID)
        c.require(d <= TOL_RIGID, f"the {key} term is not rotationally covariant")
    return c


def check_4_limits(module, positions, species, xi, params, energy, rng):
    c = Check("4", "limiting cases")
    zeros = np.zeros_like(xi)

    # (i) xi = 0: the stiffness term and both coordination half-terms vanish.
    e_x0, f_x0 = call_ef(module, positions, species, zeros, params, where="at xi = 0")
    t_x0 = call_terms(module, positions, species, zeros, params, where="at xi = 0")
    scale_x0 = max(float(np.linalg.norm(f_x0, axis=1).mean()), 1.0e-300)
    for key in COORD_TERMS:
        d = float(np.abs(t_x0[key]).max()) / scale_x0
        c.record(f"xi=0: max |{key}| / F_scale", d, ok=d <= TOL_LIMIT)
        c.require(d <= TOL_LIMIT, f"at xi = 0 the {key} term does not vanish")
    d = float(np.abs(f_x0 - t_x0["pair"]).max()) / scale_x0
    c.record("xi=0: max |F - F_pair| / F_scale", d, ok=d <= TOL_LIMIT)
    c.require(d <= TOL_LIMIT, "at xi = 0 the total force is not the pair force")

    # (ii) alpha = 0: omega_i is uniform, the coordination force vanishes, and the
    #      stiffness energy takes its stated closed form.
    flat = json.loads(json.dumps(params))
    flat["alpha"] = 0.0
    e_a0, f_a0 = call_ef(module, positions, species, xi, flat, where="at alpha = 0")
    t_a0 = call_terms(module, positions, species, xi, flat, where="at alpha = 0")
    scale_a0 = max(float(np.linalg.norm(f_a0, axis=1).mean()), 1.0e-300)
    for key in COORD_TERMS:
        d = float(np.abs(t_a0[key]).max()) / scale_a0
        c.record(f"alpha=0: max |{key}| / F_scale", d, ok=d <= TOL_LIMIT)
        c.require(d <= TOL_LIMIT, f"at alpha = 0 the {key} term does not vanish")

    e_x0a0, _ = call_ef(module, positions, species, zeros, flat, where="at xi = 0, alpha = 0")
    d = abs(e_x0a0 - e_x0) / max(abs(e_x0), 1.0)
    c.record("alpha independence at xi = 0", d, ok=d <= TOL_LIMIT)
    c.require(d <= TOL_LIMIT, "at xi = 0 the energy still depends on alpha")

    omega0 = float(params["omega0"])
    m_osc = np.asarray(params["m_osc"], dtype=np.float64)
    expected = 0.5 * float((m_osc[species] * (xi * xi).sum(axis=1)).sum()) * (0.5 * omega0) ** 2
    got = e_a0 - e_x0a0
    d = abs(got - expected) / max(abs(expected), 1.0)
    c.record("alpha=0: stiffness energy, closed form", expected)
    c.record("alpha=0: stiffness energy, measured", got)
    c.record("relative deviation", d, ok=d <= TOL_LIMIT)
    c.require(
        d <= TOL_LIMIT,
        "at alpha = 0 the stiffness energy does not equal "
        "(1/2) (omega0/2)^2 sum_i m_i sum_g xi_ig^2",
    )

    # the pinned alpha must not itself be degenerate
    non_degenerate = abs(e_a0 - energy) > 1e-9 * max(abs(energy), 1.0)
    c.record("alpha = 0 differs from the pinned alpha", non_degenerate, ok=non_degenerate, fmt="{}")
    c.require(non_degenerate, "the energy does not depend on alpha at all")
    return c


def check_5_report(report_path: Path):
    c = Check("5", "certification report and error-class mapping")
    exists = report_path.is_file()
    c.record("report path", str(report_path), fmt="{}")
    c.record("report present", exists, ok=exists, fmt="{}")
    c.require(exists, f"no certification report at {report_path}")
    if not exists:
        return c
    text = report_path.read_text()
    for heading in REPORT_SECTIONS:
        present = any(line.strip() == heading for line in text.splitlines())
        c.record(f"section {heading!r}", present, ok=present, fmt="{}")
        c.require(present, f"the report has no {heading!r} section")
    c.lines.append(
        "      note: this verifier checks that the report exists and is structured;"
    )
    c.lines.append(
        "      its content is judged against the disqualifiers in spec.md, not scored here."
    )
    return c


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", type=Path, default=HERE / "force_kernel.py")
    ap.add_argument("--config-dir", type=Path, default=HERE / "config")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()
    impl_path = args.impl.resolve()
    report_path = (args.report or impl_path.parent / "report.md").resolve()

    print("w1-01 certification battery")
    print(f"  implementation : {impl_path}")
    print(f"  configuration  : {args.config_dir.resolve()}")
    print()

    try:
        cfg, params, positions, species, xi = load_config(args.config_dir.resolve())
        module = load_impl(impl_path)
    except ContractError as exc:
        print(f"  [FAIL] interface contract\n      -> {exc}\n")
        print("RESULT: FAIL (contract)")
        return 2

    print(
        f"  configuration: N={cfg['n_particles']}, "
        f"n_osc={cfg['n_osc']}, boundary={cfg['boundary']}, seed={cfg['seed']}"
    )
    print()

    rng = np.random.default_rng(RNG_SEED)
    checks: list[Check] = []
    try:
        c0, energy, forces, terms = check_0_contract(module, positions, species, xi, params)
        checks.append(c0)
        if c0.passed:
            checks.append(
                check_1_translation(
                    module, positions, species, xi, params, energy, forces, terms, rng
                )
            )
            checks.append(check_2_gradient(module, positions, species, xi, params, forces, rng))
            checks.append(
                check_3_rotation(module, positions, species, xi, params, energy, forces, terms, rng)
            )
            checks.append(check_4_limits(module, positions, species, xi, params, energy, rng))
    except ContractError as exc:
        for c in checks:
            c.emit()
        print(f"  [FAIL] interface contract\n      -> {exc}\n")
        print("RESULT: FAIL (contract)")
        return 2

    checks.append(check_5_report(report_path))

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
