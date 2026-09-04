#!/usr/bin/env python3
"""w1-01 — mutation study: which check catches which class of error.

Every claim in the '## Error-class mapping' section of report.md is produced by
this script.  It never writes into the task directory: each mutant is a patched
copy of force_kernel.py (and, where the mutation is a *shared misreading*, of
extra_checks.py too) in a fresh temporary directory, with config/ symlinked and
a stub report.md so that check 5 does not mask checks 0-4.

Both batteries are then run against the mutant:

    verify.py --impl <tmp>/force_kernel.py    checks 0-4
    extra_checks.py from <tmp>                checks A-H

Run:  python mutation_study.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

# (id, description, [(old, new), ...] for force_kernel, [...] for extra_checks)
MUTANTS = [
    # ---- wrong physics, internally consistent -------------------------------
    ("A1", "n0 hard-coded at a wrong value (5.0)",
     [('n0 = float(params["n0"])', 'n0 = 5.0')], []),
    ("A2", "lj_switch and coord_switch windows swapped",
     [('rc_on = float(params["lj_switch"]["r_on"])\n    rc_off = float(params["lj_switch"]["r_off"])\n'
       '    rn_on = float(params["coord_switch"]["r_on"])\n    rn_off = float(params["coord_switch"]["r_off"])',
       'rc_on = float(params["coord_switch"]["r_on"])\n    rc_off = float(params["coord_switch"]["r_off"])\n'
       '    rn_on = float(params["lj_switch"]["r_on"])\n    rn_off = float(params["lj_switch"]["r_off"])')], []),
    ("A3", "tanh substituted for arctan in omega(n)",
     [('omega = omega0 * (0.5 - np.arctan(z) / np.pi)', 'omega = omega0 * (0.5 - np.tanh(z) / np.pi)'),
      ('d_omega = -omega0 * alpha / (np.pi * (1.0 + z * z))',
       'd_omega = -omega0 * alpha / (np.pi * np.cosh(z) ** 2)')], []),
    ("A4", "LJ prefactor 4 -> 1",
     [('lj = 4.0 * eps * (sr12 - sr6)', 'lj = 1.0 * eps * (sr12 - sr6)'),
      ('dlj = 4.0 * eps * (-12.0 * sr12 + 6.0 * sr6) / r',
       'dlj = 1.0 * eps * (-12.0 * sr12 + 6.0 * sr6) / r')], []),
    ("A5", "species labels swapped throughout (0=B, 1=A)",
     [('sp = np.asarray(species, dtype=np.int64)', 'sp = 1 - np.asarray(species, dtype=np.int64)')], []),
    ("A6", "n_i counts ALL neighbours, not homo-coordination only",
     [('same = off & (sp[:, None] == sp[None, :])', 'same = off.copy()')], []),
    ("A7", "C^1 cubic switch in place of the C^2 quintic",
     [('    s = 1.0 + x3 * (-10.0 + x * (15.0 - 6.0 * x))\n    ds = (-30.0 * x2 * (1.0 - x) ** 2) / width',
       '    s = 1.0 - 3.0 * x2 + 2.0 * x3\n    ds = (-6.0 * x + 6.0 * x2) / width')], []),
    ("A8", "species swapped in the eps/sigma lookup only (m_osc mapping intact)",
     [('eps = eps_mat[sp[:, None], sp[None, :]]\n    sig = sig_mat[sp[:, None], sp[None, :]]',
       'bs = 1 - sp\n    eps = eps_mat[bs[:, None], bs[None, :]]\n    sig = sig_mat[bs[:, None], bs[None, :]]')], []),
    ("A9", "n_i counts the particle itself (off-by-one)",
     [('n_coord = np.where(same, s_n, 0.0).sum(axis=1)',
       'n_coord = np.where(same, s_n, 0.0).sum(axis=1) + 1.0')], []),
    ("A10", "coord_switch radii hard-coded at their pinned values",
     [('rn_on = float(params["coord_switch"]["r_on"])\n    rn_off = float(params["coord_switch"]["r_off"])',
       'rn_on = 1.2\n    rn_off = 1.6')], []),
    ("A11", "alpha sign flipped consistently: omega INCREASING in n",
     [('z = alpha * (n_coord - n0)', 'z = -alpha * (n_coord - n0)'),
      ('d_omega = -omega0 * alpha / (np.pi * (1.0 + z * z))',
       'd_omega = +omega0 * alpha / (np.pi * (1.0 + z * z))')], []),
    # ---- broken gradient / bookkeeping --------------------------------------
    ("B1", "pair force drops the phi * S' term",
     [('dphi = np.where(off, dlj * s_lj + lj * ds_lj, 0.0)', 'dphi = np.where(off, dlj * s_lj, 0.0)')], []),
    ("B2", "coord_cross suppressed",
     [('f_cross = -np.einsum("j,kj,kjm->km", c_coeff, w, unit)', 'f_cross = np.zeros_like(f_self)')], []),
    ("B3", "whole coordination force attributed to coord_self",
     [('f_self = -c_coeff[:, None] * np.einsum("kj,kjm->km", w, unit)',
       'f_self = -c_coeff[:, None] * np.einsum("kj,kjm->km", w, unit)'
       ' - np.einsum("j,kj,kjm->km", c_coeff, w, unit)'),
      ('f_cross = -np.einsum("j,kj,kjm->km", c_coeff, w, unit)', 'f_cross = np.zeros_like(f_self)')], []),
    ("B4", "chain-rule factor wrong: C_i halved",
     [('c_coeff = mass * xi2 * omega * d_omega', 'c_coeff = 0.5 * mass * xi2 * omega * d_omega')], []),
    ("B5", "stiffness prefactor 1/2 dropped",
     [('e_stiff = 0.5 * float((mass * omega**2 * xi2).sum())',
       'e_stiff = 1.0 * float((mass * omega**2 * xi2).sum())')], []),
    ("B6", "omega0 hard-coded at its pinned value",
     [('omega0 = float(params["omega0"])', 'omega0 = 1.0')], []),
    # ---- a misreading shared by BOTH transcriptions --------------------------
    ("C1", "LJ 12-6 -> 9-6 in the kernel AND in the naive cross-check",
     [('    sr6 = (sig / r) ** 6\n    sr12 = sr6 * sr6\n    lj = 4.0 * eps * (sr12 - sr6)\n'
       '    dlj = 4.0 * eps * (-12.0 * sr12 + 6.0 * sr6) / r',
       '    sr6 = (sig / r) ** 6\n    sr12 = (sig / r) ** 9\n    lj = 4.0 * eps * (sr12 - sr6)\n'
       '    dlj = 4.0 * eps * (-9.0 * sr12 + 6.0 * sr6) / r')],
     [('4.0 * e * ((g / r) ** 12 - (g / r) ** 6) * s', '4.0 * e * ((g / r) ** 9 - (g / r) ** 6) * s'),
      ('lj = 4.0 * e * ((g / r) ** 12 - (g / r) ** 6)', 'lj = 4.0 * e * ((g / r) ** 9 - (g / r) ** 6)'),
      ('dlj = 4.0 * e * (-12.0 * g**12 / r**13 + 6.0 * g**6 / r**7)',
       'dlj = 4.0 * e * (-9.0 * g**9 / r**10 + 6.0 * g**6 / r**7)')]),
]

STUB_REPORT = ("## Implementation\n## Certification battery\n## Error-class mapping\n"
               "## What no check catches\n## Underdetermined in the specification\n")


def patch(text: str, edits, tag: str) -> str:
    for old, new in edits:
        if old not in text:
            raise SystemExit(f"mutant {tag}: pattern not found:\n{old[:80]}")
        text = text.replace(old, new)
    return text


def run_mutant(tag, desc, kedits, eedits, tmp: Path):
    d = tmp / tag
    d.mkdir()
    (d / "config").symlink_to(HERE / "config")
    (d / "report.md").write_text(STUB_REPORT)
    (d / "force_kernel.py").write_text(patch((HERE / "force_kernel.py").read_text(), kedits, tag))
    if eedits:
        (d / "extra_checks.py").write_text(patch((HERE / "extra_checks.py").read_text(), eedits, tag))
    else:
        shutil.copy(HERE / "extra_checks.py", d / "extra_checks.py")

    v = subprocess.run([PY, str(HERE / "verify.py"), "--impl", str(d / "force_kernel.py"),
                        "--report", str(d / "report.md")], capture_output=True, text=True)
    vfail = re.findall(r"\[FAIL\] check (\S+)", v.stdout)
    vres = "pass" if v.returncode == 0 else "check " + ", ".join(vfail)

    e = subprocess.run([PY, str(d / "extra_checks.py")], capture_output=True, text=True, cwd=d)
    sec, efail = None, []
    for line in e.stdout.splitlines():
        if len(line) > 1 and line[0].isupper() and line[1] == " ":
            sec = line[0]
        if "[FAIL]" in line and sec:
            efail.append(sec)
    eres = "pass" if e.returncode == 0 else ", ".join(sorted(set(efail)))
    return vres, eres


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="w1-01-mut-"))
    print(f"w1-01 mutation study   (mutants in {tmp})\n")
    print(f"  {'id':<5s} {'mutation':<58s} {'verify.py':<20s} extra_checks")
    print("  " + "-" * 104)
    for tag, desc, kedits, eedits in MUTANTS:
        vres, eres = run_mutant(tag, desc, kedits, eedits, tmp)
        print(f"  {tag:<5s} {desc:<58s} {vres:<20s} {eres}")
    print(f"\n  (temporary tree left at {tmp} for inspection; delete when done)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
