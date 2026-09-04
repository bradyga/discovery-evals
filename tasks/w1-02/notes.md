# w1-02 — provenance, licences, checksums

Built 2026-09-04, default Claude Code, in this repository. This session is a
build session: it did not create or run the graded session, and is not submitted.

## Directory contract

This task follows the contract in the repository README exactly. It amends
nothing.

```
spec.md                the specification, the criterion, the interface contract
prompt.txt             the session's first user turn, verbatim
verify.py              the verifier; imports forms.py and runs every check
env.yaml               pinned environment, uv-managed, CPU only
notes.md               this file
refs/README.md         DOIs, licences and fetch notes; no PDFs
config/params.json     model and criterion parameters
config/config.json     the pinned configuration and the demixing maps
config/make_config.py  generator for config.json, seed pinned
```

The graded session adds `forms.py` and `report.md`.

## What this task is, and what it is not

The comparison is **methodological**: two published functional forms for a
bounded, monotonically decreasing dependence of a frequency on a coordination
number, put on a common footing and compared under a criterion stated in
advance. Nothing in this task asserts, or requires, that either form is or is not
the one used in any particular implementation of any particular model, and
`spec.md` says so in its `## Method` section. The framing is deliberate: the
question of which law a specific implementation uses is not this task's, and is
not posed here.

## Relationship to w1-01

`tasks/w1-01` certifies an implementation of the arctan form. Every check in its
battery is invariant to which frequency law was chosen — the momentum sum rule,
the force-energy consistency, the rotational covariance and the `alpha -> 0`
limit all hold identically for either form — so `w1-01` cannot have settled, and
did not settle, the choice between them. Its battery is reused here as a **gate
on each form**, and `spec.md` states that relationship explicitly rather than
leaving it to be inferred.

## Literature, per source, licence read at build time

| source | DOI | licence | body obtained? | how used |
|---|---|---|---|---|
| Gobbo, Ballone & Garabato, *J. Phys. Chem. B* 2020, 124(41), 9267–9274 | 10.1021/acs.jpcb.0c07575 | ACS, not open access | **no** — HTTP 403; abstract and record only | cited by DOI as the model's source; nothing in `spec.md` taken from it |
| Forero-Martinez, Cortes-Huerto, Benedetto & Ballone, *Molecules* 2022, 27, 1647 | 10.3390/molecules27051647 | CC BY 4.0 | yes — PMC8912101, HTML | Eq. 17 restated with attribution in `spec.md`; not bundled |
| Iannuzzi, Laio & Parrinello, *Phys. Rev. Lett.* 2003, 90, 238302 | 10.1103/PhysRevLett.90.238302 | APS, not open access | **no** — record only | the rational switching function, restated in `spec.md` and adopted there as a law in `n`; cited by DOI |

Retrieved 2026-09-04. Full fetch notes, including the two provenance limits, in
`refs/README.md`.

A published equation is restated with attribution. No paper text, figure or table
is reproduced. This is the rule the pool records for literature under the §5a
data rule, applied here per source.

**The provenance limit on the rational form is recorded rather than glossed.**
Its primary source was not read at build time, so `spec.md` states the form
explicitly and pins the exponent pair, the substitution of `n` for a distance,
and the calibration as this task's own. The specification under test is the one
in `spec.md`; the citation is not load-bearing.

## What the specification owes to this task rather than to any source

Recorded so that any gap the session finds stays attributable.

- the quintic C^2 switching function, used for both the pair cutoff and the
  homo-coordination count;
- the cutoff radii `lj_switch` and `coord_switch`;
- every parameter value in `config/params.json`, including `alpha`, `n0`,
  `omega0`, `m_osc` and the whole `criterion` block;
- the calibration `p = 4 n0 alpha / pi`, and the decision to calibrate on value
  and slope at `n0` rather than on value alone;
- the reduced-unit convention and the free-cluster boundary;
- the treatment of `xi` as fixed inputs rather than propagated degrees of freedom;
- the three-way `pair` / `coord_self` / `coord_cross` force decomposition;
- the two demixing maps;
- the criterion in full — the decision statistic, the parameter band, the margin
  threshold, and the `undecided` verdict.

## Configuration provenance

`config/config.json` was produced by `config/make_config.py`, seed `20260905`,
`numpy.random.default_rng` (PCG64), numpy 2.3.3, CPython 3.13.15. Re-running the
generator reproduces the file. Nothing in the configuration comes from any
private tree, any prior run or any collaborator's data: it is generated at build
time by the script that ships with the task.

- 240 particles, 120 A and 120 B, free cluster in vacuum, no periodic boundaries.
- Rejection-sampled inside a sphere of radius 4.1 `sig_AA` with a minimum pair
  separation of 0.95 `sig_AA`; recentred on the centroid.
- 3 oscillator displacements per particle, Gaussian, width 0.30 `sig_AA`.
- Two `demix_maps`, `plane_x` and `radial`: alternative species labellings of the
  same positions, produced by sorting on a coordinate and on distance from the
  centroid. Position sorts and nothing else.
- The cluster radius and the coordination switching window were chosen at build
  time so that the homo-coordination spans a wide range straddling `n0` rather
  than sitting on a plateau, and so that the two maps separate the species
  differently. Both were settled by measurement; **the measurements are not
  recorded here**, because a coordination-number distribution computed from the
  specification is a partial answer to what the session is asked to compute.

The generator takes no options: it writes `config.json` and nothing else. It
carries no switching function, no coordination calculation, no frequency law and
no energy; it never reads `params.json`.

## Checksums

SHA-256, at build time:

```
53208919e877196aa8ff1b92f62454fde33594c0d8a5f8aa655822355829975a  config/config.json
e6d6cb8661a92c784ff4300b37984695624d66dadef20358ec83e51f4f3beeeb  config/params.json
2c8417584dcb03a30e7225513fd0cd8b41a69bb5a8afc9d655510be96453d91a  config/make_config.py
```

`verify.py` is deliberately not checksummed here: the session may read it, and a
checksum in the task directory invites the impression that it is an input to be
matched rather than a fixed verifier.

## Verifier dry-run

`verify.py` was exercised before the task was considered built, against one
correct reference implementation and **sixteen** deliberately wrong ones, one per
named failure mode enumerated in the adversarial pass. It passed the first and
rejected all sixteen, each on the check it was built to fail. No mutant survived.

Two of them are worth recording as design evidence rather than as bookkeeping,
because each demonstrates a loophole rather than arguing for one:

- an implementation whose `compare()` returns the **correct** numbers as literals
  is internally consistent and matches the verifier's recomputation exactly; it is
  caught only by requiring those numbers to move when `alpha` and `n0` change;
- an implementation that codes the second form as a **logistic** sigmoid has the
  same value and the same calibrated slope at `n0`, is a genuinely different
  function from the first, and passes every other check; it is caught only by the
  paired reflection and inversion identities, because a logistic is
  reflection-symmetric about `n0` exactly as the arctan form is.

`verify.py` was edited after the first dry-run — check 4 is now recorded as not
run when check 2 fails, and a degenerate `R` reports why instead of reporting a
non-finite number. **The entire dry-run was regenerated after that edit**, both
halves, per the rule that any change to the verifier expires its evidence.

**The dry-run material is not in this repository and must never be added to it.**
A correct implementation of both forms is a correct version of the artifact the
session is asked to produce, so it lives in the WSL-private store beside the
w1-01 dry-run material. Index form here, substance there — the standing
pointer-not-copy rule.

Nothing from that run is recorded in this directory: no entropy change, no `R`,
no margin, no residual, no energy, no force scale, no tolerance margin, and **no
record of which form the criterion selects**. The thresholds in `spec.md` and
`verify.py` are criteria, fixed in advance; the numbers the reference produced
against them are not.

## Not in this directory, deliberately

- **Which form the criterion prefers.** `verify.py` applies the rule in `spec.md`
  to the numbers the session's own module returns. It has no stored verdict, and
  `undecided` is a passing verdict when the numbers give it.
- Any measured `dS`, `R`, margin, energy, force or residual for the pinned
  configuration. None is needed by the verifier and none exists here.
- Any reference implementation of either form.
- Any legacy implementation, harness or output, from any private tree.
