# w1-01 — provenance, licences, checksums

Built 2026-09-04, default Claude Code, in this repository. This session is a
build session: it did not create or run the graded session, and is not submitted.

## Directory contract

The repository README pins `spec.md`, `verify.py`, `notes.md`, `env.yaml` and
`inputs/`. This task follows the w1-01 proposal's builder brief instead for the
data directory, which names `config/` and `refs/`. `config/` is this task's
`inputs/`; `refs/` carries citations and holds no data.

```
spec.md                the specification, the interface contract and the criteria
prompt.txt             the session's first user turn, verbatim
verify.py              the verifier; imports force_kernel.py and runs every check
env.yaml               pinned environment, uv-managed, CPU only
notes.md               this file
refs/README.md         DOIs, licences and the fetch note; no PDFs
config/params.json     model parameters
config/config.json     the pinned configuration
config/make_config.py  generator for config.json, seed pinned
```

The graded session adds `force_kernel.py` and `report.md`.

## Literature, per source, licence read at build time

| source | DOI | licence | how used |
|---|---|---|---|
| Gobbo, Ballone & Garabato, *J. Phys. Chem. B* 2020, 124(41), 9267–9274 | 10.1021/acs.jpcb.0c07575 | ACS, not open access | cited by DOI only; never bundled, never quoted |
| Forero-Martinez, Cortes-Huerto, Benedetto & Ballone, *Molecules* 2022, 27, 1647 | 10.3390/molecules27051647 | CC BY 4.0 | equations restated with attribution in `spec.md`; not bundled |

Retrieved 2026-09-04. Full fetch note in `refs/README.md`.

A published equation is restated with attribution. No paper text, figure or table
is reproduced. This is the rule the pool records for literature under the §5a
data rule, applied here per source.

## What the specification owes to this task rather than to the papers

Recorded so that the sufficiency question stays attributable. Pinned here, not
taken from either source:

- the quintic C^2 switching function, used for both the pair cutoff and the
  homo-coordination count;
- the cutoff radii `lj_switch` and `coord_switch`;
- every parameter value in `config/params.json`, including `alpha`, `n0`,
  `omega0` and `m_osc`;
- the reduced-unit convention;
- the free-cluster (non-periodic) boundary;
- the treatment of `xi` as fixed inputs rather than propagated degrees of
  freedom;
- the three-way `pair` / `coord_self` / `coord_cross` force decomposition.

`spec.md` states all of this in its `## Method` section, under "Sources, and what
is and is not taken from them".

## Configuration provenance

`config/config.json` was produced by `config/make_config.py`, seed `20260904`,
`numpy.random.default_rng` (PCG64), numpy 2.3.3, CPython 3.13.12. Re-running the
generator reproduces the file. Nothing in the configuration comes from any
private tree, any prior run or any collaborator's data: it is generated at build
time by the script that ships with the task.

- 200 particles, 100 A and 100 B, free cluster in vacuum, no periodic boundaries.
- Rejection-sampled inside a sphere of radius 3.9 `sig_AA` with a minimum pair
  separation of 0.95 `sig_AA`; recentred on the centroid.
- 3 oscillator displacements per particle, Gaussian, width 0.30 `sig_AA`.
- `n0` is pinned so that `omega(n)` is exercised across its responsive range on
  this configuration rather than sitting on a plateau, and the coordination
  switching window is pinned so that a substantial number of homo-pairs falls
  inside it. Both were chosen by measurement at build time; the measurements are
  not recorded here, because a coordination-number distribution computed from
  the specification is a partial answer to what the session is asked to compute.

The generator takes no options: it writes `config.json` and nothing else. It
carries no statistics helper and no switching function, and it never reads
`params.json`. An earlier draft printed configuration statistics, which put a
working homo-coordination calculation inside the task directory; that code was
removed, and `config.json` is byte-identical either way.

## Checksums

SHA-256, at build time:

```
231fed4207512c6037f2feae719e5d11f2545f0331a70d51d70ca506153654e9  config/config.json
ac2fd4c47728103a31627887042cb33ff8b5e213e71ebe2e2d2943e8a66596c3  config/params.json
38735ca4224ed0d8cbf763aeb815ef7b62563e3fcdae6f7de502d090e9b7b334  config/make_config.py
```

`verify.py` is deliberately not checksummed here: the session may read it, and a
checksum in the task directory invites the impression that it is an input to be
matched rather than a fixed verifier.

## Verifier dry-run

`verify.py` was exercised before the task was considered built, against one
correct reference implementation and four deliberately wrong ones: it passed the
first and rejected all four, each on the checks it was built to fail.

**The dry-run material is not in this repository and must never be added to it.**
A correct implementation of the force law is a correct version of the artifact
the session is asked to produce, so it lives in the WSL-private store, beside the
method audit and the release-prep record. Index form here, substance there —
the standing pointer-not-copy rule.

Nothing from that run — no residual, no energy, no force scale, no measured
tolerance margin — is recorded in this directory. The thresholds in `spec.md` and
`verify.py` are criteria, fixed in advance; the numbers the reference produced
against them are not, and quoting them here would put a measured answer inside
the task.

## Not in this directory, deliberately

- Any residual magnitude measured during the private method audit. Those are
  evidence that the checks discriminate; they are not a key, and quoting them
  would make this an exam question with a known answer.
- Any reference energy, force or trajectory for the pinned configuration. None
  exists, and the verifier needs none.
- Any legacy implementation, harness or output.
