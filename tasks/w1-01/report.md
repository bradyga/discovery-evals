# w1-01 — certification report

Implementation of the force law in `spec.md § Method`, and the argument for
whether it is correct. There is no reference implementation and no reference
value anywhere in this argument: every criterion below is an identity, an
invariance, a limit, or a comparison between two transcriptions of the same
specification.

**Status.** `python verify.py` exits 0, all six checks PASS (checks 0–4 physics,
check 5 report structure). `python extra_checks.py` exits 0, all eight extra
checks A–H pass. `python mutation_study.py` reproduces the error-class mapping
below. Environment: the pinned one from `env.yaml` (CPython 3.13.15, NumPy
2.3.3, created with `uv venv --python 3.13 .venv` and
`uv pip install --python .venv/bin/python numpy==2.3.3`).

---

## Implementation

`force_kernel.py`. Pure NumPy, O(N²) dense, no files, no argv, no network, no
import side effects. `N` comes from `positions.shape[0]` and `l` from
`xi.shape[1]`; every parameter is read from `params`.

**Energy.** With `r_ij = |r_i − r_j|` (free cluster: full differences, no box,
no minimum image),

```
U = sum_{i<j} 4 eps_ab [(sig_ab/r)^12 − (sig_ab/r)^6] S(r; rc_on, rc_off)
  + (1/2) sum_i m_{s_i} omega(n_i)^2 sum_g xi[i,g]^2

n_i      = sum_{j != i, s_j = s_i} S(r_ij; rn_on, rn_off)
omega(n) = omega0 [1/2 − (1/pi) arctan(alpha (n − n0))]
```

The pair sum is evaluated as `0.5 * sum_{i != j}`, identical to `sum_{i<j}`.

**Switching function.** `S` is computed branch-free as the quintic in
`x = clip((r − a)/(b − a), 0, 1)`. The quintic is exactly 1 at `x = 0` and
exactly 0 at `x = 1`, and its first two derivatives vanish at both endpoints, so
clipping reproduces the three stated branches and stays C² across `r = a` and
`r = b`. `dS/dr = −30 x²(1 − x)² / (b − a)`, which is likewise exactly zero
outside the window. `r_on >= r_off` raises `ValueError`.

**Forces.** Let `u_ij = (r_i − r_j)/r_ij`, and

```
C_i  = dU_stiff/dn_i = m_{s_i} (sum_g xi[i,g]^2) omega(n_i) omega'(n_i)
omega'(n) = − omega0 alpha / (pi (1 + alpha^2 (n − n0)^2))
W_kj = S'(r_kj; rn_on, rn_off)  if s_k == s_j and k != j, else 0   (symmetric)
```

Then

```
pair_k        = − sum_{j != k} phi'(r_kj) u_kj
coord_self_k  = − C_k sum_j W_kj u_kj                 (from k's own n_k)
coord_cross_k = − sum_j C_j W_kj u_kj                 (from every other n_i)
```

`coord_cross` follows from the definition `−sum_{i != k} C_i ∂n_i/∂r_k`: only
`i` with `s_i = s_k` contribute, through the single term `S(r_ik)` in `n_i`,
whose derivative is `S'(r_ik) u_ki = −S'(r_ik) u_ik`. Using `W` symmetric and
relabelling gives the form above. Their sum,
`−sum_j (C_k + C_j) W_kj u_kj`, is pairwise antisymmetric and therefore
conserves momentum, while neither half does on its own — as `spec.md` requires
and check 1 enforces.

`energy_and_forces` returns `pair + coord_self + coord_cross`, so the term sum
rule of check 0 holds bitwise (`0.000000e+00`).

**Result on the pinned configuration.** `U = −1.281713454e+02` (reduced),
`F_scale = mean_k |F_k| = 7.469672e+01`, homo-coordination `n_i ∈ [0.000, 6.964]`
with mean `3.000`, recovered `omega_i ∈ [0.097, 0.874]` against `omega0 = 1`.

---

## Certification battery

Two batteries were run. Measured values are from the runs reported here.

### `verify.py` — checks 0–4 (all PASS)

| check | what it measured | result |
|---|---|---|
| 0 contract | shapes, finiteness, no input mutation, repeat-call identity, `sum(terms) − F`, energy response to `epsilon`, `omega0`, `m_osc` | terms sum `0.0`; all responses `True` |
| 1 translation | `|Σ F|/Σ|F|` for total and `pair`; magnitude and net force of each coordination half; cancellation of the two halves; explicit rigid shift | total `2.4e−17`, pair `5.8e−17`; `coord_self` mag `1.44e−3` net `9.30e−2`, `coord_cross` mag `1.70e−3` net `7.87e−2`; cancellation `5.8e−16`; shift `dE` `5.1e−15`, `dF` `2.8e−14` |
| 2 gradient | central FD of the implementation's *own* energy at 12 (particle, axis) pairs over a six-step ladder | worst best-step residual `1.03e−8`; worst observed order `1.994` |
| 3 rotation | `E(Qr) − E(r)`; `F(Qr) − Q F(r)` for the total and each named term | `dE` `7.3e−15`; `dF` `1.9e−14`; terms `≤ 3.2e−17` |
| 4 limits | coordination halves at `xi = 0` and at `alpha = 0`; `F = F_pair` at `xi = 0`; alpha-independence at `xi = 0`; the closed-form stiffness energy at `alpha = 0` across three `(omega0, m_osc)` settings; non-degeneracy of the pinned `alpha` | all vanishing terms exactly `0.0`; closed form matches to `≤ 1.1e−15` at all three settings |

### `extra_checks.py` — checks A–H (all pass)

These target what checks 0–4 leave open.

| check | what it measured | result |
|---|---|---|
| A independent re-implementation | a naive O(N²) Python-loop kernel written from `spec.md` by a different route — explicit `if` branches in `S` instead of clipping, and the coordination derivative taken from the *definition* `−Σ_i C_i ∂n_i/∂r_k` rather than from the pairwise identity — compared term by term | `dE` `2.0e−15`; `pair` `1.1e−14`; `coord_self` `2.2e−17`; `coord_cross` `1.9e−17` |
| B complex-step gradient | `d/dx` of the *naive* energy by a complex step `h = 1e−30` (exact to roundoff, no subtractive cancellation), against the vectorised kernel's forces, 6 samples | worst `1.2e−15` — nine orders tighter than check 2, and across two implementations |
| C split attribution | `coord_self` against `−C_k ∂n_k/∂r_k` and `coord_cross` against `−Σ_{i≠k} C_i ∂n_i/∂r_k`, with `∂n_i/∂r_k` obtained numerically, at 5 particles × 3 axes | `2.7e−13` / `2.7e−13` |
| D scale covariance | `r`, `sigma`, and *both* switch windows scaled by `λ = 1.7`: `U` invariant, `F → F/λ` | `3.3e−15` / `1.2e−14` |
| E strict locality | a particle placed beyond both cutoffs: forces on the others, and on it | all exactly `0.0` |
| F parity | `r → −r` (improper; `verify.py` tests only proper rotations): `U` invariant, `F → −F`, per term | all exactly `0.0` |
| G particle-order covariance | random relabelling: `U` invariant, `F` permutes, per term | `0.0` / `1.9e−15` |
| H `omega(n)` from the kernel | `omega_i` read *out of the kernel* by isolating one particle's `xi` (`U(xi_iso) − U(0) = ½ m_i omega_i² Σ_g xi²`, the pair energy cancelling), then tested against the spec's stated properties: `omega_i ∈ (0, omega0)`, strictly decreasing in `n_i`, single-valued in `n_i` | in range `True`; 0 monotonicity violations over 200 particles; ties agree to `1.9e−12`; spread `0.777` |

### `mutation_study.py` — 18 deliberately wrong kernels

Each mutant is a patched copy of `force_kernel.py` in a temporary directory (the
task tree is never written to), run against both batteries. `C1` additionally
patches `extra_checks.py` with the *same* error, to model a misreading shared by
both transcriptions. The table it produces is the evidence for the next section.

---

## Error-class mapping

Read column 3 as *what the check does not rule out*, not as a weakness.

### Check 0 — interface contract

*Catches:* wrong shapes/types, non-finite output, in-place mutation of the
caller's arrays, non-determinism, a term decomposition that does not add up to
the returned force, identically zero forces, and hard-coding of the three
parameters it perturbs. Confirmed: **B6** (`omega0` hard-coded) is caught here
and *nowhere else* — no other check in either battery perturbs `omega0`.

*Does not rule out:* hard-coding of any parameter it does not perturb. It
perturbs `epsilon`, `omega0`, `m_osc` only. **A1** (wrong `n0`) and **A10**
(`coord_switch` radii hard-coded) pass check 0 outright. It tests no physics
whatever.

### Check 1 — translational invariance and the term sum rule

*Catches:* loss of pairwise antisymmetry in the pair force (Newton's third law);
suppression of a coordination half-term (**B2**, which trips both the magnitude
floor and the cancellation test); mis-attribution of the whole symmetric
coordination force to one half (**B3**, which trips the `coord_cross` magnitude
floor); leakage of absolute coordinates into the energy, via the explicit rigid
shift.

*Does not rule out:* that the force is the gradient of the energy. This is worth
stating flatly, because it is the natural over-claim: **any** central pair force
`f(r_ij) u_ij` with **any** radial function `f` sums to zero exactly, whether or
not `f` is `−φ'` for the `φ` in the returned energy. **A4** (LJ prefactor 4→1)
passes check 1 at `5.8e−17`. Check 1 also says nothing about the radial form,
about `n_i`, or about the energy at all — it never looks at `U`.

### Check 2 — force–energy consistency by finite difference

*Catches:* every inconsistency between the returned force and the returned
energy. Confirmed: **B1** (dropped `φ·S'` term), **B4** (chain-rule factor: `C_i`
halved), **B5** (stiffness prefactor), **B2** (a suppressed half-term is also a
gradient error). The convergence-order sub-test is what does the work in several
of these: the best-step residual alone can be small for the wrong reason.

*Does not rule out:* that `U` is the specified `U`. Check 2 is a *relative*
statement — it ties `F` to whatever energy the module returns. Every mutant in
the `A` family that changed `U` self-consistently (**A1, A2, A3, A4, A6, A8, A9,
A10, A11**) passes check 2. It also does not test the term decomposition:
`verify.py` finite-differences only the total force, so a wrong `self`/`cross`
split that still sums correctly is invisible to it (that gap is closed by
extra check C, not by check 2).

One catch here is **contingent, not designed**: **A7** (a C¹ cubic switch instead
of the C² quintic) fails check 2 at observed order `1.27 < 1.5`. That happens
because one of the twelve seed-pinned FD samples sits near a switching knot, so
the `h = 1e−2` stencil straddles the discontinuity in `S''`. With a different
pinned seed the same mutant could pass. Check 2 should not be described as a
smoothness test.

### Check 3 — rotational invariance and covariance

*Catches:* explicit coordinate-frame dependence — a stray fixed axis, a
transposed rotation applied internally, a component mix-up in the force
assembly, a term accumulated in a different frame from the others.

*Does not rule out:* almost anything else, and it is close to **vacuous for this
implementation**. Every quantity in `force_kernel.py` is built from `|r_i − r_j|`
and the unit vectors `u_ij`, so rotational covariance is structural rather than
earned; the check can only fail if a bug *introduces* frame dependence. No
mutant in the study — including the ones that get the physics badly wrong —
fails check 3. Extra check F (parity) has the same character: it is the improper
counterpart, and equally structural here.

### Check 4 — limiting cases

*Catches:* the stiffness normalisation and its `(omega0/2)²` factor (**B5**); any
coordination force that fails to vanish with `xi` or with `alpha`; hard-coding of
`omega0` or `m_osc` in the stiffness term, through the three-setting sweep; and
the species → `m_osc` mapping (**A5**, species labels swapped, is caught here and
only here in `verify.py`, because the verifier evaluates the closed form using
its *own* `species` array).

*Does not rule out:* anything about `U_pair`. Both limits difference two energies
at identical positions, so the pair term cancels out of every check-4 identity;
**A4** passes. It does not constrain the shape of `omega(n)` away from `alpha = 0`
— **A3** (tanh for arctan) passes, because both forms reduce to `omega0/2` at
`alpha = 0` and both are smooth and monotone. And it does not constrain `n_i`:
**A1, A6, A9, A10** all pass. It pins the species mapping only through `m_osc`,
not through `epsilon`/`sigma`: **A8** (species swapped in the `eps`/`sigma`
lookup only) passes every check in `verify.py`.

### Extra checks A–H

*A (independent re-implementation) catches* transcription and vectorisation
slips — a wrong `einsum` index, a broadcast over the wrong axis, a sign, a
mis-typed constant. It caught every mutant in the study except **A10** and
**B6**. *Does not rule out* a misreading shared by both transcriptions, which is
the whole of the next section: I wrote both, from the same reading.

*B (complex step) catches* the same gradient errors as check 2 but at `1e−15`
instead of `1e−8`, and against a *different* implementation's energy, so it also
cross-validates A. It has no cancellation error and no step-size choice. *Does
not rule out* a wrong `U`, for exactly the reason check 2 does not.

*C (split attribution) catches* a wrong `coord_self`/`coord_cross` split that
still sums correctly — the one thing `verify.py` cannot see, since it
finite-differences only the total. It caught **B3** and **B4**. *Does not rule
out* a wrong `n_i`: it differentiates the same `n_i` the kernel uses.

*D (scale covariance) catches* a length-dimensioned parameter that is hard-coded
or enters at the wrong power. It is the only check in either battery that catches
**A10** (`coord_switch` radii hard-coded), because it is the only one that
perturbs those radii. *Does not rule out* dimensionless errors — `alpha`, `n0`,
the LJ prefactor all survive it.

*E (locality) catches* leakage past the cutoff: an accidental minimum-image
convention, a wrong branch in `S`, a neighbour list that keeps out-of-range
pairs. *Does not rule out* anything about the in-range functional form.

*F (parity)* and *G (order covariance)*: F catches chirality-dependent bugs (a
cross product, a signed volume); G catches order-dependent accumulation and
indexing tied to particle order. Both are structural here and caught no mutant.

*H (`omega` recovered from the kernel) catches* violations of the properties
`spec.md` states but `verify.py` never tests: `omega ∈ (0, omega0)`, strictly
decreasing, single-valued in `n`. It caught **A2** (swapped windows: `omega` is
then not a function of the homo-coordination at all), **A6** (all-neighbour
count, same reason), **A11** (`omega` increasing in `n` — a mutant that passes
every single check in `verify.py`), and **B5** (recovered `omega > omega0`).
*Does not rule out* any `omega(n)` that is decreasing and single-valued in the
same `n`: **A3** (tanh), **A1** (wrong `n0`) and **A9** (off-by-one in the count)
all pass H.

---

## What no check catches

Three concrete classes. The first is demonstrated, not asserted.

**1. A wrong pair potential that is still a smooth, species-symmetric,
compactly-supported function of `r_ij`, when the error is present in both
transcriptions.** Demonstrated as mutant **C1**: replacing the 12-6
Lennard-Jones with a 9-6 form (`4 eps [(sig/r)^9 − (sig/r)^6] S(r)`, with the
matching analytic derivative) in `force_kernel.py` *and* in the naive
cross-check inside `extra_checks.py`. `verify.py` reports `RESULT: PASS — all
checks satisfied`, and all of A–H pass. The mechanism is that no criterion in
either battery compares any quantity against a number that does not come from
the implementation: check 2 and check B tie `F` to whatever `U` is returned;
checks 1, 3, D, E, F, G are invariances that any central pair potential
satisfies; check 4's two limits difference energies at fixed positions, so
`U_pair` cancels out of both; and check A compares against a second
transcription that carries the same error. The battery certifies *`F = −∇U`, and
`U` has the right symmetries and degenerate limits* — not *`U` is the specified
`U`*. Closing this would require a reference number, a symbolic differentiation
of the published equation by a tool that parsed it independently, or a second
reader.

**2. Errors in regions of parameter or configuration space that neither the
pinned inputs nor any perturbation visits.** Concretely: the index order of the
`epsilon`/`sigma` matrices (`eps[s_i][s_j]` vs `eps[s_j][s_i]`) is unobservable,
because both matrices in `config/params.json` are symmetric and nothing supplies
a non-symmetric one — check 0's `epsilon` perturbation is a uniform 1.5×
scaling, which preserves symmetry. Likewise: a branch that is wrong only for
`alpha < 0`, only for `r_on >= r_off`, only for `l != 3`, or only for `N = 1` is
never executed; and the absence of a coincident-particle guard is never exposed,
because `make_config.py` enforces a `0.95 sig_AA` minimum separation, so `r → 0`
never occurs. Every check in both batteries is evaluated on this one
configuration and small perturbations of it.

**3. A common offset or scaling inside `n_i` present in both transcriptions.**
The single mutant **A9** (`n_i` counting the particle itself) is caught only
because it lives in one transcription and check A sees the disagreement; written
into both, it would shift every particle along the *same* monotone `omega(n)`
curve, so check H would still see a single-valued strictly decreasing `omega`,
check C would differentiate the shifted `n_i` consistently, and every identity,
invariance and limit would hold exactly. The same is true of a shared
misreading of the coordination window as `r_on sig_ab .. r_off sig_ab` rather
than as absolute radii (see the first entry of the next section): check D would
only reject it because I wrote D to scale `coord_switch` by `λ`, which is itself
a consequence of my reading, so D adjudicates my own convention rather than the
specification.

---

## Underdetermined in the specification

Everything below is a place where `spec.md § Method` plus `config/params.json`
admits more than one implementation. In each case the convention adopted is
stated; none was resolved silently.

**1. Whether the switch windows are absolute radii or multiples of `sig_ab`.**
`params` gives one `(r_on, r_off)` pair for `lj_switch` and one for
`coord_switch`, and the spec writes `S(r; rc_on, rc_off)` with no species
subscript. *Adopted:* absolute radii in `sig_AA`, identical for every species
pair. The common alternative in a binary mixture — `rc^{ab} = rc · sig_ab`, so
that A–A, A–B and B–B are cut at 2.2/2.31/2.42 `sig_AA` rather than all at 2.2 —
is equally consistent with the text and with the parameter file. Nothing in
either battery distinguishes them (see uncaught class 3).

**2. The index order of the `epsilon` and `sigma` matrices.** The spec says
"`2 × 2` matrix, indexed by species code" and does not fix whether the entry is
`M[s_i][s_j]` or `M[s_j][s_i]`, nor whether an implementation should symmetrise.
*Adopted:* `M[s_i][s_j]`, used as given, no symmetrisation. Both pinned matrices
are symmetric, so on this configuration the choice is unobservable.

**3. Whether `alpha < 0` is admissible.** The spec asserts that `omega(n)` is
"strictly decreasing in `n` and lies in `(0, omega0)`". With
`omega = omega0 [1/2 − arctan(alpha(n − n0))/pi]` that assertion holds iff
`alpha > 0`; the pinned `alpha = 0.8` satisfies it, and `alpha = 0` (used by
check 4) is the boundary case where `omega` is constant. *Adopted:* `alpha` is
used exactly as supplied, with no sign check and no absolute value, so a negative
`alpha` would silently produce an increasing `omega(n)`. The spec does not say
whether that should be an error, and nothing in `verify.py` would reject it
(mutant **A11**).

**4. No cutoff energy shift.** The spec's `phi_ab` is switched to zero by `S` and
nothing is subtracted. *Adopted:* literally — no shift, no tail correction. Worth
recording because switched-and-shifted Lennard-Jones is the more usual pairing in
practice, and the spec's silence on a shift is what settles it.

**5. `params["n_osc"]` versus `xi.shape[1]`.** The interface contract says `l` is
read from the argument shapes; `params` also carries `n_osc = 3`. *Adopted:*
`l = xi.shape[1]`; `params["n_osc"]` is never read. The spec fixes no precedence
if the two disagreed.

**6. Behaviour on inputs the configuration never produces.** *Adopted:*
`r_on >= r_off` raises `ValueError` (the switch would divide by zero or invert);
`N = 1` and `l = 0` fall out correctly as zero coordination and zero stiffness;
coincident particles (`r_ij = 0`) are **not** guarded and would overflow in the
`(sig/r)^12` term. The spec specifies none of these, and the free-cluster
boundary means there is no box length to regularise against.

**7. The summation form of the pair term.** `sum_{i<j}` is implemented as
`0.5 · sum_{i != j}` over the full distance matrix. Mathematically identical;
they differ in floating-point accumulation order at the `1e−15` level. *Adopted:*
the half-of-full-matrix form, chosen for vectorisation. The naive cross-check in
`extra_checks.py` uses the literal `i < j` loop instead, and the two agree to
`2.0e−15` relative (extra check A).

**8. Which species-pair parameters govern the coordination count.** `n_i` counts
only same-species neighbours, and `params` supplies a single `coord_switch`.
*Adopted:* the same window for A–A and B–B, with `j != i`. The spec does not
raise the possibility of per-species coordination windows and none is supplied,
but it also never states that one window governs both — this is read off the
parameter file, not off the equations.
