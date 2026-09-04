# w1-02 — two functional forms, established non-equivalent, chosen under a stated criterion

Two published functional forms give a bounded, monotonically decreasing
dependence of a harmonic-oscillator frequency on a coordination number. They are
not algebraic reformulations of one another. Establish the difference, and
defend a choice between them under the criterion stated below.

---

## Intent

I want to establish whether two published functional forms for a
coordination-dependent oscillator stiffness are separable in practice — whether,
once they are put on a common footing, a criterion stated in advance picks one of
them on a real configuration, and what the choice rests on.

### What this task does not settle

Recorded so the result is not read as more than it is.

1. **Whether the two forms are separable at all by a criterion that is not
   decidable from the algebra.** The criterion below is one such criterion. That
   it exists does not show the class of them is rich, and a different criterion
   may order the forms differently. Nothing here establishes that one form is
   better *simpliciter*.
2. **Whether the margin survives a sensitivity analysis the session designs
   itself.** The parameter band and the margin threshold below are pinned in
   advance. Whether the ordering holds under a perturbation this specification
   did not anticipate is open, and is the report's job to probe.

---

## Prompt

The session's first user turn is in `prompt.txt`, and is reproduced here verbatim.

```
Read spec.md first. It states two functional forms, the calibration that puts
them on a common footing, and the criterion you are to decide under. The
criterion is fixed: it is not yours to choose, and not yours to change.

Implement both forms behind the interface spec.md names, so that each runs on
the configuration in config/. Establish that the two forms are not equivalent —
not that they differ numerically at the pinned parameters, but that no choice of
parameters makes one into the other.

Then evaluate the criterion and defend a choice. "Defend" means the argument
runs on the numbers your own implementation produces. That one of the two forms
is the more commonly used is not a reason, and a report that leans on it fails
whether or not it names the same form the criterion does. If the criterion does
not decide, say so and say what would.

Where the specification leaves something underdetermined, say so explicitly
rather than picking a convention silently.

verify.py is in this directory. You may run it. Do not edit it.
```

### Assumptions

1. **The functional forms and the calibration in `## Method` below are the only
   specification.** No reference implementation of either form is provided in
   this directory, and no reference value for any quantity exists.
2. **The configuration and the parameters in `config/` are as pinned and are not
   to be changed.** `config/make_config.py` records how `config.json` was
   generated; it is provenance, not an input to be re-run with other settings.
3. **The criterion in `## The criterion` is stated before this session and is
   fixed.** It is not the only defensible criterion; the report is asked to say
   where it is weak, not to substitute another.
4. **Units and conventions are as stated below.** Anything the specification
   leaves underdetermined is to be reported as underdetermined, not silently
   resolved.

### Relationship to `tasks/w1-01`

The arctan form is the subject of sibling task `w1-01`, which certifies an
implementation of it. Reusing that implementation, if it is available to you, is
legitimate and expected: it is a starting point, not the deliverable. What
`w1-01` does not settle is this task's question — the two forms' non-equivalence,
and the choice between them under the criterion below. `w1-01`'s battery is
reused here as a **gate on each form**, not as the decision: every check in it is
invariant to which frequency law was chosen, so passing it bears on whether an
implementation is correct and not at all on which form to prefer.

---

## Method

### The model

`N` particles with positions `r_i` in R^3 and species `s_i` in {A, B}. The system
is a **free cluster in vacuum**: there are no periodic boundary conditions, no
box and no minimum-image convention. Each particle also carries `l` classical
harmonic-oscillator displacements `xi[i, g]`, `g = 1 .. l`.

The potential energy is a pair term plus a coordination-dependent stiffness term:

```
    U(r, xi) = U_pair(r) + U_stiff(r, xi)
```

**Pair term.** With `r_ij = |r_i - r_j|`:

```
    U_pair(r) = sum_{i<j} phi_{s_i s_j}(r_ij)

    phi_ab(r) = 4 eps_ab [ (sig_ab / r)^12 - (sig_ab / r)^6 ] * S(r; rc_on, rc_off)
```

**Homo-coordination.** `n_i` counts, smoothly, the neighbours of particle `i` of
its *own* species:

```
    n_i = sum_{j != i, s_j = s_i} S(r_ij; rn_on, rn_off)
```

**Switching function.** One C^2 quintic switch is used for both the pair cutoff
and the coordination count. With `x = (r - a) / (b - a)`:

```
    S(r; a, b) = 1                              for r <= a
               = 1 - 10 x^3 + 15 x^4 - 6 x^5    for a < r < b
               = 0                              for r >= b
```

**Coordination-dependent stiffness.** The oscillator frequency decreases with
increasing homo-coordination:

```
    U_stiff(r, xi) = (1/2) * sum_i m_{s_i} * omega(n_i)^2 * sum_{g=1..l} xi[i, g]^2
```

`omega(n)` is one of the two forms below. Both are strictly decreasing in `n` and
lie in `(0, omega0)` for `n >= 0`.

### The two forms

**Form `"arctan"`.**

```
    omega_A(n) = omega0 * [ 1/2 - (1/pi) * arctan( alpha * (n - n0) ) ]
```

**Form `"rational"`.** The rational switching function, written in `n` with
reference value `n0` and exponent pair `(p, 2p)`:

```
    omega_R(n) = omega0 * [ 1 - (n/n0)^p ] / [ 1 - (n/n0)^(2p) ]
               = omega0 / [ 1 + (n/n0)^p ]                 for n >= 0
```

**Use the second, closed, expression.** The first is the form in which the
function is published and is given so the reference is legible; as written it is
`0/0` at `n = n0`, and the two are equal everywhere else.

### Calibration — the common footing

Two forms with different parameterizations cannot be compared as written: any
difference found would be a difference of parameter values, not of forms. Both
are therefore pinned to agree in **value and in first derivative** at the
reference coordination `n0`:

```
    omega(n0) = omega0 / 2        and        omega'(n0) = - omega0 * alpha / pi
```

Form `"arctan"` satisfies both for every `alpha`. For form `"rational"` this
fixes the exponent:

```
    p = 4 * n0 * alpha / pi
```

`p` is **derived from `alpha` and `n0` at call time**. It is not a pinned
parameter, it does not appear in `config/params.json`, and it must be recomputed
whenever either is changed — including inside the criterion's parameter sweep,
where `alpha` is varied. Note the consequence at `alpha = 0`: `p = 0`, and *both*
forms reduce to the constant `omega0 / 2`.

### Forces

The force on particle `k` is the negative gradient of `U` with respect to `r_k`
at fixed `xi`:

```
    F_k = - dU / d r_k        (xi held fixed)
```

The oscillator displacements `xi` are part of the pinned configuration. They are
inputs, not degrees of freedom to be propagated; no force on `xi` is required.

### Named force terms

The force decomposes into three named terms, which the interface contract below
requires the implementation to expose separately, for each form:

| term | definition |
|---|---|
| `pair` | `-d U_pair / d r_k` |
| `coord_self` | the part of `-d U_stiff / d r_k` that comes from the dependence of **particle k's own** `n_k` on `r_k` |
| `coord_cross` | the part of `-d U_stiff / d r_k` that comes from the dependence of **other particles'** `n_i`, `i != k`, on `r_k` |

`pair + coord_self + coord_cross` is the total force.

### Units

Reduced Lennard-Jones units throughout: lengths in `sig_AA`, energies in
`eps_AA`, forces in `eps_AA / sig_AA`, oscillator displacements `xi` in `sig_AA`,
oscillator masses in units of the A-particle oscillator mass, `omega0` in
`sqrt(eps_AA / (m sig_AA^2))`. `n_i`, `alpha * n` and `p` are dimensionless.
Entropies are reported in units of `l k_B` and are therefore pure numbers.
Nothing in this task requires a temperature, a timestep or a physical constant.

### Sources, and what is and is not taken from them

**What is being compared is a methodological question, and only that:** two
published functional forms for a bounded, monotonically decreasing dependence of
a frequency on a coordination number, put on a common footing and compared under
a stated criterion. Nothing in this task asserts, or requires, that either form
is or is not the one used in any particular implementation of any particular
model.

- **The model** — a binary Lennard-Jones mixture whose particles carry classical
  harmonic oscillators whose frequency decreases with increasing
  homo-coordination — is that of D. Gobbo, P. Ballone and B. D. Garabato,
  *Coarse-Grained Model of Entropy-Driven Demixing*, *J. Phys. Chem. B* **2020**,
  124(41), 9267–9274. DOI
  [10.1021/acs.jpcb.0c07575](https://doi.org/10.1021/acs.jpcb.0c07575).
  Published by ACS; **cited by DOI only, never redistributed.** Its text was not
  available at build time and nothing in this specification is taken from it.
- **Form `"arctan"`** is Eq. 17 of N. C. Forero-Martinez, R. Cortes-Huerto,
  A. Benedetto and P. Ballone, *Thermoresponsive Ionic Liquid/Water Mixtures:
  From Nanostructuring to Phase Separation*, *Molecules* **2022**, 27, 1647. DOI
  [10.3390/molecules27051647](https://doi.org/10.3390/molecules27051647). CC BY
  4.0. Restated with attribution.
- **Form `"rational"`** is the rational switching function of M. Iannuzzi,
  A. Laio and M. Parrinello, *Efficient Exploration of Reactive Potential Energy
  Surfaces Using Car-Parrinello Molecular Dynamics*, *Phys. Rev. Lett.* **90**,
  238302 (2003). DOI
  [10.1103/PhysRevLett.90.238302](https://doi.org/10.1103/PhysRevLett.90.238302).
  **It is published as a switching function of an interatomic distance, and is
  adopted here as a frequency law in the coordination number `n`**, with the
  exponent pair `(p, 2p)` and the calibration above. That adoption is this task's,
  not the paper's, and is stated rather than implied.

**Pinned by this task rather than taken from any source, and therefore part of
the specification under test:** the quintic switching function `S` and its use
for both the pair cutoff and the coordination count; the cutoff radii; every
parameter value in `config/params.json`; the calibration `p = 4 n0 alpha / pi`;
the reduced-unit convention; the free cluster boundary; the treatment of `xi` as
fixed inputs; the three-way force decomposition; the demixing maps; and the
criterion in full. `refs/README.md` records what was read, when, and under which
licence.

The equations are restated with attribution. No paper text, figure or table is
reproduced in this directory, and no PDF is bundled.

---

## The criterion

**Stated here, before the session, so that the choice cannot be
reverse-engineered from the answer.**

### The decision statistic

For a classical harmonic oscillator the entropy is `s = - l k_B ln(omega) +
const`, so the oscillator entropy of a configuration is fixed by `omega(n_i)`
alone. `config/config.json` carries two `demix_maps`: alternative species
labellings of the **same positions** in which the two species are spatially
separated. Writing `n_i^0` for the homo-coordination under the pinned labelling
and `n_i^m` for the homo-coordination under map `m` — same positions, same
`coord_switch` — the oscillator entropy change on demixing is

```
    dS(form, alpha, m) = - sum_i [ ln omega(n_i^m ; form, alpha)
                                 - ln omega(n_i^0 ; form, alpha) ]
```

in units of `l k_B`. With `alpha_lo = 0.75 * alpha` and `alpha_hi = 1.25 * alpha`
(`criterion.alpha_scale_lo`, `criterion.alpha_scale_hi` in
`config/params.json`), the decision statistic is

```
    R(form, m) = | dS(form, alpha, m) |
                 / | dS(form, alpha_hi, m) - dS(form, alpha_lo, m) |
```

the demixing signal divided by its sensitivity to the model's own tuning
parameter. `p` is recalibrated at each `alpha`; `n0`, `omega0`, the positions,
the maps and the switching windows are held fixed throughout.

### The rule

For each map `m` the form with the **larger** `R` is preferred, with relative
margin

```
    margin(m) = | R(arctan, m) - R(rational, m) | / min( R(arctan, m), R(rational, m) )
```

**The choice is the form preferred on both maps, with `margin(m) >= 0.10`
(`criterion.margin`) on both.** If the two maps prefer different forms, or either
margin falls below the threshold, the verdict is **`undecided`**.

`undecided` is a passing verdict when it is what the numbers say. A verdict of
`arctan` or `rational` that the numbers do not support is not.

### Why this criterion and not a simpler one

Recorded so the criterion is not mistaken for the only one available, and so its
design is auditable.

Two simpler candidates were considered at build time and rejected, both because
they are decidable **from the algebra alone**, with nothing implemented and the
configuration untouched: the value of `omega(0)` relative to `omega0`, and the
fraction of each form's total variation falling inside the sampled coordination
range. A criterion a session can answer without evaluating the model is not a
criterion for choosing between implementations of it.

The statistic above requires the homo-coordination of the pinned configuration
under three labellings, at three parameter settings, for two forms. It cannot be
evaluated without implementing both.

It is not the only defensible criterion. `## The choice, defended` in the report
is asked to say where it is weak.

---

## Inputs

```
config/params.json     model and criterion parameters (JSON)
config/config.json     the pinned configuration and the demixing maps (JSON)
config/make_config.py  the generator that produced config.json, seed pinned
```

`config/config.json` carries `species` (int, 0 = A, 1 = B), `positions`
(shape `(N, 3)`), `xi` (shape `(N, l)`), and `demix_maps` — a dict of name to
int list of length `N` — with `N = 240` and `l = 3`.

`config/params.json` carries:

| key | meaning |
|---|---|
| `epsilon` | `2 x 2` matrix `eps_ab`, indexed by species code |
| `sigma` | `2 x 2` matrix `sig_ab`, indexed by species code |
| `lj_switch` | `r_on`, `r_off` for the pair switch (`rc_on`, `rc_off`) |
| `coord_switch` | `r_on`, `r_off` for the coordination switch (`rn_on`, `rn_off`) |
| `omega0`, `alpha`, `n0` | the parameters of `omega(n)`; `p` is derived, not pinned |
| `m_osc` | length-2 list, oscillator mass per species |
| `n_osc` | `l`, the number of oscillator displacements per particle |
| `criterion` | `alpha_scale_lo`, `alpha_scale_hi`, `margin` |

---

## Interface contract

`verify.py` imports the implementation and runs every check itself. The contract
is fixed here so that it can.

**File.** `forms.py`, in this directory, importable with no side effects and with
no arguments read from `sys.argv`, no files read, and no network access. It is
imported with the working directory set elsewhere, so it must not depend on the
current directory.

**Module attribute.**

```python
FORMS = ("arctan", "rational")
```

**Functions.**

```python
def calibrate(params):
    """params : dict, exactly as parsed from config/params.json

       returns a dict with at least the key "p": the exponent of the rational
       form, derived from params["alpha"] and params["n0"].
    """

def omega(n, form, params):
    """n      : float64 ndarray of any shape, or a real scalar; n >= 0
       form   : one of FORMS
       params : dict, exactly as parsed from config/params.json

       returns omega(n) for that form, float64, same shape as n.
    """

def coordination(positions, species, params):
    """positions : float64 ndarray, shape (N, 3)
       species   : int64 ndarray, shape (N,), 0 = A, 1 = B
       params    : dict

       returns the homo-coordination n_i, float64 ndarray, shape (N,).
    """

def energy_and_forces(positions, species, xi, params, form):
    """positions, species, params as above
       xi   : float64 ndarray, shape (N, l)
       form : one of FORMS

       returns (energy, forces)
         energy : a real Python/NumPy scalar, the total U(r, xi) for that form
         forces : float64 ndarray, shape (N, 3), F_k = -dU/dr_k
    """

def force_terms(positions, species, xi, params, form):
    """same arguments; returns a dict with exactly the keys

           "pair", "coord_self", "coord_cross"

       each a float64 ndarray of shape (N, 3), summing to the forces returned
       by energy_and_forces() for the same arguments.
    """

def compare(positions, species, demix_maps, params):
    """positions, species, params as above
       demix_maps : dict of map name -> int64 ndarray, shape (N,)

       Evaluates the criterion in spec.md section '## The criterion'.
       Returns a dict:

         {"delta_s": {map_name: {form: {"lo": float, "mid": float, "hi": float}}},
          "R":       {map_name: {form: float}},
          "choice":  "arctan" | "rational" | "undecided"}

       "lo", "mid", "hi" are dS at alpha_lo, alpha and alpha_hi.  Keys are
       exactly the names in demix_maps and exactly the entries of FORMS.
    """
```

**Required of all of them.**

- Every parameter is read from `params`. Nothing from `config/params.json` may be
  hard-coded, and `p` is derived from `alpha` and `n0` on every call:
  `verify.py` calls with modified parameter values and requires the results to
  respond.
- `N` and `l` are read from the shapes of the arguments, not hard-coded.
- The input arrays are not modified in place.
- Repeated calls with the same arguments return identical results.
- All returned values are finite.

`verify.py` fails with an explicit contract message, and runs no physics checks,
if the file is missing, the attribute or functions are absent or not callable,
the return types, keys or shapes are wrong, or any of the requirements above is
violated.

---

## Verifier

`verify.py` is in this directory, pre-exists this session, and may be run at any
time. Do not edit it. It contains no reference energy, no reference force, no
reference entropy, no reference `R`, **and no record of which form the criterion
prefers**: it applies the rule above to the numbers your own module returns.
Every criterion below is an identity, an invariance, a limit, a structural
requirement, or the stated rule recomputed.

Run it with:

```
python verify.py
```

Tolerances are dimensionless ratios taken against the configuration's own force
scale `F_scale = mean_k |F_k|`, against `omega0`, or against the compared
quantity's own magnitude. None is an absolute number in reduced units.

### Check 0 — interface contract

The requirements listed under "Interface contract", plus: the named terms sum to
the returned total force to `1e-12` of `max |F|` for each form; the forces are
not identically zero; the energy responds to `epsilon`, `omega0`, a
**differential** perturbation of `m_osc` (one species up, the other down) and to
`alpha`; and `calibrate()` responds to both `alpha` and `n0` and returns `p = 0`
at `alpha = 0`.

### Check 1 — the two forms are not equivalent

Calibration, then two exact identities, each of which one form satisfies and the
other violates.

| criterion | tolerance |
|---|---|
| both forms: `\|omega(n0) - omega0/2\| / omega0` | `<= 1e-12` |
| both forms: central-difference `omega'(n0)` vs `-omega0 alpha / pi`, relative | `<= 1e-7` |
| **reflection**, `\|omega(n0+d) + omega(n0-d) - omega0\| / omega0`, `"arctan"` | `<= 1e-12` |
| **reflection**, worst over the `d` ladder, `"rational"` | **`>= 1e-3`** |
| **inversion**, `\|omega(n0 e^t) + omega(n0 e^-t) - omega0\| / omega0`, `"rational"` | `<= 1e-12` |
| **inversion**, worst over the `t` ladder, `"arctan"` | **`>= 1e-3`** |
| the same two violations, at every point of a grid in `(alpha, n0)` | **`>= 1e-3`** |
| both forms strictly decreasing on `[0, n_max]`, and in `(0, omega0)` | — |
| `max_i \|omega_A(n_i) - omega_R(n_i)\| / omega0` on the configuration's own `n_i` | **`>= 1e-3`** |

The lower bounds are as binding as the upper ones. Form `"arctan"` is symmetric
under reflection of `n` about `n0`; form `"rational"` is symmetric under
inversion of `n` about `n0`. Each identity is exact for one form at every
parameter setting and violated by the other at every parameter setting, so
neither form is a reparameterization of the other. An implementation that
returns the same function for both forms satisfies the upper bounds and fails the
lower ones.

### Check 2 — the coordination is homo-coordination

`verify.py` computes the criterion from the implementation's own
`coordination()`, so a coordination that counts the wrong neighbours would be
self-consistent and invisible. It is pinned structurally instead, with no
reference value:

| criterion | tolerance |
|---|---|
| relabelling every particle to one species: `n_i` does not decrease, elementwise | `>= -1e-12` |
| and **strictly increases** for at least a quarter of particles | **`> 1e-6`** |
| and is **positive** for at least one particle | **`> 0`** |
| global A/B swap `s -> 1 - s`: `n_i` unchanged, elementwise | `<= 1e-12` |
| flipping one particle's species changes `n_i` **only** within `rn_off` of it | exact |
| and changes that particle's own `n` | **`> 1e-6`** |
| two-particle probe, same species, `r > rn_off`: `n = 0`; `r < rn_on`: `n = 1` | `<= 1e-12` |
| two-particle probe: `n(r)`, `n'(r)`, `n''(r)` continuous at `rn_on` and `rn_off` | `<= 1e-4` of the window scale |

A count over all neighbours is invariant under the first perturbation and fails
the strict-increase row; a hetero-coordination count fails the positivity row; a
switching function that is not `C^2` fails the last.

### Check 3 — each form passes the w1-01 battery

Run **per form**. A gate, not the decision: every check here is invariant to
which frequency law was chosen.

| criterion | tolerance |
|---|---|
| `\|sum_k F_k\| / sum_k \|F_k\|`, and the same for `pair` alone | `<= 1e-10` |
| `coord_self`, `coord_cross`: `sum_k \|F^t_k\| / sum_k \|F_k\|` | **`>= 1e-6`** |
| `coord_self`, `coord_cross`: `\|sum_k F^t_k\| / sum_k \|F^t_k\|` | **`>= 1e-3`** |
| cancellation of the two coordination half-terms | `<= 1e-8` |
| rigid translation: `\|dE\|/\|E\|`, `max \|dF\| / F_scale` | `<= 1e-9` |
| finite difference vs analytic force: best residual over a six-step ladder | `<= 1e-7` |
| observed convergence order over the three coarsest steps | **`>= 1.5`** |
| rotation: `\|dE\|/\|E\|`; `max \|F(Qr) - Q F(r)\| / F_scale`, total and each term | `<= 1e-9` |
| `xi = 0`: both coordination half-terms vanish; total force equals `pair` | `<= 1e-12` |
| `alpha = 0`: `omega = omega0/2` for **both** forms; both half-terms vanish | `<= 1e-12` |
| `alpha = 0`: stiffness energy equals `(1/2)(omega0/2)^2 sum_i m_i sum_g xi^2`, at the pinned setting and at two perturbed `(omega0, m_osc)` settings | `<= 1e-12` |
| the pinned `alpha` is not itself degenerate | — |

### Check 4 — the criterion, recomputed

`verify.py` recomputes the criterion **from the implementation's own `omega()`
and `coordination()`**, applies the rule from `spec.md`, and compares.

| criterion | tolerance |
|---|---|
| returned `delta_s` vs `verify.py`'s recomputation, every map, form and setting | `<= 1e-9` relative |
| returned `R` vs `verify.py`'s recomputation | `<= 1e-9` relative |
| returned `choice` equals the rule applied to `verify.py`'s recomputed `R` | exact |
| the `CHOICE:` line in `report.md` equals the returned `choice` | exact |
| `delta_s` **moves** when `alpha` is scaled by 1.3, and when `n0` is perturbed | **`>= 1e-6`** relative |

The last row is the second half of this check: an implementation that returns a
hard-coded verdict with plausible numbers is internally consistent and is caught
only by requiring the numbers to respond to the parameters.

### Check 5 — the report

`verify.py` checks that `report.md` exists, carries the seven required headings,
and carries exactly one line of the form

```
CHOICE: arctan
```

(or `rational`, or `undecided`). **Its content is judged against the
disqualifiers below, not scored by `verify.py`.** A verdict that matches the
numbers but is defended on the wrong grounds does not pass this task.

---

## Deliverables

1. **`forms.py`** — both forms, meeting the interface contract.
2. **`report.md`** — with exactly these seven headings, spelled as written, and
   one `CHOICE:` line:

   ```
   ## Implementation
   ## Non-equivalence
   ## Calibration
   ## The criterion, measured
   ## The choice, defended
   ## What would overturn it
   ## Underdetermined in the specification
   ```

   - `## Non-equivalence` — the argument that no choice of parameters makes one
     form into the other, with the measurements that support it.
   - `## Calibration` — what the common footing does and does not remove, and
     what a comparison would have shown without it.
   - `## The criterion, measured` — the `dS` and `R` values, per map, form and
     parameter setting, and the margins.
   - `## The choice, defended` — the argument, running on those numbers; and
     where the criterion is weak.
   - `## What would overturn it` — a concrete measurement or criterion that would
     reverse the choice, named specifically enough to be run.
   - `## Underdetermined in the specification` — everything in `## Method` and
     `## The criterion` this specification leaves open, and the convention
     adopted for each.

### Disqualifiers

The report fails on any of these, whether or not `verify.py` exits 0:

- **defending the choice by appeal to which form is published, more common, or
  conventional, rather than to the criterion's measured values** — this fails
  whether or not it names the same form the criterion does;
- reporting a verdict the returned numbers do not support, or reporting numbers
  the implementation did not produce;
- claiming the two forms are equivalent, or that their difference is a
  reparameterization, or that calibration removed it;
- claiming a check proves something it cannot — in particular, that passing the
  check-3 battery bears on the choice between the forms, which it cannot, since
  every check in it is invariant to the frequency law;
- reporting a pass on a check that was not actually run, or reporting `verify.py`
  as passing when it did not;
- resolving an ambiguity in the specification silently, without recording it
  under `## Underdetermined in the specification`;
- naming under `## What would overturn it` something abstract ("a better
  criterion", "more data") rather than a measurement that could be run.

---

## Submission mapping

```
task_summary_skeleton: establish that two published functional forms for a coordination-
                       dependent stiffness are non-equivalent, and choose between them
                       under a criterion stated up front /
                       a result means the difference is pinned and the choice is defended
                       on measured values /
                       what remains is whether the forms are separable by criteria other
                       than this one, and whether the margin survives a sensitivity
                       analysis the session designs itself
objective_criterion:   the verifier's own run of checks 0-5 against the implementation
                       passes, its recomputation of the criterion agrees with the
                       implementation's, and the defence is not an appeal to convention
supporting_files:      spec.md, refs/ (DOIs, licences and a fetch note, no PDFs),
                       config/, verify.py, env.yaml, notes.md
starting_files:        the pinned configuration, the demixing maps and the parameters
                       in config/
final_deliverables:    forms.py, report.md, and the defended choice within it
```

Environment: `env.yaml`. Provenance, licences and checksums: `notes.md`.
