# w1-01 — force-implementation certification from a published description

Implement a published coarse-grained force law from a stated specification, then
establish whether the implementation is correct without a reference
implementation and without reference values.

---

## Intent

I want to establish whether the published force law can be implemented correctly
from its papers alone, and which falsifier in the certification battery catches
which class of implementation error.

---

## Prompt

The session's first user turn is in `prompt.txt`, and is reproduced here verbatim.

```
Read spec.md first. Then implement the force law it specifies so that it runs on
the configuration in config/.

Once you have an implementation, establish whether it is correct. You have no
reference implementation and no reference values — the argument has to come from
the model's own symmetries and limits. For each test you run, say what it rules
out and what it does not, and name at least one class of error that none of your
tests would catch.

Where the specification leaves something underdetermined, say so explicitly
rather than picking a convention silently.

verify.py is in this directory. You may run it. Do not edit it.
```

### Assumptions

1. **The functional forms in `## Method` below are the only specification.** No
   reference implementation is available to consult, and none exists in this
   directory.
2. **The configuration and the parameters in `config/` are as pinned and are not
   to be changed.** `config/make_config.py` records how the configuration was
   generated; it is provenance, not an input to be re-run with other settings.
3. **"Correct" means self-consistent under the model's own symmetries and
   limits**, not agreement with any external number. No reference energy, force
   or trajectory exists for this configuration, and none is expected.
4. **Units and conventions are as stated below.** Anything the specification
   leaves underdetermined is to be reported as underdetermined, not silently
   resolved.

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

**Coordination-dependent stiffness.** The oscillator frequency decreases with
increasing homo-coordination:

```
    omega(n) = omega0 * [ 1/2 - (1/pi) * arctan( alpha * (n - n0) ) ]

    U_stiff(r, xi) = (1/2) * sum_i m_{s_i} * omega(n_i)^2 * sum_{g=1..l} xi[i, g]^2
```

`omega(n)` is strictly decreasing in `n` and lies in `(0, omega0)`.

**Switching function.** One C^2 quintic switch is used for both the pair cutoff
and the coordination count. With `x = (r - a) / (b - a)`:

```
    S(r; a, b) = 1                              for r <= a
               = 1 - 10 x^3 + 15 x^4 - 6 x^5    for a < r < b
               = 0                              for r >= b
```

**Forces.** The force on particle `k` is the negative gradient of `U` with
respect to `r_k` at fixed `xi`:

```
    F_k = - dU / d r_k        (xi held fixed)
```

The oscillator displacements `xi` are part of the pinned configuration. They are
inputs, not degrees of freedom to be propagated; no force on `xi` is required.

### Named force terms

The force decomposes into three named terms, which the interface contract below
requires the implementation to expose separately:

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
`sqrt(eps_AA / (m sig_AA^2))`. `n_i` is dimensionless. Nothing in this task
requires a temperature, a timestep or a physical constant.

### Sources, and what is and is not taken from them

The model — a binary Lennard-Jones mixture whose particles carry classical
harmonic oscillators whose frequency decreases with increasing homo-coordination,
with the arctan form of `omega(n)` — is that of:

- **Primary method reference.** D. Gobbo, P. Ballone and B. D. Garabato,
  *Coarse-Grained Model of Entropy-Driven Demixing*, *J. Phys. Chem. B* **2020**,
  124(41), 9267–9274. DOI [10.1021/acs.jpcb.0c07575](https://doi.org/10.1021/acs.jpcb.0c07575).
  Published by ACS; **cited by DOI only, never redistributed**.
- **Restatement used for the equations here.** N. C. Forero-Martinez,
  R. Cortes-Huerto, A. Benedetto and P. Ballone, *Thermoresponsive Ionic
  Liquid/Water Mixtures: From Nanostructuring to Phase Separation*, *Molecules*
  **2022**, 27, 1647. DOI [10.3390/molecules27051647](https://doi.org/10.3390/molecules27051647).
  CC BY 4.0. The model Hamiltonian and the frequency law appear there as
  Eqs. 15–17.

**Pinned by this task rather than taken from either source, and therefore part of
the specification under test:** the quintic switching function `S` and its use
for both the pair cutoff and the coordination count; the cutoff radii; every
parameter value in `config/params.json`; the reduced-unit convention; the free
cluster boundary; the treatment of `xi` as fixed inputs; and the three-way force
decomposition named above. `refs/README.md` records what was read, when, and
under which licence.

The equations are restated with attribution. No paper text, figure or table is
reproduced in this directory, and no PDF is bundled.

---

## Inputs

```
config/params.json     model parameters (JSON)
config/config.json     the pinned configuration (JSON)
config/make_config.py  the generator that produced config.json, seed pinned
```

`config/config.json` carries `species` (int, 0 = A, 1 = B), `positions`
(shape `(N, 3)`), `xi` (shape `(N, l)`), with `N = 200` and `l = 3`.

`config/params.json` carries:

| key | meaning |
|---|---|
| `epsilon` | `2 x 2` matrix `eps_ab`, indexed by species code |
| `sigma` | `2 x 2` matrix `sig_ab`, indexed by species code |
| `lj_switch` | `r_on`, `r_off` for the pair switch (`rc_on`, `rc_off`) |
| `coord_switch` | `r_on`, `r_off` for the coordination switch (`rn_on`, `rn_off`) |
| `omega0`, `alpha`, `n0` | the parameters of `omega(n)` |
| `m_osc` | length-2 list, oscillator mass per species |
| `n_osc` | `l`, the number of oscillator displacements per particle |

---

## Interface contract

`verify.py` imports the implementation and runs every check itself. The contract
is fixed here so that it can.

**File.** `force_kernel.py`, in this directory, importable with no side effects
and with no arguments read from `sys.argv`, no files read, and no network access.

**Functions.**

```python
def energy_and_forces(positions, species, xi, params):
    """positions : float64 ndarray, shape (N, 3)
       species   : int64 ndarray, shape (N,), 0 = A, 1 = B
       xi        : float64 ndarray, shape (N, l)
       params    : dict, exactly as parsed from config/params.json

       returns (energy, forces)
         energy : a real Python/NumPy scalar, the total U(r, xi)
         forces : float64 ndarray, shape (N, 3), F_k = -dU/dr_k
    """

def force_terms(positions, species, xi, params):
    """same arguments; returns a dict with exactly the keys

           "pair", "coord_self", "coord_cross"

       each a float64 ndarray of shape (N, 3), summing to the forces returned
       by energy_and_forces() for the same arguments.
    """
```

**Required of both functions.**

- Every parameter is read from `params`. Nothing from `config/params.json` may be
  hard-coded: `verify.py` calls both functions with modified parameter values and
  requires the results to respond.
- `N` and `l` are read from the shapes of the arguments, not hard-coded.
- The input arrays are not modified in place.
- Repeated calls with the same arguments return identical results.
- All returned values are finite.

`verify.py` fails with an explicit contract message, and runs no physics checks,
if the file is missing, the functions are absent or not callable, the return
types or shapes are wrong, or any of the requirements above is violated.

---

## Verifier

`verify.py` is in this directory, pre-exists this session, and may be run at any
time. Do not edit it. It contains no reference energy, no reference force and no
reference implementation: every criterion is an identity, an invariance or a
limit of the model as specified above. It imports `force_kernel.py` and computes
every quantity itself, so a claimed pass on a check the implementation does not
satisfy does not survive.

Run it with:

```
python verify.py
```

All tolerances are dimensionless ratios taken against the configuration's own
force scale `F_scale = mean_k |F_k|` or against the compared quantity's own
magnitude. None is an absolute number in reduced units.

### Check 0 — interface contract

The requirements listed under "Interface contract" above, plus: the named terms
sum to the returned total force to `1e-12` of `max |F|`, and the forces are not
identically zero.

### Check 1 — translational invariance, both directions

| criterion | tolerance |
|---|---|
| `\|sum_k F_k\| / sum_k \|F_k\|` | `<= 1e-10` |
| `pair`: `\|sum_k F_k\| / sum_k \|F_k\|` | `<= 1e-10` |
| `coord_self`, `coord_cross`: `sum_k \|F^t_k\| / sum_k \|F_k\|` | **`>= 1e-6`** |
| `coord_self`, `coord_cross`: `\|sum_k F^t_k\| / sum_k \|F^t_k\|` | **`>= 1e-3`** |
| cancellation of the two coordination half-terms | `<= 1e-8` |
| rigid translation: `\|dE\| / \|E\|`, `max \|dF\| / F_scale` | `<= 1e-9` |

The two lower bounds are as binding as the upper ones. The total force conserves
momentum; **`coord_self` and `coord_cross` must each fail to conserve it on their
own, and must cancel against each other.** An implementation that makes the total
residual vanish by suppressing the coordination force fails here even though its
total force sums to zero.

### Check 2 — force–energy consistency

The analytic force is compared with a central finite difference of the
implementation's *own* returned energy, at 12 randomly chosen (particle, axis)
pairs drawn from a seed pinned inside `verify.py`, over a six-point step ladder
from `1e-2` to `10^-4.5` in `sig_AA`.

| criterion | tolerance |
|---|---|
| best residual over the ladder, relative to `F_scale` | `<= 1e-7` |
| observed convergence order over the three coarsest steps | `>= 1.5` |

Step-size convergence is required, not agreement at one step.

### Check 3 — rotational invariance and covariance

Under a random proper rotation `Q` drawn from the same pinned seed:

| criterion | tolerance |
|---|---|
| `\|E(Qr) - E(r)\| / \|E(r)\|` | `<= 1e-9` |
| `max \|F(Qr) - Q F(r)\| / F_scale`, total and each named term | `<= 1e-9` |

### Check 4 — limiting cases

Two documented degenerate limits, both exact:

- **`xi = 0`.** The stiffness term vanishes, both coordination half-terms vanish
  identically, and the total force equals the `pair` term.
- **`alpha = 0`.** `omega(n) = omega0 / 2` for every particle, independent of
  coordination, so both coordination half-terms vanish identically and

  ```
      U(r, xi; alpha = 0) - U(r, 0; alpha = 0)
          = (1/2) (omega0 / 2)^2 sum_i m_{s_i} sum_g xi[i, g]^2
  ```

  which `verify.py` evaluates in closed form from `config/` and compares.
  The energy at `xi = 0` must not depend on `alpha`, and the energy at the pinned
  `alpha` must differ from the energy at `alpha = 0`.

All four hold to `1e-12`, relative.

### Check 5 — the mapping is the deliverable

`verify.py` checks that `report.md` exists and carries the five required section
headings below. **Its content is judged against the disqualifiers below, not
scored by `verify.py`.** A passing battery with a wrong account of what it proves
does not pass this task.

---

## Deliverables

1. **`force_kernel.py`** — the implementation, meeting the interface contract.
2. **`report.md`** — the certification report, with exactly these five headings,
   spelled as written:

   ```
   ## Implementation
   ## Certification battery
   ## Error-class mapping
   ## What no check catches
   ## Underdetermined in the specification
   ```

   - `## Certification battery` — the tests run, including any beyond the four in
     `verify.py`, with what each one measured.
   - `## Error-class mapping` — which check catches which class of implementation
     error, and, for each check, what it does **not** rule out.
   - `## What no check catches` — at least one concrete class of implementation
     error that no check in the battery detects. Named concretely, not as "bugs
     in general".
   - `## Underdetermined in the specification` — everything in `## Method` above
     that this specification leaves open, and the convention adopted for each,
     stated rather than assumed.

### Disqualifiers

The report fails on any of these, whether or not `verify.py` exits 0:

- claiming a check proves something it cannot — for example, that the momentum
  sum rule establishes that the force is the gradient of the energy, or that a
  finite-difference agreement establishes rotational covariance;
- naming as an uncaught error class something that one of the four checks does in
  fact catch;
- reporting a pass on a check that was not actually run, or reporting `verify.py`
  as passing when it did not;
- resolving an ambiguity in the specification silently, without recording it
  under `## Underdetermined in the specification`;
- naming an uncaught error class only in the abstract ("coding mistakes",
  "numerical issues") rather than as a concrete class with a mechanism.

---

## Submission mapping

```
task_summary_skeleton: certify a from-the-paper implementation of a published force law /
                       a result means the battery is fixed and its blind spot named /
                       what remains is whatever the papers leave underdetermined
objective_criterion:   the verifier's own run of all four checks against the agent's
                       implementation passes, and the error-class mapping is defensible
supporting_files:      spec.md, refs/ (DOIs and a fetch note, no PDFs), config/,
                       verify.py, env.yaml, notes.md
starting_files:        the pinned configuration and parameters in config/
final_deliverables:    force_kernel.py, report.md, and the error-class mapping within it
```

Environment: `env.yaml`. Provenance, licences and checksums: `notes.md`.
