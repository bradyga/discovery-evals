# w1-02 — two frequency laws, non-equivalence, and the criterion applied

CHOICE: arctan

Every number below was produced by `forms.py` on the pinned configuration.
`analysis.py` (in this directory, not a deliverable and not imported by
`verify.py`) prints all of them; each table names the section of its output it
came from. Nothing here is quoted from a reference value, and no reference
value for any quantity in this task exists.

## What this establishes, and the one thing it does not

Stated here rather than left to a weakness list, because it bounds the verdict
and a reader who stops after the `CHOICE:` line should have it.

**Established.** The two forms are not equivalent, and no choice of parameters
makes one into the other (`## Non-equivalence`). On the pinned configuration the
criterion returns `arctan` on both maps with margins 0.9020 and 0.8136 against a
threshold of 0.10, and the margin is carried entirely by the two laws'
sensitivity to `alpha`, not by their demixing signal
(`## The criterion, measured`).

**Not established: that this is a property of the two forms rather than of this
configuration.** The verdict is decided by where the configuration's
coordination sits relative to a crossover at `n = 3.4920` — the point above
which the rational law becomes the more `alpha`-sensitive of the two. Under the
pinned labelling the mean `n_i` is 3.0216 and under the two maps 5.636 and
4.284, so **74.0% of the demixed population sits above the crossover**, on the
side that favours arctan.

The other side of the crossover **cannot be reached by any relabelling of these
positions**. The pinned labelling is a random mixture and is already near the
minimum-homo-coordination end: 400 random relabellings span mean `n` in
`[2.841, 3.541]`, 200 spatial cuts sit at `[5.494, 5.741]`, and a hill-climb
that explicitly minimises mean `n` reaches only 2.1197 (§I). Demixing on these
positions can only raise coordination. So the criterion's two maps — and every
map that could have been supplied with them — sample one side of the crossover
only, and the `arctan` verdict is a joint fact about the two laws *and* this
cluster's density.

**The decisive experiment, and its result.** Changing the configuration does
reach the other side. Diluting the cluster with `config/make_config.py`'s own
sampling procedure — same `N = 240`, same `MIN_SEP`, same map construction, same
`config/params.json`, the criterion untouched — and sweeping the radius through
`4.1 -> 5.2 -> 5.6` moves the demixed mean `n` through `4.960 -> 2.464 -> 1.961`
and the verdict through **`arctan` (margins 0.902, 0.814) -> `undecided`
(0.003, 0.010) -> `rational` (0.324, 2.634)**, reproducibly at three seeds each
(§J). The criterion separates the two forms; which one it selects is set by the
cluster density. Details and the full sweep are under
`## What would overturn it`, item 1.

---

## Implementation

`forms.py` implements the model of `spec.md` `## Method` once and the two
frequency laws twice. `_kernel()` computes the pair term, the homo-coordination
`n_i`, the stiffness term and the three named force terms in a single pass; the
**only** place the two forms differ is `_omega_and_deriv()`, which returns
`omega(n)` and `d omega / d n`. Everything downstream — the switch, the
coordination sum, `C_i = m_{s_i} (sum_g xi_ig^2) omega(n_i) omega'(n_i)`, the
`coord_self` / `coord_cross` split — is shared code. Any difference measured
between the forms is therefore a difference of frequency laws and of nothing
else.

The arctan implementation is carried over from sibling task `w1-01`, as
`spec.md` `### Relationship to tasks/w1-01` permits, and the rational law is
added beside it behind the same interface.

Analytic gradients, with `u_ij = (r_i - r_j)/r_ij` and
`W_kj = S'(r_kj; rn_on, rn_off)` for `s_k == s_j`, `k != j`, else 0:

```
    pair         F_k = -sum_{j != k} phi'(r_kj) u_kj
    coord_self   F_k = -C_k sum_j W_kj u_kj
    coord_cross  F_k = -sum_j C_j W_kj u_kj
```

The calibration is applied on every call: `calibrate()` derives
`p = 4 n0 alpha / pi` from `params` and nothing caches it, so `p` moves whenever
`alpha` or `n0` moves — including inside the criterion's own `alpha` sweep,
where `p` runs over `2.2918 -> 3.0558 -> 3.8197` as `alpha` runs over
`0.60 -> 0.80 -> 1.00`.

**Verifier.** `python verify.py` exits 0: checks 0, 1, 2, 3, 4 and 5 all report
PASS. Representative values from that run — the total force sums to zero to
`3.4e-17` (arctan) and `4.9e-17` (rational) of `sum_k |F_k|`; the worst
best-step finite-difference residual is `3.4e-8` (arctan) and `1.5e-8`
(rational) of `F_scale`, converging at observed order `2.00`; the returned
`delta_s` and `R` agree with the verifier's independent recomputation to
`0.0e+00` relative.

Check 3 is the `w1-01` battery run per form. It is a **gate on each
implementation and carries no information about the choice**: the momentum sum
rule, the rigid-motion invariances, the force–energy consistency, the `xi = 0`
limit and the `alpha = 0` limit hold identically for any differentiable
`omega(n)`, so both forms pass it and neither could have been preferred by it.
Nothing in `## The choice, defended` rests on it.

---

## Non-equivalence

The claim is that **no choice of parameters turns one law into the other** — not
that they differ at the pinned values. Three independent arguments, two of them
measurements and one analytic with a measured counterpart.

### 1. Two exact symmetries, each satisfied by one family and violated by the other

For every `(omega0, alpha, n0)`:

```
    arctan    omega_A(n0 + d) + omega_A(n0 - d) = omega0     (reflection about n0)
    rational  omega_R(n0 e^t) + omega_R(n0 e^-t) = omega0    (inversion about n0)
```

Both are algebraic identities of the closed forms — `arctan(z) + arctan(-z) = 0`
and `1/(1+e^{pt}) + 1/(1+e^{-pt}) = 1` — so each holds at *every* parameter
setting, not at the pinned one. Measured (analysis §B1; residuals relative to
`omega0`, worst over `verify.py`'s own `d` and `t` ladders):

| `(alpha, n0)` | refl. arctan | refl. rational | inv. rational | inv. arctan |
|---|---|---|---|---|
| (0.80, 3.00) | 0.0e+00 | **1.351e-01** | 2.2e-16 | **1.098e-01** |
| (0.40, 2.00) | 1.1e-16 | **1.921e-01** | 0.0e+00 | **1.508e-01** |
| (0.40, 4.50) | 0.0e+00 | **1.819e-01** | 0.0e+00 | **1.300e-01** |
| (1.50, 2.00) | 1.1e-16 | **1.131e-01** | 0.0e+00 | **9.347e-02** |
| (1.50, 4.50) | 0.0e+00 | **5.030e-02** | 2.2e-16 | **4.590e-02** |

The violations are 46–192 times the `1e-3` floor `verify.py` requires, while
the form that satisfies each identity sits at machine precision (`0` or
`2.2e-16`) at every one of the five settings. Within the
calibrated family — where both laws share `omega0` and `n0` — this settles it:
the two are separated by an exact structural property, over the whole
`(alpha, n0)` grid, and no member of one family has the other's symmetry.

### 2. The families are disjoint even with all parameters free

Argument 1 compares laws that share `omega0` and `n0`. The stronger statement —
no `(omega0, alpha, n0)` and *independently chosen* `(omega0', p, n0')` make
`omega_A ≡ omega_R` on `n >= 0` — follows in two steps.

**Step 1, the tail exponent.** As `n -> inf`, with `d = n - n0`,

```
    omega_A(n) = (omega0 / (pi alpha)) d^-1 - (omega0 / (3 pi alpha^3)) d^-3 + O(d^-5)
    omega_R(n) = omega0' (n0')^p n^-p + O(n^-2p)
```

so `omega_A` decays as `n^-1` for **every** member of the arctan family, and
`omega_R` as `n^-p`. Equality forces `p = 1`. Measured (§B2), the local exponent
`-d ln omega / d ln n` at the pinned parameters:

| `n` | 10 | 30 | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|---|---|
| arctan | 1.39902 | 1.10953 | 1.03081 | 1.01009 | 1.00301 | 1.00100 |
| rational | 2.98053 | 3.05309 | 3.05571 | 3.05577 | 3.05577 | 3.05578 |

converging to `1` and to `p = 3.055775` respectively.

**Step 2, `p = 1` is still not enough.** With `p = 1` the rational law is Möbius,
`omega_R = C/(d + c)` with `C = omega0' n0' > 0` and `c = n0' + n0`, whose
expansion `C/d - Cc/d^2 + Cc^2/d^3 - ...` carries a `d^-2` term. The arctan
expansion above has **no** `d^-2` term — only odd powers of `1/d`. Matching
forces `Cc = 0`, hence `n0' = -n0`, which no positive pair of reference
coordinations can satisfy. Equivalently and measurably: a Möbius `omega` has
`1/omega` exactly affine in `n`, so `d^2(1/omega)/dn^2 ≡ 0`. Measured (§B4), at
`alpha = pi/(4 n0) = 0.261799`, the setting at which `p` is exactly `1`:

| `n` | 4 | 6 | 10 | 30 | 100 |
|---|---|---|---|---|---|
| arctan, `d^2(1/omega)/dn^2` | 1.088e-01 | 6.427e-02 | 1.539e-02 | 3.938e-04 | 8.744e-06 |
| rational at `p = 1` | 0.0e+00 | 1.2e-11 | -1.8e-11 | 0.0e+00 | 0.0e+00 |

and the smallest `|d^2(1/omega)/dn^2|` for the arctan law at `n = 2 n0`, over
the whole `(alpha, n0)` grid of §B1, is `1.480e-02` — bounded away from the
identically-zero value the rational law is forced to. The two families are
disjoint.

### 3. The best possible collapse is far from zero

A direct numerical counterpart: fit the *whole* rational family — `omega0'`, `p`
and `n0'` all free, calibration dropped — to the pinned arctan curve on
`n in [0, 10]` (across the three labellings the configuration's `n_i` lie in
`[7.07e-06, 9.585]`; under the pinned labelling alone, `[7.07e-06, 6.765]`),
minimising the max deviation. The best fit (§B3) is `omega0' = 0.8553`,
`p = 3.0132`, `n0' = 3.3232`, and it still leaves

```
    max |omega_A - omega_R| / omega0  =  2.639e-02
```

— 26 times `verify.py`'s `1e-3` separation floor. On the configuration's own
`n_i`, the two **calibrated** laws differ by `1.472e-01` of `omega0` at worst
(§B3, and check 1's last row). The difference is not a reparameterisation and
cannot be fitted away.

---

## Calibration

**What the common footing does.** Both laws are pinned to agree at `n = n0` in
value and in first derivative, which fixes the rational exponent at
`p = 4 n0 alpha / pi`. Measured (§C1), at three `alpha` settings:

| `alpha` | `p` | `omega_A(n0)/omega0` | `omega_A'(n0)` | `omega_R(n0)/omega0` | `omega_R'(n0)` |
|---|---|---|---|---|---|
| 0.600 | 2.2918 | 0.5000000000 | -0.19098593 | 0.5000000000 | -0.19098593 |
| 0.800 | 3.0558 | 0.5000000000 | -0.25464791 | 0.5000000000 | -0.25464791 |
| 1.000 | 3.8197 | 0.5000000000 | -0.31830989 | 0.5000000000 | -0.31830989 |

So the calibration removes exactly two degrees of freedom of disagreement — the
zeroth and first order Taylor coefficients at `n0` — and it removes them at
every `alpha`, which is what makes the criterion's `alpha` sweep a comparison of
forms rather than of tunings. It also makes `alpha` a *shared* parameter: a 25%
change in `alpha` is, by construction, the same 25% change in each law's slope
at `n0`.

**What it does not remove.** Everything beyond first order at `n0`. Measured
(§C3), `|omega_A - omega_R| / omega0` as a function of distance from `n0`:

| `n` | 3.00 | 2.90 / 3.10 | 2.75 / 3.25 | 2.50 / 3.50 | 2.00 / 4.00 | 1.00 / 5.00 | 0.00 / 6.00 | 0.00 / 9.00 |
|---|---|---|---|---|---|---|---|---|
| below `n0` | 0 | 4.65e-04 | 3.25e-03 | 1.47e-02 | 6.06e-02 | 1.441e-01 | 1.257e-01 | 1.257e-01 |
| above `n0` | 0 | 3.82e-04 | 1.99e-03 | 5.49e-03 | 8.14e-03 | 4.30e-03 | 1.83e-02 | 3.17e-02 |

The agreement is local and second-order: it decays as `d^2` out of `n0` and is
gone by `|n - n0| ~ 1`. The configuration's mean `n_i` is 3.022 under the pinned
labelling and 5.636 / 4.284 under the two maps, so most of the model's weight
sits where the calibration has no purchase. The calibration makes the comparison
fair; it does not make it small.

**What a comparison without it would have shown.** Nothing usable. With `p`
pinned as a free parameter rather than derived, the rational law has no `alpha`
dependence at all, so `dS(alpha_hi) - dS(alpha_lo) = 0` exactly and
`R(rational) = inf` on both maps for **any** `p` (§C2: `dS` is `+86.957`,
`+183.853`, `+294.167`, `+507.024` on `plane_x` at `p = 1, 2, 3.056, 5` — each
value identical at all three `alpha` settings). The criterion would have handed
the rational law an infinite score for the trivial reason that it was not being
perturbed. The calibration is what makes the two laws respond to the same knob;
without it the statistic is not a comparison but an artefact of which
parameterisation was chosen for each side.

---

## The criterion, measured

`dS` in units of `l k_B`; `alpha_lo = 0.60`, `alpha = 0.80`, `alpha_hi = 1.00`;
`p` recalibrated at each (`2.2918`, `3.0558`, `3.8197`); `n0 = 3.0`,
`omega0 = 1.0`, positions, maps and switching windows fixed. From
`forms.compare()`, reproduced by `verify.py` check 4 to `0.0e+00` relative
(§D1).

| map | form | `dS(alpha_lo)` | `dS(alpha)` | `dS(alpha_hi)` | spread | `R` |
|---|---|---|---|---|---|---|
| `plane_x` | arctan | 205.7116 | 245.5158 | 277.3267 | 71.6151 | **3.42827** |
| `plane_x` | rational | 213.7022 | 294.1673 | 376.9049 | 163.2027 | **1.80247** |
| `radial` | arctan | 95.1962 | 113.1180 | 127.4598 | 32.2636 | **3.50606** |
| `radial` | rational | 101.7525 | 136.8792 | 172.5562 | 70.8037 | **1.93322** |

| map | preferred | `margin` | threshold |
|---|---|---|---|
| `plane_x` | arctan | **0.9020** | 0.10 |
| `radial` | arctan | **0.8136** | 0.10 |

Both maps prefer the same form and both margins clear the threshold by a factor
of eight. `compare()["choice"] = "arctan"`, and `verify.py`'s independent
application of the rule to its own recomputation agrees.

**Where `R` comes from.** `R` is signal over sensitivity, so the ordering
decomposes exactly (§D2):

| map | signal ratio (rat/arc) | sensitivity ratio (rat/arc) | `R_arc / R_rat` | margin |
|---|---|---|---|---|
| `plane_x` | 1.19816 | 2.27889 | 1.90199 | 0.9020 |
| `radial` | 1.21006 | 2.19454 | 1.81358 | 0.8136 |

The rational law's demixing signal is 20–21% larger — which *favours* it — but
its sensitivity to `alpha` is 119–128% larger, and the second effect is about
twice the first. The entire margin is the sensitivity term.

---

## The choice, defended

**CHOICE: arctan**, on the measured values above: the arctan law's `R` exceeds
the rational law's on both maps, by 0.9020 and 0.8136 relative, against a
threshold of 0.10.

### The mechanism behind the numbers

The margin is not an accident of the pinned `alpha`. Define the per-particle
`alpha`-sensitivity of the log frequency, which is what `dS`'s spread is built
from:

```
    s(n) = - d ln omega / d ln alpha
```

Both laws have `s(n0) = 0` — that is the calibration, which pins the value at
`n0` independently of `alpha`. Away from `n0` they part company. From the closed
forms, with `L = ln(n/n0)` and `sigma(x) = 1/(1+e^-x)`:

```
    s_A(n) -> 1              as n -> inf     (bounded, for every omega0, alpha, n0)
    s_R(n) =  p L sigma(pL)  ->  p ln(n/n0)  as n -> inf     (unbounded)
```

Measured (§D3, §B5), at the pinned parameters:

| `n` | 0.0 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 8.0 | 10.0 | 30.0 | 100.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `s_A(n)` | -0.129 | -0.174 | -0.217 | 0.000 | 0.544 | 0.805 | 0.899 | 0.960 | 0.979 | 0.999 | 1.000 |
| `s_R(n)` | 0.000 | -0.113 | -0.278 | 0.000 | 0.621 | 1.290 | 1.891 | 2.855 | 3.588 | 7.030 | 10.715 |
| `p ln(n/n0)` | — | — | — | 0 | — | — | 2.118 | — | 3.679 | 7.036 | 10.715 |

The arctan law's response to its own tuning parameter **saturates** at 1 above
`n0`; the rational law's **grows without bound**, as `p ln(n/n0)`. That is a
structural difference, visible in the closed forms and confirmed to five digits
by the measurement.

The configuration decides which regime is sampled. Under the pinned labelling
`n_i` has mean 3.022 and 45.8% of particles above `n0 = 3`; both demixing maps
raise it — mean 5.636 with 92.5% above `n0` for `plane_x`, mean 4.284 with 72.1%
above `n0` for `radial` (§A). Demixing on this configuration therefore pushes
the population **into** the region where `s_R` diverges and `s_A` has already
flattened, and the measured aggregate sensitivity ratio of 2.28 / 2.19 is the
population average of the per-particle ratio in the table above. The rational
law buys its 20% larger signal at 2.2 times the cost in `alpha`-sensitivity, and
loses.

### That the mechanism explains the overturns, and where it does not

A mechanism that only survives the perturbations is a story; one that predicts
them is an explanation. The narrow-band limit makes `R_A/R_R` decompose exactly
into `(sensitivity ratio) / (signal ratio)`, so the test is **which factor
moves** when a setting flips (§H3):

| setting | observed | signal ratio | sensitivity ratio | `R_A/R_R` |
|---|---|---|---|---|
| pinned, `plane_x` / `radial` | arctan | 1.198 / 1.210 | 2.317 / 2.230 | 1.934 / 1.843 |
| `n0 = 7.0` | rational | 1.124 / 1.106 | **0.944 / 1.015** | 0.840 / 0.917 |
| `alpha = 0.1, n0 = 6.0` | rational | 1.397 / 1.512 | **1.234 / 1.239** | 0.884 / 0.819 |
| restricted, `L = 1.00` | arctan | 1.169 / 1.188 | 2.223 / 2.232 | 1.902 / 1.878 |
| restricted, `L = 0.35` | undecided | 0.982 / 0.971 | **1.149 / 0.885** | 1.170 / 0.912 |
| restricted, `L = 0.25` | undecided | 0.995 / 1.010 | **1.008 / 1.040** | 1.014 / 1.030 |

In every case the signal ratio barely moves and the **sensitivity ratio carries
the flip** — which is what the mechanism asserts. Two of these are genuine
predictions rather than fits:

- The `L`-restriction *had* to converge to a tie, because the calibration pins
  value and slope at `n0`, leaving the contrast second order:
  `s_R - s_A = -alpha eps^2 / (pi n0) + O(eps^3)` for `n = n0 + eps`. Measured
  against that closed form (§H2), the ratio of measured to predicted runs
  0.834, 0.921, 0.962 as `eps` falls to 0.025. Both ratios must go to 1 near
  `n0`, and they do (1.008 / 1.040 at `L = 0.25`).
- The crossover location predicts the `n0 = 7.0` flip: the last `|s_R| = |s_A|`
  crossing there is at `n = 7.196`, and the demixed population sits at mean
  4.28–5.64, i.e. entirely below it, so the rational law is the *less*
  `alpha`-sensitive of the two and wins. Sensitivity ratio 0.944 / 1.015,
  measured.

**Where it fails.** A one-number predictor built from the mechanism — the
fraction of demixed `n_i` above the crossover — gets five of six settings right
and misses `alpha = 0.2, n0 = 3.0`, where 0% of the population is above the
crossover (13.524) yet `R_A/R_R` is still 1.055–1.101 (§H1, §H3 discussion).
And it does not explain the `n0`-band overturn at all — see
`## What would overturn it`, item 1. The mechanism is an explanation of the
`alpha`-band statistic on populations away from `n0`; it is not a general
account of the criterion.

### That the margin survives perturbations this session chose

`spec.md` records as open whether the ordering holds under a perturbation it did
not anticipate. Five, all in §E:

- **Band width** (§E1). The criterion pins `±25%`. Sweeping `±1%` to `±80%`, the
  verdict is `arctan` at every width, margins running `0.934 -> 0.629`
  (`plane_x`) and `0.843 -> 0.561` (`radial`). The band-free limit is well
  defined: `R_arc/R_rat` converges to `1.93393` (`plane_x`) and `1.84278`
  (`radial`) as the band shrinks, against `1.90199` / `1.81358` at the pinned
  `±25%`. The pinned width is doing no work.
- **Extra demixing maps** (§E3). Four labellings the task did not pin —
  `plane_y`, `plane_z`, a `[111]` plane sort, and a random relabelling — give
  `R_arctan / R_rational` of `3.393/1.807`, `3.404/1.808`, `3.392/1.808` and
  `2.345/1.011`: arctan preferred on all four, margins 0.876 to 1.319.
- **Position jitter** (§E4). 200 Gaussian displacements at each of 0.02, 0.05
  and 0.10 `sig_AA`, recomputing coordination and the whole statistic: `arctan`
  200/200 at every width, smallest margin over 600 draws 0.7183.
- **Coordination window** (§E5). Moving `coord_switch` over `(1.0,1.4)` to
  `(2.0,2.6)` — mean `n_i` from 1.794 to 12.140 — gives `arctan` at all five.
- **`(alpha, n0)` grid** (§E2). `arctan` at 15 of 18 settings. The three
  exceptions are all `undecided`, and they are not all the same kind: at
  `(0.2, 3.0)` arctan has the larger `R` on both maps but `radial`'s margin is
  0.057; at `(0.2, 5.0)` the **rational** law has the larger `R` on both maps
  (2.514 vs 2.321, 2.860 vs 2.430) with margins 0.083 and 0.177, so only
  `plane_x` keeps it from reading `rational`; at `(0.4, 5.0)` the two maps
  prefer different forms outright. The `alpha` sweep at the pinned `n0 = 3.0`
  is cleaner: `undecided` for `alpha <= 0.20` and `arctan` from `alpha = 0.30`
  up, with the rational law never preferred on both maps. This is a limit on
  the result, not robustness — see below.

### Where the criterion is weak

Five weaknesses, each measured rather than asserted.

1. **The verdict tracks where the configuration sits relative to a crossover
   near `n0 exp(1/p)`, not relative to `n0`, and not the forms alone.**
   `s_A -> 1` while `s_R -> p ln(n/n0)`, so the rational law only becomes the
   more `alpha`-sensitive one once `p ln(n/n0)` clears `1`. Measured (§H1),
   `|s_R|` and `|s_A|` cross **three** times, and the last crossing at the
   pinned parameters is at `n = 3.492`, against `n0 exp(1/p) = 4.161` — *not*
   at `n0 = 3.0`. Between `n0` and 3.492 the arctan law is still the more
   sensitive of the two. The asymptotic estimate tracks the measured crossing
   for `p >~ 1` (5.773/5.154, 6.085/5.280, 8.054/7.196) and fails badly below
   it (563.7 against 2.994 at `p = 0.191`), which is why the small-`alpha`
   corner of §E2 does not follow the rule. So a configuration sitting *below*
   the crossover should favour the rational law, and does: at `alpha = 0.8`
   with `n0` re-pinned to 7.0 — where
   only 18.3% of the demixed `n_i` exceed `n0` — the rule returns
   **`rational`** (`R`: 6.716/8.120 on `plane_x`, 6.669/7.446 on `radial`;
   margins 0.209, 0.117). At `alpha = 0.1, n0 = 6.0` it returns **`rational`**
   with margins 0.129 and 0.216. The criterion is answering a question about
   this configuration's coordination distribution as much as about the two laws.
2. **The two pinned maps are distinct measurements, but "agreeing on both" is
   nearly uninformative, and no map can reach the regime that would flip the
   verdict.** This is the limitation stated at the top of this report; the
   measurements behind it are these. The two labellings agree at 0.5167 up to
   the global A/B swap — chance — and their per-particle contributions correlate
   at only 0.43–0.57 (§I), so they are *not* one observation repeated. `radial`
   is not even drawn from `plane_x`'s family: against an ensemble of 200
   random-direction half-space cuts (`R_arctan = 3.4063 +- 0.0327`,
   `R_rational = 1.8069 +- 0.0046`), `plane_x` sits at `+0.67` and `-0.98` sigma
   while `radial` sits at `+3.05` and `+27.58` sigma. What is uninformative is
   the *agreement*: that ensemble has a relative spread of 0.96% and 0.25% and
   prefers arctan 200/200, so two draws agreeing says almost nothing. And the
   family cannot reach the other side of the crossover at all — every spatial
   cut raises homo-coordination (mean `n^m` in `[5.494, 5.741]` against 3.0216
   pinned), 400 non-spatial relabellings stay in `[2.841, 3.541]`, and a
   hill-climb minimising mean `n` reaches only 2.1197. On these positions no
   relabelling demixes downward. The 18 of 400 random relabellings that do
   prefer the rational law (16 clearing the margin) are near-random mixtures,
   not demixings. Also, `1 - radial` scores *identically* to `radial`
   (`3.50606 / 1.93322`, §E3), since `n_i` is swap-invariant: the map set has
   that exact redundancy built in. What the maps cannot test, a different
   configuration can, and does — `## What would overturn it`, item 1, where
   diluting the cluster flips the verdict to `rational`.
3. **`R` rewards insensitivity without bound, and diverges where `dS(alpha)`
   turns over.** At `alpha = 0.8, n0 = 7.5` the rational law's
   `dS(lo, mid, hi)` is `72.3027, 71.4675, 72.1664` — non-monotone in `alpha`,
   so the denominator nearly cancels and `R = 61.4` and `45.6` on the two maps;
   at `n0 = 8.0` the same near-cancellation gives the arctan law `R = 80.1`.
   Those are numerically enormous scores earned by a turning point, not by
   robustness. Nothing in the rule detects it. (Both are outside the pinned
   setting: at `n0 = 3.0` both laws' `dS` are strictly monotone in `alpha`, as
   the `## The criterion, measured` table shows.)
4. **The statistic is blind to `omega0`, and to everything except `omega(·)` and
   three coordination vectors.** `ln omega0` cancels in `ln omega(n^m) - ln
   omega(n^0)`, so `dS` is *exactly* invariant under rescaling `omega0`: a
   `±25%` `omega0` band gives a spread that is pure floating-point roundoff and
   `R ~ 8.6e+15` or `inf` (§F2). `epsilon`, `sigma`, `lj_switch`, `m_osc`, `xi`
   and every force are equally absent — the criterion never calls
   `energy_and_forces()`. The verdict is a statement about two curves evaluated
   on three coordination vectors, and about nothing else in the model.
5. **The margin is carried by particles away from the calibration point, which
   is exactly where the specification pinned the least.** Restricting *both* the
   signal and the sensitivity to particles with `|ln(n_i/n0)| <= L` in every
   labelling (§D4): `L = 1.0` (194 of 240 particles) still gives `arctan`,
   margins 0.872 / 0.849; `L = 0.5` (45 particles) gives `arctan`, margins
   0.290 / 0.117; `L = 0.35` (20 particles) gives **`undecided`**, with
   `plane_x` preferring arctan (0.171) and `radial` preferring rational (0.095);
   `L = 0.25` (10 particles) gives `undecided` with margins 0.014 and 0.031. The
   choice is a statement about the laws' tails. It is a real statement — the
   tails are where the model does its work on this configuration — but it is not
   a statement about the neighbourhood the calibration controls.

None of these overturns the verdict at the pinned setting; all of them bound
what it means. In particular the criterion has **not** shown that either law is
better *simpliciter*, only that on this configuration, under this statistic, one
of them delivers 1.8–1.9 times more demixing signal per unit of tuning
sensitivity.

---

## What would overturn it

Five measurements. All were run, and their results are given, so the claim that
they would overturn the choice is not a conjecture. They are ordered by how much
they are worth: the first reverses the verdict on a configuration built by the
task's own generator; the last is included with the reason it should be
discounted.

1. **Dilute the cluster until the reachable coordination straddles the
   crossover — the decisive experiment.** Re-run `config/make_config.py`'s
   sampling procedure with `RADIUS` raised from its pinned 4.1, keeping
   `N_PARTICLES = 240`, `MIN_SEP = 0.95`, the two position-sort maps and every
   value in `config/params.json` exactly as they are, and score the result with
   the criterion unchanged. The crossover — the last `n` at which `|s_R|` and
   `|s_A|` cross — is at `n = 3.4920` for the pinned `(alpha, n0)`. Measured
   (§J), with the monotonicity guard below applied to every row:

   | `RADIUS` | mean `n^0` | mean `n^m` | frac `n^m` above 3.492 | guard | verdict (margins) |
   |---|---|---|---|---|---|
   | 3.90 | 3.5323 | 5.7362 | 0.860 | pass | `arctan` (1.189, 0.941) |
   | **4.10** (pinned) | 3.0216 | 4.9602 | 0.740 | pass | `arctan` (0.902, 0.814) |
   | 4.40 | 2.5117 | 4.2243 | 0.615 | pass | `arctan` (0.626, 0.634) |
   | 4.80 | 2.0681 | 3.2186 | 0.415 | pass | `arctan` (0.334, 0.359) |
   | 5.20 | 1.5074 | 2.4640 | 0.225 | pass | **`undecided`** (0.003, 0.010) |
   | 5.60 | 1.2124 | 1.9612 | 0.117 | pass | **`rational`** (0.324, 2.634) |
   | 6.00 | 0.8811 | 1.6174 | 0.069 | FLAG | (`rational`, discounted) |
   | 6.50 | 0.8050 | 1.2497 | 0.023 | pass | `arctan` — see below |
   | 7.20 | 0.4994 | 0.9309 | 0.017 | FLAG | (`arctan`, discounted) |

   The margin falls monotonically to zero and changes sign, and the flip is not
   one draw: at three seeds each, `RADIUS = 4.8` gives `arctan` 3/3,
   `RADIUS = 5.2` gives `undecided` 3/3, and `RADIUS = 5.6` and `6.0` give
   `rational` 3/3. **`RADIUS = 5.6` is the single run that overturns the
   choice**: one number changed in the generator, the criterion untouched, the
   verdict `rational` with margins 0.324 and 2.634. The `arctan` at
   `RADIUS = 6.5` is not a return: by there the mean demixed `n` is 1.25, both
   laws' `dS(alpha)` have nearly stopped responding, and the rational law's
   `dS` has begun *decreasing* in `alpha` while the arctan law's still
   increases — the large `R` values (21.4, 30.3) are small denominators, and
   rows 6.00 and 7.20 fail the guard outright.

   This is the experiment the report owes, because it is the only one that
   changes the thing the verdict actually depends on. Items 2–4 vary the
   statistic or the parameters; this one varies the configuration, which is
   where §I shows the constraint lives.

2. **Perturb `n0` instead of `alpha`.** Replace the band in `compare()` with
   `R' = |dS(n0)| / |dS(1.25 n0) - dS(0.75 n0)|`, holding `alpha` at 0.80 and
   recalibrating `p` at each `n0`. `n0` is as much "the model's own tuning
   parameter" as `alpha` is, and the criterion does not say why `alpha` and not
   `n0`. Measured (§F1): `R'` is `20.28034` (arctan) against `7.77056`
   (rational) on `plane_x`, but `19.50739` against `21.22230` on `radial` —
   **the maps disagree**, and the rule returns `undecided`. One line changed,
   verdict overturned.
   **But this overturn is the weakest of the five, and the mechanism does not
   account for it.** The `n0`-sensitivity `t(n) = -d ln omega / d ln n0` has
   none of the saturate-versus-diverge structure `s(n)` has: `t_R` changes sign
   (`-0.0558` at `n = 8`, `+0.6079` at `n = 10`, `+3.977` at `n = 30`) while
   `t_A` stays negative, and `|t_R| > |t_A|` flips back and forth across the
   sampled range (§H4). And the `radial` flip is a near-cancellation, not a
   contrast: the per-particle `Delta t` terms cancel to 90.6% (arctan) and 91.9%
   (rational) in the sum, against 36.4% and 28.0% for the `alpha` perturbation
   (§H5). The `radial` `R'` values 19.51 and 21.22 are therefore the ratio of
   two sums that have each lost ~91% of their mass to cancellation, and which is
   larger is settled by the ~9% residual. This is weakness 3 in a different
   guise, and the monotonicity guard at the end of this section catches it.
3. **Restrict the sums to the calibration neighbourhood.** Keep only particles
   with `|ln(n_i/n0)| <= L` under both the pinned labelling and the map, in both
   `dS` and its spread. Measured (§D4): `L = 0.35` gives `undecided` (maps
   split, margins 0.171 and 0.095); `L = 0.25` gives `undecided` (margins 0.014,
   0.031). Running this at `L = 0.35` and `L = 0.25` overturns the choice.
4. **Re-pin `n0` to 7.0 in `config/params.json`, changing nothing else.**
   Measured (§E2 extended, reported under weakness 1 above): `R` becomes
   `6.7162` (arctan) / `8.1197` (rational) on `plane_x` and `6.6687` / `7.4455`
   on `radial` — the rational law preferred on both maps with margins 0.209 and
   0.117, i.e. **`CHOICE: rational`**. `alpha = 0.10` with `n0 = 6.0` does the
   same (margins 0.129, 0.216). This is the sharpest overturn: it reverses the
   verdict rather than merely voiding it, and it requires one parameter to move.
5. **Score a configuration whose coordination sits at the calibration point.**
   Draw `n_i` from `N(n0, sigma_n)` and demix by shifting the mean by
   `0.5 sigma_n`, for `sigma_n` in `0.15 ... 1.60` (§F3): margins are 0.010,
   0.042, 0.121, 0.256, 0.421. At `sigma_n <= 0.30` the verdict is `undecided`.
   Generating a real cluster with the same property — a denser, more uniform
   packing whose homo-coordination is narrowly distributed about `n0`, produced
   by the same `make_config.py` procedure with a smaller `RADIUS` or a larger
   `MIN_SEP` — and re-running `compare()` on it is the physical version of this
   test.

A sixth entry, a cheap guard rather than an overturn: **before trusting any `R`, sample
`dS(alpha)` on 30 points across `[0.5 alpha, 1.5 alpha]` and check it is
monotone.** Where it is not — measured at `n0 = 7.5` and `n0 = 8.0`, weakness 3
above — `R` is a near-zero denominator and means nothing. At the pinned setting
`dS` is monotone for both laws and the guard passes.

---

## Underdetermined in the specification

Everything `## Method` and `## The criterion` leave open, and the convention
adopted. The first four are marked `[UNDET-n]` in `forms.py` at the point of
use.

1. **`0^0` in the rational law at `alpha = 0`.** `spec.md` states the intended
   consequence — "`p = 0`, and *both* forms reduce to the constant
   `omega0 / 2`" — but `omega_R(n) = omega0 / [1 + (n/n0)^p]` at `n = 0` is
   `omega0 / (1 + 0^0)`, and `omega_R(0) = omega0` for *every* `p > 0`.
   **Adopted:** `p == 0` is handled as the constant `omega0/2` on all `n`, i.e.
   `0^0 = 1`, the only reading under which the law is constant.
   **Consequence, recorded:** `omega_R(0)` is discontinuous in `alpha` at
   `alpha = 0`, jumping from `omega0` to `omega0/2`. `verify.py` check 3's
   `alpha = 0` rows exercise the constant and pass; nothing tests the
   neighbourhood.
2. **The rational law's derivative at `n = 0` when `0 < p < 1`.** For
   `alpha < pi/(4 n0)` (= 0.261799 at the pinned `n0`), `d omega_R/dn ~ n^{p-1}`
   diverges at `n = 0`, and the force needs it through
   `C_i = m_i (sum_g xi^2) omega omega'`. The specification does not say what
   the force on a particle with exactly zero homo-coordination is in that
   regime. **Adopted:** the non-finite entry is replaced by 0.
   **Not reached here:** the smallest `n_i` is `7.07e-06`, not 0, and the
   criterion's `alpha` band gives `p in [2.2918, 3.8197]`, all above 1. It *is*
   reachable in the `(alpha, n0)` sweep of §E2, where `p` falls to 0.382 — but
   that sweep uses `omega()` only, never `omega'()`.
3. **The domain of `omega()`.** The contract says `n >= 0`; `omega_R` is not
   real for `n < 0` at non-integer `p`, while `omega_A` is. **Adopted:** a
   negative `n` raises rather than returning `nan`.
4. **Return type for a scalar `n`.** The contract says "float64 ndarray of any
   shape, or a real scalar ... same shape as `n`", which does not say what shape
   a scalar has. **Adopted:** always a float64 ndarray — 0-d for a scalar input.
5. **Ties in `R`.** The rule says "the form with the larger `R` is preferred" and
   is silent on `R_arctan == R_rational`. **Adopted:** an arbitrary fixed break;
   it is never load-bearing, because a tie has margin 0, below any positive
   threshold, so the verdict is `undecided` either way.
6. **`margin` when `min(R_arctan, R_rational) = 0`.** The denominator is
   undefined. **Adopted:** an infinite margin — a form with zero demixing signal
   against a form with nonzero signal is a decisive separation. Not reached.
7. **`R` when `dS(alpha_hi) = dS(alpha_lo)`.** The criterion's denominator
   vanishes and `R` is undefined. **Adopted:** raise, rather than return `inf`
   or `nan`; `verify.py` treats a non-finite `R` as an implementation failure
   rather than a verdict, and this matches that reading. **This is reachable**,
   not hypothetical: `dS` is non-monotone in `alpha` near `n0 = 7.5` (weakness 3
   above), so a nearby parameter setting makes the denominator pass through zero
   exactly.
8. **`dS` is exactly invariant under `omega0`.** `ln omega0` cancels in
   `ln omega(n^m) - ln omega(n^0)`. `spec.md` lists `omega0` among the
   quantities "held fixed throughout" without saying it is inert; measured, an
   `omega0` band produces only roundoff (§F2). **Recorded, not resolved** — no
   convention is needed, but it bounds what the criterion can be about.
9. **Where `l` comes from, and whether it enters.** `l` is available both as
   `params["n_osc"]` (= 3) and as `xi.shape[1]` (= 3). **Adopted:** the kernel
   reads it from `xi.shape[1]`, per "N and `l` are read from the shapes of the
   arguments"; `params["n_osc"]` is never used. `dS` is stated in units of
   `l k_B` and its formula carries no `l`, and `R` is a ratio, so `l` cancels
   twice and `compare()` does not reference it at all.
10. **The scope of `sum_i` in `dS`.** The specification writes `sum_i` without
    restriction. **Adopted:** all `N` particles, including any whose label the
    map leaves unchanged. A sum restricted to relabelled particles would be a
    different statistic; `verify.py`'s recomputation uses the unrestricted sum,
    and the two agree.
11. **Symmetry of `epsilon` and `sigma`.** `U_pair = sum_{i<j} phi_{s_i s_j}` is
    well defined only if the matrices are symmetric; the specification does not
    require it. **Adopted:** indexed directly as `eps[s_i][s_j]` with no
    symmetrisation. The pinned matrices are symmetric, so the choice is
    invisible here and would matter only for a non-symmetric `params`.
12. **What the criterion does *not* read.** Neither `epsilon`, `sigma`,
    `lj_switch`, `m_osc` nor `xi` enters `dS`, `R` or the verdict, and
    `compare()` never calls `energy_and_forces()` or `force_terms()`. The
    specification does not state this; it follows from the definition.
    **Recorded** because it bounds the verdict: the choice rests on `omega(·)`
    evaluated on three coordination vectors, and the force machinery that
    `verify.py` checks 0 and 3 certify is a gate on the implementations and no
    part of the decision.
