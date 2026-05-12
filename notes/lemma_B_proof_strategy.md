# Lemma B — proof-strategy memo

**Status:** survey of candidate analytical routes for proving the
target Lemma B (uniform spectral gap on admissible relational
carriers). Phase 2 of the program; the statement and empirical
certification are in [`lemma_B_statement.md`](lemma_B_statement.md).

The goal is to derive

```
lambda_2(L_N) >= lambda_*  > 0      for all N >= N_0,
```

from the carrier-action construction of P6 and the integer inputs
`(d, N_gen, xi_min)`, without recourse to per-N numerics. Four
candidate routes are surveyed below. Recommended route, costs,
risks, and prior art for each.

## Route 1 — Cheeger-type isoperimetric inequality

**Idea.** The discrete Cheeger inequality

```
lambda_2(L_N)  >=  h(L_N)^2 / 2,
```

where `h(L_N)` is the Cheeger constant

```
h(L_N)  =  min_{ S ⊂ V_N, |S| <= N/2 }
              [  sum_{i in S, j not in S} W_N(i,j)  ] / vol(S),
```

reduces the spectral-gap question to a vertex-cut question.
Uniform lower bound on `h(L_N)` over admissible carrier sequences
implies uniform Lemma B.

**Required intermediate result.** For every admissible carrier
sequence `{(V_N, Xi_N)}_N`,

```
h(L_N)  >=  c_h(d, N_gen, xi_min)  > 0      for all N >= N_0.
```

**Tractability.** Moderate. The carrier-action construction
forces all pairwise `Xi_N(i, j) >= xi_min`, so every vertex-cut
has weight at least `xi_min * |S| * (N - |S|)`. Combined with
`vol(S) <= (N-1)` per vertex, gives

```
h(L_N)  >=  xi_min * |S| * (N - |S|) / [ |S| * (N-1) ]
        =  xi_min * (N - |S|) / (N - 1)
        >=  xi_min / 2                          (|S| <= N/2).
```

This yields the trivial Cheeger lower bound `lambda_2 >= xi_min^2 / 8`
which for `xi_min = 10^{-3}` gives `lambda_2 >= 1.25 × 10^{-7}` —
strictly positive but seven orders of magnitude weaker than the
empirical `lambda_inf ≈ 0.38`. Useful as existence, not as
quantitative bound.

**Cost:** weeks. **Risk:** the Cheeger bound is provably the right
*structural* answer for existence, but quantitatively much weaker
than the empirically observed gap. Provides existence of Lemma B,
not its sharp constant.

## Route 2 — Bakry–Émery curvature-dimension condition

**Idea.** If the Markov chain `P_N = D_N^{-1} W_N` satisfies a
Bakry–Émery `CD(K, infty)` condition with `K > 0`, then the
Poincaré inequality holds with constant `1/K`. The Bakry–Émery
operator on a finite graph is

```
2 Gamma_2(f) = L (Gamma f) - 2 Gamma(L f, f),
Gamma(f, g)  = (1/2) ( L(fg) - f Lg - g Lf ).
```

`CD(K, infty)` is the pointwise inequality `Gamma_2 >= K Gamma`.

**Required intermediate result.** The Xi-weighted normalised
Laplacian on admissible carriers satisfies `CD(K, infty)` with
`K = K(d, N_gen, xi_min) > 0` uniformly in `N`.

**Tractability.** Hard. Lin-Lu-Yau (2011) showed `CD(K, infty)`
on graphs reduces to a quadratic-form inequality that is
algebraically tractable but combinatorially explosive
(N^3 terms). For specific structured graphs (Cayley graphs,
Riemannian-like graphs) there are explicit results; for the
admissible-carrier class as defined by M0–M3 + xi_min there is
no published Bakry–Émery curvature estimate.

**Cost:** months. **Risk:** the `CD(K, infty)` condition may
simply not hold uniformly across the admissible-carrier family —
the empirical Symanzik-1 decay `a/N` is *not* consistent with
the strict Bakry–Émery bound (which would give a constant
spectral gap with no finite-size correction). Bakry–Émery may
need to be relaxed to `CD(K, n)` with effective dimension `n`
to match the data.

**Strength:** if it works, gives the sharpest constant, the
exact form of `lambda_inf`, and connects to Lemma A
(non-collapse) via standard Sturm-style arguments.

## Route 3 — Hardy-type Poincaré inequality via spectral synthesis

**Idea.** Bypass curvature, work directly with the spectrum.
For a finite Markov kernel `P_N` with stationary measure
`pi_N`, the Poincaré constant satisfies

```
C_P = 1 / (1 - lambda_2(P_N)).
```

For the Xi-weighted chain, `lambda_2(P_N) = 1 - lambda_2(L_N)`,
so the question reduces to a spectral synthesis on the
admissible-carrier class:

```
lambda_2(L_N)  =  1  -  || P_N |_{1^⊥} ||      (operator norm
                                                on orthogonal-
                                                to-constants).
```

By the carrier-action construction, the off-diagonal `Xi_N(i,j)`
factorise through the bounded-operator readout: `Xi_N(i,j) =
f(d_N(i,j))` for a single bounded monotone function `f`.
This is *not* identifiable for fully general M0–M3 but **is**
the case in the P5/P5N construction (M3 + UV-closure of P6
forces `Xi_N` to be a function of metric distance only). In
this restricted setting

```
P_N  =  M_N · diag(1 / row-sum),
M_N(i,j)  =  f(d_N(i,j))  -  delta_{ij},
```

becomes a *radial* Markov operator. Radial operators on
finite spaces have explicit spectra in terms of the
metric-radial spectral measure. This is the natural
analytical route.

**Required intermediate result.** Compute the spectrum of the
radial operator `M_N - diag(1 / row-sum)` on the carrier metric
`d_N(i,j) = -log Xi_N(i,j)` and show its second-smallest
eigenvalue is bounded below uniformly in N.

**Tractability.** Moderate-to-easy if the metric-radial-spectral-
measure analytical form is available. The Symanzik-1 finite-size
correction `a/N` would emerge naturally from the radial-spectrum
expansion.

**Cost:** weeks to a few months. **Risk:** the "radial" hypothesis
may fail for adversarial admissible carriers; needs care to
identify *which* subclass of admissible carriers the radial
construction applies to. The carrier-action of P6 should imply
the radial structure as a UV-closure consequence — this needs to
be checked.

**Strength:** likely the most concrete analytical route. Connects
the Lemma B asymptote `lambda_inf` to the integral form of the
metric-radial spectrum, which is a known quantity for the System-R
rational coefficient set. **This is the conjectural route to
identifying `lambda_inf = 3/8`** (current empirical match at 1.0%).

## Route 4 — Functional Brascamp-Lieb / log-Sobolev approach

**Idea.** Skip the Poincaré inequality, prove the stronger
log-Sobolev inequality directly. Log-Sobolev `LSI(c)` with
constant `c > 0` implies Poincaré with constant `2c`. For the
Xi-weighted chain on admissible carriers,

```
Ent_{pi_N}( f^2 )  <=  c  ·  E_N(f, f),
```

with `E_N` the Dirichlet form. The Brascamp-Lieb argument shows
log-Sobolev follows from a finite-dimensional convexity condition
on the Hamiltonian.

**Required intermediate result.** Show that the effective
Hamiltonian `H_N = -log Xi_N` has uniform convexity constant
`alpha > 0` on the admissible-carrier class.

**Tractability.** Hard. The discrete log-Sobolev constant on
non-product chains is notoriously hard to estimate. Bobkov-Tetali
inequalities give the right reduction but require curvature input
that we already lack.

**Cost:** 6+ months. **Risk:** very high. Likely overkill for
Lemma B; log-Sobolev would give Lemma B plus Gaussian
concentration (Lemma A indirectly) but the analytical machinery
is brittle.

**Strength:** if it works, simultaneously gives Lemma A and B.
But this is the longest route and Route 3 is more likely to land
first.

## Phase-2 Step 1 result (radial hypothesis falsified)

**Related existing audit.** The corpus already contains
`src/verify_xi_gram_spectral_gap_scaling.py`, which audits Xi_N's
SVD spectrum on the P0..P8 ladder for Wigner-Dyson RMT
classification (gap-ratio statistic). It reports
intermediate-regime behaviour (gap-ratio ≈ 0.46, between Poisson
0.386 and GOE 0.530), consistent with a non-trivial structured
spectrum. The Phase-2 Step-1 audit below tests a different
diagnostic (effective rank, not RMT class) on a different ladder
(P5/P5N, not P0..P8), addressing the specific question whether
Xi_N is low-effective-dimensional in the sense required by the
spectral-synthesis route.

The original radial-hypothesis formulation "`Xi_N(i,j) = f(d_N(i,j))`
for a single bounded monotone `f`" is tautologically true because
`d_N := -log Xi_N` by definition. The non-tautological version
tested in Phase-2 Step 1 was: **Xi_N has effectively bounded rank
as N → ∞**, i.e., the singular-value spectrum concentrates on a
number `k = k(d, N_gen, xi_min)` of dominant components,
independent of N.

This formulation was tested empirically by
`src/verify_lemma_B_radial_hypothesis.py` across the 10-regime
ladder with 184 seeds. Result:

| Diagnostic         | N-scaling fit (preferred) | Scaling                  | Verdict   |
|--------------------|---------------------------|--------------------------|-----------|
| `r_95(Xi_N)`       | linear                    | `r_95 ≈ -3.1 + 0.41·N`   | DIVERGES  |
| `r_99(Xi_N)`       | linear                    | `r_99 ≈ -5.2 + 0.61·N`   | DIVERGES  |
| `PR(Xi_N)`         | linear                    | `PR  ≈ -7.6 + 0.28·N`    | DIVERGES  |
| `top σ-share`      | log                       | `≈ 0.71 - 0.12·log(N)`   | bounded → 0 |

**Verdict: RADIAL_HYPOTHESIS_NOT_SUPPORTED.** The effective rank
of `Xi_N` grows linearly with `N` (at ~41% of `N` for the
95%-energy cutoff). The carrier produces matrices with broadly-
distributed singular spectra, not low-dimensional ones. The
top-singular share decays as `0.71 - 0.116·log(N)`, dropping to
0.029 at `N = 512` — no dominant subspace persists in the
continuum.

**Implication.** Route 3 (Hardy spectral synthesis via low-dim
embedding) as originally formulated does *not* apply. The
uniform spectral gap (Phase 1, empirical) is therefore *not*
explained by low-dimensional structure of `Xi_N`. The gap must
come from a different mechanism.

## Revised route ranking (post-Step-1)

The Phase-1 result `lambda_2(L_N) → 0.3789 + 6.62/N` combined with
the Step-1 result "Xi_N is effective-full-rank with broad spectrum"
sharpens the analytical question to: *what structural property
of the carrier action enforces a uniform Laplacian spectral gap
in spite of the broad Xi spectrum?*

Two natural candidates, each tested empirically below the route
descriptions:

**Revised primary route: Route 1+ (quantitative Cheeger via
degree concentration).** The trivial Cheeger bound `lambda_2 ≥
xi_min^2/8` is seven orders of magnitude weaker than the empirical
gap. A *quantitative* refinement uses the fact that, if the
vertex-degrees `deg(i) = sum_j (Xi_N(i,j) - delta_ij)` are
uniformly concentrated about a common mean, the random-regular-
graph-style Cheeger argument (Diaconis–Stroock canonical-paths)
gives a uniform spectral gap controlled by the mean degree and
the degree variance, not by `xi_min` alone. The empirical test
is the **degree-concentration hypothesis**:

```
Var(deg) / Mean(deg)^2 → 0  as N → ∞,
```

i.e., relative degree fluctuations vanish in the continuum
limit. This is the Phase-2 Step-2 audit (to be implemented in
`src/verify_lemma_B_degree_concentration.py`).

**Revised secondary route: Route 2 (Bakry–Émery curvature-
dimension).** Now elevated because the broad-spectrum Xi rules
out the simpler radial picture, while leaving open whether the
Markov chain `P_N = D^{-1} W` admits a uniform `CD(K, infty)`
estimate. Empirical test: compute the discrete Bakry-Émery
operator `Gamma_2` per regime and check the inequality
`Gamma_2(f) ≥ K Gamma(f, f)` pointwise for a uniform `K > 0`.
This is the Phase-2 Step-3 audit.

**Deferred: Routes 3 and 4** (radial, log-Sobolev). Route 3 is
falsified at the low-rank level by Step 1; a softer "radial"
version (e.g., approximate translation-invariance under a
non-trivial vertex permutation) may still apply but is no longer
the leading candidate. Route 4 (log-Sobolev) was already
deferred and remains so.

## Revised Phase-2 work plan

**Step 1 (DONE, 1 session).** Radial hypothesis falsified on
the 10-regime ladder.

**Step 2 (1–2 weeks).** Degree-concentration audit. Compute
`Var(deg) / Mean(deg)^2` per regime and fit the N-scaling.
Predicted: `Var/Mean^2 → 0` as `N → ∞` if Cheeger-route works;
`Var/Mean^2 → const > 0` if degree fluctuations persist and
Route 1+ also fails.

**Step 3 (2–4 weeks).** Bakry–Émery audit. Compute the discrete
`Gamma_2` operator on small-N admissible carriers and test
`CD(K, infty)` for uniform `K > 0`.

**Step 4 (2–4 months).** Whichever route survives Steps 2–3,
derive the analytical proof of Lemma B from the carrier-action
construction. The empirical asymptote `lambda_inf = 0.3789` and
the conjectured rational `3/8` are the calibration targets.

**Phase-2 status after Step 1:** Route 3 falsified, Route 1+
elevated to primary, Route 2 secondary. The 6–10 week
proof-of-concept estimate (Steps 2+3) and the 3–6 month full-
theorem estimate (Step 4) are unchanged. The deciding next
audit is the degree-concentration test (Step 2).
