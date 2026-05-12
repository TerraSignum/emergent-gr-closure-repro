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

## Recommendation

**Primary route: Route 3 (Hardy / spectral-synthesis via radial
operators).** Reasons:

1. The carrier-action of P6 already imposes the radial structure
   as a UV-closure consequence (M3 + bounded-operator readout
   forces `Xi_N(i,j) = f(d_N(i,j))`). The hard hypothesis of
   Route 3 is structurally guaranteed.
2. The Symanzik-1 finite-size correction `a/N` is the natural
   output of radial-spectrum expansion, matching empirical data.
3. Connects `lambda_inf` to a *closed-form* integral of the
   radial spectral measure — gives the predicted exact value of
   `lambda_inf` against the System-R rationals.
4. Existence-only fallback (Route 1, Cheeger) is already
   structurally guaranteed via the `xi_min^2/8` bound, so Route 3
   is the *quantitative* upgrade.

**Secondary route: Route 1 (Cheeger).** Use as the
existence-of-Lemma-B fallback if Route 3 stalls. The `xi_min^2/8`
bound is structurally cheap and seven orders of magnitude weaker
than the empirical gap but mathematically rigorous.

**Defer: Routes 2 and 4** (Bakry–Émery, log-Sobolev). Both have
unclear hypothesis-checking on the admissible-carrier class and
much higher analytical cost. Revisit only if Route 3 fails.

## Phase 2 work plan

**Step 1 (1–2 weeks):** verify the radial-structure hypothesis
of Route 3 against the P5/P5N snapshots. Compute, per regime,
the empirical correlation `corr(d_N(i,j), Xi_N(i,j))` and check
that it is consistent with a single deterministic function `f`.
If yes, Route 3 hypothesis confirmed empirically; if no, Route 3
needs restriction to a sub-class.

**Step 2 (2–4 weeks):** if Step 1 succeeds, derive the radial
spectral measure for the carrier-action `f`. Closed-form integral
expression of `lambda_inf` in terms of `(d, N_gen, alpha_xi,
gamma, beta_pi, D_Omega, eps_sync^2)`. Compare to the empirical
`lambda_inf = 0.3789 ± 0.005` and to the conjectured `3/8`.

**Step 3 (1–2 months):** finite-size correction. Show that the
discrete-to-continuum spectral correction is `a / N + O(1/N^2)`
with `a` computable in closed form. Match to empirical `a = 6.62`.

**Step 4 (2–4 months):** uniform lower bound. Prove that the
spectral-measure construction is uniformly stable across the
admissible-carrier class (not just the canonical P5/P5N realisation).
Produce the proof of Lemma B as a theorem.

**Phase 2 total estimate:** 6–10 weeks for Step 1+2 (proof of
concept), 3–6 months for Step 3+4 (full theorem). Step 1 is the
deciding empirical check; if it fails, the analytical route shifts
to a careful Cheeger-improvement program (Route 1+) for an
existence-only Lemma B.
