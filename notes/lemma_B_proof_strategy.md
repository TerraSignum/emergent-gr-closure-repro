# Lemma B — proof-strategy memo

## Status of the original 3-Lemma decomposition (A, B, C)

The initial Phase-2 framing introduced three target lemmas:

  - **Lemma A** — Non-collapse / volume-doubling theorem
    (uniform Ahlfors-doubling A8 on admissible carriers).
  - **Lemma B** — Uniform Poincaré / spectral-gap theorem
    on admissible carriers (this memo's central object).
  - **Lemma C** — Operator / tensor convergence theorem
    (G_N → G_∞, T_N^Ξ → T_∞^Ξ as distributions on the mm-GH
    limit).

**Current status of each:**

- **Lemma A is SUBSUMED by Lemma B.** P4
  `thm:sg_implies_admissibility` proves that the uniform
  spectral gap `(SG)` (= Lemma B) implies the full
  admissibility set A1+A8 via uniform Poincaré + Cheeger-
  Buser machinery. Lemma A therefore does not need a
  separate analytical attack; once Lemma B is established
  analytically, Lemma A follows.

- **Lemma B is the open analytical target** addressed by
  this memo. Phase-1 empirical certification complete
  (P4 prop:sg_empirical, λ_inf = 0.3789, conjectured 3/8);
  Phase-2 empirical characterisation complete (7 P4
  propositions); analytical Step 4 (small-world spectral
  theory, Friedman-Kahale type bounds) remains
  multi-month research.

- **Lemma C is a three-axis target, all three currently
  numerically certified.** Initially this memo treated
  Lemma C as the Dal Maso Γ-convergence audit alone
  (γ-score 0.849, GAMMA_CONVERGED). A subsequent
  inventory of the corpus's actual CLP framework
  (outputs/clp_full_report.json,
  outputs/clp_full_families_extended_audit.json) corrected
  this to a three-axis decomposition:

    **Axis 1 — CLP-A (continuum-limit existence):**
      score 0.813, CLOSED (T1=1.0, T2=0.90, T3=0.45, T4=1.0).

    **Axis 2 — CLP-B (operator convergence):**
      score 0.598 (outputs/clp_full_report.json,
      clp_full_families_extended_audit.json clp_b_b4 key).
      Four sub-components:
        - absorption: 0.514  (bottleneck)
        - locality:   0.542
        - density:    0.652
        - spectral:   0.682
      A B-family restructuring (B-fast/B-slow/B-internal)
      with score 0.801 was once proposed in internal
      working notes (Papers/python/...) but is NOT in the
      published repo outputs; it is therefore not cited
      here.

    **Axis 3 — CLP-C (Γ-convergence, the Dal Maso 5-proxy
      audit):** score 0.849, GAMMA_CONVERGED. Sub-scores
      C1=0.73 (liminf), C2=1.00 (recovery), C3=0.83 (equi-
      coercivity), C4=0.82 (minimiser convergence), C5=0.87
      (= ½ C1 + ½ C2, δS_N → δS_∞ link).

    **CLP-N4 Bridge (internal-notes only, not in repo):**
      Γ-convergence → first variation → operator-limit chain.
      Mentioned in internal working notes
      (Papers/python/08_Gap_Closure_Results.md) but no
      reproducible output in the repo; not cited as a
      formal closure component.

    **CLP-D Overall:** 0.738 (weights A:0.3, B:0.4, C:0.3),
    status CLP_PROVEN.

  The analytical-replacement programme therefore proceeds
  along three axes simultaneously, not one:
    (a) tighten the weakest C_i (currently C1=liminf at 0.73)
        by replacing the numerical liminf proxy with a
        rigorous analytical argument;
    (b) tighten the weakest B sub-component (currently
        absorption at 0.514) by replacing the
        absorption/locality/density/spectral numerical
        proxies with analytical arguments;
    (c) (optional/future) operationalise the analytical
        bridge between Γ-convergence and operator-limit
        sides; not yet present as a reproducible audit in
        the repo.

  All three are independent multi-month research
  deliverables. The conditional master closure theorem of
  P6 holds at CLP_PROVEN level (0.738) on all three axes;
  the unconditional theorem requires replacement of the
  weakest-scoring sub-component on each axis.

**Net Phase-2 deliverable count:**

  - Subsumed: 1 (Lemma A → follows from Lemma B).
  - Open analytical: 1 (Lemma B).
  - Numerically certified, analytically partial: 1 (Lemma C).

The session work has focused on Lemma B because it is the
only one of the three that lacks both an analytical proof
and a clear bridge from one of the others. Lemma C's
numerical certification is robust enough to support the
conditional master closure theorem (M3 in P6 master
synthesis); analytical replacement would harden the result
but is not the deciding gap.



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

## Phase-2 Step 2 result (degree concentration also falsified)

Reproducer: `src/verify_lemma_B_degree_concentration.py`.
Output: `outputs/verify_lemma_B_degree_concentration.json`.

Same 10-regime ladder, 184 seeds:

| Diagnostic                  | Scaling        | Asymptote N→∞    | Verdict |
|-----------------------------|----------------|------------------|---------|
| `mean(deg)`                 | Symanzik-1     | 4.24 + 53/N      | constant (no concentration) |
| `CV(deg) = std/mean`        | const          | 0.168            | constant (no concentration) |
| `NSM(deg) = var/mean^2`     | const          | 0.029            | **NOT → 0**, no concentration |

**Verdict: DEGREE_CONCENTRATION_NOT_SUPPORTED.** Relative degree
fluctuations persist at the ~17% level in the continuum limit.
Route 1+ (quantitative Cheeger via degree concentration) is
falsified as a route to the uniform spectral gap.

### Positive structural information from the negative result

The Step-2 audit also produces a **positive structural finding**
that reorients the analytical strategy:

```
mean(deg) -> 4.24   (constant in N, with Symanzik-1 1/N correction)
```

The carrier produces a **sparse weighted graph** where each
vertex's mean weighted degree is N-independent. Combined with
the empirical findings of Phase 1 (`lambda_2 → 0.3789`) and
Step 1 (Xi_N effective-full-rank), the carrier signature is:

  - sparse: mean weighted degree O(1) as N → ∞
  - full-rank: bulk spectrum is N-extensive (not low-rank)
  - uniformly gapped: spectral gap bounded below uniformly in N
  - constant relative degree fluctuation: ~17%

This is the **classical signature of a sparse expander-like
graph**. The appropriate analytical machinery is expander-
graph theory, not classical isoperimetric Cheeger.

## Revised route ranking (post-Step-2)

**New primary route (Route 5): Expander-graph theory.**

For a weighted graph with mean degree `d_eff` constant in N and
uniform Laplacian spectral gap `lambda_*`, the appropriate
analytical machinery is:

  - **Alon-Boppana bound** (sharpness of spectral gap on
    regular-like graphs): `lambda_2 ≤ d_eff - 2*sqrt(d_eff - 1) + o(1)`.
    For `d_eff = 4.24`, this gives `lambda_2 ≤ 1.04`, consistent
    with the empirical 0.3789.

  - **Friedman's theorem** (random d-regular graphs are
    "near-Ramanujan"): typical sparse graphs achieve spectral gap
    close to the Alon-Boppana bound with high probability. The
    carrier-action equilibrium construction may be reducible to a
    Friedman-type random-construction argument.

  - **Kahale's spectral bound for irregular expanders**: extends
    expander analysis to graphs with non-uniform degree, which
    the carrier exhibits (CV = 17%).

  - **Mossel-Neeman-Sly stochastic block model** and similar
    structured-random-graph results: provide the analytical
    framework for spectral gaps in graphs generated by a
    structured stochastic process (which the carrier-action
    arguably is).

**Phase-2 Step 3 (revised, 2-4 weeks):** Decide which of these
expander-theoretic frameworks the carrier-action equilibrium
construction fits into. Key empirical follow-up audits:

  - Edge-weight distribution at the cluster level: are there a
    small number of distinct edge-weight values (block-model
    signature) or a continuous distribution (random-graph
    signature)?

  - Spectral fingerprint: does the Laplacian spectrum bulk
    follow Marchenko-Pastur (random matrix) or a structured
    multimodal distribution (block-model)?

  - "Effective regularity": is there a vertex permutation that
    makes Xi approximately block-Toeplitz?

**Deferred again:** Routes 1, 2, 3, 4 (classical Cheeger, Bakry-
Émery, low-rank radial, log-Sobolev). Route 1 will only re-
emerge if Step 3 fails and we have to fall back to an existence-
only result via the trivial `xi_min^2/8` Cheeger bound.

**Phase-2 status after Step 2:** Routes 1+ and 3 both falsified;
Route 5 (expander theory) elevated to primary. The 6-10 week
proof-of-concept estimate is unchanged; the route is just
different. Step 3 is the deciding empirical follow-up.

## Phase-2 Step 3a result (near-Ramanujan skeleton identified)

Reproducer: `src/verify_lemma_B_edge_weight_structure.py`.
Output: `outputs/verify_lemma_B_edge_weight_structure.json`.

Same 10-regime ladder, 184 seeds. Edge-weight distribution
combined with threshold-dependent effective degree
`d_eff(tau, N) = mean_i |{j: Xi(i,j) > tau}|`:

| Threshold τ | d_eff^∞ (Symanzik or const) | Alon-Boppana λ_AB(d_eff^∞) |
|------------:|----------------------------:|----------------------------:|
| 0.01        | 36 (const)                  | 0.6709                      |
| 0.05        | 26 (const)                  | 0.6138                      |
| **0.10**    | **12 (Symanzik-1)**         | **0.4499 ← skeleton**       |
| 0.20        | 5.5 (Symanzik-1)            | 0.2277 (RULED OUT)          |
| 0.50        | 0.67 (Symanzik-1)           | — (below threshold)         |
| ≥10% rowsum | 1.38 (const)                | — (near-tree)               |

**Empirical λ_inf / λ_AB(12) = 0.3789 / 0.4499 = 0.842.**

**Interpretation.** The carrier has a discrete connectivity
hierarchy:

  - A *near-tree skeleton* of ~1.38 "row-significant" edges per
    vertex (carrying ≥10% of row weight).
  - A *strong-connectivity skeleton* of ~5.5 edges per vertex
    (Xi > 0.2) — too sparse for Alon-Boppana.
  - A *structural skeleton* of ~12 edges per vertex (Xi > 0.1) —
    Alon-Boppana λ_AB = 0.45, and the empirical λ_inf = 0.38
    sits at **84%** of this bound. **This is the
    Friedman-near-Ramanujan threshold for the carrier.**
  - A *weak-connectivity halo* of 26–36 edges per vertex.

The 84% saturation ratio is the empirical signature of an
expander whose spectral gap saturates close to (but not at)
Alon-Boppana, exactly as predicted by Friedman's theorem for
typical sparse regular graphs.

## Revised Phase-2 work plan (post-Step-3a)

**Step 3b (1–2 weeks):** Extract the τ=0.10 skeleton adjacency
matrix `A_skel = 1[Xi > 0.10]` and audit its spectral gap
directly. Predicted: `lambda_2(L(A_skel)) ≈ 0.38` (matching
the weighted-Laplacian gap up to skeleton-projection error).
If yes, the analytical question reduces to "is the
carrier-action-generated skeleton a near-Ramanujan expander?"
and the weighted-vs-unweighted Laplacian discrepancy is
controlled by Kahale-type bounds.

**Step 3c (2–4 weeks):** Spectral fingerprint of the skeleton.
Test whether the skeleton Laplacian bulk spectrum follows
Marchenko-Pastur (random-regular-like, Friedman regime) or
shows block-modular structure (stochastic-block-model regime).
The two regimes have different analytical tractability.

**Step 3d (2–4 weeks):** Effective-regularity / block-Toeplitz
test. Search for vertex permutations that make A_skel
approximately regular (constant unweighted degree) or
approximately block-Toeplitz. If found, the System-R
algebraic structure is preserved by the skeleton extraction
and the Phase-2 Step 4 analytical bound becomes computable in
closed form.

**Step 4 (2–4 months, unchanged):** Analytical proof of the
uniform spectral gap from the carrier-action construction,
using whichever framework Steps 3b/c/d identify.

**Phase-2 status after Step 3a:** Route 5 (expander theory)
confirmed empirically; the τ=0.10 skeleton is the analytical
handle. Next deciding test is the direct skeleton-spectrum
audit (Step 3b).

## Phase-2 Step 3b result (skeleton-Laplacian asymptote at 7/24)

Reproducer: `src/verify_lemma_B_skeleton_laplacian.py`.
Output: `outputs/verify_lemma_B_skeleton_laplacian.json`.

Same 10-regime ladder, 184 seeds. Computed the normalised
Laplacian spectral gap of the τ=0.10 unweighted skeleton
`A_skel = 1[Xi > 0.10]` per snapshot, cross-seed averaged
per regime, Symanzik-1 N-scaling fit.

| Quantity                   | Symanzik-1 asymptote | Notes                |
|----------------------------|---------------------:|----------------------|
| λ₂(L_weighted)             | 0.3789               | Phase-1 reproduction |
| λ₂(L_skeleton)             | **0.2924**           | NEW                  |
| ratio λ_skel / λ_w         | 0.789                | NEW                  |
| skeleton mean_deg          | 12.14                | matches Step 3a (12) |
| skeleton CV_deg            | 0.263                | higher than weighted |

**Key findings:**

1. **The skeleton itself has a uniform spectral gap.** The
   τ=0.10 unweighted skeleton is a sparse Ramanujan-like
   expander with λ_∞^skel = 0.2924, asymptotic regular degree
   12, and CV ≈ 0.26.

2. **Weighting improves the gap by 30%.** The empirical ratio
   λ_∞^w / λ_∞^skel ≈ 1.30 quantifies the spectral-gap
   improvement from including the edge weights.

3. **Two complementary rational conjectures:**
   - `λ_∞^weighted ≈ 3/8 = 0.3750` (Δ=1.0%) — already registered
     in Phase 1.
   - `λ_∞^skeleton ≈ 7/24 = 0.29167` (Δ=0.24%) — NEW from
     Step 3b. Tighter match than the weighted conjecture.
   - Algebraically consistent ratio: `(7/24)/(3/8) = 7/9 = 0.7778`
     vs empirical 0.789 (Δ=1.4%).

4. **Analytical-route refinement.** The 30%-improvement
   factor `λ_∞^w / λ_∞^skel ≈ 9/7` is now the analytical bridge:
   if the skeleton is shown to be a near-Ramanujan expander
   (Friedman-type, Step 4a), the weight-distribution lift
   (Kahale-type, Step 4b) closes the gap to the empirical
   weighted asymptote. Both sub-bounds become computable in
   closed form against the System-R rationals 3/8 and 7/24.

### Updated Phase-2 work plan (post-Step-3b)

**Step 4a (2-3 months):** Prove the τ=0.10 skeleton is a
near-Ramanujan expander with `d = 12`. Friedman's theorem for
random regular graphs is the natural starting point; if the
carrier-action construction reduces to a structured random-
graph ensemble, Friedman applies directly. Predicted
closed-form: `λ_∞^skel = 7/24`.

**Step 4b (1-2 months):** Prove the edge-weight distribution
lifts the unweighted gap by the rational factor `9/7`. Kahale-
type bounds for irregular weighted expanders. Predicted
closed-form: `λ_∞^w / λ_∞^skel = 9/7`.

**Step 4c (optional, 2-4 months):** Universality. Show that
both Steps 4a and 4b extend uniformly across the admissible-
carrier class (not just the canonical P5/P5N realisation).

**Phase-2 status after Step 3b:** Route 5 fully empirically
mapped. The analytical question is decomposed into two
algebraically discrete sub-bounds (3/8 weighted, 7/24
skeleton, ratio 9/7). Both are closed-form rationals in
the System-R primitive set, ready for Step-4 analytical
attack.

## Phase-2 Step 3c result (structured-random spectral fingerprint)

**Relation to existing audit.** The corpus already contains
`src/verify_xi_gram_spectral_gap_scaling.py`, which classifies
the Xi-Gram spectrum on the P0..P8 ladder via the Wigner-Dyson
gap-ratio statistic (Poisson 0.386 / GOE 0.530 / GUE 0.603) —
a level-spacing classifier. Step 3c is a *complementary* RMT
fingerprint on a *different* operator (skeleton normalised
Laplacian L_skel, not Xi-Gram) and a *different* ladder
(P5/P5N, not P0..P8), using the Kesten-McKay BULK density
(not level spacing) as discriminator. The two together build
the full spectral-fingerprint picture.

Reproducer: `src/verify_lemma_B_spectral_fingerprint.py`.
Output: `outputs/verify_lemma_B_spectral_fingerprint.json`.

| Diagnostic            | Signal       | Friedman-pure | SBM-pure | Observed |
|-----------------------|--------------|---------------|----------|----------|
| n_isolated_eigs(N)    | bounded → ∞ | O(1)          | O(N)     | ~N/3 (linear) |
| frac_bulk_in_KM       | ≤ 1         | ~ 1           | < 1      | 0.999     |
| bulk kurtosis_excess  | ≈ -0.5      | ≈ -0.3       | > 0      | -0.7      |

**Verdict: STRUCTURED_RANDOM** — neither pure Friedman nor
pure SBM. The bulk lies inside the Kesten-McKay arch (Friedman-
like global shape), but the isolated-eigenvalue count grows
roughly linearly with N (block-model-like). The natural
interpretation is a **Cayley-graph-like construction**: the
skeleton resembles a Cayley graph on a finite group, whose
spectrum has a KM bulk per irreducible representation class
plus discrete isolated eigenvalues counted by the
representation lattice.

**Important caveat.** The MAD-based isolated-eigenvalue
detector with threshold 3 × MAD(spacings) may misclassify
fine bulk clustering as isolated; in the Cayley-graph
interpretation, the "isolated" eigenvalues would actually be
narrow internal bands corresponding to non-trivial
representations of the underlying group, not true isolated
peaks. Step 3d (effective-regularity / permutation search)
is the deciding follow-up.

### Updated Phase-2 work plan (post-Step-3c)

**Step 3d (1-2 weeks):** Effective-regularity search on the
skeleton. Test whether there exists a vertex permutation
σ such that A_skel(σ(i), σ(j)) is approximately translation-
invariant (Cayley/circulant signature) or block-permutable
(SBM signature). Test the strongest candidates:
  - Z/N cyclic Cayley structure: sort by degree, check
    autocorrelation of permuted adjacency.
  - Cuthill-McKee reordering bandwidth: small bandwidth
    indicates geometric/manifold-like structure.
  - Block-Toeplitz pattern via random projection: random
    sub-matrices should reveal periodicity.

**Step 4 (analytical, unchanged):** Once Step 3d identifies
the structural class, the analytical derivation of
lambda_inf^skel = 7/24 follows from:
  - Cayley case: representation theory of the underlying
    finite group acting on the carrier.
  - Block case: Mossel-Neeman-Sly stochastic-block-model
    bounds, adapted to System-R rational parameters.

**Phase-2 status after Step 3c:** Bulk is Friedman-shape
(Kesten-McKay), discrete spectrum has structured complement
(Cayley-like signature). Step 3d will discriminate; Step 4
follows once the class is fixed.

## Phase-2 Step 3d result (Friedman regime confirmed; Cayley falsified)

Reproducer: `src/verify_lemma_B_skeleton_diameter.py`.
Output: `outputs/verify_lemma_B_skeleton_diameter.json`.

10-regime ladder, 8 seeds per regime (the O(N²) shortest-path
cost limits seed multiplicity; still 80 snapshots total).

| Diagnostic                  | Result                  | Friedman-pure | Cayley-geometric |
|-----------------------------|-------------------------|---------------|------------------|
| diameter scaling            | **−0.47 + 0.84·log(N)** | log(N) ✓      | N^(1/k) ✗        |
| AICc(log) − AICc(power)     | −3.07 (log wins)        | log ✓         | power ✗          |
| giant-component fraction    | 1.000                   | connected ✓   | connected ✓      |
| girth                       | 3                       | typical sparse| typical for d>2  |
| skeleton degree CV          | 0.24                    | ≈ 1/√12 = 0.29| 0 (regular)      |

**Verdict: FRIEDMAN_LIKE.** Diameter grows logarithmically, the
graph is connected with girth 3, and degree heterogeneity is
~24% (close to the 1/√d Friedman-random value of 29%). The
skeleton is a **sparse near-regular random expander**, not a
Cayley graph on a finite group.

### Reinterpretation of Step 3c result

The Step-3c finding that "isolated-eigenvalue count grows
linearly with N" is re-interpreted in light of the Step-3d
verdict:

  - The MAD-based detector (threshold 3·MAD of bulk spacings)
    over-reports "isolated" eigenvalues in a dense bulk with
    fine spacing structure.
  - The actual structure is a Kesten-McKay bulk with O(1)
    truly-isolated eigenvalues (Perron value 0 and a few
    near-Perron stragglers), plus many fine-spacing
    bulk-internal gaps that the detector mislabels.
  - The 99.9% KM-support coverage in Step 3c was the
    correct signal; the "structured-random" verdict was
    over-stated.

### Updated Phase-2 analytical route

**Route 5a (locked-in):** Friedman's theorem direct application.

The carrier-action equilibrium construction produces a
structured-random graph ensemble whose τ=0.10 skeleton is in
the Friedman class:

  - d ≈ 12 (constant in N, Step 3a)
  - bulk follows Kesten-McKay (Step 3c)
  - diameter scales logarithmically (Step 3d)
  - degree heterogeneity is 1/√d-like (Step 3d)

If the System-R rationals (γ, α_ξ, β_π, D_Ω, ε²_sync, d=4,
N_gen=3) can be shown to *uniquely* determine an
edge-formation probability that reduces to a structured
sparse Erdős-Rényi-like construction with mean degree 12,
Friedman's theorem gives λ_inf^skel within o(1) of the
Alon-Boppana bound 0.4499. The conjectured rational
λ_inf^skel = 7/24 = 0.29167 corresponds to a 65%
Alon-Boppana saturation — within the typical Friedman regime
but not at the Ramanujan optimum (which would be 0.4499).

**Step 4a (revised, 2-3 months):** Prove from the
carrier-action construction that the τ=0.10 skeleton is a
Friedman-random d=12 expander, deriving λ_inf^skel = 7/24
in closed form.

**Step 4b (revised, 1-2 months):** Prove the edge-weight
distribution lifts the unweighted spectral gap by the rational
9/7 (Kahale-type bound applied to the carrier-action weight
distribution).

**Steps 4c+ (universality, optional):** Extend across the
admissible-carrier class.

**Phase-2 status after Step 3d:** Friedman-random class
preliminarily confirmed by log(N) diameter; Cayley hypothesis
rejected. (See Step 4a pre-flight result below for the final
refinement of this classification.)

## Phase-2 Step 4a pre-flight result (small-world, not pure Friedman)

Reproducer: `src/verify_lemma_B_edge_correlation.py`.
Output: `outputs/verify_lemma_B_edge_correlation.json`.

Same 10-regime ladder, 184 seeds. Tests whether the τ=0.10
skeleton edges are independently formed (Erdős-Rényi limit)
or have persistent triangle correlation (small-world / SBM
signal).

| Diagnostic              | Asymptote (N→∞)    | Interpretation                    |
|-------------------------|--------------------:|-----------------------------------|
| p_edge                  | -0.025 + 19/N → 0  | edge density → 0 (sparse)         |
| triangles_observed/ER   | **5.31** (Symanzik-1)  | **5x more triangles than ER**   |
| global clustering coeff | **0.142** (asymptote)  | **finite asymptotic clustering** |

**Verdict: CORRELATED edge formation — small-world, not
Erdős-Rényi.** The combined signature

  - log(N) diameter (Step 3d, small-world ✓)
  - girth = 3 (Step 3d, triangles exist ✓)
  - global clustering → 0.142 finite (Step 4a, NEW)
  - triangle excess → 5.3× ER (Step 4a, NEW)

is the **classical Watts-Strogatz small-world graph
signature**: short paths plus high local clustering, distinct
from both pure Erdős-Rényi (low clustering) and regular
lattices (long paths). Friedman's theorem applies to typical
random regular graphs which have *vanishing* clustering as
N → ∞; the observed persistent clustering rules out
Friedman-pure interpretation.

### Refined analytical Phase-2 route (post Step 4a pre-flight)

**Route 5b (locked-in):** Small-world spectral theory.

For Watts-Strogatz small-world graphs, the spectral gap can
be analysed via resolvent methods that combine:
  - Random-regular Friedman contribution (Kesten-McKay-like
    bulk from the underlying d=12 random regular skeleton),
  - Triangle/clique structural correction (the 5.3× excess
    encoded as a structural perturbation).

The Symanzik-1 fit lambda_inf^skel = 7/24 (from Step 3b) is
now interpreted as the small-world-spectral-gap asymptote,
not the pure-Friedman asymptote. The conjectured rational
7/24 should follow from:

  - Carrier-action edge-formation rule reduces to a
    structured edge-inclusion probability p_inc(i, j)
    depending on the carrier dynamics; total mean degree
    constraint d_eff = 12 fixes the global edge density.
  - Triangle-clustering correction is computable from the
    second-order moment of the carrier-action inclusion
    rule.

**Updated Step 4a analytical target:** derive 7/24 from a
carrier-action edge-formation rule that produces a
small-world graph with mean degree 12 and clustering 0.142.

**Step 4b (unchanged):** 9/7 lift factor from weight
distribution.

**Phase-2 status after Step 4a pre-flight:** Classification
refined to small-world (Watts-Strogatz-like), away from pure
Friedman-random. Both rational conjectures (7/24, 9/7)
unchanged; the analytical machinery changes from Friedman's
theorem to small-world spectral theory.

## Phase-2 W-loop result (t_00 corpus-canonical cross-check)

Reproducer: `src/verify_lemma_B_fiedler_vs_t00.py`.
Output: `outputs/verify_lemma_B_fiedler_vs_t00.json`.

Earlier audits (S-loop row-sum, T-loop row-variance) used
proxies for the matter-core classifier. The W-loop closes
this by running the corpus-canonical Galerkin pipeline
(`per_seed_galerkin` from
`verify_galerkin_runner_A_hessian_ricci.py`) which computes
the per-node energy density `t_00(a)` directly from ψ, K, Q
fields available in the snapshots. AUC for t_00 as
matter-core classifier is 0.85 (corpus, anisotropic
companion paper).

9-regime ladder, 6 seeds per regime (Galerkin-throttled):

| Overlap | Empirical | Random | × Random |
|---------|-----------|--------|----------|
| J(Fiedler30%, t00-top10%)  | 0.137  | 0.057  | 2.4× |
| J(Fiedler30%, t00-top30%)  | 0.249  | 0.18   | 1.38× |
| J(Fiedler30%, t00-top5%)   | 0.078  | 0.029  | 2.7× |
| J(Fiedler30%, t00-top1%)   | 0.039  | 0.006  | ~6× (abs small) |
| J(rowvar30%, t00-top30%)   | 0.430  | 0.18   | 2.4× (proxy validation) |

**Verdict:**
1. Canonical t_00 audit qualitatively confirms the S+T loop
   conclusions: Fiedler-set has weak positive matter-core
   bias (1.4-2.4× random), strongly suppressed at the deep
   C99 cusp.
2. Row-variance(Ξ) is an *adequate* matter-core proxy
   (Jaccard 0.43 with canonical t_00, 2.4× random), but not
   perfect. Earlier S/T-loop findings stand.
3. Halo-interpretation (Fiedler ~ matter-core neighbourhood,
   not the cusp itself) is reinforced.
4. Regime-stability: J(Fiedler30%, t00-top10%) is constant
   at 0.12-0.16 across N=64..512, confirming the Halo
   structure is θ(N)-stable (not dependent on the running
   chirality angle).

## Cross-sector independence of the 3/8 conjecture

**Branch scope.** The entire Phase-2 ladder (N ∈ [50, 512])
sits *pre-inversion* of the chirality flip — empirical
inversion at N_inversion ≈ 591–600 (memory:
project_chirality_flip_pi_over_4 2026-05-05). All Phase-2
data therefore characterise the **vacuum branch**
(θ_chir < π/4). The post-inversion (matter-branch) regime
N ≥ 600 is *not* yet covered and is registered as an open
audit obligation (FB-w4 below).

The weighted-Laplacian asymptote `lambda_w_inf = 3/8 = (d-1)/(2d)`
isolates a **pure spectral-graph identity** in the carrier
dimension `d=4` alone. It does **not** factor through any
System-R primitive (γ, α_ξ, β_π, ε²_sync, D_Ω) or any
matter-sector observable. Numerical checks (Y2 cross-
connection audit, both branches):

  - `3/8 / Λ_t^vac = 3/8 / (33/40)   = 5/11 ≈ 0.4545` (trivially
    rational; numerators/denominators share no physical
    primes — coincidence of two reducible rationals, not a
    System-R relation)
  - `3/8 / Λ_t^mat = 3/8 / (α_ξ²)    = 0.4630` (no rational)
  - `3/8 / T_00^vac = 3/8 / (84/100) = 0.4464` (no rational)
  - `3/8 / Λ_trace = 3/8 / (161/200) = 0.4658` (no rational)
  - Closest System-R combination `γ + 1/d = 0.350`:
    +6.7% off; no clean rational within 5%.

This confirms the **sector-decomposition principle**: the
geometric closures (`Λ_t^vac = 33/40`, `Λ_t^mat = α_ξ² =
81/100`, `Λ_μν trace = 161/200`, `T_00^vac = 84/100`) live
in the System-R coefficient algebra (α_ξ-dependent,
branch-resolved at θ_chir = π/4), while the spectral-gap
asymptote lives in the dimensional-graph algebra
(d-dependent only). They share *no* algebraic cross-term
and must be derived from disjoint analytical mechanisms:

  - Λ_t: branch-resolved Lagrangian (chirality-locked,
    P4-B anisotropic source; jumps 33/40 ↔ 81/100 at the
    flip).
  - 3/8: small-world spectral theory on the τ=0.10
    skeleton, Friedman-Bulk + isolated-eigenvalue
    decomposition (Lemma B Step 4). **Currently certified
    on the vacuum branch only.**

The independence is a *feature*, not a defect: it permits
Lemma B (analytical) to be settled without re-opening any
Lambda-closure, and conversely a future revision of Λ_t
would not propagate into the spectral-gap target. Whether
the **value** of `lambda_w_inf` itself jumps across θ_chir = π/4
is an open empirical question (FB-w4).

## Falsification triggers for the 3/8 weighted-Laplacian conjecture

The 3/8 = (d−1)/(2d) conjecture for
`lambda_inf^weighted` is empirical (Symanzik-1 fit at 1.0%
relative residual). It is currently certified on the
vacuum branch only (N < 591..600 = N_inversion). Four
explicit falsifiers:

- **(FB-w1)** On an extended ladder N ≥ 1024 the cross-seed
  mean λ_2(L_w) drops outside [0.365, 0.385] (i.e., outside
  ±2.7% of 3/8). The current Symanzik-1 fit predicts
  asymptote 0.379, well within the band.

- **(FB-w2)** A higher-order N-scaling model (e.g.,
  Symanzik-2 with parameters λ_∞ + a/N + b/N²) prefers an
  asymptote distinct from 3/8 by more than 2σ on the
  bootstrap-CI of the asymptote across an extended ladder.

- **(FB-w3)** The full-Laplacian asymptote on the alt-anchor
  ladder (P0..P8) at any single physics regime (rather than
  pooled) is more than 5% away from 3/8 at the corresponding
  N. (Pointwise check at the overlap N=50: λ_2 = 0.515 for
  both canonical and alt; consistency holds.)

- **(FB-w4)** Post-inversion regime (N ≥ N_inversion ≈
  591–600, θ_chir > π/4 → matter branch). A future ladder
  N ∈ [600, 1024] that yields a Symanzik-1 asymptote
  outside [0.365, 0.385] would either falsify the conjecture
  outright OR force a **branch-resolved** restatement
  `lambda_w_inf^vac = 3/8` (already empirical) and
  `lambda_w_inf^mat = …` (to be determined). The pre-flip
  evidence does not predict the post-flip value because the
  3/8 mechanism (small-world spectral geometry of the
  τ=0.10 skeleton) plausibly survives the chirality flip,
  but the |Ξ|/K/Q field-content does jump (cf. Λ_t
  33/40 ↔ 81/100). A jump in the skeleton edge-distribution
  would translate to a jump in λ_w_inf.

Trigger (FB-w1) is the deciding empirical falsifier *within
the vacuum branch*; FB-w4 is the deciding test for the
universality across the chirality flip. The current data
do not distinguish between (a) `lambda_w_inf = 3/8`
universally and (b) `lambda_w_inf` branch-resolved with
`vac = 3/8`.
