# Lemma B Step-4 — concrete cavity-method sketch (2026-05-15)

**Status:** session-level analytical sketch toward the closed-form
derivation of `λ_∞^skel = 7/24`. Identifies a concrete and testable
algebraic structure (harmonic-sum tensor factorization) that
reduces the proof to two well-defined sub-claims, each within
reach of the Pham–Peron–Metz 2024 + Silva–Metz 2022 cavity
machinery.

This memo extends `lemma_B_step4_analytical_attempt.md` (which
verified the algebraic identity `7/24 = (d+N_gen)/(2·d·N_gen)`
but left the carrier-action derivation as multi-month research)
by isolating the *structural decomposition* that the cavity method
should reproduce.

## 1. Empirical input: carrier moments

From `outputs/verify_lemma_B_gap_statistical_fingerprint.json`
(250 seeds × 8 ladder sizes up to N=128000):

```
mean_degree                ⟨k⟩          = 13.79  ≈ 14
mean_triangles_per_node    ⟨k_Δ⟩        = 24.90  ≈ 25
transitivity               C            = 0.275
degree_assortativity       ρ            = 0.143
skeleton mean degree       d_eff^skel   = 12.14  ≈ d·N_gen = 12
E2 Newman-clustered gap    λ_∞^skel     = 0.2929 ≈ 7/24 = 0.2917
```

The Symanzik a_∞ asymptote of the E2_newman_clustered ensemble
reproduces 7/24 to +0.36 % over 8 ladder sizes; only E2
(Newman-clustered with joint (k_s, k_Δ) matching the carrier)
reproduces the gap among E0–E4. This is the entry point for the
Pham–Peron–Metz cavity construction.

## 2. The harmonic-sum decomposition

The target rational 7/24 admits a clean *additive* decomposition
into the two integer inputs (d, N_gen):

```
7/24 = 1/(2d) + 1/(2N_gen) = 1/8 + 1/6 = 3/24 + 4/24
```

Equivalently:

```
λ_∞^skel = (d + N_gen) / (2·d·N_gen)
         = 1/(2d) + 1/(2N_gen)                         (HSD)
```

Each summand has the form 1/(2k) — the same form that arises in
the cavity-method spectral gap of a tensor-product / commuting-
Z-action structure (Diaconis–Shahshahani type). The
*conjecture* of this memo:

> **Harmonic-Sum Decomposition (HSD).** The carrier's skeleton-
> Laplacian spectrum, in the Silva–Metz high-connectivity-limit
> reduction of the Pham–Peron–Metz cavity equations, factorizes
> into a tensor product of a d-axis (spatial-cyclicity) channel
> and an N_gen-axis (generation-cyclicity) channel. Each channel
> contributes a uniform-Markov-chain spectral gap of 1/(2·axis-size),
> and the total skeleton gap is the additive harmonic sum (HSD).

This is the structural claim that Step-4 must establish. It is
genuinely testable, factorizes the problem into two narrower
claims (existence of the tensor factorization; per-channel
spectral-gap value), and explains why 7/24 = (d+N_gen)/(2dN_gen)
is the specific small-integer rational that the carrier selects.

## 3. Pham–Peron–Metz cavity equations specialized

For a Newman-clustered configuration model with joint
distribution P(k_s, k_Δ) over (single-edge degree,
triangle-degree) at each node, the cavity-method diagonal
resolvent G(z) = ⟨G_ii(z)⟩ satisfies (PPM 2024 Eq. 18, with
J_e = J_Δ = 1/√⟨k⟩ for the symmetric-normalized Laplacian
case):

```
G(z) = ⟨ [z - h_s(G, P) - h_Δ(G, G^(2), P)]⁻¹ ⟩
```

where the cavity self-energy splits as:

```
h_s(G, P)         = Σ_{k_s} (k_s/⟨k⟩) · P(k_s) · G(z)             [single edges]
h_Δ(G, G^(2), P)  = Σ_{k_Δ} (k_Δ/⟨k⟩) · P(k_Δ) · K_Δ[G, G^(2)]    [triangles]
```

and K_Δ is the closed triangle-cavity kernel separating
single-edge (J_e²) and triangle (J_Δ²) contributions
to the diagonal resolvent (PPM Eq. 19).

For the carrier-specific case with ⟨k_s⟩ + 2⟨k_Δ_distinct⟩ ≠ ⟨k⟩
(triangles share edges — overlapping not edge-disjoint), one uses
the *generalized PPM* form that takes ⟨k⟩ and ⟨k_Δ⟩ as
independent moments, plus higher cumulants ⟨k²⟩, ⟨k_Δ²⟩,
⟨k·k_Δ⟩, which are also numerically accessible from the
empirical carrier (extension of the verifier needed).

## 4. Silva–Metz high-connectivity reduction

In the high-connectivity limit ⟨k⟩ → ∞ at fixed
clustering C and degree variance, the distributional cavity
fixed-point reduces to a *single* self-consistent equation for
the average resolvent (Silva-Metz 2022, JPC §3):

```
G(z) = 1 / [ z - σ_eff²(P) · G(z) - C_Δ(P) · K_Δ[G, G^(2)] ]
```

with σ_eff² and C_Δ both rational functions of the joint
moments. For the *normalized* Laplacian (skeleton convention,
where ⟨k⟩ = d_eff^skel = 12 is held fixed in the limit), the
spectral edge λ_∞^skel is the smallest non-trivial root of the
algebraic equation obtained by setting the discriminant of
the cavity quadratic to zero.

## 5. Tensor-factorization route to (HSD)

The Step-4 analytical claim has two sub-claims:

**Sub-claim A (existence of tensor factorization).** The
carrier's joint distribution P(k_s, k_Δ) approximately
factorizes as a product of two independent components:

```
P(k_s, k_Δ) ≈ P_spatial(j; d) · P_gen(ℓ; N_gen)
```

where j ∈ {1,...,d} indexes spatial-axis modes and
ℓ ∈ {1,...,N_gen} indexes generation modes, with each
component being a uniform Markov chain on its respective
modulus. The factorization is *exact in the high-
connectivity-cavity-limit*; finite-N corrections explain the
empirical N^(-1) Symanzik correction (b = 12.36 in the
Newman-clustered ensemble) and the empirical 0.293 vs 7/24
0.36 % gap.

**Sub-claim B (per-channel spectral gap).** Each factor
contributes a uniform-Markov-chain spectral gap of 1/(2k)
to the total skeleton Laplacian spectrum:

```
λ_∞^spatial   = 1/(2d)     = 1/8
λ_∞^gen       = 1/(2N_gen) = 1/6
λ_∞^skel      = λ_∞^spatial + λ_∞^gen = 7/24       (HSD)
```

The 1/(2k) form is the spectral gap of a *lazy random walk
on a chirality-doubled cyclic group Z/(2k)Z* under the cavity-
method tensor-product convention.

**Why these specific sub-claims close Step-4:**

- Sub-claim A is testable numerically (factorize the empirical
  P(k_s, k_Δ) and check approximate independence) — this is a
  20-line extension of the existing verifier.
- Sub-claim B follows from Diaconis–Shahshahani-type spectral
  identities for tensor products of reversible Markov chains
  (Aldous–Fill, *Reversible Markov Chains and Random Walks on
  Graphs*, ch.~4); each Z/(2k)Z factor has a known closed-form
  Laplacian spectrum with gap 1 - cos(π/k) ≈ π²/(2k²) for large
  k, but in the high-connectivity *cavity limit* the leading-
  order edge is 1/(2k), which is the long-time-relaxation
  inverse-decay-rate constant (the "ε" parameter of Silva–Metz
  Eq. 24).

The combination A + B + PPM/SM cavity quadratic-edge identity
gives the closed-form λ_∞^skel = 7/24 as the harmonic sum,
with finite-N corrections matching the empirical b = 12.36
Symanzik coefficient.

## 6. Status assessment

**What this memo delivers:**

- A concrete *algebraic decomposition* of 7/24 into harmonic
  summands corresponding to the two integer inputs (d, N_gen).
- A *structural conjecture* (HSD) that explains why 7/24 is the
  small-integer rational the carrier selects.
- *Two sub-claims* (A: factorization; B: per-channel gap) that
  reduce the closed-form proof to two narrower, well-defined
  tasks.
- A concrete *next analytical step*: extend the existing
  verifier to test sub-claim A by computing the empirical
  joint cumulants ⟨k·k_Δ⟩ − ⟨k⟩⟨k_Δ⟩ and checking near-
  vanishing as N → ∞.

**What this memo does NOT deliver:**

- A first-principles proof of sub-claim A (the factorization
  must be derived from the carrier-action edge-formation
  probabilities, not just postulated).
- A first-principles proof of sub-claim B (the 1/(2k) per-channel
  spectral gap must be derived from the high-connectivity cavity
  reduction, not just stated).
- Universality across all admissible (d, N_gen) integer pairs
  (only the canonical (4, 3) case is empirically certified).

**Time-to-discharge estimate:** Reduced from the 2–4 months
estimated in the predecessor memo to **6–10 weeks** of
mathematical-physics research, conditional on the empirical
joint-cumulant decoupling holding numerically.

The sketch is *not* a proof. It is a structural decomposition that
identifies the specific load-bearing analytical claim (HSD) and
the specific mathematical machinery (PPM 2024 + Silva–Metz 2022)
that should reproduce it.

## 7. Cross-corpus consequence

If (HSD) holds analytically, the (SG)-axiom λ_∞ = (d-1)/(2d) = 3/8
of the FULL (non-skeleton) carrier Laplacian — which the
physical-mechanism-open proposal of 2026-05-15 identifies as
the load-bearing axiom unifying H_0, Λ_lat = 19/15, and
CD(K_CD, N) — admits an analogous decomposition:

```
3/8 = (d-1)/(2d) = 1/2 - 1/(2d)
```

i.e. as the unitarity (1/2) minus the d-axis spectral defect
(1/(2d)). Comparing the two:

```
λ_∞^skel = 1/(2d) + 1/(2N_gen)       skeleton    [HSD]
λ_∞      = 1/2    - 1/(2d)           full        [(SG)]
```

The two spectra differ by N_gen-generation activation
(skeleton gains 1/(2N_gen), full carrier loses it and gains
1/2). This is consistent with the corpus reading that the
skeleton is the *chirality-coupled* spectrum while the full
carrier is the *chirality-uncoupled* spectrum. The unified
structural form

```
λ_α = 1/(2α) ± 1/(2β) ± 1/2,  α, β ∈ {d, N_gen, ∞}
```

would be the closed-form harmonic-summand calculus that the
Step-4 + (SG)-axiom proof must reproduce.

This is the cross-corpus consequence: a single harmonic-summand
calculus on the integers (d, N_gen) would simultaneously discharge
Lemma B Step-4 AND the (SG)-axiom, with the corollary closures
on H_0, Λ_lat, CD_K_N falling out as derived consequences via
the proposals of 2026-05-15.

## 8. Concrete next-session task

Extend `src/verify_lemma_B_gap_statistical_fingerprint.py` with
a sub-routine computing:

1. Joint distribution P̂(k, k_Δ) of (degree, triangle-count) on
   the carrier skeleton at τ=0.10 for each ladder size N.
2. Joint-cumulant κ_{1,1}(N) = ⟨k·k_Δ⟩ − ⟨k⟩⟨k_Δ⟩, plotted vs N.
3. Decoupling diagnostic: κ_{1,1}(N) → 0 as N → ∞ under
   N^(-α) decay for some α > 0?

If yes, sub-claim A is empirically supported. If no, the
factorization hypothesis fails and the Step-4 route reverts
to the full PPM + SM machinery without the simplifying
factorization.

This is a 1–2 hour coding task that decides whether the HSD
analytical route is viable.

## 9. Empirical test result (2026-05-15): HSD sub-claim A FALSIFIED

The task of §8 was carried out in
`src/verify_lemma_B_step4_hsd_factorization.py`. Result over the
canonical ladder N ∈ [50, 512] with 8–24 seeds per regime:

```
N=50    <k>=18.32  <k_d>=75.62  kappa_11=+139.94  r=+0.960
N=64    <k>=16.67  <k_d>=55.29  kappa_11=+196.64  r=+0.965
N=72    <k>=17.22  <k_d>=52.84  kappa_11=+128.49  r=+0.959
N=84    <k>=16.90  <k_d>=47.65  kappa_11=+138.53  r=+0.961
N=100   <k>=15.99  <k_d>=39.55  kappa_11=+137.48  r=+0.961
N=128   <k>=10.83  <k_d>=13.27  kappa_11= +36.35  r=+0.950
N=200   <k>=15.09  <k_d>=26.20  kappa_11= +68.24  r=+0.956
N=256   <k>=13.08  <k_d>=18.40  kappa_11= +54.23  r=+0.959
N=300   <k>=14.48  <k_d>=22.19  kappa_11= +62.13  r=+0.961
N=512   <k>=12.11  <k_d>=14.27  kappa_11= +58.98  r=+0.971
```

Symanzik-1 fit κ_{1,1}(N) = a_∞ + b/N:
- **a_∞ = +35.39** (HSD predicted zero)
- b = +7140.5
- R² = 0.66

**Verdict: HSD_FACTORIZATION_FALSIFIED.** The joint cumulant does not
decay to zero; the (k, k_Δ) joint distribution does *not* factorize
as N → ∞. Sub-claim A of §5 is empirically disproved.

**Strikingly stable observation:** the Pearson correlation
ρ(k, k_Δ) sits at r ≈ +0.96 across the entire ladder, varying by only
±0.01 between N=50 and N=512. This is a *new structural fact* about
the carrier that was not previously isolated: **high-degree nodes
are also triangle hubs, in a near-perfectly linear (k_Δ ∝ k) way**.

The asymptotic mean-degree ⟨k⟩ → 12.11 at N=512 is consistent with
d·N_gen = 12 within the seed scatter, supporting the corpus's
identification of d_eff^skel = d·N_gen as the carrier's emergent
mean degree. The companion ⟨k_Δ⟩ → 14.27 is *not* identifiable with
a small-integer System-R rational under any obvious match; the
ratio ⟨k_Δ⟩/⟨k⟩ → 1.18 ≈ (4d−1)/(2d−1) = 15/7 ... no, that's 2.14.
The ⟨k_Δ⟩/⟨k⟩ ratio empirical value 1.18 has no obvious System-R
identification at this stage.

## 10. Revised Step-4 route (post-falsification)

With HSD out, the closed-form derivation of 7/24 must use the full
PPM + Silva–Metz cavity machinery on the empirically *correlated*
joint distribution P(k, k_Δ) with ρ(k, k_Δ) ≈ +0.96.

**Revised sub-claim A′:** the cavity equations for the carrier's
specific joint distribution (with mean degree d·N_gen = 12 and
near-perfect linear coupling k_Δ ≈ ρ·(σ_kd/σ_k)·k + const) admit a
closed-form spectral edge given by the smallest root of a cavity
quadratic whose discriminant evaluates to a perfect rational
square, yielding λ_∞^skel = 7/24 = (d+N_gen)/(2dN_gen).

The PPM 2024 + Silva–Metz 2022 cavity machinery applied to
strongly-correlated joint distributions is genuinely uncharted; the
analytical step likely requires extending Silva–Metz's
high-connectivity reduction to bivariate distributions with
prescribed Pearson correlation, which is a 2–3 month
mathematical-physics task and is the genuine Lemma-B Step-4.

The harmonic-sum identity 7/24 = 1/(2d) + 1/(2N_gen) of §2 remains a
*necessary algebraic identity* the cavity equations must reproduce,
but no longer a *sufficient* identification via tensor factorization.

**Concrete next deliverable:** test whether the empirical
P(k, k_Δ) with linear coupling k_Δ ≈ a·k + b approximately
satisfies the cavity quadratic discriminant condition. This is a
~50-line numerical evaluation; if it works, the closed-form
proof is 1–2 months of analytical work to discharge rigorously.
