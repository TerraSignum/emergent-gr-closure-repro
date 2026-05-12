# Lemma B — uniform Poincaré inequality on admissible relational carriers

**Status:** empirically certified on the 10-regime canonical-physics
ladder (184 seeds); analytical proof open. This document is the
formal statement, the empirical certification summary, and the open
analytical question for downstream work.

## 1. Setting

Fix integers `d = 4` (relational-spacetime dimension) and
`N_gen = 3` (fermion-generation count). Let

```
{ (V_N, Xi_N, mu_N) }_{N >= N_0}
```

be a sequence of finite relational carriers in the sense of P6:

- `V_N` is a finite vertex set of cardinality `N`;
- `Xi_N : V_N × V_N -> [xi_min, 1]` is symmetric with unit
  diagonal, satisfies the submultiplicative-triangle inequality
  `Xi_N(i,j) * Xi_N(j,k) <= Xi_N(i,k)` (M3), and is bounded below
  uniformly by `xi_min > 0` (admissibility floor);
- `mu_N` is the counting measure on `V_N`.

The associated metric is `d_N(i,j) = -log Xi_N(i,j)`. The
Xi-weighted symmetric normalised graph Laplacian on `V_N` is

```
L_N  =  I - D_N^{-1/2} W_N D_N^{-1/2},
W_N  =  Xi_N - I,
D_N  =  diag(W_N · 1).
```

`L_N` is symmetric positive semidefinite with spectrum
`{0 = lambda_1(L_N) < lambda_2(L_N) <= ... <= lambda_N(L_N) <= 2}`.

## 2. Statement (target Lemma B)

**Lemma B (uniform spectral gap / Poincaré inequality).**
There exist constants `lambda_* > 0` and `N_0 < infty`, depending
only on `(d, N_gen, xi_min)` and the carrier-action construction
of P6, such that

```
lambda_2(L_N)  >=  lambda_*       for all  N  >=  N_0.    (B)
```

Equivalently, the discrete Poincaré inequality

```
sum_i mu_N(i) ( f(i) - <f>_N )^2
    <=  C_P  ·  sum_{i,j} W_N(i,j) ( f(i) - f(j) )^2 / sum_i deg_N(i)    (P)
```

holds with `C_P = 1/lambda_*` for every `f : V_N -> R`, uniformly
in `N >= N_0`. Here `<f>_N` is the unweighted mean.

**Asymptotic statement (Lemma B').** The map `N -> lambda_2(L_N)`
admits the Symanzik-1 expansion

```
lambda_2(L_N)  =  lambda_inf  +  a / N  +  o(1/N)      (B')
```

with `lambda_inf > 0` and `a > 0`, finite-size correction.

`(B')` implies `(B)` for `N_0` large enough.

## 3. Empirical certification

Reproducer: `src/verify_lemma_B_uniform_poincare.py`.
Output:    `outputs/verify_lemma_B_uniform_poincare.json`.

Ladder `N ∈ {50, 64, 72, 84, 100, 128, 200, 256, 300, 512}`,
184 seeds total (28+24+24+24+24+12+8+12+12+12), per-regime
last-timestep `Xi_N` snapshot, normalised graph Laplacian
`lambda_2(L_N)` per seed, cross-seed mean ± bootstrap-95% CI.

### 3.1 Per-regime results

| Regime  |   N | seeds | mean λ₂ | min    | max    | 95% CI         | C_P max |
|---------|----:|------:|--------:|-------:|-------:|----------------|--------:|
| P5      |  50 |    28 | 0.5146  | 0.4487 | 0.5812 | [0.505, 0.526] |   2.229 |
| P5N64   |  64 |    24 | 0.4862  | 0.4327 | 0.5388 | [0.476, 0.497] |   2.311 |
| P5N72   |  72 |    24 | 0.4754  | 0.4342 | 0.5352 | [0.466, 0.485] |   2.303 |
| P5N84   |  84 |    24 | 0.4609  | 0.4048 | 0.5000 | [0.451, 0.470] |   2.470 |
| P5N100  | 100 |    24 | 0.4410  | 0.4126 | 0.4940 | [0.434, 0.450] |   2.424 |
| P5N128  | 128 |    12 | 0.3969  | 0.3799 | 0.4274 | [0.390, 0.406] |   2.632 |
| P5N200  | 200 |     8 | 0.4198  | 0.4067 | 0.4361 | [0.413, 0.427] |   2.459 |
| P5N256  | 256 |    12 | 0.4067  | 0.3717 | 0.4276 | [0.398, 0.414] |   2.690 |
| P5N300  | 300 |    12 | 0.4049  | 0.3932 | 0.4339 | [0.399, 0.411] |   2.543 |
| P5N512  | 512 |    12 | 0.4012  | 0.3824 | 0.4133 | [0.396, 0.406] |   2.615 |

Cross-ladder minimum: `lambda_* = 0.3717` (worst-seed, P5N128);
empirical Poincaré-constant upper bound `C_P <= 2.69`.

### 3.2 N-scaling model selection

Five competing models, AICc-ranked:

| Model                             | params               | AICc    | ΔAICc  |
|-----------------------------------|----------------------|--------:|-------:|
| **Symanzik-1: λ_inf + a/N**       | λ_inf=0.379, a=6.62  | −83.17  |  0.00  |
| Symanzik-2: λ_inf + a/N + b/N²    | λ_inf=0.391, a=3.16  | −80.75  | +2.42  |
| Symanzik-1/2: λ_inf + a/√N        | λ_inf=0.332, a=1.19  | −78.17  | +5.01  |
| Power-law: c · N^(-α)             | c=0.738, α=0.106     | −74.03  | +9.14  |
| Const: c                          | c=0.441              | −62.19  | +20.99 |

**Symanzik-1 is the preferred model (ΔAICc=2.42 over Symanzik-2,
20.99 over const).** The continuum asymptote is

```
lambda_inf  =  0.3789,     C_P^inf  =  1 / lambda_inf  =  2.6392.
```

The empirical evidence is consistent with Lemma B' with
`lambda_inf > 0`; in particular Lemma B holds on the canonical
ladder with `N_0 = 50` and `lambda_* >= 0.37`.

### 3.3 Counterexample reference

The constant-Xi family `C_N(alpha) = alpha · J + (1-alpha) · I`
with `alpha in (0, 1)` satisfies M0–M3 and admissibility uniformly
in `N`, but its doubling ratio grows linearly with `N` (see
`verify_admissibility_counterexample_and_spectral_gap.py`,
section 1). For `C_N(alpha)` one computes directly:

```
lambda_2(L_{C_N(alpha)})  =  1  -  alpha / 1  =  1     (alpha < 1)
```

…wait, this is *not* the right computation. The normalised
Laplacian of a complete graph with uniform weights `alpha` has

```
W_N  =  alpha · (J - I),     D_N  =  alpha (N-1) · I,
norm_W  =  W_N / (alpha (N-1))  =  (J-I) / (N-1),
L_N  =  I - (J-I)/(N-1)  =  N/(N-1) · I  -  J/(N-1),
eigenvalues of L_N : 0 (mult 1) and N/(N-1) (mult N-1).
```

So `lambda_2(C_N(alpha)) = N/(N-1) -> 1` as `N -> infty`. The
counterexample violates A8 (Ahlfors-regular doubling) *not*
because the spectral gap collapses, but because the metric
becomes degenerate (all off-diagonal pairs have identical
distance). This is consistent with the picture:
**uniform spectral gap is necessary but not sufficient for A8**;
Lemma B is the *sharpened* missing structural axiom, not the
unique one.

## 4. Open analytical questions

### 4.1 The principal open question

> **Open (Lemma B, analytical).** Given the carrier-action
> construction of P6 and the integer inputs `(d, N_gen) = (4, 3)`,
> derive an explicit lower bound `lambda_*((d, N_gen, xi_min)) > 0`
> on `lambda_2(L_N)` valid for all `N >= N_0`.

Three candidate analytical routes are surveyed in
[`lemma_B_proof_strategy.md`](lemma_B_proof_strategy.md).

### 4.2 What `lambda_inf` is *not*

- It is **not** the Yang–Mills mass gap (this is a graph-Laplacian
  spectral gap, not the gap of a self-adjoint Hamiltonian on a
  QFT Hilbert space).
- It is **not** automatic from M0–M3 + admissibility (constant-Xi
  has full spectral gap but trivial metric).
- It is **not** identical to the percentile-spectrum-law exponents
  of P4 §13 (those concern the asymptotic distribution of the
  Galerkin-Frobenius residual percentile spectrum, not the
  Laplacian spectrum).

### 4.3 Conjectured form of `lambda_inf`

From the rational structure of the carrier-action and the empirical
`lambda_inf ≈ 0.379`:

```
lambda_inf  =?  alpha_xi - gamma   =  9/10 - 1/10  =  0.8        (mismatch)
lambda_inf  =?  alpha_xi^2 - gamma  =  81/100 - 1/10  =  0.71    (mismatch)
lambda_inf  =?  3/8                                   =  0.375   (close, 1.0%)
lambda_inf  =?  alpha_xi/N_gen - gamma                =  0.2     (mismatch)
lambda_inf  =?  alpha_xi · gamma · (d-1)              =  0.27    (mismatch)
lambda_inf  =?  d * gamma + alpha_xi / d              =  0.4 + 0.225 = 0.625 (mismatch)
lambda_inf  =?  alpha_xi - 2 * gamma * (d-1)          =  0.3     (mismatch)
lambda_inf  =?  (2 * d + N_gen - 1) / d^2             =  10/16   = 0.625   (mismatch)
```

The closest System-R rational so far is `3/8 = 0.375` (empirical
asymptote 0.3789, relative residual 1.0%). The next-best candidate,
testable with more N values, is `lambda_inf = 3/8`. **This is a
conjecture to register, not a closure to claim.**

## 5. Reproducer chain

- Generation of `Xi_N` snapshots: existing `results_d1_*/*.npz`
  bundles, produced by the d1 / P5N pipelines of the parent
  package.
- `lambda_2(L_N)` audit: `src/verify_lemma_B_uniform_poincare.py`.
- Counterexample (A8 not implied by M0–M3 + admissibility):
  `src/verify_admissibility_counterexample_and_spectral_gap.py`.
- Output JSON: `outputs/verify_lemma_B_uniform_poincare.json`.
- This statement document:
  `notes/lemma_B_statement.md` (the present file).

## 6. Falsification triggers

- (FB-1) On any extended ladder `N >= 1024` the cross-seed mean
  `lambda_2(L_N)` drops below 0.3 with bootstrap 95% upper-CI
  also below 0.3: Symanzik-1 fit broken, Lemma B' falsified.
- (FB-2) On any admissible carrier sequence (not necessarily
  `(d, N_gen) = (4, 3)`) the Symanzik-1 asymptote `lambda_inf`
  is non-positive: the carrier-construction does not enforce a
  uniform Poincaré inequality, and the unconditional continuum
  theorem program fails.
- (FB-3) The Symanzik-1 model loses AICc preference to a
  model with `lambda_inf <= 0` on a larger ladder (N >= 1024):
  the asymptote claim is overturned.
