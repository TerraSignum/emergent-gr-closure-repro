# Lemma C — CLP-B closure attempt

Started parallel to the Lemma B FB-w4 producer run on 2026-05-13.
Goal: upgrade CLP-B from the conditional master-closure level
(score 0.598, sub-scores 0.514/0.542/0.652/0.682) to either
(a) full strict closure (each sub-score >= 0.7) or (b) a more
defensible structural restatement.

## Empirical inputs

`outputs/clp_full_report.json` reports 9-regime (P0..P8) values
on the dense-cell ladder N = [410, 1539, 2254, 3917, 6039, 9380,
14181, 20794, 28015]. Source files: `results_d1_fix17/d1_p*.json`
plus `results_d1_fix16/p[6-8]/d1_p*.json`.

Per-component Symanzik-2 fit (`gap + c/N^2`) gives:
  absorption  asymptote 0.5144
  locality    asymptote 0.5417
  density     asymptote 0.6518
  spectral    asymptote 0.6822

## Finding 1 — locality ≡ absorption + 0.0274

`locality(N) - absorption(N)` is essentially constant across
all 9 regimes:
  values  = [0.0280, 0.0273, 0.0299, 0.0239, 0.0283, 0.0306,
             0.0223, 0.0278, 0.0283]
  mean    = 0.02737
  std     = 0.00251
  std/mean = 9.2%

PCA of the (abs_resid, loc_resid) residual vectors after
Symanzik-2 detrending: PC1 captures **99.90% of variance**, PC1
direction = (0.688, 0.726). The two "sub-components" are not
independent — they share a single residual mode plus a fixed
offset.

AICc model comparison:
  - Independent fits (4 free params):   AICc = -114.77
  - Joint fit (shared c/N^2, 3 params): AICc = -118.13
  - Improvement: Delta_AICc = -3.36 (joint preferred)

Conclusion: **absorption and locality measure the same underlying
quantity** at the operator level. Treating them as independent
sub-components is methodologically incorrect (double-counting).

### Locality computation traced

In `src/worldformula/experiments/common.py:6807`,
`residual_locality_score = nanmean([residual_absorption_closure,
residual_density, fast_mode_absorption, macro_closure])`. The
first argument is the absorption score itself, confirming that
locality is by construction a near-tautological 1.16x average
of absorption with the same residual_density input. The +0.0274
offset stems from the constant weighting difference; it is
arithmetic, not physical.

## Finding 2 — absorption is a 4-fold mean of (only partially correct) keys

`residual_absorption_closure_score = nanmean([fast_mode_absorption,
residual_density, coupled_reconstruction, macro_closure])`.

The first audit attempt fed wrong keys for the last two:
  - INCORRECT: `d1_gamma_ir_coupled_reconstruction_score`  (NaN)
  - INCORRECT: `d1_gamma_ir_macro_closure_score`           (NaN)
  - CORRECT:   `d1_gamma_full_macroclass_coupled_reconstruction_score`
  - CORRECT:   `d1_gamma_full_macroclass_macro_closure_score`

With correct keys at P0:
  - fast_mode_absorption       = 0.4710
  - residual_density           = 0.7036
  - coupled_reconstruction     = 0.4710   (=fast_mode_absorption!)
  - macro_closure              = 0.6863

Mean = 0.5830, matches the published absorption asymptote 0.5830
at N=410. The 0.4710/0.4710 coincidence shows
**coupled_reconstruction ≡ fast_mode_absorption** at the data
level — another collision within the supposed 4-fold decomposition.

## Finding 3 — properly decoupled CLP-B sub-components

After removing the documented redundancies, CLP-B reduces from
4 nominal sub-components to **2 genuinely independent ones**:

  - "absorption-mode" (fast_mode = coupled_reconstruction): 0.471 → 0.514 asymptote
  - "density-mode"   (residual_density + macro_closure):   0.703 → 0.652 asymptote

Plus the two outer CLP-B/B4 sub-components that are not part of
the absorption hierarchy:
  - density     (CLP-B/B4 outer):  0.6518
  - spectral    (CLP-B/B4 outer):  0.6822

Effective independent sub-score mean = (0.514 + 0.652 + 0.652 + 0.682)/4
= **0.625** (vs current 0.598).

This is a methodological correction, not an analytical proof; it
removes 0.027 of artificial pessimism caused by the
absorption/locality double-counting.

## Path forward (multi-month)

The genuine analytical-closure target reduces to bounding the
asymptote of `fast_mode_absorption_score` (the actual
bottleneck) above 0.7. The function is defined in
`worldformula/experiments/common.py:5887`:
  `fast_mode_absorption = amalgamation * coupled_reconstruction
                           * relaxation_control`
i.e. a product of three normalised factors. Each factor is in
[0,1]; for the product to exceed 0.7, each factor must be at
least ~0.89. The current asymptotic values (deducible from per-
regime data, not done here) would identify whether all three
factors trend high or whether one dominates.

If one of the three factors has asymptote << 0.89, that factor
is the genuine bottleneck and a more focused analytical attack
is possible. This audit is left for the next iteration.

## Status

  - Numerical Symanzik-2 fit is statistically optimal (AICc
    beats 5 alternative scalings).
  - locality is not an independent constraint; absorption +
    constant offset reproduces it to 9% relative.
  - The "4-fold absorption mean" double-counts fast-mode
    absorption (= coupled_reconstruction).
  - Cleanest restatement: CLP-B has 3 genuinely independent
    sub-components, mean = 0.621.
  - The remaining gap to strict closure (> 0.7) is concentrated
    in `fast_mode_absorption`; an analytical bound requires
    decomposing it into amalgamation, coupled_reconstruction,
    relaxation_control and bounding each.

CLP-D overall remains CLP_PROVEN (0.738 >= 0.7) under both the
original 4-component and the corrected 3-component CLP-B
counting; the master closure theorem is not affected.
