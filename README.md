# emergent-gr-closure-repro

**Emergent Einstein dynamics from relational metric closure on finite correlation lattices.**

[![CI: reproduce](https://github.com/[anonymized]/emergent-gr-closure-repro/actions/workflows/reproduce.yml/badge.svg)](https://github.com/[anonymized]/emergent-gr-closure-repro/actions/workflows/reproduce.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository reproduces the emergent-Einstein closure of the Emergence
program: a relational similarity Xi -> distance d_ij = -ell_0 * log(Xi_ij)
under M1-M3, a fast-slow metric closure, four convergence-axis scores,
the Einstein-identity gap at exponent 2/3, and PPN gamma=beta=1.

## Result in one line

```
Metric axioms M1-M3 hold under stated assumptions.
Fast-slow flow stabilizes the quasi-metric tube on the canonical regime.
Convergence axes (A)/(B)/(C)/(D) = 0.8157 / 0.8046 / 0.8712 / 0.8300
  (all >= 0.70).
Einstein-gap two-point Richardson at three candidate exponents
  (Ricci-order 2/3, linear 1.0, empirical 0.8477) all yield
  |gap_inf| < 0.05 from N1=1534 and N2=2254.
PPN gamma_PPN = beta_PPN = 1 within Cassini/LLR experimental band.
Lambda_lat^infty = 19/15 = 17/20 + 5/12 unconditional System-R
  algebraic identity in Q, with 17/20 = alpha_xi + eps_sync^2 - gamma
  (non-scalar Clifford-channel, K_rec component) and 5/12 =
  alpha_xi/2 - gamma/N_gen (Lemma 1 spinor-trace minus Lemma 5
  generation, gradient component); under per-component-appropriate
  disciplines (K_rec asymptotic-window mean P6..P8 = 0.848 vs 17/20
  at 0.20%; gradient alpha=2/3 extrapolation = 0.416 vs 5/12 at
  0.06% with LOO-jackknife 0.05 sigma) the combined Lambda_inf =
  1.265 matches 19/15 to 0.16% relative.
Hilbert-variation source-side: emergent T_munu^Xi is anisotropic
  (T_00 ~ +1.36, T_ii ~ -0.42 in the asymptotic window),
  w_eff = T_ii/T_00 ~ -0.31 at the accelerated-expansion threshold;
  NEC, WEC, DEC robustly satisfied 9/9 across the lattice ladder,
  SEC saturated at the boundary (8/9 pass; only P0=N=18 marginally
  fails as a finite-size effect). The asymptotic-window EOS is
  close to the cosmic-string-network analytical EOS w = -1/3
  (Vilenkin-Shellard isotropized network), with the lattice
  carrying actual U(1) topological vortex defects (integer-
  quantized per-triangle winding, max residual 0.0000 across
  all 9 regimes; corpus-canonical vortex_count field).
  Caveat (Phase O'): the linear-regression intercept "vortex-
  density-zero limit at w_bg = -1/3 exact" reading is not
  uniquely supported by the data -- regressions against any
  monotonic function of N give intercept ratios in the band
  -0.27 to -0.39, so the 0.4% Phase-O tightness is a regressor-
  choice coincidence rather than a structural decoupling
  identification. Caveat (Phase R): per-seed dispersion test
  shows chi^2/dof = 1.28 (T_00) and 0.80 (T_ii) for the
  N-independent constant hypothesis; the apparent N-trend is
  STATISTICALLY CONSISTENT WITH NO TREND modulated by seed
  noise. Pooled w_eff = -0.310 (= -0.426 / +1.373).
  Load-bearing reading (Phase S): the diagonal-block T_munu^Xi
  decomposes into a two-component dark-matter + dark-energy
  mixture, with vortex-topology pressureless DM (w_DM = 0)
  and a dark-energy component w_DE = -1 + eps^4/gamma = -0.975
  established independently in the companion landing-protocol
  paper (closes at 0.05% there). Self-consistent extraction
  from T_00 = +1.373, T_ii = -0.426: rho_DM = +0.936,
  rho_DE = +0.437, f_DM = 0.682, f_DE = 0.318; predicted
  w_eff = -0.310 matches lattice exactly. Phase T per-seed
  robustness: NEC margin +0.947 +/- 0.001 (~1666 sigma above
  0; non-phantom + anisotropy claim solid); SEC margin +0.095
  +/- 0.013 (~7.5 sigma above 0; SEC-positive boundary
  saturation). Falsifiable cosmological-epoch prediction:
  rho_DM/rho_DE = 2.14 corresponds to redshift z ~ 0.67 in
  standard LCDM (matter-DE equality at z ~ 0.30). w_DE = -0.975
  sits within Pantheon+CMB+BAO 1-sigma band w_0 = -0.978(+24,-31)
  and is consistent with post-DESI w(z) > -1 trends.
```

## Scope

This package presents the emergent-GR closure conditional on:
- the M1-M3 metric axioms on Xi_ij;
- the fast-slow flow stabilizing a quasi-metric tube on the canonical regime;
- the four convergence-axis scores (axes (A)/(B)/(C) are aggregate-only here;
  axis (D) is the two-point Richardson bridge fully exposed in
  data/einstein_gap_results.json);
- the structural target alpha_gap = 2/3 with the two-point Richardson
  construction; supplementary multi-observable convergence witnesses
  are bundled and reproduce as follows:
    * five-point chirality-deviation fit, alpha_fit = 0.6355, R^2 = 0.78
      (data/einstein_gap_5point_fit.json,
       src/verify_einstein_gap_5point_fit.py);
    * nine-point R_bar load-bearing curvature-side ladder,
      R_bar^infty = -0.004, R^2 = 0.83 at alpha = 2/3
      (data/einstein_gap_9point_witnesses.json,
       src/verify_einstein_gap_9point_witnesses.py); same file carries
      the nine-point chirality-balance ladder. The global-mean
      chirality fit at alpha = 2/3 gives Delta_infty = +0.022 with
      R^2 = 0.55, but the global mean is noise-dominated on the
      seed-corrected within-canonical-regime ladder
      (per-seed std/mean ~ 65%, statistically indistinguishable from
      a Wishart Xi baseline). The chirality witness is therefore
      retracted in its global-mean form and replaced by the per-node
      bulk-percentile sup-decay analogue (P4-B / H3c v9 style):
      sup_i |chir_i(N)| follows a clean power law with R^2 = 0.99
      and free-fit alpha = 0.8136, bootstrap CI95 [0.76, 1.02].
      The closest framework rational to the free-fit is
      alpha_xi^2 = 81/100 = Lambda_t (the time-time cosmological-
      constant tensor coefficient in the emergent Einstein equation,
      L9 row of the P2 landings table), within 0.44% of the
      free-fit and statistically tied with the AICc-best
      single-parameter model (Delta AICc = +0.001). The diffusion
      identity D_Omega = beta_pi - gamma = 67/80 = 0.8375 that
      anchors the strict-EXACT charged-lepton mass and primordial
      scalar amplitude closures of the loop-class library (P3) is
      also in CI95 (Delta AICc = +0.033) but 2.86% from free-fit;
      retained as alternative structural candidate. The previous
      alpha = 2/3 chirality identification is excluded by the
      bootstrap CI95 (Delta AICc = +1.25). alpha_xi^2 > 2/3, so
      the chirality witness saturates the Theorem 15.18 P2 bound
      at the larger exponent (faster decay than the analytical
      bound requires); the cross-observable match
      alpha = alpha_xi^2 = Lambda_t on independently-defined
      observables (chirality sup-decay vs. time-time backreaction
      asymptote y_inf = 0.8134 on the same lattice) is the new
      empirical signature. The R_bar nine-point ladder
      (curvature-side primary witness) gives free-fit alpha =
      0.843, R^2 = 0.87, also in [0.80, 0.90] band well above 2/3,
      with closest match to D_Omega = 67/80 (0.68%) or 17/20
      = alpha_xi - gamma/2 = 0.85 (0.80%) -- distinct from
      chirality at alpha_xi^2; the two witnesses are
      structurally distinct observables with distinct decay rates
      both saturating the Theorem 15.18 P2 bound with margin.
      See emergent-gr-h3c-witnesses-repro/paper for details;
    * eight-point T_00^Xi + Lambda source-side consistency on the
      time-time component of the full Einstein equation
      G_munu + Lambda * g_munu = 8 pi G * T_munu^Xi, with
      Lambda_eff = 0.314 (CV = 3.3% over the asymptotic window
      P_4..P_8) and pointwise |residual| < 0.05 for N >= 30
      (data/einstein_with_lambda_8point.json,
       src/verify_einstein_with_lambda.py);
    * three convention-dependent System-R rational identifications
      of Lambda_lat^infty: proxy convention -> 1/4 = alpha_xi/2 -
      2*gamma (Bekenstein-Hawking 1/d_spacetime constant, 0.40%
      match); row-mean Definition 12.20 -> 17/20 = alpha_xi +
      eps_sync_sq - gamma (non-scalar Clifford-channel reaction
      rate, 0.12% match); Section-14.1 Laplace-Beltrami -> ~6/5
      (1.64% match, look-elsewhere boundary)
      (src/verify_lambda_system_R.py reproduces algebraically
      from the rational System-R coefficients);
    * classical energy-condition test on the Phase-G anisotropic
      diagonal block (rho = T_00 ~ +1.36, p = T_ii ~ -0.42,
      w_eff ~ -0.31 over the asymptotic window): NEC, WEC, DEC
      robustly satisfied 9/9 across the lattice ladder; SEC
      saturated at the boundary (rho + 3p ~ +0.11, ~8% of rho;
      8/9 pass with only P0=N=18 marginally failing by -0.009
      as a finite-size effect). The emergent source is NOT
      phantom, NOT a pure cosmological constant, and NOT ordinary
      matter -- it sits precisely on the gravitational dividing
      line between attractive and acceleration-driving sources
      (data/lattice_diagonal_T_munu_9point.json,
       src/verify_lambda_energy_conditions.py);
    * trace anomaly and continuum extrapolation:
      T^mu_mu / rho ~ -1.92 in the asymptotic window (within 4%
      of SEC saturation -2; far from radiation 0 or pure-Lambda
      -4); alpha=2/3 continuum extrapolation gives
      w_eff^inf ~ -0.28 with r^2 = 0.80, sitting ~5.6% on the
      SEC-positive side of the saturation line; the SEC margin
      grows from ~8% rho at finite N to ~18% rho in the
      continuum. The emergent source is non-conformal,
      non-pure-Lambda, and remains physically admissible in the
      strict continuum (src/verify_lambda_trace_and_continuum.py);
    * vortex-background structural decomposition: per-regime
      Pearson correlations across the full 9-point ladder show
      r(T_00, N_KZM) = -0.79, r(T_ii, N_KZM) = +0.80,
      r(w_eff, N_KZM) = +0.82 between (T_00, T_ii) and the
      Kibble-Zurek family-resolved defect density. Linear
      decomposition T_munu = T_munu^bg + N_KZM * T_munu^per-vortex
      extracts a vortex-density-independent background at
      w_bg = -0.3347 (within 0.4% of SEC saturation -1/3) and
      a per-vortex modulation at w_pv = -0.78 (phantom-like).
      Leave-one-out jackknife: w_bg^LOO = -0.335 +/- 0.003,
      range [-0.342, -0.329] containing -1/3. Independent
      cross-check using the corpus-canonical vortex_count
      field (per-seed total of active winding triangles, NOT
      subject to the winding_map truncation), regressing T_munu
      against per-node density v/N: w_bg^geometric = -0.3252
      (within 2.4% of -1/3, slope ratio -0.775 consistent
      with Phase O -0.78). The five KZM classes have fixed
      proportional ratios across regimes ~ sqrt(1, 1.5, 2, 3, 4)
      and are graduated quench-rate levels, NOT 5 independent
      fermion generations
      (data/lattice_topological_observables_9point.json,
       src/verify_lambda_vortex_background_decomposition.py,
       src/verify_lambda_vortex_quantization_geometry.py);
    * analytical cosmic-string-network stress-tensor comparison:
      the Vilenkin-Shellard isotropized random-orientation
      string network has w_string = -1/3 EXACTLY (textbook
      result; Cosmic Strings & Other Topological Defects, 1994,
      Sec. 11). The Phase O extracted background w_bg = -0.3347
      matches w_string = -1/3 to 0.4% relative; the Phase G/L/M/O
      "anisotropy headline" has a clean analytical structural
      analog as a cosmic-string-network background plus
      finite-density per-vortex modulation. The per-vortex
      w_pv = -0.78 sits between domain-wall (-2/3) and vacuum
      (-1) without a pure-defect-class match, reported as a
      finite-density modulation
      (src/verify_lambda_cosmic_string_network_comparison.py).
  All four supplementary witnesses are explicitly framed as indirect
  multi-observable consistency checks of Theorem 15.18 P2; none of
  them evaluates the pointwise tensor-residual identity
  ||G_munu - 8 pi G T_munu^Xi|| / ||G_munu|| -> 0. Reviewer-hedging
  caveats (free-fit exponent degeneracy, N=18 finite-size skip
  caveat, chirality-bridge heuristic, Lambda empirically extracted)
  are reproduced inline by each verification script.

## What this is **not**

- Not a complete Quantum-Gravity theory
- Not a UV completion
- Not a claim of universal validity outside the closure domain
- Not a derivation of singularity behavior

## Installation (Windows PowerShell)

```powershell
git clone https://github.com/[anonymized]/emergent-gr-closure-repro.git
cd emergent-gr-closure-repro

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce the result

The repository is split into two reproduction tiers:

**Tier 1 (lightweight, no external data).** The bundled JSON
certificates in `data/` already contain the principal closure
results. A direct content-verification recomputation runs from
the bundled data:

```powershell
python .\src\recompute_gr_summary.py
python .\src\compare_gap_definitions.py
pytest
```

These reproduce all four CLP axes, the two-point Richardson on
the Einstein-identity gap, the PPN limits, the
Frobenius-decomposed cosmological-constant proxy, and the
9-point witness ladder on $\bar R$ and chirality balance —
i.e.\ everything covered by the bundled JSON certificates.

**Tier 2 (per-node Galerkin, requires D1 NPZ files).** The
per-node 4×4 Galerkin Frobenius residual scripts (Runner A
Hessian-Ricci, Runner B 3-Lambda variants, Runner D
basis-invariance, etc.) require the per-node lattice state:
the full $\Xi_{ab}$ matrix, $\Psi_a$ field, and $K(x)$/$Q(x)$
fields per seed and regime. These are bundled per regime as
`d1_p<X>.npz` files (4 MB at $N=18$ to 365 MB at $N=84$,
$\sim 1.15$ GB total for the canonical nine-point ladder).

The Galerkin scripts auto-discover the NPZ files via
`src/_d1_npz_discovery.py`, with the following search order:

1. `data/d1_runs/d1_p<X>.npz` — bundled location (preferred for
   standalone reproduction)
2. external developer paths (`<repo-parent>/results_d1_fix17/`,
   `<repo-parent>/results_d1_fix16/p<X>/`, etc.)

To reproduce Tier 2 in standalone mode, place the D1 NPZ files
into `data/d1_runs/`. The frozen Galerkin output JSON
certificates are bundled in `data/galerkin_*.json` so the test
suite verifies the recomputed numbers against frozen targets
even when the NPZ inputs are unavailable. The NPZ-dependent
scripts gracefully skip with a clear message if their input is
not found.

## Repository structure

```
emergent-gr-closure-repro/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── xi_metric_inputs.json
│   ├── a1_regime_constants.json
│   ├── clp_scores.json
│   ├── einstein_gap_results.json
│   ├── einstein_gap_5point_fit.json
│   ├── einstein_metric_stress_multiN.json
│   ├── lattice_diagonal_T_munu_9point.json
│   ├── lattice_diagonal_T_munu_per_seed_9point.json
│   ├── lattice_topological_observables_9point.json
│   ├── lattice_trivial_contributions_9point.json
│   ├── lorentzian_3plus1.json
│   ├── multi_n_extension.json
│   ├── ppn_results.json
│   ├── einstein_gap_9point_frobenius.json
│   ├── einstein_gap_18point_frobenius.json
│   ├── galerkin_runner_A_hessian_ricci.json   (Runner A frozen output)
│   ├── galerkin_runner_B_lambda_variants.json (Runner B frozen output)
│   ├── galerkin_runner_D_basis_invariance.json(Runner D frozen output)
│   ├── galerkin_calibrated_gpu.json
│   ├── galerkin_robustness_gpu.json
│   ├── galerkin_per_node_full_gpu.json
│   ├── galerkin_schwarzschild_defect_gpu.json
│   ├── lambda_offdiagonal_Tij_spectral_multiN.json
│   ├── d1_runs/      (Tier-2 NPZ files; populate from external dataset)
│   │   └── (d1_p0.npz .. d1_p8.npz, optional)
│   └── black_hole/
│       ├── bekenstein_hawking.json
│       ├── binary_inspiral.json
│       ├── hawking_spectrum.json
│       ├── horizon_threshold.json
│       ├── information_paradox.json
│       ├── kerr_geometry.json
│       ├── penrose_process.json
│       └── schwarzschild_far_field.json
├── src/
│   ├── recompute_gr_summary.py
│   ├── compare_gap_definitions.py
│   ├── recompute_lorentzian_3plus1.py
│   ├── recompute_schwarzschild_defect.py
│   ├── recompute_bh_sector.py
│   ├── verify_curvature_fixed_point.py
│   ├── verify_einstein_metric_stress.py
│   ├── verify_einstein_gap_5point_fit.py
│   ├── verify_einstein_gap_9point_witnesses.py
│   ├── verify_einstein_with_lambda.py
│   ├── verify_lambda_system_R.py
│   ├── verify_lambda_frobenius_residual.py
│   ├── verify_lambda_frobenius_exact_offdiagonal.py
│   ├── verify_lambda_cc_dressing_cross_check.py
│   ├── verify_lambda_LEE_bonferroni.py
│   ├── verify_lambda_energy_conditions.py
│   ├── verify_lambda_trace_and_continuum.py
│   ├── verify_lambda_vortex_background_decomposition.py
│   ├── verify_lambda_cosmic_string_network_comparison.py
│   ├── verify_lambda_vortex_quantization_geometry.py
│   ├── verify_lambda_per_seed_dispersion_constancy.py
│   ├── verify_lambda_DM_DE_mixture_reconciliation.py
│   ├── verify_lambda_anisotropy_NEC_SEC_robustness.py
│   ├── verify_lambda_DM_cosmo_gate_self_consistency.py
│   ├── verify_lambda_offdiagonal_Tij_spectral.py
│   ├── verify_lambda_19_15_breakthrough.py
│   ├── verify_lambda_extrapolation_discipline_audit.py
│   ├── verify_lambda_offdiagonal_Tij_spectral_multiN.py
│   ├── _d1_npz_discovery.py             (NPZ path-resolution helper)
│   ├── verify_galerkin_runner_A_hessian_ricci.py    (Runner A)
│   ├── verify_galerkin_runner_B_lambda_variants.py  (Runner B)
│   ├── verify_galerkin_runner_D_basis_invariance.py (Runner D)
│   ├── verify_galerkin_calibrated_gpu.py
│   ├── verify_galerkin_robustness_gpu.py
│   ├── verify_galerkin_per_node_full_gpu.py
│   ├── verify_galerkin_per_node_gpu.py
│   ├── verify_galerkin_schwarzschild_defect_gpu.py
│   ├── verify_full_tensor_frobenius_d1.py
│   ├── compute_true_einstein_residual.py
│   ├── verify_hawking_spectrum.py
│   └── make_figures.py
├── tests/
│   ├── test_metric_axioms.py
│   ├── test_a1_thresholds.py
│   ├── test_clp_thresholds.py
│   ├── test_gap_2_3.py
│   ├── test_gap_definitions_agree.py
│   ├── test_ppn.py
│   ├── test_falsification.py
│   ├── test_curvature_fixed_point.py
│   ├── test_einstein_metric_stress.py
│   ├── test_einstein_gap_5point_fit.py
│   ├── test_lorentzian_3plus1.py
│   ├── test_schwarzschild_defect.py
│   ├── test_bh_sector.py
│   └── test_hawking_spectrum.py
├── outputs/
│   ├── expected_output.txt
│   ├── recompute_gr_summary.json
│   └── gap_comparison_table.{json,csv}
├── paper/
│   ├── manuscript.tex
│   ├── manuscript.pdf
│   └── figures/
└── .github/workflows/
    └── reproduce.yml
```

## Falsification

The closure mechanism fails if:

1. M1-M3 axioms are violated by Xi_ij data;
2. Fast-slow thresholds (lambda_triangle >= 1, epsilon <= 0.10) fail on
   the canonical regime;
3. Any convergence-axis aggregate score falls below 0.70;
4. Any of the three Richardson candidate exponents yields |gap_inf| > 0.05
   on the two clean lattice points, or a future >= 3-point fit selects a
   single alpha_gap outside 2/3 +/- 30%;
5. PPN gamma or beta falls outside the Cassini/LLR experimental band.

## Citation

```bibtex
@misc{bucciarelli2026emergentgr,
  author    = {Bucciarelli, Sandro},
  title     = {Emergent Einstein dynamics from relational metric closure},
  year      = {2026},
  version   = {0.1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

## License

MIT License. See [LICENSE](LICENSE).
