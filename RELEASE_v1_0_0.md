# RELEASE_v1_0_0

**Repo:** `emergent-gr-closure-repro` (P4)
**Title:** Emergent-gravity Einstein-residual closure on the relational lattice
**Freeze date:** 2026-05-12
**State:** publication-ready (pre-arXiv)

## Headline numerical state

This release is the publication-prep frozen state after the
9-round review/optimization autopilot of 2026-05-12. Cross-corpus
invariants verified at freeze:

- 507/507 tests pass across 10 test-bearing repos
- 11/11 manuscripts compile cleanly with tectonic 0.15.0
- 0 unresolved citations, 0 unresolved references, 0 unused bibitems
  across all 11 manuscripts
- 9 PRL-target short-form abstracts (`paper/abstract_short.tex`,
  138-177 words) in addition to the long-form abstracts in the
  manuscripts
- 11 arXiv-ready tarballs `paper-arxiv.tar.gz` (10 KiB - 6.8 MiB)
- 10 `data/SHA256SUMS` files with LF line endings (GNU
  `sha256sum -c` compatible)
- 10 `.github/workflows/reproduce.yml` CI workflows (P0 manifest is
  documentation-only)

## Cosmological-constant closure (P4B headline)

The cosmological-constant 122-OoM hierarchy closes at the EXACT
tier under all three corpus EXACT cuts:

- 9-layer parameter-free dressing
- ratio rho_pred / rho_obs = 1.001 (Planck 2018 anchor)
- residual +0.0004 OoM (0.1% above Planck)
- The ninth layer (`H_sync`, synchronization-channel un-cancellation)
  supplies the multiplicative factor 10^(eps_sync^2) = 10^(1/20),
  with the log10-magnitude eps_sync^2 = gamma/2 = 1/20 being the
  derived fluctuation-dissipation identity `C_3` of P2 (causal-wave
  landings, section 5)
- All 9 layer log10_contributions are listed in
  `data/cosmological_constant_closure.json`; the reproducer
  `src/verify_cosmological_constant.py` recomputes the ratio from
  the layer sum + M_Pl^4 and asserts it matches the bundled
  closure_result within 0.02 OoM (verified by the
  `test_repro_match` test).

## Pre-registration

Five near-term cosmology / neutrino predictions are stamped at
freeze date with content hash `aa563454e9ff5404...` (full 64-char
SHA-256 in `data/preregistration_2026_05_11.json`):

- `P-W_A`: |w_a| <= 0.01 (DESI DR3 decisive)
- `P-SIGMA_M_NU`: Sigma m_nu = 0.0591 eV (CMB-S4 / Euclid)
- `P-H0`: H_0 = 67.5 km/s/Mpc (structurally locked)
- `P-DELTA_CP`: delta_CP = 1.13 rad (next NuFIT)
- `P-OMEGA_DM_H2`: Omega_DM h^2 * (60/59) = LCDM ref (DESI Year-3)

## Companion-paper graph

This repo is part of the 11-repo corpus organised under the reader's
guide at `relational-carrier-manifest/paper/manifest.tex` (P0).

## Reproducibility quick-check

```bash
# Test suite
pytest tests/ -v

# CC closure recompute (P4B only)
python src/verify_cosmological_constant.py

# Tectonic compile
tectonic -X compile paper/manuscript.tex

# Data integrity (Linux/macOS)
cd data && sha256sum -c SHA256SUMS
```

Expected: all green.
