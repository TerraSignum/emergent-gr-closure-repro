# Lemma B (uniform spectral gap / Poincaré inequality) — notes

**Naming note.** Internally these notes use "Lemma B" as a
working name for the target uniform-spectral-gap theorem.
In the P4 manuscript the same theorem is called the
**"target (SG)-Lemma"** to avoid confusion with the
loop-class library entries Lemma 1, ..., Lemma 10 of P3
(Pure-Sync, Yukawa-Damping, PMNS-Self-Energy, etc.). The
two names refer to the same object; the P4 nomenclature is
authoritative for citation purposes.

**Authoritative formal statement is in the P4 manuscript:**

- Definition: `def:uniform_spectral_gap` (the `(SG)` axiom on
  the Xi-weighted normalised Laplacian).
- Theorem: `thm:sg_implies_admissibility` ((SG) ⇒ uniform
  Poincaré with constant `C_P ≤ 1/lambda_*` ⇒ A1+A8 via
  Cheeger-Buser).
- Counterexample: `prop:m0m3_insufficient` (constant-Xi family
  satisfies M0-M3 + admissibility but violates A8 uniformly).
- Empirical certification: `prop:sg_empirical` (10-regime
  ladder, 184 seeds, Symanzik-1 fit, `lambda_inf = 0.3789`,
  conjectured `3/8`).

This `notes/` document records (a) the in-package reproducer
chain, (b) the rational-conjecture audit, and (c) the
near-term falsification triggers. The proof-strategy survey
for Phase 2 is in [`lemma_B_proof_strategy.md`](lemma_B_proof_strategy.md).

## Reproducer chain

- Generation: `Xi_N` snapshots in `results_d1_*/*.npz`
  bundles, produced by the d1 / P5N pipelines.
- Audit: `src/verify_lemma_B_uniform_poincare.py` (this repo).
- Counterexample audit (constant-Xi):
  `src/verify_admissibility_counterexample_and_spectral_gap.py`.
- Output JSON: `outputs/verify_lemma_B_uniform_poincare.json`.

## Rational closed-form audit

The Symanzik-1 fit on the 10-regime ladder gives
`lambda_inf = 0.3789 ± 0.005` (empirical asymptote). Tested
System-R rational candidates:

| Candidate                                  | Value | Δ      |
|--------------------------------------------|------:|-------:|
| `3/8`                                      | 0.375 | +1.0%  |
| `alpha_xi - gamma`                         | 0.800 | mismatch |
| `alpha_xi^2 - gamma`                       | 0.710 | mismatch |
| `alpha_xi / N_gen - gamma`                 | 0.200 | mismatch |
| `alpha_xi * gamma * (d-1)`                 | 0.270 | mismatch |
| `d * gamma + alpha_xi / d`                 | 0.625 | mismatch |
| `(2*d + N_gen - 1) / d^2`                  | 0.625 | mismatch |
| `alpha_xi - 2 * gamma * (d-1)`             | 0.300 | mismatch |

The closest match is `3/8`. **Registered as conjecture, not
closure** — the 1.0% residual is well above any of the EXACT
or PRECISE tier thresholds. Falsification trigger FB-1 below
is the discriminator.

## Falsification triggers

- **FB-1.** An extended ladder `N ≥ 1024` with cross-seed mean
  `lambda_2(L_N) < 0.3` and bootstrap 95% upper-CI also below
  `0.3` falsifies the Symanzik-1 asymptote and the `3/8`
  conjecture together.
- **FB-2.** An admissible carrier sequence outside the P5/P5N
  construction (different `(d, N_gen)`) with non-positive
  Symanzik-1 asymptote falsifies the universality of (SG).
- **FB-3.** The Symanzik-1 model loses AICc preference to a
  model with `lambda_inf ≤ 0` on a larger ladder.

## What this document does *not* re-prove

- The implication `(SG) ⇒ A1 + A8` (P4
  `thm:sg_implies_admissibility`).
- The constant-Xi counterexample (P4 `prop:m0m3_insufficient`).
- The definition of the (SG) axiom (P4 `def:uniform_spectral_gap`).

All three are stated and proved in the P4 manuscript; this
document is a navigational artefact for the empirical
extension (Phase 1) and the open analytical question
(Phase 2).
