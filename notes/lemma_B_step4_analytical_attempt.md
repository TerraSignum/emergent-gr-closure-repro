# Lemma B Step 4a/4b — analytical attempt

**Status:** session-level attempt at the analytical derivation of
the two rational conjectures `λ_skel^∞ = 7/24` and
`λ_w^∞ = 3/8`. Honest verdict: the algebraic identities in
`(d, N_gen)` are trivially verified, but the *carrier-action
realisation* of those rationals — the actual proof that the
relational-carrier construction produces these asymptotes —
remains open. This memo records what was checked, what is
plausibly within reach, and where the genuine analytical
work remains.

## 1. Algebraic verification of the conjectures

**Conjecture 4a:** `λ_skel^∞ = 7/24`.

Algebraic identity:
```
7/24 = (d + N_gen) / (2 · d · N_gen)
     = 7 / (2·4·3)
     = 7/24                                                     ✓
```

with `(d, N_gen) = (4, 3)` the relational-spacetime dimension
and fermion-generation count. The numerator `d + N_gen = 7` is
the corpus's "mass-mode dimensional primitive" (P0 §1.2,
recurs in `n_s`, `Y_p`, `A_s`, neutrino mass identities).
The denominator `2 d N_gen = 24` is the "occupancy-doubled"
combination.

**Conjecture 4b:** `λ_w^∞ = 3/8`.

Algebraic identity:
```
3/8 = (d - 1) / (2 · d)
    = 3 / (2 · 4)
    = 3/8                                                       ✓
```

The numerator `d - 1 = 3` is the "spatial-dimension primitive";
the denominator `2 d = 8` is the same doubled dimension.

**Ratio identity (algebraic consequence):**
```
λ_w^∞ / λ_skel^∞ = (3/8) / (7/24) = 72/56 = 9/7                ✓
```

These three identities are pure algebra; they verify that
*if* the empirical asymptotes are exactly `7/24` and `3/8`,
they fit naturally into the System-R rational primitive set.

**They do not constitute a derivation.** The empirical match
is 0.24% for 7/24 and 1.0% for 3/8 — well above the corpus
EXACT/PRECISE tier thresholds.

## 2. Structural skeleton degree algebra

The empirical structural-skeleton degree at `τ = 0.10` is
`d_eff^∞ = 12.14` (from `verify_lemma_B_edge_weight_structure.py`).
Algebraic candidates:

```
d · N_gen = 4 · 3 = 12                                          ✓
(d + N_gen) · (d - 1) / N_gen = 7 · 3 / 3 = 7                  ✗
(d + 1) · (d - 1) = 5 · 3 = 15                                 ✗
```

The natural identification `d_eff^∞ = d · N_gen` gives 12
exactly; the empirical 12.14 sits 1.2% above (consistent with
finite-N corrections, Symanzik-1 fit gave c_inf = 12.14 with
correction +312/N). This is the simplest System-R-rational
read-off for the skeleton degree.

## 3. Alon–Boppana saturation ratio

For `d = d_eff^∞ = 12`:
```
λ_AB(12) = 1 - 2√(d-1)/d = 1 - 2√11/12 ≈ 0.4499
```

Conjectured saturation:
```
λ_skel^∞ / λ_AB(d=12) = (7/24) / (1 - 2√11/12)
                       = (7/24) / 0.4499
                       ≈ 0.6483
                       ≈ 13/20 (numerical match 0.18%)
```

The saturation ratio `13/20` is suggestive but not an algebraic
identity in (d, N_gen) — Alon-Boppana involves `√11 = √(d_eff-1)`
which is irrational, so the ratio of two rationals can't
equal `13/20` exactly. This means: **the carrier skeleton is
NOT at exactly Alon-Boppana saturation** by an irrational
factor; the empirical `0.65` saturation is approximate.

This rules out pure Friedman/Alon-Boppana saturation as the
mechanism for `7/24`. The skeleton is sub-Ramanujan, but not
sub-Ramanujan-Alon-Boppana-ratio (which is generally
distributed for random d-regular graphs); rather, the carrier
realises a specific spectral gap determined by the carrier-
action equilibrium, not the random-graph optimum.

## 4. Honest scope of what is/isn't doable in this memo

**Doable (this session):**

- Algebraic identities ✓ (Section 1, above).
- Numerical saturation analysis ✓ (Section 3, above).
- Falsification checks (FB-1, FB-2, FB-3 of statement memo).

**Not doable (multi-month analytical research):**

- **Carrier-action → graph-ensemble characterisation.** Show
  that the τ=0.10 skeleton is a *specific* graph ensemble
  whose ensemble-averaged spectral gap is `7/24`. This
  requires deriving the edge-formation probability from the
  carrier dynamics and computing the ensemble spectral gap.
  Tools: cavity-method / Boltzmann-equation approach on
  small-world graphs; or finite-N enumeration. Estimated 2-3
  months of mathematical-physics research.

- **Weight-distribution lift derivation.** Show that the
  carrier-action edge weight distribution lifts the
  unweighted-skeleton gap to `3/8` via a Kahale-type bound.
  Tools: Kahale's irregular-expander spectral bound applied
  to the carrier-action weight Markov chain. Estimated 1-2
  months.

- **Small-world spectral synthesis.** Combine the Friedman-
  Kesten-McKay bulk with the persistent clustering correction
  (Step 4a pre-flight). Tools: small-world spectral
  resolvent methods (Newman 2003 review; Newman, Strogatz,
  Watts 2001). Estimated 1-2 months.

- **Universality.** Show that the same rationals appear for
  any admissible carrier sequence with the same `(d, N_gen)`
  integers, not just the canonical P5/P5N realisation.
  Tools: invariance arguments under carrier-action
  reparametrisation. Estimated 2-4 months.

## 5. Status of the analytical Phase-2 work

The algebraic identities of Section 1 are *necessary
conditions* for the conjectured asymptotes — they fit into
the System-R framework. They are not *sufficient* — many
algebraic rationals fit the framework, and the empirical
selection of `7/24` and `3/8` requires the carrier-action
derivation to actually produce these specific values.

**Concrete next session-tauglich step**: numerically
sample carrier-action edge-formation probabilities (from
the d1_*.npz snapshots) and check whether they match a
specific structured small-world rewiring rule with mean
degree exactly `d · N_gen = 12`. If yes, the random-graph
construction is empirically pinned down; if no, the
carrier-action is more complex and Step 4a requires
different machinery.

**Verdict:** the analytical Phase-2 work is registered as
open. The algebraic skeleton (Section 1) is the
calibration target; the empirical signature (Phase-2
empirical work, propositions in P4) constrains the proof
strategy; the actual derivation from carrier-action
remains multi-month research. This memo documents what
this session could verify and what it explicitly could not.
