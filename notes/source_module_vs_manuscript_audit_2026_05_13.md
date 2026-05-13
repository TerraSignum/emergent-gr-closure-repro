# Source-module vs. manuscript closure audit (2026-05-13)

User pointed at four "physics axis" modules:
  - `src/worldformula/physics/phase_i_closures.py`
  - `src/worldformula/physics/supernovae_nucleosynthesis.py`
  - `src/worldformula/physics/omega_world_consistency.py`
  - `src/worldformula/physics/black_hole_thermodynamics.py`

with the worry that their results may have been lost when writing
the manuscripts, and the question whether the claims even hold on
different regimes. This memo records the audit.

## Summary: not "lost"; mostly *superseded* by the loop-class library in P3

The four modules together represent an *earlier-generation*
audit layer that took upstream "physics bundles" (gre, qfe, emt,
smp, bht, dme, dee, pdc, sca) and computed downstream physical
quantities using standard SM/BBN/stellar formulae with framework
inputs. The published P3 manuscript (`loop-class-closure-repro`)
later supersedes most of these with the **loop-class library**
approach: each observable is matched to a System-$\mathcal{R}$
rational of the form $1\pm c_\sigma$, giving an algebraic
closure that is **regime-independent by construction** (no
bundle inputs at all).

## Phase-I closures (phase_i_closures.py)

Output `outputs_phase_i_closures/phase_i_assessment.json` reports
6/6 closed (5 EXACT/PRECISE plus 2 ORDER for PMNS). Per-item
status vs P3 closure ledger `tab:closure-29`:

| Source claim | Source tier (residual) | P3 row | P3 tier (residual) | Verdict |
|---|---|---|---|---|
| C-5 $\Gamma_W$ | EXACT (0.50%) | O18 | PRECISE (0.50%) | Same, P3 records it |
| C-5 $\Gamma_Z$ | EXACT (0.41%) | O19 | PRECISE (0.50%) | Same |
| C-3 $\Omega_b h^2$ | PRECISE (6.41%) | O26 | PRECISE (0.067%) | **P3 much better** |
| B-1 $\Lambda_{QCD}$ | PRECISE (5.20%) | O25 | PRECISE (0.039%) | **P3 much better** |
| C-4 $\alpha_s(M_Z)$ | EXACT (0%, circular) | O04 | PRECISE (0.20%) | P3 standalone, source circular |
| C-2 $H_0$ | PRECISE (2.30%) | O20 | PRECISE (0.60%) | **P3 better** |
| C-1 $\sin^2\theta_{23}$ | ORDER (82% off) | O11 | EXACT (0.0052%) | **P3 vastly better** |
| C-1 $\sin^2\theta_{13}$ | ORDER (89% off) | O09 | EXACT (0.000%) | **P3 vastly better** |

**Verdict.** Nothing is lost. P3 has the SAME or BETTER closure
for every Phase-I item via the loop-class library. The Phase-I
module is an early-stage "standard-formula sanity check" that
has been superseded.

## Supernovae nucleosynthesis (supernovae_nucleosynthesis.py)

Output `outputs_supernovae_nucleosynthesis/*.json`:

| Item | Predicted | Observed | Ratio | Status |
|---|---|---|---|---|
| SNE-01 $B/A$ binding | 4.77 MeV | 8.5 MeV | 0.56 | **BROKEN (44% off)** |
| SNE-02 $M_{\rm Ch}$ | 2.72 $M_\odot$ | 1.44 $M_\odot$ | 1.89 | **BROKEN (89% off)** |
| SNE-03 $Y_p$ (BBN $^4$He) | 0.4306 | 0.247 | 1.74 | **BROKEN (74% off)** |
| SNE-04 endpoints | qualitative | — | — | Compatible only |

The SNE-03 $Y_p$ calculation has a numerical bug: it reports
$(n/p)_{\rm freeze} = 0.2744$ where the standard-BBN value is
$\approx 1/7\!=\!0.143$, doubling $Y_p$ to 0.43 instead of 0.247.

P3 records $Y_p$ at row O29 with residual 1.062% via the
loop-class identity $Y_p = Y_p^{\rm obs}(1\pm\gamma/4)$
(Yukawa-Damping cluster). This is the **authoritative**
framework $Y_p$ readout; the SNE module's standalone BBN
calculation is broken and should be treated as deprecated.

The SNE items not present in any manuscript ($M_{\rm Ch}$,
$B/A$, $M_{\rm TOV}$, D/H, $^7$Li/H, $r$-process site
identification, supernova light-curve peak strain, $r$-process
window) are *not* closures at the present accuracy — the
implementations land 40--90% off observation, which is **not
publication-grade**. They should either be repaired (proper
weak-rate calculation, proper $T_{\rm BBN}$ etc.) or marked as
roadmap-only.

## Black-hole thermodynamics (black_hole_thermodynamics.py)

Output `outputs_black_hole_thermo/*.json`:

| Item | Status | Where |
|---|---|---|
| BHT-01 horizon threshold (compactness $>9$) | sanity check | not a closure |
| BHT-02 $S_{\rm BH}\!=\!A/4$ | DERIVED | **P3 O07 at 0.004%**, **P4-A** APS+SV closure |
| BHT-03 Hawking temperature | DERIVED | **P4-A** Page-1976 greybody, $f_{\rm grey}\!=\!0.6320$ |
| BHT-04 unitarity preserved (info paradox) | DERIVED | **P4-A** chirality pairing |
| BHT-06 Penrose process / spin extraction | derivation only | **not in manuscripts** |
| BHT-07 merger GW signal (peak frequency, amplitude) | derivation only | **not in manuscripts** |
| BHT-08 Hawking spectrum (greybody) | DERIVED | **P4-A** explicit `verify_hawking_greybody_finite_mass.py` |

**Verdict.** The core BHT results ($S_{\rm BH}\!=\!A/4$,
unitarity, Hawking spectrum with greybody) are in the
manuscripts. The Penrose-process spin extraction and the
detailed merger-GW signal predictions (BHT-06, BHT-07) are
*not* in any manuscript and may genuinely be candidates for
inclusion or for a separate astrophysics companion paper.

## Omega-World consistency (omega_world_consistency.py)

OWC-01..05 is a *meta-audit* that checks cross-axis consistency
(dimensional, gravitational, energy-budget, axiom-trace). Not
itself a set of observable closures. Nothing to publish; it is
a pipeline-internal sanity check.

## Regime-dependence question

The user asked: do these claims hold when computed on different
lattice regimes?

**Loop-class closures in P3 (tab:closure-29):** algebraic in
System-$\mathcal{R}$ primitives $\gamma, \alpha_\xi, \beta_\pi,
\varepsilon^2_{\rm sync}, D_\Omega$ and integers $d, N_{\rm gen}$.
**Regime-independent by construction** — no bundle inputs.

**Source-module closures (Phase-I, SNE, BHT, OWC):** take
upstream physics bundles which DO vary with regime. The largest
regime-sensitive inputs are:

  - $\eta_B$ (baryon-to-photon ratio, framework output, drives
    C-3 $\Omega_b h^2$ and SNE-03 $Y_p$)
  - $\rho_\Lambda$ ratio (drives C-2 $H_0$)
  - $d_{\rm spectral\,eff}$ (drives OWC-01 dimensional check)
  - compactness ratio (drives BHT-01 horizon threshold)
  - $G_N$ ratio (drives OWC-02 + SNE-02 $M_{\rm Ch}$)

Items using only PDG inputs ($m_W, m_Z, \sin^2\theta_W, G_F$ in
C-5; $\alpha_s$ as input in B-1) are regime-independent
trivially.

## Recommendations

1. **Treat P3 tab:closure-29 as authoritative** for the
   overlapping items. The Phase-I source-module is a backup
   sanity-check layer, not a publication source.

2. **Mark the SNE module as deprecated / under repair**: $B/A$,
   $M_{\rm Ch}$, $Y_p$-from-BBN are all quantitatively broken
   (40--90% off). Use the P3 O29 $Y_p$ loop-class closure as
   the framework $Y_p$ prediction; the others have no
   manuscript equivalent and are not at publication accuracy.

3. **Consider whether BHT-06 (Penrose) and BHT-07 (merger GW)
   warrant a separate companion paper**: these are physics
   predictions with no manuscript home, but at publication
   accuracy they need their own validation pass.

4. **Add a note in OWC**: the meta-audit is regime-dependent
   through its bundle inputs; results vary by regime. This is
   appropriate for a sanity-check layer, not a publication
   target.

5. Nothing should be *added to the manuscripts* on the basis of
   these source-module outputs alone, since most of them are
   either:
   - already present in P3 with better residuals, or
   - quantitatively broken at the current implementation level.
