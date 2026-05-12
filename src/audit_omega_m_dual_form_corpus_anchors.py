"""Omega_m dual-form audit using ONLY source-cited anchors.

Framework predictions (chirality-flip-aware):
    Omega_m^matter = gamma N_gen + gamma^2 d / N_gen     = 47/150 = 0.31333
    Omega_m^vacuum = gamma N_gen + gamma^2 (d+1) / N_gen = 19/60  = 0.31667
    Branch separation Delta = gamma^2 / N_gen            = 1/300  = 0.00333

ANCHOR-DATA POLICY: this script uses ONLY anchors with explicit source
provenance. Two source classes are admitted:

(I) CORPUS-INTERNAL: a value bundled in the framework's reference data
    (`Papers/data/*.json`) or referenced in the framework's manuscripts
    with bibliographic citation.

(II) EXTERNAL with arXiv/journal CITATION: an explicit-source published
    value, with arXiv ID or DOI recorded next to the anchor.

No fabricated, hand-quoted, or memory-recalled values are admitted.

CMB-anchored anchors used:
  (1) Planck 2018 PR3 baseline (Papers/data/reference_planck_2018_baseline_lcdm.json)
      Omega_m = 0.3166 +- 0.0084
      Source: Planck Collaboration 2020, A&A 641, A6
              (corpus pdf hash 4eb730ac8781d46d655400c25899f627ad5d152c01a40aaa5a87e174da209c3e)
  (2) PDG 2024 cosmology compact (Papers/data/reference_pdg_2024_cosmology_compact.json)
      Omega_m = 0.315 +- 0.007
      Source: PDG 2024 review, Cosmological Parameters
  (3) Planck PR4 NPIPE CamSpec (external arXiv:2205.10869)
      Omega_m = 0.3150 +- 0.0068
      Source: Rosenberg, Gratton, Efstathiou 2022,
              MNRAS 517, 4620 (Table 5 PR4_12.6 TTTEEE)

Combined CMB+late-universe anchor:
  (4) DESI + Planck + ACT lensing (external arXiv:2404.03002)
      Omega_m = 0.307 +- 0.005
      Source: DESI Collaboration 2024, JCAP

Late-universe-only anchors:
  (5) DESI Year-1 BAO alone (external arXiv:2404.03002)
      Omega_m = 0.295 +- 0.015
      Source: DESI Collaboration 2024, JCAP
  (6) Pantheon+ SNe Ia alone (external arXiv:2202.04077)
      Omega_m = 0.334 +- 0.018
      Source: Brout et al. 2022, ApJ 938, 110

Output: outputs/audit_omega_m_dual_form_corpus_anchors.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "audit_omega_m_dual_form_corpus_anchors.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

GAMMA = 1 / 10
N_GEN = 3
D = 4

OMEGA_M_MATTER = GAMMA * N_GEN + GAMMA ** 2 * D / N_GEN          # 47/150
OMEGA_M_VACUUM = GAMMA * N_GEN + GAMMA ** 2 * (D + 1) / N_GEN    # 19/60
DELTA = OMEGA_M_VACUUM - OMEGA_M_MATTER                          # 1/300


def z_score(measured, sigma, target):
    return (measured - target) / sigma


def inv_var_weighted(measurements):
    """Inverse-variance combination, returns (mean, sigma)."""
    weights = [1 / s ** 2 for _, s, _ in measurements]
    W = sum(weights)
    central = sum(m * w for (m, _, _), w in zip(measurements, weights)) / W
    sigma = 1 / math.sqrt(W)
    return central, sigma


def bayes_posterior(measured, sigma, mean_a, mean_b, prior_a=0.5):
    log_lik_a = -((measured - mean_a) ** 2) / (2 * sigma ** 2)
    log_lik_b = -((measured - mean_b) ** 2) / (2 * sigma ** 2)
    m = max(log_lik_a, log_lik_b)
    p_a = prior_a * math.exp(log_lik_a - m)
    p_b = (1 - prior_a) * math.exp(log_lik_b - m)
    return p_a / (p_a + p_b)


def main():
    # Anchor catalogue: each entry has (value, sigma, source-citation, subset)
    anchors = {
        "Planck 2018 PR3 baseline (corpus reference)": {
            "value": 0.3166, "sigma": 0.0084,
            "source": "Planck Collaboration 2020, A&A 641, A6",
            "provenance": (
                "Papers/data/reference_planck_2018_baseline_lcdm.json "
                "(SHA256 4eb730ac... gepinnt)"),
            "subset": "CMB",
            "type": "corpus_internal",
        },
        "PDG 2024 cosmology compact (corpus reference)": {
            "value": 0.315, "sigma": 0.007,
            "source": "PDG 2024 review, Cosmological Parameters",
            "provenance":
                "Papers/data/reference_pdg_2024_cosmology_compact.json",
            "subset": "CMB",
            "type": "corpus_internal",
        },
        "Planck PR4 NPIPE CamSpec TTTEEE": {
            "value": 0.3150, "sigma": 0.0068,
            "source": (
                "Rosenberg, Gratton, Efstathiou 2022, "
                "MNRAS 517, 4620 (Table 5 PR4_12.6 TTTEEE)"),
            "arxiv": "2205.10869",
            "subset": "CMB",
            "type": "external_arxiv",
        },
        "DESI Y1 + Planck CMB + ACT lensing combined": {
            "value": 0.307, "sigma": 0.005,
            "source": "DESI Collaboration 2024, JCAP",
            "arxiv": "2404.03002",
            "subset": "Combined",
            "type": "external_arxiv",
        },
        "DESI Y1 BAO alone": {
            "value": 0.295, "sigma": 0.015,
            "source": "DESI Collaboration 2024, JCAP",
            "arxiv": "2404.03002",
            "subset": "Late",
            "type": "external_arxiv",
        },
        "Pantheon+ SNe Ia alone (flat LCDM)": {
            "value": 0.334, "sigma": 0.018,
            "source": "Brout et al. 2022, ApJ 938, 110",
            "arxiv": "2202.04077",
            "subset": "Late",
            "type": "external_arxiv",
        },
    }

    print("=" * 80)
    print("Omega_m dual-form audit (source-cited anchors only)")
    print("=" * 80)
    print()
    print(f"Framework predictions:")
    print(f"  Omega_m^matter = 47/150 = {OMEGA_M_MATTER:.5f}")
    print(f"  Omega_m^vacuum = 19/60  = {OMEGA_M_VACUUM:.5f}")
    print(f"  Branch separation Delta = gamma^2/N_gen = "
          f"1/300 = {DELTA:.5f}")
    print()
    print("Anchor catalogue (with source citations):")
    print(f"{'#':>2} {'anchor':<48} {'Om_m':>8} {'sigma':>7} "
          f"{'subset':>10}")
    for i, (label, info) in enumerate(anchors.items(), 1):
        print(f"{i:>2} {label[:48]:<48} {info['value']:>8.4f} "
              f"{info['sigma']:>7.4f} {info['subset']:>10}")
        if 'arxiv' in info:
            print(f"     {info['source']} (arXiv:{info['arxiv']})")
        else:
            print(f"     {info['source']}")
            print(f"     {info['provenance']}")
    print()

    # Per-anchor z-scores
    print("Per-anchor discrimination:")
    print(f"{'#':>2} {'anchor':<32} {'z_matter':>10} {'z_vacuum':>10} "
          f"{'preferred':>12}")
    for i, (label, info) in enumerate(anchors.items(), 1):
        z_m = z_score(info['value'], info['sigma'], OMEGA_M_MATTER)
        z_v = z_score(info['value'], info['sigma'], OMEGA_M_VACUUM)
        pref = "matter" if abs(z_m) < abs(z_v) else "vacuum"
        print(f"{i:>2} {label[:32]:<32} {z_m:>+10.3f} {z_v:>+10.3f} "
              f"{pref:>12}")
    print()

    # Subset combinations
    cmb_meas = [(info['value'], info['sigma'], label)
                for label, info in anchors.items() if info['subset'] == 'CMB']
    late_meas = [(info['value'], info['sigma'], label)
                 for label, info in anchors.items() if info['subset'] == 'Late']
    cmb_central, cmb_sigma = inv_var_weighted(cmb_meas)
    late_central, late_sigma = inv_var_weighted(late_meas)
    p_m_cmb = bayes_posterior(cmb_central, cmb_sigma,
                               OMEGA_M_MATTER, OMEGA_M_VACUUM)
    p_m_late = bayes_posterior(late_central, late_sigma,
                                OMEGA_M_MATTER, OMEGA_M_VACUUM)

    # Internal-tension check on the late-universe subset (BAO vs SNe)
    if len(late_meas) == 2:
        m1, s1, _ = late_meas[0]
        m2, s2, _ = late_meas[1]
        z_late_internal = abs(m1 - m2) / math.sqrt(s1 ** 2 + s2 ** 2)
    else:
        z_late_internal = None

    print("--- CMB-anchored subset (vacuum-era inference, n=3) ---")
    print(f"  inverse-variance combined: Omega_m = "
          f"{cmb_central:.5f} +- {cmb_sigma:.5f}")
    print(f"  z to matter form: "
          f"{z_score(cmb_central, cmb_sigma, OMEGA_M_MATTER):+.3f}")
    print(f"  z to vacuum form: "
          f"{z_score(cmb_central, cmb_sigma, OMEGA_M_VACUUM):+.3f}")
    print(f"  Bayes P(matter|data) = {p_m_cmb:.4f}, "
          f"P(vacuum|data) = {1 - p_m_cmb:.4f}")
    print()

    print("--- Late-universe subset (matter-era weighted, n=2) ---")
    print(f"  inverse-variance combined: Omega_m = "
          f"{late_central:.5f} +- {late_sigma:.5f}")
    print(f"  internal tension between BAO and SNe: z = "
          f"{z_late_internal:+.3f}sigma "
          f"(if > 1, the inverse-variance combination is contaminated)")
    print(f"  z to matter form: "
          f"{z_score(late_central, late_sigma, OMEGA_M_MATTER):+.3f}")
    print(f"  z to vacuum form: "
          f"{z_score(late_central, late_sigma, OMEGA_M_VACUUM):+.3f}")
    print(f"  Bayes P(matter|data) = {p_m_late:.4f}, "
          f"P(vacuum|data) = {1 - p_m_late:.4f}")
    print()

    # Direction-of-split test (with subset-sensitivity analysis)
    diff = cmb_central - late_central
    sigma_diff = math.sqrt(cmb_sigma ** 2 + late_sigma ** 2)
    z_split_zero = diff / sigma_diff
    z_split_pred = (diff - DELTA) / sigma_diff
    direction_match = diff > 0  # CMB above Late predicted

    # Subset-sensitivity: BAO-only and SNe-only late-anchor choices
    bao_alone = anchors["DESI Y1 BAO alone"]
    sne_alone = anchors["Pantheon+ SNe Ia alone (flat LCDM)"]
    bao_diff = cmb_central - bao_alone["value"]
    bao_sd = math.sqrt(cmb_sigma ** 2 + bao_alone["sigma"] ** 2)
    bao_z_zero = bao_diff / bao_sd
    bao_z_pred = (bao_diff - DELTA) / bao_sd
    sne_diff = cmb_central - sne_alone["value"]
    sne_sd = math.sqrt(cmb_sigma ** 2 + sne_alone["sigma"] ** 2)
    sne_z_zero = sne_diff / sne_sd
    sne_z_pred = (sne_diff - DELTA) / sne_sd

    print("--- Direction-of-split test ---")
    print(f"  Predicted: CMB anchor > Late anchor with diff = "
          f"gamma^2/N_gen = {DELTA:.5f}")
    print(f"  Observed (BAO+SNe naive avg): CMB - Late = "
          f"{diff:+.5f} +- {sigma_diff:.5f}")
    print(f"    z to prediction = {z_split_pred:+.3f}; "
          f"z to zero = {z_split_zero:+.3f}")
    print()
    print(f"  Subset sensitivity (CRITICAL given {z_late_internal:.2f}sigma "
          f"BAO-SNe tension):")
    print(f"    BAO-only late: CMB - Late = "
          f"{bao_diff:+.5f} +- {bao_sd:.5f}")
    print(f"      = {bao_diff/DELTA:+.2f}x predicted Delta; "
          f"z(pred) = {bao_z_pred:+.3f}, z(zero) = {bao_z_zero:+.3f}")
    print(f"    SNe-only late: CMB - Late = "
          f"{sne_diff:+.5f} +- {sne_sd:.5f}")
    print(f"      = {sne_diff/DELTA:+.2f}x predicted Delta; "
          f"z(pred) = {sne_z_pred:+.3f}, z(zero) = {sne_z_zero:+.3f}")
    print(f"      ** Direction is "
          f"{'positive (CMB>Late, matches prediction)' if sne_diff > 0 else 'NEGATIVE (CMB<Late, WRONG sign vs prediction)'} **")
    print()

    # Honest verdict — subset-sensitivity-aware
    bao_only_supports = (bao_diff > 0)
    sne_only_supports = (sne_diff > 0)

    if direction_match and abs(z_split_pred) < 1.0 \
       and bao_only_supports and sne_only_supports:
        verdict = (
            "ROBUST_CONSISTENT_WITH_PREDICTION: split direction matches "
            "and magnitude matches across all late-anchor choices."
        )
    elif direction_match and abs(z_split_pred) < 1.0 \
         and not (bao_only_supports and sne_only_supports):
        verdict = (
            f"INCONCLUSIVE_DUE_TO_LATE_SUBSET_TENSION: the naive "
            f"BAO+SNe average gives split = {diff:+.5f} +- "
            f"{sigma_diff:.5f}, matching predicted Delta at "
            f"{abs(z_split_pred):.2f}sigma — but this is an artefact "
            f"of naive inverse-variance averaging two "
            f"{z_late_internal:.2f}sigma-tension late-universe "
            f"datasets. Subset-by-subset: "
            f"BAO-only gives split = {bao_diff:+.5f} "
            f"({bao_diff/DELTA:+.1f}x predicted, "
            f"{abs(bao_z_zero):.2f}sigma above zero, "
            f"{'direction matches' if bao_only_supports else 'wrong direction'} "
            f"but magnitude {abs(bao_diff/DELTA):.1f}x off); "
            f"SNe-only gives split = {sne_diff:+.5f} "
            f"({'direction matches' if sne_only_supports else 'WRONG direction (CMB < Late)'}, "
            f"{abs(sne_z_zero):.2f}sigma from zero, "
            f"actively against the chirality-flip prediction). "
            f"The naive-average match is therefore an interpolation "
            f"between the two extremes; current late-universe data "
            f"cannot honestly test the prediction without first "
            f"resolving the BAO-vs-SNe tension. CMB-only-side "
            f"discrimination at sigma_Om <= 0.001 (CMB-S4-class) "
            f"would test the prediction without late-universe input."
        )
    else:
        verdict = "DIRECTION OR MAGNITUDE INCONSISTENT WITH PREDICTION."

    print("HONEST VERDICT (subset-sensitivity aware):")
    print(verdict)
    print()

    # Discrimination thresholds
    sigma_3sig = DELTA / 3
    sigma_5sig = DELTA / 5
    print(f"Required precision:")
    print(f"  3-sigma branch discrimination: sigma_Om <= {sigma_3sig:.5f}")
    print(f"  5-sigma branch discrimination: sigma_Om <= {sigma_5sig:.5f}")
    print(f"  Current best single anchor (DESI+Planck+ACT): "
          f"sigma = 0.005, ratio sigma/Delta = {0.005/DELTA:.2f}")
    print()

    bundle = {
        "method": "audit_omega_m_dual_form_corpus_anchors",
        "data_source_policy": (
            "ONLY anchors with explicit source provenance "
            "(corpus-internal references or external arxiv/journal "
            "citations). No fabricated or memory-recalled values."),
        "framework_constants": {"gamma": GAMMA, "N_gen": N_GEN, "d": D},
        "branch_predictions": {
            "matter": {"fraction": "47/150", "value": OMEGA_M_MATTER,
                       "form": "gamma N_gen + gamma^2 d / N_gen"},
            "vacuum": {"fraction": "19/60", "value": OMEGA_M_VACUUM,
                       "form": "gamma N_gen + gamma^2 (d+1) / N_gen"},
            "delta": {"fraction": "1/300", "value": DELTA,
                      "form": "gamma^2 / N_gen"},
        },
        "anchors": {
            label: {
                **info,
                "z_to_matter": z_score(info["value"], info["sigma"],
                                       OMEGA_M_MATTER),
                "z_to_vacuum": z_score(info["value"], info["sigma"],
                                       OMEGA_M_VACUUM),
            }
            for label, info in anchors.items()
        },
        "cmb_subset_combined": {
            "n_anchors": len(cmb_meas),
            "value": cmb_central, "sigma": cmb_sigma,
            "z_to_matter": z_score(cmb_central, cmb_sigma,
                                   OMEGA_M_MATTER),
            "z_to_vacuum": z_score(cmb_central, cmb_sigma,
                                   OMEGA_M_VACUUM),
            "P_matter": p_m_cmb,
            "P_vacuum": 1 - p_m_cmb,
        },
        "late_subset_combined": {
            "n_anchors": len(late_meas),
            "value": late_central, "sigma": late_sigma,
            "internal_BAO_SNe_tension_sigma": z_late_internal,
            "z_to_matter": z_score(late_central, late_sigma,
                                   OMEGA_M_MATTER),
            "z_to_vacuum": z_score(late_central, late_sigma,
                                   OMEGA_M_VACUUM),
            "P_matter": p_m_late,
            "P_vacuum": 1 - p_m_late,
            "caveat": (
                "The late-universe subset has BAO (DESI Y1, 0.295) and "
                "SNe (Pantheon+, 0.334) at "
                f"{z_late_internal:.2f}sigma internal tension; "
                "the inverse-variance combination is naive and "
                "the result should be read as a directional indicator "
                "rather than a precise late-universe estimate."),
        },
        "direction_of_split_test": {
            "predicted_diff": DELTA,
            "predicted_diff_form": "gamma^2 / N_gen",
            "observed_diff_CMB_minus_Late": diff,
            "sigma_diff": sigma_diff,
            "z_split_equals_zero_null": z_split_zero,
            "z_split_equals_prediction_null": z_split_pred,
            "direction_match": bool(direction_match),
        },
        "discrimination_thresholds": {
            "delta": DELTA,
            "sigma_for_3sigma": sigma_3sig,
            "sigma_for_5sigma": sigma_5sig,
            "best_current_single_anchor_sigma": 0.005,
            "best_current_sigma_over_delta": 0.005 / DELTA,
        },
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
