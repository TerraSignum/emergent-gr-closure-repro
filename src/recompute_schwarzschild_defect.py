r"""
Recompute the Schwarzschild-defect block of Section 3 of the manuscript.

Loads the bundled Schwarzschild far-field data file and surfaces the
load-bearing numbers that the manuscript references:

  - effective spectral dimension d_eff^spec = 3.170;
  - emergent spatial dimension d_spatial (nearest integer) = 3;
  - emergent spacetime dimension d_spacetime = 4;
  - Newton-constant ratio G_eff / G_N (residual <= 0.5%);
  - Planck-length ratio (residual <= 0.001%);
  - g_00 Schwarzschild-match residual (canonical / extended regimes);
  - Jacobson 3/3 area-law-derivation checks pass in both regimes.

Usage:
    python ./src/recompute_schwarzschild_defect.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "black_hole"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_schwarzschild():
    with open(DATA / "schwarzschild_far_field.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    sw = load_schwarzschild()
    print("=" * 72)
    print("Schwarzschild-defect recompute (Section 3 of the manuscript)")
    print("=" * 72)
    print()

    # Program-wide tier thresholds:
    #   EXACT     :  |residual| <  0.40 %
    #   PRECISE   :  0.40 % <= |residual| < 2.50 %
    EXACT_THRESHOLD_PCT = 0.40
    PRECISE_THRESHOLD_PCT = 2.50

    def _tier_from_residual(res_pct):
        if abs(res_pct) < EXACT_THRESHOLD_PCT:
            return "EXACT"
        if abs(res_pct) < PRECISE_THRESHOLD_PCT:
            return "PRECISE"
        return "OUT_OF_TIER"

    nc = sw["newton_constant"]
    # Recompute the tier from the residual rather than trust the JSON
    # label. If the data file ever drifts (e.g. residual > 0.40 % but
    # tier = EXACT), the recompute catches it loudly.
    tier_recomputed = _tier_from_residual(nc["residual_pct"])
    tier_label = nc["tier"]
    tier_consistent = tier_label == tier_recomputed

    print("--- Newton-constant test ---")
    print(f"  G_N predicted (SI):    {nc['G_N_predicted_SI']:.5e}")
    print(f"  G_N measured (SI):     {nc['G_N_measured_SI']:.5e}")
    print(f"  G_N ratio:             {nc['G_N_ratio']:.6f}")
    print(f"  Residual:              {nc['residual_pct']:.4f}%")
    print(f"  Tier (label in data):  {tier_label}")
    print(f"  Tier (recomputed):     {tier_recomputed}  "
          f"(threshold EXACT < {EXACT_THRESHOLD_PCT}%)")
    print(f"  Tier label consistent: {tier_consistent}")
    print()

    ed = sw["emergent_spacetime_dimension"]
    print("--- Emergent spacetime dimension ---")
    print(f"  d_eff^spec:            {ed['d_spectral_eff']:.5f}")
    print(f"  d_spatial (nearest):   {ed['d_spatial_nearest_integer']}")
    print(f"  d_spacetime:           {ed['d_spacetime']}")
    print(f"  d_residual:            {ed['d_residual']:.5f}")
    print()

    p1 = sw["metric_quality_canonical"]
    p2 = sw["metric_quality_extended"]
    print("--- Metric-quality match (canonical / extended regimes) ---")
    print(f"  Canonical g_00 residual via tensor_projector_closure: "
          f"{p1['tensor_projector_closure']:.4f}")
    print(f"  Extended  g_00 residual via tensor_projector_closure: "
          f"{p2['tensor_projector_closure']:.4f}")
    print(f"  Canonical Einstein-identity gap:  {p1['einstein_identity_gap']:.6f}")
    print(f"  Extended  Einstein-identity gap:  {p2['einstein_identity_gap']:.6f}")
    print()

    je = sw["jacobson_einstein"]
    # Recompute Jacobson-Einstein status from checks_passed/checks_total
    # rather than trusting the JSON label.
    def _je_status(passed, total):
        if total > 0 and passed == total:
            return "EMERGENT_GR"
        return "FAIL"

    je_status_canonical_recomp = _je_status(je["checks_passed_canonical"],
                                     je["checks_total_canonical"])
    je_status_p2_recomp = _je_status(je["checks_passed_extended"],
                                     je.get("checks_total_extended",
                                            je["checks_total_canonical"]))
    je_label_canonical_consistent = (je["einstein_equation_status_canonical"]
                              == je_status_canonical_recomp)
    je_label_p2_consistent = (je["einstein_equation_status_extended"]
                              == je_status_p2_recomp)

    print("--- Jacobson-Einstein area-law derivation ---")
    print(f"  Canonical (P1):   {je['checks_passed_canonical']}/{je['checks_total_canonical']}"
          f" -- label={je['einstein_equation_status_canonical']!r} "
          f"recomp={je_status_canonical_recomp!r} "
          f"(consistent={je_label_canonical_consistent})")
    print(f"  Extended (P2'):   {je['checks_passed_extended']}/"
          f"{je.get('checks_total_extended', je['checks_total_canonical'])}"
          f" -- label={je['einstein_equation_status_extended']!r} "
          f"recomp={je_status_p2_recomp!r} "
          f"(consistent={je_label_p2_consistent})")
    print()

    out = {
        "G_N_ratio": nc["G_N_ratio"],
        "G_N_residual_pct": nc["residual_pct"],
        "G_N_tier_label": tier_label,
        "G_N_tier_recomputed": tier_recomputed,
        "G_N_tier_label_consistent": tier_consistent,
        "tier_thresholds_pct": {
            "EXACT": EXACT_THRESHOLD_PCT,
            "PRECISE": PRECISE_THRESHOLD_PCT,
        },
        "d_eff_spec": ed["d_spectral_eff"],
        "d_spatial": ed["d_spatial_nearest_integer"],
        "d_spacetime": ed["d_spacetime"],
        "metric_canonical": {
            "tensor_projector_closure": p1["tensor_projector_closure"],
            "einstein_identity_gap": p1["einstein_identity_gap"],
        },
        "metric_extended": {
            "tensor_projector_closure": p2["tensor_projector_closure"],
            "einstein_identity_gap": p2["einstein_identity_gap"],
        },
        "jacobson_einstein": {
            "canonical_status_label": je["einstein_equation_status_canonical"],
            "canonical_status_recomputed": je_status_canonical_recomp,
            "canonical_status_label_consistent": je_label_canonical_consistent,
            "canonical_checks": [je["checks_passed_canonical"], je["checks_total_canonical"]],
            "extended_status_label": je["einstein_equation_status_extended"],
            "extended_status_recomputed": je_status_p2_recomp,
            "extended_status_label_consistent": je_label_p2_consistent,
            "extended_checks": [je["checks_passed_extended"],
                               je.get("checks_total_extended",
                                      je["checks_total_canonical"])],
        },
    }
    out_path = OUTPUTS / "schwarzschild_defect_recompute.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
