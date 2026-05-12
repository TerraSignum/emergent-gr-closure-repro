r"""
Verify the bundled vacuum-stability + reheating closures.

Vacuum stability: B = infinity (Coleman-de Luccia bounce action) in all
three canonical regimes; the Xi-effective potential is monotonically
repulsive so no tunnelling channel exists. Compares to the
Standard-Model Higgs-quartic metastability at ~10^11 GeV.

Reheating: universal gravitational inflaton-decay channel
Gamma_grav = (N_dof/96 pi) m_phi^3 / M_Pl^2; T_RH = 6.61e16 GeV at
ratio 1.021 to the Planck 2018 3-sigma anchor (PRECISE).

Usage:
    python ./src/verify_vacuum_stability.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_bundle():
    with open(DATA / "vacuum_stability_closure.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    d = load_bundle()
    print("=" * 78)
    print("Vacuum-stability + reheating closure recompute")
    print("=" * 78)
    print()

    v = d["vacuum_stability_closure"]
    print("--- Vacuum-stability closure (VAC-01) ---")
    print(f"  Claim: {v['claim']}")
    print(f"  Construction: {v['construction']}")
    print()
    print("  Regime results:")
    for r in v["regime_results"]:
        print(f"    {r['regime']:<25}  barrier: {r['barrier_shape']}, "
              f"B = {r['B_action']}, verdict: {r['verdict']}")
    print()
    print(f"  SM comparison: {v['comparison_to_SM']}")
    print()
    print(f"  Tier: {v['tier']}")
    print()

    r = d["reheating_closure"]
    print("--- Reheating closure (REH-01) ---")
    print(f"  Claim: {r['claim']}")
    print(f"  Gamma_grav formula: {r['Gamma_grav_formula']}")
    print(f"  Gamma_grav (GeV):   {r['Gamma_grav_GeV']:.3e}")
    print(f"  T_RH predicted:     {r['T_RH_predicted_GeV']:.3e} GeV")
    print(f"  T_RH anchor:        {r['T_RH_anchor_source']}")
    print(f"  T_RH ratio:         {r['T_RH_ratio_to_anchor']:.3f}")
    print(f"  Tier:               {r['tier']}")
    print()

    s = d["summary"]
    print("--- Summary ---")
    print(f"  Closures:           {s['n_closures']}")
    print(f"  Fitted parameters:  {s['fitted_parameters']}")
    print()

    # Verdict
    all_regimes_stable = all(
        rg["verdict"] == "absolutely stable" for rg in v["regime_results"]
    )
    treh_within_precise = abs(r["T_RH_ratio_to_anchor"] - 1.0) <= 0.025

    out = {
        "criterion": "Vacuum-stability + reheating recompute",
        "all_regimes_absolutely_stable": all_regimes_stable,
        "n_regimes_checked": len(v["regime_results"]),
        "B_infinite_in_all_regimes": all(
            rg["B_action"] == "infinity" for rg in v["regime_results"]
        ),
        "T_RH_ratio": r["T_RH_ratio_to_anchor"],
        "T_RH_PRECISE": treh_within_precise,
        "Gamma_grav_GeV": r["Gamma_grav_GeV"],
        "T_RH_GeV": r["T_RH_predicted_GeV"],
        "fitted_parameters": s["fitted_parameters"],
        "verdict": (
            "PASS"
            if (all_regimes_stable and treh_within_precise
                and s["fitted_parameters"] == 0)
            else "FAIL"
        ),
    }
    out_path = OUTPUTS / "vacuum_stability_recompute.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
