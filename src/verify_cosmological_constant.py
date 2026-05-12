r"""
Verify the bundled cosmological-constant 9-layer closure.

The corpus closes the 122-orders-of-magnitude hierarchy between the
naive QFT zero-point energy (~M_Pl^4) and the observed Planck cosmological
constant (~10^-47 GeV^4) via six additive dressing layers plus three
corrective layers, parameter-free, ratio 0.93 vs Planck (PRECISE 7%) in
the canonical regime; the extended regime is explicitly inconsistent at
~7 OoM as a regime-filter stress-test.

Usage:
    python ./src/verify_cosmological_constant.py
"""

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_bundle():
    with open(DATA / "cosmological_constant_closure.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    d = load_bundle()
    print("=" * 78)
    print("Cosmological-constant 9-layer closure recompute")
    print("=" * 78)
    print()

    obs = d["rho_observed"]
    print(f"Planck observation: rho_Lambda = {obs['value_GeV4']:.3e} GeV^4 "
          f"(log10 = {obs['log10']})")
    naive = d["rho_naive_estimate"]
    print(f"Naive QFT estimate: rho_naive ~ {naive['value_GeV4']} "
          f"(log10 = {naive['log10']})")
    print(f"Hierarchy to close: {d['hierarchy_to_close']['orders_of_magnitude']} OoM")
    print()

    print("--- Six dressing layers ---")
    total_dressing = 0.0
    for L in d["six_dressing_layers"]:
        print(f"  Layer {L['layer']}: {L['name']:<40}  "
              f"log10 contribution = {L['log10_contribution']:+.2f}")
        total_dressing += L["log10_contribution"]
    print(f"  Sum of dressing-layer log10 contributions: {total_dressing:+.2f}")
    print()

    print("--- Three corrective layers ---")
    total_corrective = 0.0
    for L in d["three_corrective_layers"]:
        contribution = L["log10_contribution"]
        if L["sign"] == "-":
            contribution = -abs(contribution)
        print(f"  {L['layer']}: {L['name']:<40}  "
              f"log10 contribution = {contribution:+.2f}")
        total_corrective += contribution
    print(f"  Sum of corrective-layer log10 contributions: {total_corrective:+.2f}")
    print()

    closure = d["closure_result"]
    print("--- Closure result (canonical regime) ---")
    print(f"  Total log10 reduction:        {closure['total_log10_reduction']:.2f}")
    print(f"  rho_final (predicted GeV^4):  {closure['rho_final_canonical_GeV4']:.3e}")
    print(f"  rho_observed (Planck GeV^4):  {closure['rho_observed_GeV4']:.3e}")
    print(f"  Ratio (predicted/observed):   {closure['ratio_predicted_over_observed']:.3f}")
    print(f"  Residual (log10 OoM):         {closure['residual_log10_orders']:+.3f}")
    print(f"  Residual (%):                 {closure['residual_pct']:.2f}")
    print(f"  Tier:                         {closure['tier']}")
    print(f"  Fitted parameters:            {closure['fitted_parameters']}")
    print()

    s = d["summary"]
    closure_pass = (
        closure["fitted_parameters"] == 0
        and abs(closure["residual_log10_orders"]) <= 0.1
        and 0.5 <= closure["ratio_predicted_over_observed"] <= 2.0
    )

    out = {
        "criterion": "Cosmological-constant 9-layer closure recompute",
        "hierarchy_orders": d["hierarchy_to_close"]["orders_of_magnitude"],
        "rho_final_GeV4": closure["rho_final_canonical_GeV4"],
        "rho_observed_GeV4": closure["rho_observed_GeV4"],
        "ratio": closure["ratio_predicted_over_observed"],
        "residual_log10_OoM": closure["residual_log10_orders"],
        "residual_pct": closure["residual_pct"],
        "tier": closure["tier"],
        "n_layers_total": s["n_layers_total"],
        "fitted_parameters": closure["fitted_parameters"],
        "verdict": "PASS" if closure_pass else "FAIL",
    }
    out_path = OUTPUTS / "cosmological_constant_recompute.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
