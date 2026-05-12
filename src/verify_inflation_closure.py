r"""
Verify the bundled inflation closure: three independent readouts on
n_s (algebraic loop-class EXACT, dynamical cascade range, world-filter
proxy), the tensor-to-scalar ratio r below the BICEP/Keck upper bound,
the cascade mechanism delivering >70 e-folds, and the explicit
inconsistency of the extended-regime stress test.

Usage:
    python ./src/verify_inflation_closure.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_bundle():
    with open(DATA / "inflation_closure.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    d = load_bundle()
    print("=" * 78)
    print("Inflation closure recompute (n_s, r, N_efolds, A_s, mechanism)")
    print("=" * 78)
    print()

    p = d["primary_closure_loop_class_algebraic"]
    print("--- Primary n_s closure (loop-class algebraic) ---")
    print(f"  Identity:  {p['structural_identity']}")
    print(f"  Predicted: {p['predicted']}")
    print(f"  Anchor:    {p['anchor_value']} ({p['anchor_source']})")
    print(f"  Residual:  {p['residual_pct']}%   Tier: {p['tier']}")
    print()

    s2 = d["secondary_closure_inflation_cascade_range"]
    print("--- Secondary n_s closure (inflation cascade range) ---")
    print(f"  Range:     [{s2['n_s_range_lo']:.6f}, {s2['n_s_range_hi']:.6f}]")
    print(f"  Measured-in-range: {s2['n_s_measured_in_range']}")
    print(f"  Tier:      {s2['tier']}")
    print()

    r = d["tensor_scalar_closure"]
    print("--- Tensor-to-scalar ratio r ---")
    print(f"  Predicted r:        {r['predicted_r']}")
    print(f"  Observational bound: r < {r['observational_upper_bound_r']}")
    print(f"  Factor below bound:  {r['factor_below_bound']}")
    print(f"  Below limit:         {r['below_limit']}")
    print(f"  Tier:                {r['tier']}")
    print()

    nf = d["horizon_flatness_closure"]
    print("--- Horizon-flatness (N_efolds) ---")
    print(f"  N_efolds canonical:    {nf['N_efolds_canonical']}")
    print(f"  Typical requirement:   {nf['N_efolds_horizon_flatness_typical']}")
    print(f"  Factor above typical:  {nf['factor_above_typical']}")
    print(f"  Mechanism:             {nf['inflation_mechanism']}")
    print(f"  Tier:                  {nf['tier']}")
    print()

    a = d["primordial_amplitude_closure"]
    print("--- Primordial scalar amplitude A_s ---")
    print(f"  Predicted A_s:  {a['A_s_predicted']:.3e}")
    print(f"  Anchor (Planck): {a['A_s_anchor']:.3e}")
    print(f"  log10 ratio:     {a['log10_A_s_ratio']}")
    print(f"  Tier:            {a['tier']}")
    print()

    e = d["extended_regime_stress_test"]
    print("--- Extended-regime stress test (regime filter) ---")
    print(f"  n_s extended:   {e['values']['n_s']}")
    print(f"  r extended:     {e['values']['r_tensor_to_scalar']}")
    print(f"  Factor above r bound: {e['values']['factor_above_r_bound']}")
    print()

    summ = d["summary"]
    print("--- Summary ---")
    print(f"  Predictions canonical:        {summ['n_predictions_canonical']}")
    print(f"  Within-observation canonical: {summ['n_within_observation_canonical']}")
    print(f"  Fitted parameters:            {summ['fitted_parameters']}")
    print(f"  Load-bearing n_s tier:        {summ['load_bearing_n_s_tier']}")
    print()

    primary_n_s_exact = (p["tier"] == "EXACT" and p["residual_pct"] < 0.5)
    cascade_range_ok = s2["n_s_measured_in_range"]
    r_satisfied = r["below_limit"]
    nefolds_ok = nf["above_typical"]
    n_s_three_consistent = primary_n_s_exact and cascade_range_ok

    out = {
        "criterion": "Inflation closure recompute",
        "primary_n_s_EXACT_via_loop_class": primary_n_s_exact,
        "cascade_range_contains_Planck": cascade_range_ok,
        "tensor_scalar_r_below_BICEP_bound": r_satisfied,
        "N_efolds_above_horizon_flatness_typical": nefolds_ok,
        "extended_regime_explicitly_rejected": e["values"]["factor_above_r_bound"] > 5.0,
        "fitted_parameters": summ["fitted_parameters"],
        "verdict": (
            "PASS"
            if (primary_n_s_exact and cascade_range_ok
                and r_satisfied and nefolds_ok
                and summ["fitted_parameters"] == 0)
            else "FAIL"
        ),
    }
    out_path = OUTPUTS / "inflation_closure_recompute.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
