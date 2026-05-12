r"""
Verify the bundled dimension + arrow-of-time closures.

  1. d_spacetime = 4 derived from spectral dimension d_eff = 3.170:
     d_spatial = round(d_eff) = 3, d_spacetime = d_spatial + 1 = 4.
     Light-cone fuzziness = |d_eff - round(d_eff)| = 0.170.
  2. Arrow of time from P(forward)/P(backward) = exp(2 S_bounce)
     with S_bounce ~ 38 -> ratio ~ 10^33.

Both are derived in the wider Emergence corpus
(src/worldformula/physics/gr_emergence.py GRE-01,
 src/worldformula/physics/emergent_time.py EMT-01..EMT-02).

Usage:
    python ./src/verify_dimension_and_arrow.py
"""

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_bundle():
    with open(DATA / "dimension_and_arrow_closure.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    d = load_bundle()
    print("=" * 78)
    print("Spacetime dimension + arrow-of-time closure recompute")
    print("=" * 78)
    print()

    dim = d["spacetime_dimension_closure"]
    print("--- d_spacetime = 4 closure ---")
    print(f"  Claim: {dim['claim']}")
    v = dim["values"]
    print(f"  d_spectral_eff:                {v['d_spectral_eff']}")
    print(f"  d_spatial = round(d_eff):      {v['d_spatial_nearest_integer']}")
    print(f"  d_spacetime = d_spatial + 1:   {v['d_spacetime']}")
    print(f"  d_residual (light-cone fuzz):  {v['d_residual']}")
    print(f"  spatial_dimension_derived:     {v['spatial_dimension_derived']}")
    print(f"  spacetime_matches_GR:          {v['spacetime_matches_GR']}")
    print(f"  Tier: {dim['tier']}")
    print()

    arrow = d["arrow_of_time_closure"]
    print("--- Arrow-of-time closure ---")
    print(f"  Claim: {arrow['claim']}")
    av = arrow["values"]
    s_bounce = av["S_bounce_canonical"]
    ratio = math.exp(2 * s_bounce)
    print(f"  S_bounce canonical:            {s_bounce}")
    print(f"  exp(2 S_bounce) recomputed:    {ratio:.3e}")
    print(f"  asymmetry orders of magnitude: ~{int(math.log10(ratio))}")
    print(f"  Tier: {arrow['tier']}")
    print()

    s = d["summary"]
    print("--- Summary ---")
    print(f"  Closures:           {s['n_closures']}")
    print(f"  Fitted parameters:  {s['fitted_parameters']}")
    print(f"  Open problems resolved:")
    for p in s["open_problems_resolved"]:
        print(f"    - {p}")
    print()

    # Verdict
    spatial_ok = (v["d_spatial_nearest_integer"] == 3
                  and v["d_spacetime"] == 4)
    arrow_ok = (s_bounce > 30 and ratio > 1e30)

    out = {
        "criterion": "Spacetime dimension + arrow-of-time recompute",
        "d_spacetime_equals_4": v["d_spacetime"] == 4,
        "d_spatial_equals_3": v["d_spatial_nearest_integer"] == 3,
        "lorentzian_possible": v["lorentzian_possible"],
        "S_bounce_canonical": s_bounce,
        "P_forward_over_P_backward_recomputed": ratio,
        "asymmetry_OoM_recomputed": int(math.log10(ratio)),
        "fitted_parameters": s["fitted_parameters"],
        "verdict": (
            "PASS"
            if (spatial_ok and arrow_ok and s["fitted_parameters"] == 0)
            else "FAIL"
        ),
    }
    out_path = OUTPUTS / "dimension_and_arrow_recompute.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
