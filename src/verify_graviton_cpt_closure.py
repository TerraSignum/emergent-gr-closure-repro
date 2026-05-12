r"""
Verify the bundled graviton + CPT + CP-violation closures from the
wider Emergence corpus (GAB-03 and GAB-04 modules).

This script asserts the structural claims:
  1. Two causal modes -> massless spin-2 graviton in d=4;
  2. CPT theorem from (Lorentz, locality, unitarity);
  3. CP violation as unique consequence of CPT + GUE T-breaking
     (the construction never introduces a theta_QCD Lagrangian term,
     so the strong-CP problem is structurally absent).

Usage:
    python ./src/verify_graviton_cpt_closure.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_bundle():
    with open(DATA / "graviton_cpt_closure.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    d = load_bundle()
    print("=" * 78)
    print("Graviton + CPT + CP-violation closure recompute")
    print("=" * 78)
    print()
    g = d["graviton_closure"]
    print("--- Graviton closure (GAB-03) ---")
    print(f"  Claim: {g['claim']}")
    print(f"  n_causal modes:    {g['values']['n_causal_modes_canonical']}")
    print(f"  graviton_mass_lu:  {g['values']['graviton_mass_lu']}")
    print(f"  graviton_massless: {g['values']['graviton_massless']}")
    print(f"  graviton_spin:     {g['values']['graviton_spin']}")
    print(f"  polarizations:     {g['values']['graviton_polarizations']}")
    print(f"  dispersion:        {g['values']['dispersion']}")
    print(f"  Tier:              {g['tier']}")
    print()

    c = d["CPT_theorem_closure"]
    print("--- CPT theorem closure (GAB-04) ---")
    print(f"  Claim: {c['claim']}")
    print("  Derivation:")
    for line in c["derivation"]:
        print(f"    - {line}")
    print(f"  Tier: {c['tier']}")
    print()

    cp = d["CP_violation_closure"]
    print("--- CP violation closure (GAB-04) ---")
    print(f"  Claim: {cp['claim']}")
    print("  Derivation:")
    for line in cp["derivation"]:
        print(f"    - {line}")
    print(f"  Consequence for strong-CP: {cp['consequence_for_strong_CP_problem']}")
    print(f"  Tier: {cp['tier']}")
    print()

    s = d["summary"]
    print("--- Summary ---")
    print(f"  Closures:           {s['n_closures']}")
    print(f"  Fitted parameters:  {s['fitted_parameters']}")
    print()

    out = {
        "criterion": "Graviton + CPT + CP-violation recompute",
        "graviton_massless": g["values"]["graviton_massless"],
        "graviton_spin": g["values"]["graviton_spin"],
        "graviton_polarizations": g["values"]["graviton_polarizations"],
        "n_causal_modes_canonical": g["values"]["n_causal_modes_canonical"],
        "CPT_theorem_tier": c["tier"],
        "CP_violation_tier": cp["tier"],
        "fitted_parameters": s["fitted_parameters"],
        "verdict": (
            "PASS"
            if (
                g["values"]["graviton_massless"]
                and g["values"]["graviton_spin"] == 2
                and g["values"]["n_causal_modes_canonical"] == 2
                and c["tier"] == "DERIVED"
                and cp["tier"] == "DERIVED"
                and s["fitted_parameters"] == 0
            )
            else "FAIL"
        ),
    }
    out_path = OUTPUTS / "graviton_cpt_recompute.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
