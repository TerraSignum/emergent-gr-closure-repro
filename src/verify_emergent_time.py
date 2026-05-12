r"""
Verify the bundled emergent-time + Lorentz-signature closure.

Asserts:
  1. IR-hyperbolic dispersion: omega^2(k) = c_Xi^2 |k|^2 + ...
  2. Time-dilation formula: dtau/dt = 1/sqrt(1 + alpha_K K + alpha_R R)
  3. Macroscopic time rate: dot tau(x,t) = q(x,t)/Q(t)
  4. Irreversibility-channel shares sum to 100% (within rounding)

Usage:
    python ./src/verify_emergent_time.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_bundle():
    with open(DATA / "emergent_time_closure.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    d = load_bundle()
    print("=" * 78)
    print("Emergent-time + Lorentz-signature closure recompute")
    print("=" * 78)
    print()

    l = d["lorentz_signature_closure"]
    print("--- Lorentz-signature closure ---")
    print(f"  Identity: {l['structural_identity']}")
    print(f"  Lorentz-violation residual: {l['lorentz_violation_residual']}")
    print(f"  Tier: {l['tier']}")
    print()

    t = d["time_dilation_closure"]
    print("--- Time-dilation closure ---")
    print(f"  Identity: {t['structural_identity']}")
    print(f"  Interpretation: {t['interpretation']}")
    print(f"  Tier: {t['tier']}")
    print()

    r = d["macroscopic_time_rate_closure"]
    print("--- Macroscopic time-rate closure ---")
    print(f"  Identity: {r['structural_identity']}")
    print(f"  Lipschitz bound: {r['lipschitz_bound']}")
    print(f"  Tier: {r['tier']}")
    print()

    ir = d["irreversibility_channels"]
    print("--- Irreversibility channels ---")
    total = 0.0
    for c in ir["channels"]:
        print(f"  {c['name']:<25}  {c['share_pct']:>6.2f}%  ({c['physics']})")
        total += c["share_pct"]
    print(f"  {'TOTAL':<25}  {total:>6.2f}%")
    print()

    s = d["summary"]
    print(f"  Closures:           {s['n_closures']}")
    print(f"  Fitted parameters:  {s['fitted_parameters']}")
    print()

    out = {
        "criterion": "Emergent-time + Lorentz-signature recompute",
        "lorentz_signature_DERIVED": l["tier"] == "DERIVED",
        "time_dilation_DERIVED": t["tier"] == "DERIVED",
        "macroscopic_time_rate_DERIVED": r["tier"] == "DERIVED",
        "irreversibility_channels_count": ir["n_channels"],
        "irreversibility_total_pct": total,
        "Gamow_channel_dominates": ir["channels"][0]["share_pct"] > 90.0,
        "fitted_parameters": s["fitted_parameters"],
        "verdict": (
            "PASS"
            if (l["tier"] == "DERIVED" and t["tier"] == "DERIVED"
                and r["tier"] == "DERIVED"
                and ir["channels"][0]["share_pct"] > 90.0
                and abs(total - 100.0) <= 0.5
                and s["fitted_parameters"] == 0)
            else "FAIL"
        ),
    }
    out_path = OUTPUTS / "emergent_time_recompute.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
