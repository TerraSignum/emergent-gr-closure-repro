r"""
Two-point Richardson extrapolation of the Einstein-identity gap.

Three candidate exponents (Ricci-order 2/3, linear 1.0, empirical
2-point fit 0.8477) are exercised against the two clean lattice
sizes (N1=1534, N2=2254). All three must yield gap_inf below the
0.05 closure-domain threshold for the universality claim to hold
under the present (data-poor) two-point construction.

A definitive single-exponent identification requires >=3 lattice
points; that data is in flight.

Usage:
    python ./src/compare_gap_definitions.py
"""

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def main():
    with open(DATA / "einstein_gap_results.json", "r", encoding="utf-8") as f:
        gap = json.load(f)

    target = gap["alpha_target_for_universality_claim"]
    threshold = gap["closure_threshold"]

    print("=" * 78)
    print("Einstein-identity gap: two-point Richardson universality check")
    print("=" * 78)
    print()
    print(f"  Structural target exponent: {target:.5f}  ({gap['alpha_target_form']})")
    print(f"  Closure-domain threshold:   |gap_inf| <= {threshold}")
    print()
    print(f"  --- Two clean lattice points ---")
    pts = sorted(gap["honest_two_point_data"], key=lambda p: p["N"])
    for p in pts:
        print(f"    N = {p['N']:<6}  Delta_E = {p['Delta_E']:.6f}   ({p['regime_label']})")
    print()

    # Recompute the empirical alpha as a sanity cross-check
    N1, gap1 = pts[0]["N"], pts[0]["Delta_E"]
    N2, gap2 = pts[1]["N"], pts[1]["Delta_E"]
    alpha_recompute = math.log(gap1 / gap2) / math.log(N2 / N1)
    print(f"  Empirical 2-point fit (recomputed): alpha = {alpha_recompute:.4f}")
    print()

    print(f"  --- Three Richardson candidate exponents ---")
    print(f"  {'name':<32} {'alpha':>8} {'gap_inf':>10}  {'passes <0.05'}")
    print("  " + "-" * 70)
    rows = []
    all_pass = True
    for cand in gap["richardson_candidates"]:
        passes = cand["passes_005"]
        if not passes:
            all_pass = False
        rows.append((cand["exponent_name"], cand["alpha"], cand["gap_inf"], passes))
        print(f"  {cand['exponent_name']:<32} {cand['alpha']:>8.4f} "
              f"{cand['gap_inf']:>+10.5f}  {'yes' if passes else 'NO'}")
    print()

    print("--- Verdict ---")
    if all_pass:
        print(f"  PASS: all three Richardson candidate exponents land "
              f"|gap_inf| <= {threshold}.")
    else:
        print(f"  FAIL: at least one candidate fails the closure threshold.")

    out = {
        "alpha_target": target,
        "alpha_target_form": gap["alpha_target_form"],
        "closure_threshold": threshold,
        "two_point_data": gap["honest_two_point_data"],
        "alpha_empirical_recomputed": alpha_recompute,
        "richardson_candidates": rows_as_dicts(rows),
        "all_pass": all_pass,
        "verdict": "PASS" if all_pass else "FAIL",
        "data_poverty_caveat": gap["data_poverty_caveat"],
    }
    out_path = OUTPUTS / "gap_comparison_table.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    csv_path = OUTPUTS / "gap_comparison_table.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("name,alpha,gap_inf,passes_005\n")
        for name, alpha, gap_inf, passes in rows:
            f.write(f"{name},{alpha:.5f},{gap_inf:+.6f},{int(passes)}\n")
    print()
    print(f"Saved: {out_path}")
    print(f"Saved: {csv_path}")


def rows_as_dicts(rows):
    return [
        {"name": r[0], "alpha": r[1], "gap_inf": r[2], "passes_005": r[3]}
        for r in rows
    ]


if __name__ == "__main__":
    main()
