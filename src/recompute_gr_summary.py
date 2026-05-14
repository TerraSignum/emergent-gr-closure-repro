r"""
Emergent-GR closure recompute: aggregate summary.

Loads all data files (xi_metric_inputs, a1_regime_constants, clp_scores,
einstein_gap_results, ppn_results) and prints a single coherent
summary that demonstrates the closure on the canonical regime P1.

Usage:
    python ./src/recompute_gr_summary.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load(name):
    with open(DATA / name, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    metric = load("xi_metric_inputs.json")
    a1 = load("a1_regime_constants.json")
    clp = load("clp_scores.json")
    gap = load("einstein_gap_results.json")
    ppn = load("ppn_results.json")

    print("=" * 84)
    print("Emergent Einstein dynamics from relational metric closure -- summary")
    print("=" * 84)
    print()

    print("--- (1) Metric construction ---")
    print(f"  d_ij = {metric['metric_definition']}")
    print(f"  ell_0 = {metric['ell_0']}")
    print("  Axioms:")
    for k, v in metric["Xi_axioms"].items():
        print(f"    {k}: {v}")
    print()

    print("--- (2) A1 fast-slow regime checks ---")
    p1 = a1["regimes"]["canonical"]
    print(f"  P1: lambda_triangle = {p1['lambda_triangle']} (>=1.0?)  "
          f"epsilon = {p1['epsilon']} (<=0.10?)  -- {p1['status']}")
    print(f"  Verdict (P1): {a1['verdict_canonical']}")
    print()

    print("--- (3) CLP scores ---")
    pass_threshold = True
    for axis, ax in clp["axes"].items():
        ok = "PASS" if ax["value"] >= ax["threshold"] else "FAIL"
        if ax["value"] < ax["threshold"]:
            pass_threshold = False
        print(f"  {axis} ({ax['name']}): {ax['value']:.4f}  >= {ax['threshold']}  -- {ok}")
    print(f"  Aggregate: {'PASS' if pass_threshold else 'FAIL'}")
    print()

    print("--- (4) Einstein-gap two-point Richardson extrapolation ---")
    print(f"  Target exponent (universality): {gap['alpha_target_for_universality_claim']:.5f}  "
          f"({gap['alpha_target_form']})")
    pts = sorted(gap["honest_two_point_data"], key=lambda p: p["N"])
    print(f"  Two clean points:")
    for p in pts:
        print(f"    N = {p['N']:<6}  Delta_E = {p['Delta_E']:.6f}   ({p['regime_label']})")
    print(f"  Richardson candidates (all must pass |gap_inf| <= {gap['closure_threshold']}):")
    all_pass = True
    for cand in gap["richardson_candidates"]:
        if not cand["passes_005"]:
            all_pass = False
        print(f"    alpha = {cand['alpha']:.4f} ({cand['exponent_name']}): "
              f"gap_inf = {cand['gap_inf']:+.5f}  "
              f"{'PASS' if cand['passes_005'] else 'FAIL'}")
    print(f"  Aggregate: {'PASS' if all_pass else 'FAIL'} "
          f"(>=3 N-points needed for definitive single-exponent ID)")
    print()

    print("--- (5) PPN parameters ---")
    print(f"  gamma_PPN = {ppn['predictions']['gamma_PPN']['value']:.6f}  "
          f"+/- {ppn['predictions']['gamma_PPN']['uncertainty']}")
    print(f"  beta_PPN  = {ppn['predictions']['beta_PPN']['value']:.6f}   "
          f"+/- {ppn['predictions']['beta_PPN']['uncertainty']}")
    print(f"  Tier:    {ppn['tier']}")
    print()

    out = {
        "metric_definition": metric["metric_definition"],
        "canonical_a1_status": p1["status"],
        "clp_scores": {k: v["value"] for k, v in clp["axes"].items()},
        "clp_threshold_pass": pass_threshold,
        "alpha_target": gap["alpha_target_for_universality_claim"],
        "alpha_target_form": gap["alpha_target_form"],
        "richardson_candidates_all_pass": gap["all_candidates_pass_005_threshold"],
        "ppn_gamma": ppn["predictions"]["gamma_PPN"]["value"],
        "ppn_beta":  ppn["predictions"]["beta_PPN"]["value"],
    }
    out_path = OUTPUTS / "recompute_gr_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved: {out_path}")

    print("--- Acceptance ---")
    a1_ok = p1["status"] == "PASS"
    gap_ok = gap["all_candidates_pass_005_threshold"]
    ppn_ok = (abs(ppn["predictions"]["gamma_PPN"]["value"] - 1.0) <= 1e-4
              and abs(ppn["predictions"]["beta_PPN"]["value"] - 1.0) <= 1e-4)
    if a1_ok and pass_threshold and gap_ok and ppn_ok:
        print("  PASS: emergent-Einstein closure consistent under all five axes (P1 regime).")
    else:
        print("  FAIL: at least one axis does not pass.")


if __name__ == "__main__":
    main()
