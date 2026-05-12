r"""
Phase-I: Look-Elsewhere-Effect (LEE) Bonferroni-correction test
for the three System-R rational identifications of Lambda_lat^infty.

Method: enumerate the natural search space of simple System-R
rational combinations (depth <= 3 linear combinations with
multipliers in {-2, -1, -1/2, 0, 1/2, 1, 2}, plus the topological
multipliers {1/d_st, 1/2, 1/3, 2/3, pi/4, 4/pi}), count how many
distinct rational values fall within 1%, 2%, and 5% of each
empirical Lambda_lat^infty value. The Bonferroni-corrected
p-value for the three identifications jointly is then estimated
under the null that the rationals are randomly distributed in
the [0, 2] range.

Output:
    outputs/lambda_LEE_bonferroni.json with:
      - search-space size
      - per-Lambda hit count at each tolerance
      - per-Lambda Bonferroni-corrected p-value
      - joint three-Lambda p-value

This is the peer-review-relevant LEE quantification analogous to
the parent corpus's Remark 16d.3a (which does the same for the
8 landings in the companion Paper~2).

Usage:
    python ./src/verify_lambda_LEE_bonferroni.py
"""
from __future__ import annotations
import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


# System-R rational coefficients
SYSTEM_R = {
    "alpha_xi": Fraction(9, 10),
    "gamma":    Fraction(1, 10),
    "beta_pi":  Fraction(15, 16),
    "D_Omega":  Fraction(67, 80),
    "eps_sq":   Fraction(1, 20),
}

# Topological multipliers from corpus alphabet (Paper 2 search-space spec)
MULTIPLIERS = [
    Fraction(0),
    Fraction(1, 2), Fraction(-1, 2),
    Fraction(1), Fraction(-1),
    Fraction(2), Fraction(-2),
    Fraction(1, 4), Fraction(-1, 4),
    Fraction(1, 3), Fraction(-1, 3),
    Fraction(2, 3), Fraction(-2, 3),
]

EMPIRICAL = {
    "proxy":        {"value": 0.2513, "rational_match": Fraction(1, 4),
                     "rational_str": "1/4"},
    "row_mean":     {"value": 0.8510, "rational_match": Fraction(17, 20),
                     "rational_str": "17/20"},
    "section_14_1": {"value": 1.220,  "rational_match": Fraction(6, 5),
                     "rational_str": "6/5"},
}


def enumerate_search_space():
    """Enumerate all rational values reachable by a sum of up to 3
    multiplied System-R coefficients within the multiplier set,
    plus a constant offset from {0, 1/4, 1/2, 1, 2}."""
    coeff_keys = list(SYSTEM_R.keys())
    base_coeffs = list(SYSTEM_R.values())
    # Add some topological constants the search alphabet might include
    topo_constants = [
        Fraction(0), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2),
        Fraction(2, 3), Fraction(3, 4), Fraction(1), Fraction(5, 4),
    ]

    seen = set()
    # Single coefficient with multiplier
    for c in base_coeffs:
        for m in MULTIPLIERS:
            v = m * c
            if v not in seen and float(v) > -0.5 and float(v) < 3.0:
                seen.add(v)

    # Two-coefficient sum
    for c1 in base_coeffs:
        for c2 in base_coeffs:
            for m1 in MULTIPLIERS:
                for m2 in MULTIPLIERS:
                    v = m1 * c1 + m2 * c2
                    if v not in seen and float(v) > -0.5 and float(v) < 3.0:
                        seen.add(v)

    # Three-coefficient sum (limited multipliers to keep size reasonable)
    short_mult = [Fraction(0), Fraction(1), Fraction(-1),
                  Fraction(1, 2), Fraction(-1, 2)]
    for c1 in base_coeffs:
        for c2 in base_coeffs:
            for c3 in base_coeffs:
                for m1 in short_mult:
                    for m2 in short_mult:
                        for m3 in short_mult:
                            v = m1 * c1 + m2 * c2 + m3 * c3
                            if v not in seen and float(v) > -0.5 and float(v) < 3.0:
                                seen.add(v)

    # Add additive topological constants
    for tc in topo_constants:
        for v_base in list(seen):
            vp = v_base + tc
            vm = v_base - tc
            if float(vp) > -0.5 and float(vp) < 3.0 and vp not in seen:
                seen.add(vp)
            if float(vm) > -0.5 and float(vm) < 3.0 and vm not in seen:
                seen.add(vm)

    return sorted(seen, key=lambda x: float(x))


def count_hits(target, search_space, tolerance):
    """Count how many rationals in search_space are within `tolerance`
    relative of `target`."""
    return sum(1 for r in search_space
               if target != 0 and abs(float(r) - target) / abs(target) < tolerance)


def main():
    print("=" * 78)
    print("Phase-I: Look-Elsewhere Bonferroni-correction test for the")
    print("three System-R rational identifications of Lambda_lat^infty.")
    print("=" * 78)
    print()

    search_space = enumerate_search_space()
    n_search = len(search_space)
    range_size = 3.0 - (-0.5)  # search range
    print(f"Search-space size: {n_search} distinct rationals "
          f"in [-0.5, 3.0]")
    print()

    results = {}
    for tag, emp in EMPIRICAL.items():
        target = emp["value"]
        rational_str = emp["rational_str"]
        per_tag = {"target": target, "rational_match": rational_str}
        for tol_pct in [0.1, 0.5, 1.0, 2.0, 5.0]:
            tol = tol_pct / 100.0
            n_hits = count_hits(target, search_space, tol)
            # Probability under uniform-random null:
            # density of search space = n_search / range_size
            # expected count within +- tol*target = 2*tol*target * density
            expected_random = 2 * tol * target * n_search / range_size
            per_tag[f"hits_at_{tol_pct}pct"] = n_hits
            per_tag[f"expected_random_at_{tol_pct}pct"] = expected_random
            per_tag[f"poisson_p_at_{tol_pct}pct"] = (
                1.0 - math.exp(-expected_random) if n_hits >= 1 else 1.0
            )
        results[tag] = per_tag

    print(f"{'tag':>14}  {'target':>8}  {'match':>10}  "
          f"{'hits@1%':>8}  {'p@1%':>8}  {'hits@2%':>8}  {'p@2%':>8}  "
          f"{'hits@5%':>8}  {'p@5%':>8}")
    print("-" * 110)
    for tag, r in results.items():
        print(f"{tag:>14}  {r['target']:>8.4f}  {r['rational_match']:>10}  "
              f"{r['hits_at_1.0pct']:>8}  {r['poisson_p_at_1.0pct']:>8.4f}  "
              f"{r['hits_at_2.0pct']:>8}  {r['poisson_p_at_2.0pct']:>8.4f}  "
              f"{r['hits_at_5.0pct']:>8}  {r['poisson_p_at_5.0pct']:>8.4f}")
    print()

    # Joint Bonferroni-corrected p-value at the per-Lambda 1% (or actual match)
    # Take the actual-match relative error per row; use min(p, 1)
    joint_p_2pct = 1.0
    for tag, r in results.items():
        joint_p_2pct *= max(r["poisson_p_at_2.0pct"], 1e-12)
    # Bonferroni for 3 trials
    bonf_p = min(1.0, 3 * max(r["poisson_p_at_2.0pct"]
                              for r in results.values()))
    print(f"Joint product p-value at 2% (independent null): {joint_p_2pct:.4e}")
    print(f"Bonferroni-corrected p-value at 2% "
          f"(3 trials, max-individual): {bonf_p:.4f}")
    print()

    # Specific actual-match analysis
    print("Specific actual-match relative errors and individual p-values:")
    for tag, r in results.items():
        match_value = float(EMPIRICAL[tag]["rational_match"])
        rel_err = abs(match_value - r["target"]) / r["target"]
        # Probability of random hit at this exact tolerance
        expected_at_actual = (2 * rel_err * r["target"]
                              * n_search / range_size)
        poisson_p = (1.0 - math.exp(-expected_at_actual)
                     if expected_at_actual > 0 else 1.0)
        print(f"  {tag:>14}: rel_err = {rel_err*100:.2f}%, "
              f"expected-random hits = {expected_at_actual:.2f}, "
              f"individual p = {poisson_p:.4f}")
    print()

    # Verdict
    print("--- LEE Verdict ---")
    print("Path-5 System-R rational identifications under three K_rec")
    print("conventions: each individual match has nontrivial LEE risk")
    print("at 1-5% tolerance under the present search-space scope. The")
    print("Bonferroni-corrected joint test does NOT formally exclude")
    print("LEE; structural arguments (Lemma 1 spinor-trace 1/d_st for")
    print("1/4; non-scalar Clifford-channel rate for 17/20) carry the")
    print("identification weight beyond pure numerical match.")

    out = {
        "method": "Bonferroni_LEE_test_against_System_R_rational_search_space",
        "search_space_size": n_search,
        "search_range": [-0.5, 3.0],
        "per_lambda_results": results,
        "joint_p_2pct_independent": joint_p_2pct,
        "bonferroni_p_2pct_3trials": bonf_p,
        "verdict": (
            "Path-5 System-R rational identifications carry nontrivial "
            "LEE risk at 1-5% tolerance under the search-space scope; "
            "the joint Bonferroni-corrected test does NOT formally "
            "exclude LEE. The structural identifications (Lemma 1 "
            "spinor-trace 1/d_spacetime for 1/4; non-scalar Clifford-"
            "channel rate for 17/20) provide identification weight "
            "beyond pure numerical match. The 6/5 identification "
            "for Section-14.1 sits at the look-elsewhere boundary "
            "and is reported without structural promotion."
        ),
    }
    out_path = OUTPUTS / "lambda_LEE_bonferroni.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
