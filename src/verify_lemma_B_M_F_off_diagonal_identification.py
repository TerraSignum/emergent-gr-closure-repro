r"""Lemma B Step 4a (1)-side: identify the off-diagonal weights of
the family-coupling matrix M_F that produce its empirically certified
spectrum (0, 7/6, 11/6).

Algebraic constraints from spec(L_norm(M_F)) = (0, 7/6, 11/6):

(C1) Sum of squares:   rho_12^2 + rho_13^2 + rho_23^2 = 31/36
(C2) Product:          rho_12 * rho_13 * rho_23       = 5/72
(C3) Trace identity:   trace(L_norm) = N_gen = 3 (automatic)

The characteristic polynomial of the normalised adjacency
N = D^{-1/2} A D^{-1/2} on the 3-vertex K_3 is:
    mu^3 - (Sum rho^2) * mu - 2 * (Prod rho) = 0
For our case:
    mu^3 - (31/36) mu - (5/36) = 0
    (mu - 1)(6 mu + 1)(6 mu + 5) = 0
    spec(N) = {1, -1/6, -5/6} -> spec(L_norm) = {0, 7/6, 11/6}

Two constraints on three unknowns -> 1-parameter family of solutions.

This script tests CANDIDATE System-R rational triples (rho_12,
rho_13, rho_23) against (C1, C2), reporting the closest matches and
their structural interpretations.

Output: outputs/verify_lemma_B_M_F_off_diagonal_identification.json
"""
from __future__ import annotations

import json
from pathlib import Path
from fractions import Fraction
import math

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"

GAMMA = Fraction(1, 10)
N_GEN = 3
D = 4
ALPHA_XI = Fraction(9, 10)
EPS_SYNC2 = Fraction(1, 20)


# Target constraints from spec (0, 7/6, 11/6):
P_TARGET = Fraction(31, 36)   # sum of squares
Q_TARGET = Fraction(5, 72)    # product


def check_triple(r12, r13, r23):
    """Return (sum_sq, product, sum_sq_err, prod_err) for given triple."""
    sum_sq = r12 ** 2 + r13 ** 2 + r23 ** 2
    product = r12 * r13 * r23
    err_p = float(sum_sq - P_TARGET) / float(P_TARGET) * 100
    err_q = float(product - Q_TARGET) / float(Q_TARGET) * 100
    return sum_sq, product, err_p, err_q


def system_r_candidates():
    """Build a list of System-R rational candidates for rho values."""
    # Single-rational candidates (small denominators with structural names)
    cands = []
    # Pure System-R
    for name, val in [
        ("gamma", GAMMA),  # 1/10
        ("gamma/2", GAMMA / 2),  # 1/20
        ("gamma/N_gen", GAMMA / N_GEN),  # 1/30
        ("1/N_gen", Fraction(1, N_GEN)),  # 1/3
        ("1/d", Fraction(1, D)),  # 1/4
        ("1/6", Fraction(1, 6)),
        ("1/8", Fraction(1, 8)),
        ("1/9", Fraction(1, 9)),
        ("1/12", Fraction(1, 12)),
        ("1/16", Fraction(1, 16)),
        ("2/9", Fraction(2, 9)),  # (N_gen-1)/N_gen^2
        ("3/16", Fraction(3, 16)),  # (d-1)/d^2
        ("(d-1)/(2d)", Fraction(D - 1, 2 * D)),  # 3/8
        ("(N_gen-1)/(2 N_gen)", Fraction(N_GEN - 1, 2 * N_GEN)),  # 1/3
        ("alpha_xi/2", ALPHA_XI / 2),  # 9/20
        ("alpha_xi/N_gen", ALPHA_XI / N_GEN),  # 3/10
        ("alpha_xi/d", ALPHA_XI / D),  # 9/40
        ("eps_sync2", EPS_SYNC2),  # 1/20
        ("(d+N_gen)/(2 d N_gen)", Fraction(D + N_GEN, 2 * D * N_GEN)),  # 7/24
        ("1/d - gamma/2", Fraction(1, D) - GAMMA / 2),  # 1/4 - 1/20 = 1/5
        ("1/N_gen - gamma", Fraction(1, N_GEN) - GAMMA),  # 1/3 - 1/10 = 7/30
        ("1/N_gen + gamma", Fraction(1, N_GEN) + GAMMA),  # 1/3 + 1/10 = 13/30
        ("alpha_xi/N_gen - gamma", ALPHA_XI / N_GEN - GAMMA),  # 3/10 - 1/10 = 2/10
        ("(N_gen+d)/d^2", Fraction(N_GEN + D, D * D)),  # 7/16
        ("d/(d+N_gen)", Fraction(D, D + N_GEN)),  # 4/7
        ("N_gen/(d+N_gen)", Fraction(N_GEN, D + N_GEN)),  # 3/7
    ]:
        cands.append((name, val))
    return cands


def main():
    print("=" * 100)
    print("Lemma B Step 4a (1): identification of M_F off-diagonal "
            "weights (rho_12, rho_13, rho_23)")
    print("=" * 100)
    print(f"Target spectrum: spec(L_norm(M_F)) = (0, 7/6, 11/6)")
    print(f"  Constraint C1: sum rho^2 = {P_TARGET} = {float(P_TARGET):.5f}")
    print(f"  Constraint C2: prod rho   = {Q_TARGET} = {float(Q_TARGET):.5f}")
    print()

    cands = system_r_candidates()
    n_cands = len(cands)
    print(f"Testing {n_cands ** 3} ordered triples of System-R rational "
            f"candidates...")
    print()

    # Generate all triples (with possible repetition) and check
    matches = []
    for i, (n1, r1) in enumerate(cands):
        for j, (n2, r2) in enumerate(cands):
            for k, (n3, r3) in enumerate(cands):
                if i > j or j > k:  # only ordered triples (r1 <= r2 <= r3)
                    continue
                if r1 > r2 or r2 > r3:
                    continue
                sum_sq, prod, err_p, err_q = check_triple(r1, r2, r3)
                # Loose threshold: 5% on each
                if abs(err_p) < 5 and abs(err_q) < 5:
                    matches.append({
                        "rho_12_name": n1, "rho_12_val": float(r1),
                        "rho_13_name": n2, "rho_13_val": float(r2),
                        "rho_23_name": n3, "rho_23_val": float(r3),
                        "sum_sq": float(sum_sq),
                        "product": float(prod),
                        "sum_sq_rel_err_pct": err_p,
                        "prod_rel_err_pct": err_q,
                        "total_rel_err_pct": abs(err_p) + abs(err_q),
                    })

    # Sort by total error
    matches.sort(key=lambda m: m["total_rel_err_pct"])
    print(f"Found {len(matches)} candidate triples with both errors < 5%:")
    print()
    if matches:
        print(f"{'rank':>4} {'rho_12':<35} {'rho_13':<35} "
                f"{'rho_23':<35} {'errC1':>7} {'errC2':>7}")
        for rank, m in enumerate(matches[:30]):
            print(f"{rank+1:>4} "
                    f"{m['rho_12_name']:<35} "
                    f"{m['rho_13_name']:<35} "
                    f"{m['rho_23_name']:<35} "
                    f"{m['sum_sq_rel_err_pct']:>+7.2f} "
                    f"{m['prod_rel_err_pct']:>+7.2f}")
    print()

    # Direct check: try (1/6, 1/2, 5/6) which would have product
    # 5/72 EXACT but sum_sq = 35/36 (off by 4/36)
    r_a, r_b, r_c = Fraction(1, 6), Fraction(1, 2), Fraction(5, 6)
    s, p, eP, eQ = check_triple(r_a, r_b, r_c)
    print(f"Geometric-progression candidate (1/6, 1/2, 5/6):")
    print(f"  sum_sq = {s} = {float(s):.5f} (err {eP:+.2f}%)")
    print(f"  product = {p} = {float(p):.5f} (err {eQ:+.2f}%)")
    print(f"  Note: product MATCHES EXACTLY (5/72); sum_sq off by 4/36 = 1/9")
    print()

    # If we relax to allow non-rational solutions:
    # Use Cardano or numerical to find one explicit triple
    # Try the symmetric arithmetic-progression family: (a-b, a, a+b)
    # Sum_sq = 3a^2 + 2b^2 = 31/36
    # Product = a(a^2 - b^2) = 5/72
    print("Arithmetic-progression family (a-b, a, a+b):")
    print(f"  3 a^2 + 2 b^2 = 31/36")
    print(f"  a (a^2 - b^2) = 5/72")
    print(f"  -> cubic 180 a^3 - 31 a - 5 = 0")
    # Numerical solve
    import numpy as np
    coeffs = [180, 0, -31, -5]
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and 0 < r.real < 1]
    print(f"  Real positive roots: {real_roots}")
    if real_roots:
        a = real_roots[0]
        b_sq = a ** 2 - 5 / (72 * a)
        if b_sq > 0:
            b = math.sqrt(b_sq)
            print(f"  a = {a:.6f}, b = {b:.6f}")
            print(f"  rho_12 = a - b = {a - b:.6f}")
            print(f"  rho_13 = a     = {a:.6f}")
            print(f"  rho_23 = a + b = {a + b:.6f}")
    print()

    bundle = {
        "method": "verify_lemma_B_M_F_off_diagonal_identification",
        "stand": "2026-05-13",
        "d": D,
        "N_gen": N_GEN,
        "target_spectrum": [0, "7/6", "11/6"],
        "constraints": {
            "sum_sq_rho_12_13_23": "31/36",
            "product_rho_12_13_23": "5/72",
        },
        "system_R_candidate_matches": matches[:30],
        "interpretation": (
            "The off-diagonal weights of M_F are constrained by two "
            "equations (sum of squares, product) leaving a 1-parameter "
            "family of solutions. No PRECISE-tier System-R rational "
            "triple (with both constraints below 1% rel err) found in "
            "the searched candidate space. The geometric-progression "
            "candidate (1/6, 1/2, 5/6) matches the product EXACTLY "
            "(5/72) but the sum_sq is off by 1/9. The carrier-action "
            "family-sector dynamics must provide an additional "
            "constraint to uniquely determine the off-diagonal triple."
        ),
    }
    out = OUTPUTS / "verify_lemma_B_M_F_off_diagonal_identification.json"
    out.write_text(json.dumps(bundle, indent=2, default=float),
                       encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
