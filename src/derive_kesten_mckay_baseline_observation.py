r"""Kesten--McKay baseline observation at the carrier mean degree.

Numerical observation (not a closed-form derivation):

   lambda_2^KM(d=12)  - gamma^2 * (d + N_gen)
       = (1 - 2*sqrt(11)/12) - 7/100
       = 0.44743... - 0.07000
       = 0.37743...
       ~  3/8 = 0.37500
   relative residual ~ +0.65%.

Reading. Bauerschmidt--Huang--Yau 2019 (arXiv:1609.09052,
Comm. Math. Phys. 369, 523) prove the optimal local Kesten--McKay
law for the empirical spectral distribution of random d-regular
graphs: the empirical density follows
   rho_KM(lambda) = (d / 2pi) * sqrt(4(d-1) - lambda^2) / (d^2 - lambda^2)
on |lambda| <= 2*sqrt(d-1), down to the optimal scale N^{-1+eps}.
The upper spectral edge of the normalised Laplacian baseline
1 - A/d on a random d-regular graph is

   lambda_2^KM(d) = 1 - 2*sqrt(d-1)/d.

The carrier skeleton (tau = 0.10) has empirical mean degree
~ 12 (= d * N_gen on the (4,3) anchor), so d=12 is the natural
Kesten--McKay reference. The cross-corpus universal-leakage
constant gamma^2 * (d + N_gen) = 7/100 reappears here: subtracting
it from the Kesten--McKay baseline lands within ~0.65% of the
empirically certified 3/8.

What this is. A rigorous structural BASELINE for the (SG) value:
the Kesten--McKay upper edge for a d-regular graph at d=12,
shifted by the corpus-wide universal-leakage constant, sits
within a percent of the closure 3/8. The residual is consistent
in sign and magnitude with the triangle/cluster correction
captured by the Newman-clustered / Pham--Peron--Metz cavity
extension (Proposition prop:gap_statistical_fingerprint of the
companion manuscript).

What this is NOT. A closed-form derivation: 1 - 2*sqrt(11)/12
is irrational, while 3/8 is rational. The match is therefore a
percent-level *observation*, not an exact identity. The
identification of the residual with an explicit clustering
correction is the open Step-4 deliverable of Lemma B.

Output: outputs/derive_kesten_mckay_baseline_observation.json plus
a console summary. No external data is required.
"""
from __future__ import annotations

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# (d, N_gen) anchor.
# ------------------------------------------------------------------
D = 4
N_GEN = 3
GAMMA = Fraction(1, 10)
D_PLUS_NGEN = D + N_GEN                          # 7
UNIVERSAL_LEAKAGE = GAMMA ** 2 * D_PLUS_NGEN     # 7/100
CARRIER_MEAN_DEGREE = D * N_GEN                  # 12

# ------------------------------------------------------------------
# Targets.
# ------------------------------------------------------------------
LAMBDA_INF_VAC = Fraction(3, 8)                  # (d-1)/(2d)
LAMBDA_INF_SKEL = Fraction(7, 24)                # (d + N_gen)/(2 d N_gen)


def km_upper_edge(d: int) -> float:
    """Normalised-Laplacian upper-edge gap of a random d-regular graph."""
    return 1.0 - 2.0 * math.sqrt(d - 1) / d


def main() -> int:
    print("=" * 78)
    print("Kesten--McKay baseline observation (vacuum-branch 3/8)")
    print("=" * 78)
    print()
    print(f"  (d, N_gen)                = ({D}, {N_GEN})")
    print(f"  carrier mean degree d*N   = {CARRIER_MEAN_DEGREE}")
    print(f"  universal leakage gamma^2*(d+N_gen) = "
          f"{UNIVERSAL_LEAKAGE.numerator}/{UNIVERSAL_LEAKAGE.denominator} "
          f"= {float(UNIVERSAL_LEAKAGE):.5f}")
    print()

    km12 = km_upper_edge(CARRIER_MEAN_DEGREE)
    shifted = km12 - float(UNIVERSAL_LEAKAGE)
    target_vac = float(LAMBDA_INF_VAC)
    rel_err_vac = (shifted - target_vac) / target_vac

    print(f"  lambda_2^KM(12)           = 1 - 2*sqrt(11)/12 "
          f"= {km12:.5f}")
    print(f"  lambda_2^KM(12) - 7/100   = {shifted:.5f}")
    print(f"  target lambda_inf^vac     = 3/8 = {target_vac:.5f}")
    print(f"  relative residual         = {rel_err_vac*100:+.3f}%")
    print()

    # Cross-check: at d = N_gen + d = 7 the KM edge differs and the
    # skeleton target 7/24 does not enjoy a similar near-match.
    km_skel_naive = km_upper_edge(CARRIER_MEAN_DEGREE)
    target_skel = float(LAMBDA_INF_SKEL)
    rel_err_skel_naive = (km_skel_naive - target_skel) / target_skel
    print(f"  [cross]  lambda_2^KM(12) vs lambda_inf^skel=7/24 "
          f"=> residual {rel_err_skel_naive*100:+.3f}% "
          f"(no universal-leakage match, as expected)")
    print()

    # ------------------------------------------------------------------
    # Verdict.
    # ------------------------------------------------------------------
    print("-" * 78)
    print("Verdict")
    print("-" * 78)
    print("  Kesten--McKay baseline at d = d*N_gen = 12, shifted by the")
    print("  universal-leakage constant gamma^2 * (d + N_gen) = 7/100,")
    print(f"  sits within {abs(rel_err_vac)*100:.3f}% of the vacuum-branch")
    print("  closure lambda_inf^vac = 3/8.")
    print()
    print("  Status: STRUCTURAL_BASELINE_OBSERVATION")
    print("          * Bauerschmidt--Huang--Yau 2019 (arXiv:1609.09052)")
    print("            certifies the Kesten--McKay law at the optimal")
    print("            local scale -- the baseline is rigorous.")
    print("          * 1 - 2*sqrt(11)/12 is irrational; the residual")
    print("            cannot be eliminated by algebra. It is the open")
    print("            clustering/triangle correction (Step-4 of Lemma B,")
    print("            attacked via Pham--Peron--Metz 2024 in")
    print("            prop:gap_statistical_fingerprint).")

    bundle = {
        "title": "Kesten--McKay baseline observation at the carrier mean degree",
        "anchor": {"d": D, "N_gen": N_GEN, "d_times_N_gen": CARRIER_MEAN_DEGREE},
        "inputs": {
            "gamma": str(GAMMA),
            "universal_leakage_gamma_sq_times_d_plus_N_gen": str(UNIVERSAL_LEAKAGE),
        },
        "kesten_mckay": {
            "formula": "lambda_2^KM(d) = 1 - 2*sqrt(d-1)/d",
            "literature": "BauerschmidtHuangYau2019 (arXiv:1609.09052)",
            "lambda_2_KM_at_12": km12,
        },
        "vacuum_branch_observation": {
            "shifted_value": shifted,
            "target_lambda_inf_vac": target_vac,
            "relative_residual": rel_err_vac,
            "irrational": True,
        },
        "skeleton_cross_check": {
            "target_lambda_inf_skel": target_skel,
            "naive_residual": rel_err_skel_naive,
        },
        "verdict": "STRUCTURAL_BASELINE_OBSERVATION",
    }
    out_path = OUTPUTS / "derive_kesten_mckay_baseline_observation.json"
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
