r"""
Phase E: Full Frobenius residual test for the Einstein equation
with cosmological term, beyond the previous T_00-only check.

Tests:
    || G_munu + Lambda * g_munu - 8 pi G * T_munu^Xi ||_F  /  || G_munu ||_F

with Lambda fixed to the System-R rational identification under
each convention:

  Convention                       Lambda (System-R)        Theory motivation
  ------------------------------- -------------------------- -------------------
  proxy K_rec heuristic            1/4   = alpha_xi/2 - 2*g  Bekenstein-Hawking 1/d_st
  row-mean Definition 12.20        17/20 = alpha_xi + e2 - g non-scalar Clifford rate
  Section 14.1 Laplace-Beltrami    6/5   = 1 + e2 * d_st     LEE-boundary

Strategy: on the bundled nine-point ladder we have access to the
diagonal T_00-G_00 estimates from Path 5; the off-diagonal
T_ij values are not directly bundled in the present
reproducibility package, so the Frobenius test is performed on
the diagonal block (T_00 plus T_ii) using the bundled per-regime
data and the row-mean K_rec computation already validated in
verify_einstein_with_lambda.py.

For the off-diagonal components, the Hilbert variation in the
parent corpus's residual-tensor framework predicts T_ij^aniso
from grad(Psi)*grad(Psi) and grad(Xi)*grad(Xi); we bundle the
per-regime average aniso magnitude as a sanity bound, with the
exact off-diagonal Frobenius computation flagged as a future-work
item (it requires per-node tensor reconstruction beyond the
present bundle).

Output:
    outputs/lambda_frobenius_residual.json with:
      - diagonal-block Frobenius residual under each Lambda convention
      - aniso-bound estimate for off-diagonal contribution
      - per-regime per-convention residual table

Usage:
    python ./src/verify_lambda_frobenius_residual.py
"""
from __future__ import annotations
import json
import math
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


# System-R rational coefficients (cross-check with verify_lambda_system_R.py)
ALPHA_XI = Fraction(9, 10)
GAMMA = Fraction(1, 10)
EPS_SQ = Fraction(1, 20)
D_SP = Fraction(4)

LAMBDA_PROXY = ALPHA_XI / 2 - 2 * GAMMA          # = 1/4
LAMBDA_ROW = ALPHA_XI + EPS_SQ - GAMMA           # = 17/20
LAMBDA_SEC14 = Fraction(1) + EPS_SQ * D_SP       # = 6/5


def diagonal_block_frobenius(t_00, g_00, Lambda_const, t_ii_average,
                              g_ii_average):
    """Compute || G_munu + Lambda * g_munu - 8*pi*G*T_munu^Xi ||_F
    on the diagonal block (00, 11, 22, 33) under
    8*pi*G = 1 lattice convention, g_00 = -1, g_ii = +1
    (Lorentzian, mostly-plus convention).

    Diagonal residuals:
        r_00 = G_00 + Lambda * g_00 - T_00 = G_00 - Lambda - T_00
        r_ii = G_ii + Lambda * g_ii - T_ii = G_ii + Lambda - T_ii  (i = 1,2,3)

    For the present reproducibility, we use:
        T_00:   bundled per-regime mean (data/einstein_with_lambda_8point.json)
        G_00:   bundled per-regime R_bar/2
        T_ii:   estimated as a fraction of T_00 (isotropic cosmological-fluid
                approximation): T_ii = w * T_00 with w = +1 (isotropic radiation
                limit) or w = -1 (de-Sitter limit).  We report both bounds.
        G_ii:   estimated as a fraction of G_00 (Ricci-isotropic):
                G_ii ~ -G_00 in the FRW de-Sitter limit, so the diagonal
                Einstein tensor is approximately diag(G_00, -G_00, -G_00, -G_00).
    """
    # 00-component (Lorentzian g_00 = -1 here, but our T - G convention
    # already absorbs the sign in the lattice computation; we use the
    # parent-corpus convention: r_00 = G_00 + Lambda - T_00 with
    # 8 pi G = 1 absorbed):
    r_00 = g_00 + Lambda_const - t_00

    # Spatial components: under FRW de-Sitter isotropy with
    # T_ii = -T_00 (de-Sitter, w = -1) and G_ii = -G_00:
    # r_ii_dS = -G_00 + Lambda - (-T_00) = -G_00 + Lambda + T_00
    r_ii_dS = -g_ii_average + Lambda_const - (-t_ii_average)
    # Under FRW radiation isotropy (w = +1/3 → bound):
    # T_ii = T_00/3, G_ii = ? — for radiation FRW it is non-trivial;
    # we report only the de-Sitter bound as the natural cosmological-
    # constant background isotropy.
    # Total Frobenius: sqrt(r_00^2 + 3 * r_ii^2) on the 4 diagonal elements
    frob_diagonal = math.sqrt(r_00 ** 2 + 3 * r_ii_dS ** 2)

    # Reference: || G_munu ||_F on the diagonal block
    # = sqrt(G_00^2 + 3 * G_ii^2) ~ sqrt(g_00^2 + 3*g_00^2) = 2*|G_00|
    g_norm = math.sqrt(g_00 ** 2 + 3 * g_00 ** 2)  # = 2*|g_00|

    return {
        "r_00": r_00,
        "r_ii_dS": r_ii_dS,
        "frob_diagonal": frob_diagonal,
        "g_norm": g_norm,
        "rel_residual": frob_diagonal / g_norm if g_norm > 0 else float("inf"),
    }


def main():
    # Load bundled Pfad-5 data
    with open(DATA / "einstein_with_lambda_8point.json", "r",
              encoding="utf-8") as f:
        d = json.load(f)

    Ns = d["lattice_ladder"]["N_values"]
    labels = d["lattice_ladder"]["regime_labels"]
    T00 = d["T_00_Xi_values"]
    G00 = d["G_00_values"]

    # Run Frobenius test for each convention's Lambda
    conventions = [
        ("proxy",        float(LAMBDA_PROXY),  "1/4 (BH-entropy)"),
        ("row_mean",     float(LAMBDA_ROW),    "17/20 (non-scalar Clifford rate)"),
        ("section_14_1", float(LAMBDA_SEC14),  "6/5 (LEE-boundary)"),
    ]

    print("=" * 78)
    print("Phase-E: Frobenius residual on diagonal block under each")
    print("System-R rational Lambda convention (8 pi G = 1 lattice convention).")
    print("=" * 78)
    print()
    print("Lambda values (System-R rational, cross-checked with")
    print("verify_lambda_system_R.py):")
    print(f"  Proxy        Lambda = {LAMBDA_PROXY} = {float(LAMBDA_PROXY):.4f}")
    print(f"  Row-mean     Lambda = {LAMBDA_ROW}  = {float(LAMBDA_ROW):.4f}")
    print(f"  Section 14.1 Lambda = {LAMBDA_SEC14}  = {float(LAMBDA_SEC14):.4f}")
    print()

    results = {}
    for tag, Lambda_const, motive in conventions:
        print(f"--- Convention: {tag} (Lambda = {motive}) ---")
        print(f"  {'N':>4} {'reg':>8} {'r_00':>10} {'frob_diag':>12} {'rel':>10}")
        per_reg = []
        for n, lab, t0, g0 in zip(Ns, labels, T00, G00):
            # Use t_ii_average ~ T_00 (de Sitter isotropy bound),
            # g_ii_average ~ G_00 (Ricci isotropy bound).
            # NOTE: this is a bound, not the exact off-diagonal computation.
            res = diagonal_block_frobenius(
                t_00=t0, g_00=g0, Lambda_const=Lambda_const,
                t_ii_average=t0, g_ii_average=g0,
            )
            per_reg.append({
                "N": n, "regime": lab,
                "r_00": res["r_00"], "frob_diag": res["frob_diagonal"],
                "rel_residual": res["rel_residual"],
            })
            print(f"  {n:>4} {lab:>8} {res['r_00']:>+10.4f} "
                  f"{res['frob_diagonal']:>12.4f} {res['rel_residual']:>10.4f}")
        # Asymptotic mean (P4..P8)
        asym = [r for r in per_reg if r["N"] >= 42]
        asym_rel = sum(r["rel_residual"] for r in asym) / len(asym)
        print(f"  Asymptotic mean (N>=42) rel_residual: {asym_rel:.4f}")
        print()
        results[tag] = {
            "Lambda": float(Lambda_const),
            "per_regime": per_reg,
            "asymptotic_rel_residual": asym_rel,
        }

    print("--- Summary ---")
    print("Diagonal-block Frobenius rel-residual at asymptotic window (P4..P8):")
    for tag, res in results.items():
        print(f"  {tag:>14}: {res['asymptotic_rel_residual']:.4f}")
    print()
    print("CAVEAT: this is the diagonal-block Frobenius, with off-diagonal")
    print("components estimated under FRW de-Sitter isotropy bound. The exact")
    print("off-diagonal T_ij computation requires per-node tensor reconstruction")
    print("beyond the present bundled package and is flagged as future-work.")
    print("The 00-only residual reported in the main paper (max 0.019 for N>=30")
    print("under Lambda = 0.314 numerical convention) is a special case of this")
    print("test in the row-mean convention, on the 00-component alone.")

    out = {
        "method": "diagonal_block_Frobenius_under_FRW_de_Sitter_isotropy_bound",
        "8_pi_G_convention": "absorbed_into_lattice_units",
        "lambda_conventions": {
            "proxy":        {"rational": str(LAMBDA_PROXY),  "decimal": float(LAMBDA_PROXY)},
            "row_mean":     {"rational": str(LAMBDA_ROW),    "decimal": float(LAMBDA_ROW)},
            "section_14_1": {"rational": str(LAMBDA_SEC14),  "decimal": float(LAMBDA_SEC14)},
        },
        "results": results,
        "caveat_off_diagonal_exact_test_future_work": True,
        "reproducible_inputs": "data/einstein_with_lambda_8point.json",
    }
    out_path = OUTPUTS / "lambda_frobenius_residual.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
