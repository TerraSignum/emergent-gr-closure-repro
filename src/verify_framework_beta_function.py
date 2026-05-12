"""Framework-internal beta-function: 1-loop and 2-loop coefficients
from the framework's own mode-counting (no SM inputs).

The 1-loop beta function in any non-Abelian gauge theory with
n_f Dirac fermions and n_s scalars is

    beta_1loop(g) = -b0 * g^3 / (16 pi^2)
    b0 = (11/3) N_gauge - (2/3) n_f - (1/3) n_s.

The 2-loop coefficient (MS-bar scheme):
    b1 = (34/3) N_gauge^2 - (10/3) N_gauge n_f
         - 2 C_F n_f
    C_F = (N_gauge^2 - 1) / (2 N_gauge)

This script substitutes the framework's mode-counting values:
  N_gauge = 2  (SU(2)-like emergent gauge structure from
                phase + causal mode pair)
  n_f     = 2  (correlation-stability fermion seeds)
  n_s     = 2  (structure scalars)

and reports b0, b1, asymptotic-freedom verdict, and the 2-loop
running-coupling prediction.

The 2-loop integration takes alpha_eff(M_Pl) -> alpha_eff(v_EW)
using the framework's own b0/b1 (no SM constants).

Writes:
  data/framework_beta_function.json
"""
from __future__ import annotations
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Framework mode-counting (from Feldtheorie §9.4 + spectral
# Yukawa-eigenvalue construction)
N_GAUGE = 2          # SU(N)-like (phase + causal modes)
N_FERMION = 2        # correlation-stability seeds
N_SCALAR = 2         # structure scalars
N_GEN = 3            # generation count (integer-uniqueness fixed)

# QFE-02 lattice readouts (per regime)
ALPHA_EFF_P1 = 0.422884
ALPHA_EFF_P2P = 0.202952

# Scales (this is the only externally fixed input: M_Pl/v_EW ratio,
# both natural-unit scales for the framework)
M_PL_GEV = 1.22090e19
V_EW_GEV = 246.22


def b0(n_gauge: int, n_f: int, n_s: int) -> float:
    """1-loop beta-function coefficient."""
    return (11.0 / 3.0) * n_gauge - (2.0 / 3.0) * n_f - (1.0 / 3.0) * n_s


def b1(n_gauge: int, n_f: int) -> float:
    """2-loop beta-function coefficient (MS-bar scheme)."""
    C_F = (n_gauge ** 2 - 1) / (2.0 * n_gauge)
    return (34.0 / 3.0) * n_gauge ** 2 - (10.0 / 3.0) * n_gauge * n_f \
        - 2.0 * C_F * n_f


def two_loop_run(alpha_uv: float, b0_v: float, b1_v: float,
                 log_ratio: float) -> float:
    """alpha_IR via 2-loop integration of beta(g)."""
    # 1-loop: alpha(mu) = alpha_UV / (1 + (b0 alpha_UV / 2pi) ln(mu/UV))
    # 2-loop multiplicative correction:
    # additional factor (1 + (b1/b0) (alpha_UV - alpha_IR) / (4pi))
    inv_one_loop = 1.0 / alpha_uv + (b0_v * log_ratio) / (2.0 * math.pi)
    alpha_IR_1l = 1.0 / inv_one_loop
    correction = 1.0 + (b1_v / b0_v) * (alpha_uv - alpha_IR_1l) \
        / (4.0 * math.pi)
    return alpha_IR_1l * correction


def main() -> int:
    print("=" * 78)
    print("Framework-internal 1-loop and 2-loop beta-function coefficients")
    print("=" * 78)
    print()
    print(f"Mode counting (Feldtheorie §9.4):")
    print(f"  N_gauge   = {N_GAUGE} (phase + causal modes)")
    print(f"  n_fermion = {N_FERMION} (correlation-stability seeds)")
    print(f"  n_scalar  = {N_SCALAR} (structure scalars)")
    print(f"  N_gen     = {N_GEN} (integer-uniqueness fixed)")
    print()

    b0_v = b0(N_GAUGE, N_FERMION, N_SCALAR)
    b1_v = b1(N_GAUGE, N_FERMION)
    print(f"1-loop coefficient: b0 = (11/3)*{N_GAUGE} - (2/3)*{N_FERMION}"
          f" - (1/3)*{N_SCALAR}")
    print(f"                       = {11/3*N_GAUGE:.4f} - "
          f"{2/3*N_FERMION:.4f} - {1/3*N_SCALAR:.4f}")
    print(f"                       = {b0_v:.4f}")
    print(f"  asymptotic_freedom     = {b0_v > 0}")
    print()
    C_F = (N_GAUGE ** 2 - 1) / (2.0 * N_GAUGE)
    print(f"2-loop coefficient: b1 = (34/3)*N^2 - (10/3)*N*n_f - 2*C_F*n_f")
    print(f"  C_F = (N^2-1)/(2N) = {C_F:.4f}")
    print(f"  b1 = (34/3)*{N_GAUGE**2} - (10/3)*{N_GAUGE}*{N_FERMION} "
          f"- 2*{C_F:.4f}*{N_FERMION}")
    print(f"     = {34/3*N_GAUGE**2:.4f} - {10/3*N_GAUGE*N_FERMION:.4f} "
          f"- {2*C_F*N_FERMION:.4f}")
    print(f"     = {b1_v:.4f}")
    print()

    log_ratio = math.log(M_PL_GEV / V_EW_GEV)
    print(f"Running M_Pl -> v_EW (log_ratio = ln({M_PL_GEV:.2e}/"
          f"{V_EW_GEV}) = {log_ratio:.3f}):")
    for label, alpha_UV in [("P1", ALPHA_EFF_P1),
                            ("P2'", ALPHA_EFF_P2P)]:
        # 1-loop only
        alpha_1l = 1.0 / (1.0 / alpha_UV + b0_v * log_ratio / (2 * math.pi))
        # 2-loop (framework b0+b1)
        alpha_2l = two_loop_run(alpha_UV, b0_v, b1_v, log_ratio)
        print(f"  {label}: alpha_eff(M_Pl) = {alpha_UV:.6f}")
        print(f"    1-loop: alpha_eff(v_EW) = {alpha_1l:.6f}")
        print(f"    2-loop: alpha_eff(v_EW) = {alpha_2l:.6f}")
        print(f"    2-loop correction       = "
              f"{(alpha_2l - alpha_1l) / alpha_1l * 100:+.2f}%")

    print()
    print("Comparison to SM-QCD 1-loop coefficient (b0_QCD with n_f=5):")
    b0_qcd = (11.0 - 2.0 * 5 / 3.0)
    print(f"  b0_QCD = 11 - (2/3)*5 = {b0_qcd:.4f}")
    print(f"  Framework b0 / b0_QCD = {b0_v / b0_qcd:.4f}")
    print()
    print("Reading: the framework's b0 = 5.333 sits between b0_QCD = 7.667")
    print("(positive asymptotic freedom) and the QED Landau-pole regime")
    print("(b0 < 0). Both b0 > 0 and b1 > 0 indicate a doubly")
    print("asymptotic-free emergent gauge structure under the framework's")
    print("own mode counting -- no Standard-Model coupling constants used.")

    bundle = {
        "method": "framework_internal_beta_function",
        "mode_counting": {
            "N_gauge": N_GAUGE, "n_fermion": N_FERMION,
            "n_scalar": N_SCALAR, "N_gen": N_GEN,
        },
        "coefficients": {
            "b0_one_loop": b0_v,
            "b1_two_loop": b1_v,
            "C_F": C_F,
            "asymptotic_freedom_one_loop": b0_v > 0,
            "asymptotic_freedom_two_loop": b1_v > 0,
        },
        "running_predictions": {
            "log_ratio_M_Pl_v_EW": log_ratio,
            "P1": {
                "alpha_UV": ALPHA_EFF_P1,
                "alpha_IR_1loop":
                    1.0 / (1.0 / ALPHA_EFF_P1 + b0_v * log_ratio / (2 * math.pi)),
                "alpha_IR_2loop": two_loop_run(ALPHA_EFF_P1, b0_v, b1_v,
                                               log_ratio),
            },
            "P2prime": {
                "alpha_UV": ALPHA_EFF_P2P,
                "alpha_IR_1loop":
                    1.0 / (1.0 / ALPHA_EFF_P2P + b0_v * log_ratio / (2 * math.pi)),
                "alpha_IR_2loop": two_loop_run(ALPHA_EFF_P2P, b0_v, b1_v,
                                               log_ratio),
            },
        },
        "comparison_to_SM": {
            "b0_QCD_n_f_5": b0_qcd,
            "framework_b0_over_QCD": b0_v / b0_qcd,
            "framework_uses_SM_coupling_input": False,
            "note": (
                "The framework's b0/b1 are computed from its own mode "
                "counting (N_gauge=2, n_f=2, n_s=2). The SM b0_QCD with "
                "n_f=5 is shown only for reference. The Wilson-area-law "
                "cross-check in the GAU module uses SM alpha_s(M_Z) as "
                "input separately; this beta-function is independent of "
                "that consistency check."
            ),
        },
    }
    out = REPO / "data" / "framework_beta_function.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
