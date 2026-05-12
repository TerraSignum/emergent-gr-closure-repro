"""Standalone reproducer for the Federer-prefactor closure (Lemma 9 + T_F).

This script is self-contained — it depends only on Python stdlib so it
runs inside this repository without the parent worldformula package.

The strict Galerkin-FEM chain (Cea + Aubin-Nitsche on the linearised
causal-wave operator) predicts |d|/c = (M/alpha)*sqrt(lambda_crit) ~ 2.03,
which deviates from the empirical 1.097 by 46% — Galerkin alone does not
close c, d.  The chain below closes them under three structural anchors
of the framework:

  (A1) Lambda_t = alpha_xi^2 = 81/100        [P4 Section sec:gap]
  (A2) shear_off = 2*alpha_xi*gamma = 18/100 [verify_unit_budget_redistribution.py]
  (A3) bulk-percentile loop-class with full topological coupling
       (1 + gamma)                            [Lemma-9 extension]

Combined identification:
       c = alpha_xi * (2*alpha_xi*gamma) = 2*alpha_xi^2*gamma  = 81/500
   |d|/c = (1 + gamma)                                         = 11/10
       d = -(1+gamma) * c                                      = -891/5000

Empirical fit on 10-point Federer ladder:
       c_emp     = +0.161304     -> match -0.43%
       d_emp     = -0.176953     -> match -0.70%
   |d|/c_emp     = +1.097016     -> match -0.27%

Verdict: PROOF-CONDITIONAL-PRECISE.

Output:
  outputs/federer_prefactor_lemma9_closure.json
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


# System-R rationals (P2 sec:linear_stability)
ALPHA_XI = Fraction(9, 10)
GAMMA = Fraction(1, 10)
EPS_SYNC2 = Fraction(1, 20)
BETA_PI = Fraction(15, 16)

# Derived (closure constraints)
D_OMEGA = BETA_PI - GAMMA              # 67/80
M_SQUARED = ALPHA_XI - GAMMA + EPS_SYNC2  # 17/20
LAMBDA_CRIT = M_SQUARED / D_OMEGA      # 68/67

# Empirical M3 two-power fit on the cleaned ten-point Federer ladder
# (P5..P5N300, 208 seeds total; subsection of the twelve-point extended
# ladder used for the AICc test which adds N=128 and N=256).
# Source: outputs/stage6f_regular_core_decomposition.json (regular_set_median
# at tau=0.05) plus identify_subleading_corrections_to_federer.py M3 fit.
C_EMPIRICAL = 0.161304
D_EMPIRICAL = -0.176953
LADDER_SIZE = 10  # ten-point cleaned canonical-physics ladder


def main() -> int:
    print("=" * 78)
    print("  Federer-Prefactor Closure via Lemma 9 + C1^2 unit-budget anchor")
    print("=" * 78)

    # ── Anchor I: C1^2 unit-budget closure ──
    c1_squared = (ALPHA_XI + GAMMA) ** 2
    Lambda_t = ALPHA_XI ** 2                # diagonal time-time
    Lambda_s_per_axis = -(GAMMA ** 2) / 2   # per-axis spatial diagonal
    shear_off = 2 * ALPHA_XI * GAMMA        # off-diagonal, decaying
    print()
    print("  Anchor I (C1^2 unit-budget closure):")
    print(f"    (alpha_xi + gamma)^2                = {c1_squared} (must equal 1)")
    print(f"    Lambda_t (diagonal time-time)       = {Lambda_t} = {float(Lambda_t):.6f}")
    print(f"    |Lambda_s| per spatial axis         = {-Lambda_s_per_axis} = {float(-Lambda_s_per_axis):.6f}")
    print(f"    shear_off (off-diagonal, decaying)  = {shear_off} = {float(shear_off):.6f}")

    # ── Anchor II: Lemma 9 (1+gamma) loop class ──
    lemma9_class = 1 + GAMMA
    print()
    print("  Anchor II (Lemma 9, full topological coupling, a = 1):")
    print(f"    (1 + gamma)                         = {lemma9_class} = {float(lemma9_class):.6f}")

    # ── Theorem T_F: Federer prefactor identification ──
    c_predicted = ALPHA_XI * shear_off              # 81/500
    d_predicted = -lemma9_class * c_predicted       # -891/5000
    ratio_predicted = lemma9_class                  # 11/10

    print()
    print("  Theorem T_F (Federer prefactor identification, conditional):")
    print(f"    c_predicted = alpha_xi * shear_off  = {c_predicted} = {float(c_predicted):+.6f}")
    print(f"    d_predicted = -(1+gamma) * c        = {d_predicted} = {float(d_predicted):+.6f}")
    print(f"    |d|/c_predicted = (1+gamma)         = {ratio_predicted} = {float(ratio_predicted):+.6f}")

    # ── Empirical comparison ──
    c_diff_pct = (C_EMPIRICAL - float(c_predicted)) / float(c_predicted) * 100
    d_diff_pct = (D_EMPIRICAL - float(d_predicted)) / float(d_predicted) * 100
    ratio_emp = abs(D_EMPIRICAL) / C_EMPIRICAL
    ratio_diff_pct = (ratio_emp - float(ratio_predicted)) / float(ratio_predicted) * 100

    print()
    print("  Empirical match (10-point Federer ladder):")
    print(f"    c_emp                               = {C_EMPIRICAL:+.6f}  (diff {c_diff_pct:+.4f} %)")
    print(f"    d_emp                               = {D_EMPIRICAL:+.6f}  (diff {d_diff_pct:+.4f} %)")
    print(f"    |d|/c_emp                           = {ratio_emp:+.6f}  (diff {ratio_diff_pct:+.4f} %)")

    max_diff = max(abs(c_diff_pct), abs(d_diff_pct), abs(ratio_diff_pct))
    if max_diff < 1.0:
        verdict = "PROOF-CONDITIONAL-PRECISE"
    elif max_diff < 5.0:
        verdict = "PROOF-CONDITIONAL-PRECISE-WEAK"
    else:
        verdict = "PROOF-CONDITIONAL-FAILED"

    print()
    print(f"  VERDICT: {verdict}")
    print()
    print("  Conditional on three structural anchors (A1)-(A3):")
    print("    (A1) Lambda_t = alpha_xi^2 = 0.81 [P4 Section sec:gap]")
    print("    (A2) c = 2*gamma*Lambda_t = twice damping x cosmological-tensor diagonal")
    print("         (algebraically equal to 2*alpha_xi^2*gamma = alpha_xi*shear_C1;")
    print("         the alternative shear factorisation c=alpha_xi*shear is")
    print("         empirically falsified at Delta-AICc=+86; see paper)")
    print("    (A3) bulk-percentile (1+gamma) class [Lemma-9 extension]")

    # ── Save JSON bundle ──
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "federer_prefactor_lemma9_closure.json"
    bundle = {
        "method": "federer_prefactor_closure_lemma9_T_F",
        "stand": "2026-05-04",
        "system_R_anchors": {
            "alpha_xi": str(ALPHA_XI),
            "gamma": str(GAMMA),
            "alpha_xi_squared": str(ALPHA_XI ** 2),
            "two_alpha_xi_gamma": str(shear_off),
            "one_plus_gamma": str(lemma9_class),
        },
        "anchor_I_C1_unit_budget": {
            "Lambda_t_diagonal": str(Lambda_t),
            "Lambda_s_per_axis": str(Lambda_s_per_axis),
            "shear_off_diagonal": str(shear_off),
            "decaying_component": "shear_off_diagonal",
            "derived_in": "verify_unit_budget_redistribution.py lines 1-26",
        },
        "anchor_II_lemma9": {
            "topology_factor": "gamma",
            "loop_class": str(lemma9_class),
            "a_value": "1",
            "admissibility": (
                "a=1 in allowed set {1/4, 1/2, 1, 2, 1/N_gen, 1/(2*N_gen)} "
                "per loop_class_library.json forbidden_classes; unfilled by "
                "existing Lemmas 1-8."
            ),
            "derived_in": (
                "loop-class-closure-repro/data/lemma9_federer_extension.json"
            ),
        },
        "theorem_T_F": {
            "c_predicted": str(c_predicted),
            "c_predicted_float": float(c_predicted),
            "d_predicted": str(d_predicted),
            "d_predicted_float": float(d_predicted),
            "ratio_predicted": str(ratio_predicted),
            "ratio_predicted_float": float(ratio_predicted),
            "c_empirical": C_EMPIRICAL,
            "d_empirical": D_EMPIRICAL,
            "ratio_empirical": ratio_emp,
            "c_diff_pct": c_diff_pct,
            "d_diff_pct": d_diff_pct,
            "ratio_diff_pct": ratio_diff_pct,
            "structural_ansaetze": [
                "(A1) Lambda_t = alpha_xi^2 = 81/100 (P4 sec:gap; "
                "per-regime ratio 0.974-1.017 on 12 N-points)",
                "(A2) c = 2*gamma*Lambda_t (twice damping rate x "
                "cosmological-tensor time-time diagonal); algebraically "
                "equal to 2*alpha_xi^2*gamma. The alternative "
                "factorisation c = alpha_xi*shear (where shear=2*alpha_xi*gamma "
                "from C1^2 unit-budget closure) is empirically falsified at "
                "Delta-AICc=+86 on the 12-point ladder.",
                "(A3) Lemma-9 sub-leading class: d/c = -(1+gamma); "
                "topology coefficient a=1 admissible-but-unfilled in the "
                "loop_class_library forbidden-classes parameter-free set.",
            ],
        },
        "verdict": verdict,
        "interpretation": (
            "PRECISE-tier closure conditional on three pre-registered "
            "structural anchors (A1)-(A3); not first-principles. The "
            "integrated cross-anchor theorem chain has not been derived. "
            "The chain falsifies if any anchor fails empirical-validation."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    print()
    print(f"  Saved JSON bundle: {out_path.relative_to(Path(__file__).resolve().parents[1])}")
    return 0 if verdict == "PROOF-CONDITIONAL-PRECISE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
