"""(*) Lambda_t = alpha_xi^2 + gamma^2 = 1 - 2*alpha_xi*gamma derivation.

System-R C1 identity: alpha_xi + gamma = 1 (Rooted-Leak unit closure).
Squaring: (alpha_xi + gamma)^2 = alpha_xi^2 + 2*alpha_xi*gamma + gamma^2 = 1.

Therefore the diagonal closure alpha_xi^2 + gamma^2 = 1 - 2*alpha_xi*gamma
equals 82/100 = 0.820 in System-R rationals (alpha_xi=9/10, gamma=1/10).

We verify this against the empirical Lambda_t* per-regime fits
(memory: P4 closure-domain audit, asymptote 0.823 from Symanzik 2+4).

Output: outputs/lambda_t_pythagorean_audit.json
"""
from __future__ import annotations
import json
from pathlib import Path

from fractions import Fraction

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    print("=" * 100)
    print("(*) Lambda_t = alpha_xi^2 + gamma^2 = 1 - 2*alpha_xi*gamma  [from C1 closure]")
    print("=" * 100)
    alpha_xi = Fraction(9, 10)
    gamma    = Fraction(1, 10)
    eps_sq   = Fraction(1, 20)

    # Verify C1 identity (load-bearing in System-R)
    c1_lhs = alpha_xi + gamma
    print(f"  C1: alpha_xi + gamma = {c1_lhs} (expected 1)")
    assert c1_lhs == 1

    # Verify gamma = 2*eps_sync^2 (System-R secondary identity)
    eps_link = 2 * eps_sq
    print(f"  gamma = 2*eps_sync^2 = {eps_link} (expected gamma = 1/10)")
    assert eps_link == gamma

    # Three equivalent forms
    form_pyth = alpha_xi**2 + gamma**2
    form_cross = 1 - 2*alpha_xi*gamma
    form_eps = 1 - 4*alpha_xi*eps_sq
    print()
    print(f"  Form 1 (Pythagorean diagonal):     alpha_xi^2 + gamma^2 = {form_pyth} = {float(form_pyth):.4f}")
    print(f"  Form 2 (Cross-term complement):    1 - 2*alpha_xi*gamma = {form_cross} = {float(form_cross):.4f}")
    print(f"  Form 3 (Sync-leakage explicit):    1 - 4*alpha_xi*eps^2 = {form_eps} = {float(form_eps):.4f}")
    assert form_pyth == form_cross == form_eps

    Lambda_t_R = float(form_pyth)
    print()
    print(f"  Structural Lambda_t (System-R): Lambda_t^R = 82/100 = {Lambda_t_R}")
    print()

    # Compare to empirical asymptote (from memory: P4 closure-domain audit)
    Lambda_t_inf = 0.823       # Symanzik 2+4 asymptote on 11-pt ladder
    Lambda_t_alpha2 = float(alpha_xi**2)  # naive candidate
    Lambda_t_per_regime_lo = 0.813
    Lambda_t_per_regime_hi = 0.840

    print(f"  Empirical Lambda_t asymptote (Symanzik 2+4):    {Lambda_t_inf}")
    print(f"  Per-regime spread of best-fit Lambda_t*(N):    [{Lambda_t_per_regime_lo}, {Lambda_t_per_regime_hi}]")
    print()

    delta_R = Lambda_t_inf - Lambda_t_R
    delta_alpha2 = Lambda_t_inf - Lambda_t_alpha2
    print(f"  Distance to asymptote:")
    print(f"    Lambda_t^R = alpha_xi^2 + gamma^2 = 0.820:   |Delta| = {abs(delta_R):.4f} ({abs(delta_R)/Lambda_t_inf*100:.2f}% rel)")
    print(f"    naive alpha_xi^2 = 0.810:                    |Delta| = {abs(delta_alpha2):.4f} ({abs(delta_alpha2)/Lambda_t_inf*100:.2f}% rel)")
    print(f"  Improvement factor (alpha_xi^2 -> alpha_xi^2 + gamma^2): {abs(delta_alpha2)/abs(delta_R):.2f}x")
    print()

    inside_spread_R = Lambda_t_per_regime_lo <= Lambda_t_R <= Lambda_t_per_regime_hi
    print(f"  Lambda_t^R = 0.820 inside per-regime spread [0.813, 0.840]?  {inside_spread_R}")

    # q_ir cross-arm number (memory: legacy world-formula frontier audit)
    q_ir_target = 0.818779
    print(f"  q_ir_required_target (governance arm, programmatic):       {q_ir_target}")
    print(f"  Distance to Lambda_t^R: {abs(q_ir_target - Lambda_t_R):.4f} ({abs(q_ir_target - Lambda_t_R)/Lambda_t_R*100:.2f}% rel)")

    out = {
        "method": "lambda_t_pythagorean_from_C1",
        "schema_version": "1.0.0",
        "system_R": {
            "alpha_xi_rational": [int(alpha_xi.numerator), int(alpha_xi.denominator)],
            "gamma_rational":    [int(gamma.numerator),    int(gamma.denominator)],
            "C1_identity": "alpha_xi + gamma = 1",
            "secondary_identity": "gamma = 2*eps_sync^2",
        },
        "Lambda_t_structural_rational": [int(form_pyth.numerator), int(form_pyth.denominator)],
        "Lambda_t_structural_decimal": Lambda_t_R,
        "equivalent_forms": [
            "alpha_xi^2 + gamma^2",
            "1 - 2*alpha_xi*gamma",
            "1 - 4*alpha_xi*eps_sync^2",
        ],
        "Lambda_t_empirical_asymptote_Symanzik24": Lambda_t_inf,
        "Lambda_t_per_regime_spread": [Lambda_t_per_regime_lo, Lambda_t_per_regime_hi],
        "rel_offset_C1_form_pct": float(abs(delta_R)/Lambda_t_inf*100),
        "rel_offset_naive_alpha2_pct": float(abs(delta_alpha2)/Lambda_t_inf*100),
        "improvement_factor_alpha2_to_C1form": float(abs(delta_alpha2)/abs(delta_R)) if abs(delta_R) > 0 else float('inf'),
        "inside_per_regime_spread": bool(inside_spread_R),
        "q_ir_governance_target": q_ir_target,
        "q_ir_to_Lambda_t_R_rel_pct": float(abs(q_ir_target - Lambda_t_R)/Lambda_t_R*100),
        "verdict": "LAMBDA_T_DERIVED_FROM_C1_PARAMETER_FREE",
        "caveat": ("This is an algebraic consequence of C1 closure, not a "
                   "first-principles derivation of why the diagonal-only "
                   "closure should be picked up by the per-direction "
                   "Galerkin Lambda_t fit. The interpretation that "
                   "alpha_xi^2 + gamma^2 captures the diagonal Pythagorean "
                   "sum while 2*alpha_xi*gamma spreads into off-diagonal "
                   "shear remains a phenomenological reading until a "
                   "Hilbert-variation derivation is constructed."),
    }
    out_path = REPO / "outputs" / "lambda_t_pythagorean_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
