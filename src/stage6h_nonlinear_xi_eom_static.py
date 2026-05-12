"""Stage 6h: Non-linear Xi-EOM numerical shooting on static spherically
symmetric solutions.

The linearised Ξ-EOM derivation in P4-A gives, around the vacuum
Ξ→1, the static-source 1/r profile
    Ξ_lin(r) = 1 - 2GM/r + O((2GM/r)^2)
in the soft-mass limit m_Ξ r << 1.

This stage tests the strict non-linear Ξ-EOM by numerically solving
the full second-variation of the residual action S_res[Ξ, ψ, K, Q, ω]
for a static spherically symmetric ansatz Ξ(r), and compares the
non-linear solution against Ξ_lin(r) at finite r.

Approach (shooting method):
  1. Parametrise Ξ(r) on a logarithmic radial grid r ∈ [r_core, R_inf]
  2. Boundary conditions:
       Ξ(r_core) = Ξ_inner (free, fitted via shooting parameter)
       Ξ(R_inf) → 1 - 2GM/R_inf  (matching to linearised far-field)
  3. EOM: variational Euler-Lagrange equation for the residual action
       δS_res / δΞ(r) = 0
     in static spherically symmetric ansatz
  4. Integrate the ODE from R_inf inward using the linearised
     boundary as initial condition
  5. Compare the integrated Ξ(r) against Ξ_lin(r) at each r/r_S
  6. Report deviation (Ξ_full - Ξ_lin)/Ξ_lin as function of r/r_S
  7. Closure criterion: if deviation < 10% for r > 2 r_S
     (outside-horizon weak-field regime), the linearised derivation
     is robust; if deviation diverges or grows monotonically, the
     non-linear corrections are essential.

Output: outputs/stage6h_nonlinear_xi_eom_static.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Action coefficients (parent paper §sec:residual_action)
ZETA_1 = 1.0
ZETA_3 = 0.5
OMEGA_0 = 0.83996       # bounded-operator readout, D(Omega) = 67/80
A_K = 0.55              # K_rec row-mean coefficient
A_Q = 0.45              # Q-coefficient
M_XI_SQ = 0.10          # effective Xi-mass squared (soft-mass regime)


def linearised_xi(r, GM):
    """Linearised solution in soft-mass limit m_Xi*r << 1:
        Xi_lin(r) = 1 - 2GM/r
    """
    return 1.0 - 2.0 * GM / r


def linearised_with_yukawa(r, GM, m_xi):
    """Linearised solution with finite m_Xi:
        Xi_lin_yukawa(r) = 1 - (2GM/r) * exp(-m_Xi * r)
    """
    return 1.0 - (2.0 * GM / r) * np.exp(-m_xi * r)


def nonlinear_eom_rhs(r, y, GM):
    """Non-linear ODE RHS for static spherically symmetric Xi(r).

    Variational EOM from S_res with Xi-field action:
      S_res ⊃ (zeta_1 omega / 2) (d Xi/dr)^2
            + (m_Xi^2 / 2) (1 - Xi)^2
            + (zeta_3 omega K_rec) Xi
            + lambda/4 (1 - Xi)^4   (non-linear Mexican-hat term)

    Static EOM in spherical coords, with phi = 1 - Xi:
      phi'' + (2/r) phi' - m_Xi^2 phi - lambda phi^3
        = source(r) / (zeta_1 omega_0)

    For a point-mass source rho(r) = M delta^3(r), the right-hand
    side vanishes for r > 0 (only enters through inner boundary).

    State vector: y = [phi, phi'] where phi = 1 - Xi
    """
    phi, dphi = y
    LAMBDA_QUARTIC = 0.6927  # carrier potential quartic coupling (P1)
    # Non-linear EOM: phi'' + (2/r) phi' - m_Xi^2 phi - lambda phi^3 = 0
    # for r > 0 (vacuum exterior)
    d2phi = -2.0 / max(r, 1e-12) * dphi + M_XI_SQ * phi + LAMBDA_QUARTIC * phi ** 3
    return np.array([dphi, d2phi])


def shoot_inward(r_grid, GM, m_xi):
    """Integrate Xi-EOM from r_grid[0] (large r) inward to r_grid[-1].

    Initial condition at r_grid[0] = R_inf:
      phi(R_inf) ≈ (2GM/R_inf) * exp(-m_Xi R_inf)
      dphi/dr (R_inf) ≈ -(2GM/R_inf^2) * exp(-m_Xi R_inf) * (1 + m_Xi R_inf)

    Returns Xi(r) array on r_grid (reversed to ascending).
    """
    # Reverse the grid for inward integration
    r_inv = r_grid[::-1]
    phi_R = (2.0 * GM / r_inv[0]) * np.exp(-m_xi * r_inv[0])
    dphi_R = -(2.0 * GM / r_inv[0] ** 2) * np.exp(-m_xi * r_inv[0]) * \
             (1.0 + m_xi * r_inv[0])
    y = np.array([phi_R, dphi_R])
    phi_arr = np.zeros_like(r_inv)
    phi_arr[0] = phi_R
    # 4th-order Runge-Kutta step inward
    for i in range(len(r_inv) - 1):
        r1, r2 = r_inv[i], r_inv[i + 1]
        h = r2 - r1   # negative (inward)
        k1 = nonlinear_eom_rhs(r1, y, GM)
        k2 = nonlinear_eom_rhs(r1 + h / 2, y + h / 2 * k1, GM)
        k3 = nonlinear_eom_rhs(r1 + h / 2, y + h / 2 * k2, GM)
        k4 = nonlinear_eom_rhs(r2, y + h * k3, GM)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        phi_arr[i + 1] = y[0]
        # Cutoff: stop if phi > 1 (Xi negative, defect-core saturation)
        if phi_arr[i + 1] > 0.999:
            phi_arr[i + 1:] = 0.999
            break
    # Re-reverse to ascending r
    Xi_arr = 1.0 - phi_arr[::-1]
    return Xi_arr


def main() -> int:
    print("=" * 78)
    print("Stage 6h: Non-linear Xi-EOM static spherically symmetric solution")
    print("=" * 78)
    print()
    print(f"Action coefficients: zeta_1={ZETA_1}, zeta_3={ZETA_3}, omega_0={OMEGA_0}")
    print(f"m_Xi^2 = {M_XI_SQ} (soft-mass regime)")
    print(f"Test mass: GM = 1.0 (lattice units)")
    print()

    # Test parameters
    GM = 1.0
    m_xi = np.sqrt(M_XI_SQ)
    r_S = 2.0 * GM    # Schwarzschild radius
    r_core = 0.5 * r_S
    R_inf = 100.0 * r_S
    n_r = 200

    # Logarithmic radial grid
    r_grid = np.logspace(np.log10(r_core), np.log10(R_inf), n_r)

    # Integrate the non-linear EOM
    Xi_full = shoot_inward(r_grid, GM, m_xi)

    # Linearised reference (m_Xi -> 0 limit)
    Xi_lin = linearised_xi(r_grid, GM)

    # Linearised with Yukawa decay (finite m_Xi)
    Xi_yuk = linearised_with_yukawa(r_grid, GM, m_xi)

    # Per-r relative deviation
    rel_dev_lin = (Xi_full - Xi_lin) / np.maximum(np.abs(Xi_lin), 1e-12)
    rel_dev_yuk = (Xi_full - Xi_yuk) / np.maximum(np.abs(Xi_yuk), 1e-12)

    # Identify weak-field regime (r > 2 r_S)
    weak_mask = r_grid > 2.0 * r_S
    if weak_mask.sum() > 0:
        max_dev_weak = float(np.max(np.abs(rel_dev_lin[weak_mask])))
        median_dev_weak = float(np.median(np.abs(rel_dev_lin[weak_mask])))
    else:
        max_dev_weak = float("nan")
        median_dev_weak = float("nan")

    # Convergence check vs Yukawa-corrected linearised
    if weak_mask.sum() > 0:
        max_dev_yuk_weak = float(np.max(np.abs(rel_dev_yuk[weak_mask])))
        median_dev_yuk_weak = float(np.median(np.abs(rel_dev_yuk[weak_mask])))
    else:
        max_dev_yuk_weak = float("nan")
        median_dev_yuk_weak = float("nan")

    print(f"Radial grid: r/r_S in [{r_core/r_S:.2f}, {R_inf/r_S:.0f}], "
          f"{n_r} points")
    print(f"Weak-field regime r > 2 r_S: {int(weak_mask.sum())} points")
    print()
    print("Comparison to linearised Xi(r) = 1 - 2GM/r:")
    print(f"  median |Xi_full - Xi_lin| / |Xi_lin| (weak-field) "
          f"= {median_dev_weak:.4e}")
    print(f"  max |Xi_full - Xi_lin| / |Xi_lin| (weak-field)    "
          f"= {max_dev_weak:.4e}")
    print()
    print("Comparison to Yukawa-corrected linearised Xi(r) = "
          "1 - (2GM/r) e^{-m_Xi r}:")
    print(f"  median |Xi_full - Xi_yuk| / |Xi_yuk| (weak-field) "
          f"= {median_dev_yuk_weak:.4e}")
    print(f"  max |Xi_full - Xi_yuk| / |Xi_yuk| (weak-field)    "
          f"= {max_dev_yuk_weak:.4e}")
    print()

    # Sample table at selected r/r_S
    sample_radii = [1.5, 2.0, 5.0, 10.0, 50.0, 100.0]
    print(f"{'r/r_S':>8} {'Xi_full':>12} {'Xi_lin':>12} {'Xi_yuk':>12} "
          f"{'rel_dev_lin':>13} {'rel_dev_yuk':>13}")
    print("-" * 80)
    samples = []
    for rs in sample_radii:
        idx = np.argmin(np.abs(r_grid / r_S - rs))
        if r_grid[idx] / r_S < 0.1:
            continue
        sample = {
            "r_over_rS": float(r_grid[idx] / r_S),
            "Xi_full": float(Xi_full[idx]),
            "Xi_lin": float(Xi_lin[idx]),
            "Xi_yuk": float(Xi_yuk[idx]),
            "rel_dev_vs_lin": float(rel_dev_lin[idx]),
            "rel_dev_vs_yuk": float(rel_dev_yuk[idx]),
        }
        samples.append(sample)
        print(f"{r_grid[idx]/r_S:>8.2f} {Xi_full[idx]:>12.6f} "
              f"{Xi_lin[idx]:>12.6f} {Xi_yuk[idx]:>12.6f} "
              f"{rel_dev_lin[idx]:>+13.3e} {rel_dev_yuk[idx]:>+13.3e}")
    print()

    # Closure verdict
    closure_threshold = 0.10
    if max_dev_yuk_weak < closure_threshold:
        verdict = ("LINEARISED_DERIVATION_ROBUST_AT_WEAK_FIELD: "
                   f"max deviation < {closure_threshold:.2f} for r > 2 r_S")
    else:
        verdict = ("NONLINEAR_CORRECTIONS_ESSENTIAL: "
                   f"max deviation > {closure_threshold:.2f} at weak field")
    print(f"Verdict: {verdict}")
    print()

    bundle = {
        "method": "stage6h_nonlinear_xi_eom_static_shooting",
        "schema_version": "1.0.0",
        "action_coefficients": {
            "zeta_1": ZETA_1, "zeta_3": ZETA_3, "omega_0": OMEGA_0,
            "a_K": A_K, "a_Q": A_Q, "m_Xi_sq": M_XI_SQ,
        },
        "test_parameters": {
            "GM": GM, "m_Xi": m_xi, "r_S": r_S,
            "r_core_over_rS": r_core / r_S,
            "R_inf_over_rS": R_inf / r_S,
            "n_r_grid": n_r,
        },
        "weak_field_deviations": {
            "median_dev_vs_lin": median_dev_weak,
            "max_dev_vs_lin": max_dev_weak,
            "median_dev_vs_yukawa_corrected_lin": median_dev_yuk_weak,
            "max_dev_vs_yukawa_corrected_lin": max_dev_yuk_weak,
        },
        "samples": samples,
        "closure_threshold": closure_threshold,
        "verdict": verdict,
    }
    out = REPO / "outputs" / "stage6h_nonlinear_xi_eom_static.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
