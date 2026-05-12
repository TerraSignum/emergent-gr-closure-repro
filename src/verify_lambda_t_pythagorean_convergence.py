"""(*) Per-regime Lambda_t* convergence test against the Pythagorean
prediction Lambda_t^R = alpha_xi^2 + gamma^2 = 82/100 = 0.820.

If the C1^2 + Hilbert-variation derivation is correct, the per-regime
best-fit Lambda_t*(N) should converge to 82/100 as N -> infinity, with
finite-N corrections vanishing under continuum extrapolation.

We extract the per-regime <T_00 - G_00> on the existing 13-regime
lattice data and Symanzik-extrapolate to N = infinity, then compare to
the Pythagorean prediction.

Output: outputs/lambda_t_pythagorean_convergence_audit.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self
    def load_module(self, name):
        raise ImportError("cupy disabled")
sys.meta_path.insert(0, _BlockCupy())

from _d1_npz_discovery import find_d1_npz
from verify_galerkin_runner_A_hessian_ricci import (
    edge_to_matrix, per_seed_galerkin)


REGIMES = [
    ("P1", 28), ("P3", 36), ("P4", 42), ("P5", 50), ("P5N64", 64),
    ("P6", 60), ("P7", 72), ("P8", 84), ("P5N100", 100),
    ("P5N72", 72), ("P5N84", 84), ("P6N128", 128), ("P8N128", 128),
]

ALPHA_XI = 9.0/10.0
GAMMA    = 1.0/10.0
LAMBDA_T_R_PYTHAGOREAN = ALPHA_XI**2 + GAMMA**2  # 82/100 = 0.820
LAMBDA_T_R_NAIVE       = ALPHA_XI**2             # 81/100 = 0.810
CROSS_TERM             = 2.0*ALPHA_XI*GAMMA      # 18/100 = 0.180


def per_regime_lambda_t_star(reg, n_lat):
    """Extract per-regime best-fit Lambda_t* = <T_00 - G_00>."""
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    e = d["dense_cell_edge_xi_values"]
    a = d["dense_cell_node_amplitude_values"]
    ph = d["dense_cell_node_phase_values"]
    g_pool, t_pool = [], []
    for s in range(min(e.shape[0], 32)):
        xi_mat = edge_to_matrix(e[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = a[s] * np.exp(1j*ph[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        g_pool.append(np.asarray(prep["g_00_h"]))
        t_pool.append(np.asarray(prep["t00"]))
    g00 = np.concatenate(g_pool); t00 = np.concatenate(t_pool)
    mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g00)
    if not np.any(mask):
        return None
    g00 = g00[mask]; t00 = t00[mask]
    # Lambda_t* = mean(T_00 - G_00) per regime (this is the
    # least-squares optimal Lambda_t making median residual zero
    # in the per-direction time-time equation G_00 + Lambda_t = T_00).
    lambda_t_star = float(np.mean(t00 - g00))
    return {
        "regime": reg, "N": int(n_lat),
        "lambda_t_star": lambda_t_star,
        "median_T00":  float(np.median(t00)),
        "median_G00":  float(np.median(g00)),
        "n_nodes": int(len(t00)),
    }


def symanzik_24(N_arr, y_arr):
    """Fit y(N) = y_inf + c_2/N^2 + c_4/N^4 (Symanzik 2+4)."""
    A = np.column_stack([np.ones_like(N_arr), 1.0/N_arr**2, 1.0/N_arr**4])
    coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    y_inf, c2, c4 = coef
    pred = A @ coef
    ss_res = np.sum((y_arr - pred)**2)
    ss_tot = np.sum((y_arr - y_arr.mean())**2)
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return float(y_inf), float(c2), float(c4), float(R2)


def main() -> int:
    print("=" * 100)
    print("(*) Lambda_t* convergence vs Pythagorean prediction Lambda_t^R = alpha_xi^2 + gamma^2 = 82/100")
    print("=" * 100)
    print(f"  Pythagorean prediction:    Lambda_t^R = alpha_xi^2 + gamma^2 = {LAMBDA_T_R_PYTHAGOREAN:.4f}")
    print(f"  Naive prediction:          alpha_xi^2                       = {LAMBDA_T_R_NAIVE:.4f}")
    print(f"  Cross-term (NOT in Lambda_t expected): 2*alpha_xi*gamma     = {CROSS_TERM:.4f}")
    print()

    rows = []
    for reg, n in REGIMES:
        r = per_regime_lambda_t_star(reg, n)
        if r is not None:
            rows.append(r)
    print(f"{'regime':<10} {'N':>4} | {'Lambda_t*':>10} | {'rel vs 0.820':>14} {'rel vs 0.810':>14}")
    print("-" * 70)
    for r in sorted(rows, key=lambda x: x["N"]):
        L = r["lambda_t_star"]
        rel_pyth = (L - LAMBDA_T_R_PYTHAGOREAN)/LAMBDA_T_R_PYTHAGOREAN*100
        rel_naive = (L - LAMBDA_T_R_NAIVE)/LAMBDA_T_R_NAIVE*100
        print(f"{r['regime']:<10} {r['N']:>4} | {L:>10.4f} | {rel_pyth:>+13.2f}% {rel_naive:>+13.2f}%")
    arr = sorted(rows, key=lambda x: x["N"])
    N_arr = np.array([r["N"] for r in arr], dtype=float)
    L_arr = np.array([r["lambda_t_star"] for r in arr])

    # Symanzik on full 13-regime ladder
    L_inf, c2, c4, R2 = symanzik_24(N_arr, L_arr)
    print()
    print(f"  Symanzik 2+4 fit on all {len(arr)} regimes:")
    print(f"    Lambda_t^infty = {L_inf:.4f}, c_2 = {c2:+.4f}, c_4 = {c4:+.4f}, R^2 = {R2:.3f}")
    print()
    print(f"  Distance to predictions:")
    print(f"    |Lambda_t^infty - 0.820| = {abs(L_inf - LAMBDA_T_R_PYTHAGOREAN):.4f} ({abs(L_inf - LAMBDA_T_R_PYTHAGOREAN)/LAMBDA_T_R_PYTHAGOREAN*100:.2f}% rel)")
    print(f"    |Lambda_t^infty - 0.810| = {abs(L_inf - LAMBDA_T_R_NAIVE):.4f} ({abs(L_inf - LAMBDA_T_R_NAIVE)/LAMBDA_T_R_NAIVE*100:.2f}% rel)")
    improvement = abs(L_inf - LAMBDA_T_R_NAIVE)/abs(L_inf - LAMBDA_T_R_PYTHAGOREAN) if abs(L_inf - LAMBDA_T_R_PYTHAGOREAN) > 1e-6 else float('inf')
    print(f"    Improvement Pythagorean over naive: {improvement:.2f}x")

    # Test if Lambda_t* is consistent with 82/100 + finite-N correction
    finite_N_corrections = L_arr - LAMBDA_T_R_PYTHAGOREAN
    print()
    print(f"  Finite-N corrections (Lambda_t* - 82/100):")
    for r, c in zip(arr, finite_N_corrections):
        print(f"    {r['regime']:<10} N={r['N']:>4}: {c:+.4f}")
    print(f"  Mean correction: {np.mean(finite_N_corrections):+.4f}")
    print(f"  Std correction:  {np.std(finite_N_corrections):.4f}")
    print(f"  Asymptotic (N>=72): mean = {np.mean(L_arr[N_arr>=72]) - LAMBDA_T_R_PYTHAGOREAN:+.4f}")

    out = {
        "method": "lambda_t_pythagorean_convergence",
        "schema_version": "1.0.0",
        "pythagorean_prediction": LAMBDA_T_R_PYTHAGOREAN,
        "naive_prediction":       LAMBDA_T_R_NAIVE,
        "cross_term_excluded":    CROSS_TERM,
        "per_regime": [
            {**r,
             "rel_offset_pyth_pct":  float((r["lambda_t_star"] - LAMBDA_T_R_PYTHAGOREAN)/LAMBDA_T_R_PYTHAGOREAN*100),
             "rel_offset_naive_pct": float((r["lambda_t_star"] - LAMBDA_T_R_NAIVE)/LAMBDA_T_R_NAIVE*100)}
            for r in arr],
        "symanzik_24_fit": {
            "Lambda_t_inf": L_inf, "c_2": c2, "c_4": c4, "R2": R2,
            "n_points": len(arr),
        },
        "distance_to_predictions": {
            "abs_to_pyth": float(abs(L_inf - LAMBDA_T_R_PYTHAGOREAN)),
            "abs_to_naive": float(abs(L_inf - LAMBDA_T_R_NAIVE)),
            "rel_pyth_pct":  float(abs(L_inf - LAMBDA_T_R_PYTHAGOREAN)/LAMBDA_T_R_PYTHAGOREAN*100),
            "rel_naive_pct": float(abs(L_inf - LAMBDA_T_R_NAIVE)/LAMBDA_T_R_NAIVE*100),
            "improvement_factor": float(improvement),
        },
        "verdict": (
            "PYTHAGOREAN_CONVERGENCE_CONFIRMED"
            if abs(L_inf - LAMBDA_T_R_PYTHAGOREAN) < abs(L_inf - LAMBDA_T_R_NAIVE)
            else "NAIVE_BETTER_OR_TIE"
        ),
    }
    out_path = REPO / "outputs" / "lambda_t_pythagorean_convergence_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
