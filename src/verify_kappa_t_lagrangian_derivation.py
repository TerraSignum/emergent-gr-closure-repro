"""Systematic 4-step Lagrangian derivation test for kappa_t.

  STEP 1: Linear S_back = kappa * integral T_00 -> G_00 = (1-kappa)*T_00
          Already verified empirically (kappa = 0.987 holds 6/6 DoD).

  STEP 2: Log-nonlinear S_back = kappa * integral T_00 * log(T_00/<T_00>)
          Hilbert variation produces a backreaction with log-correction.
          Test: does log-form fit the observed Lambda_t(a) better than
          linear?

  STEP 3: Spectral-mode integration: does <delta^2/d^2>_w have a
          natural 4/pi factor from the spectral-Laplacian eigenvalue
          density rho(lambda)? Test against Wigner-Dyson surmise.

  STEP 4: <xi^2> normalisation: is <(log Xi - <log Xi>)^2>_w lattice-
          asymptotic to a System-R rational number?

Output: outputs/kappa_t_lagrangian_derivation_audit.json
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
    edge_to_matrix, per_seed_galerkin, ELL_0, D_MIN, EPS_D, XI_THRESH)


REGIMES = [
    ("P1", 28), ("P3", 36), ("P4", 42), ("P5", 50), ("P5N64", 64),
    ("P5N72", 72), ("P5N84", 84), ("P5N100", 100),
    ("P6", 60), ("P7", 72), ("P8", 84),
    ("P6N128", 128), ("P8N128", 128),
]
GAMMA = 0.1
ALPHA_XI = 0.9
GAMMA_SQ = GAMMA ** 2
FOUR_OVER_PI = 4.0 / np.pi


# -------------------------------------------------------------- STEP 2
def step2_log_nonlinear_test():
    """Linear S_back: Lambda_t(a) = kappa * T_00(a)
    Log-nonlinear S_back: Lambda_t(a) = kappa * T_00 * (1 + alpha * log(T_00/<T_00>))

    Per-node: Lambda_t_observed(a) = T_00(a) - G_00(a)
    Fit kappa, alpha jointly across all nodes pooled across regimes.
    Compare R^2 vs the linear (alpha=0) form.
    """
    print("=" * 100)
    print("STEP 2: Log-nonlinear backreaction test")
    print("=" * 100)
    g_pool, t_pool = [], []
    for reg, n_lat in REGIMES:
        p = find_d1_npz(reg, REPO)
        if p is None or not p.exists(): continue
        d = np.load(p, allow_pickle=True)
        edge_arr = d["dense_cell_edge_xi_values"]
        amp_arr = d["dense_cell_node_amplitude_values"]
        phase_arr = d["dense_cell_node_phase_values"]
        n_seeds = min(edge_arr.shape[0], 32)
        for s in range(n_seeds):
            xi_mat = edge_to_matrix(edge_arr[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            psi = amp_arr[s] * np.exp(1j * phase_arr[s])
            k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
            q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
            prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
            g_pool.append(np.asarray(prep["g_00_h"]))
            t_pool.append(np.asarray(prep["t00"]))
    g00 = np.concatenate(g_pool)
    t00 = np.concatenate(t_pool)
    # Filter out zero/negative T_00 (avoid log(0))
    mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g00)
    g00 = g00[mask]; t00 = t00[mask]
    # Lambda_t observed
    lt = t00 - g00
    t00_mean = float(np.mean(t00))
    log_ratio = np.log(t00 / t00_mean)

    # Linear fit: Lambda_t = kappa * T_00
    kappa_linear = float(np.sum(lt * t00) / np.sum(t00 ** 2))
    pred_linear = kappa_linear * t00
    ss_res_linear = float(np.sum((lt - pred_linear) ** 2))
    ss_tot = float(np.sum((lt - lt.mean()) ** 2))
    r2_linear = 1.0 - ss_res_linear / ss_tot

    # Log-nonlinear fit: Lambda_t = kappa * T_00 + alpha * T_00 * log(T_00/<T>)
    # Solve normal equations: [[T·T, T·T·logr], [T·T·logr, T·T·logr²]] * [k,a] = [T·lt, T·logr·lt]
    A11 = float(np.sum(t00 ** 2))
    A12 = float(np.sum(t00 ** 2 * log_ratio))
    A22 = float(np.sum(t00 ** 2 * log_ratio ** 2))
    b1 = float(np.sum(t00 * lt))
    b2 = float(np.sum(t00 * lt * log_ratio))
    A = np.array([[A11, A12], [A12, A22]])
    b = np.array([b1, b2])
    sol = np.linalg.solve(A, b)
    kappa_log = float(sol[0]); alpha_log = float(sol[1])
    pred_log = kappa_log * t00 + alpha_log * t00 * log_ratio
    ss_res_log = float(np.sum((lt - pred_log) ** 2))
    r2_log = 1.0 - ss_res_log / ss_tot

    # AIC-style comparison: log-nonlinear has 1 extra param
    n = len(lt)
    delta_AIC = 2 - 2 * (np.log(ss_res_log / n) - np.log(ss_res_linear / n)) * n / 2
    # Simpler: BIC change
    delta_BIC = np.log(n) - n * (np.log(ss_res_log) - np.log(ss_res_linear))

    print(f"  n_nodes_pooled = {n}")
    print(f"  Linear:    Lambda_t = {kappa_linear:.4f} * T_00,    R^2 = {r2_linear:.6f}")
    print(f"  Log-nonl.: Lambda_t = {kappa_log:.4f} * T_00 + {alpha_log:+.4f} * T_00 * log(T_00/<T_00>),  R^2 = {r2_log:.6f}")
    print(f"  R^2 gain from adding log-term: {(r2_log - r2_linear)*100:.4f}%")
    print(f"  Coefficient on log: alpha = {alpha_log:+.5f}")

    if abs(alpha_log) < 0.005:
        verdict = "LOG_NONLINEARITY_NEGLIGIBLE_LINEAR_FORM_EXACT"
    elif r2_log - r2_linear > 0.001:
        verdict = "LOG_NONLINEARITY_IMPROVES_FIT_NON_NEGLIGIBLY"
    else:
        verdict = "LOG_NONLINEARITY_NO_IMPROVEMENT"
    print(f"  VERDICT: {verdict}")

    return {
        "n_nodes_pooled": int(n),
        "linear_kappa": kappa_linear,
        "linear_R_squared": r2_linear,
        "log_kappa": kappa_log,
        "log_alpha": alpha_log,
        "log_R_squared": r2_log,
        "R_squared_gain": r2_log - r2_linear,
        "verdict": verdict,
    }


# -------------------------------------------------------------- STEP 3
def step3_spectral_mode_density():
    """Test if spatial spectral-Laplacian eigenvalue density yields
    a 4/pi factor in some natural integral measure."""
    print()
    print("=" * 100)
    print("STEP 3: Spectral-Laplacian mode density yields 4/pi?")
    print("=" * 100)
    rows = []
    for reg, n_lat in REGIMES:
        p = find_d1_npz(reg, REPO)
        if p is None or not p.exists(): continue
        d = np.load(p, allow_pickle=True)
        edge_arr = d["dense_cell_edge_xi_values"]
        n_seeds = min(edge_arr.shape[0], 32)
        eigs_pool = []
        for s in range(n_seeds):
            xi_mat = edge_to_matrix(edge_arr[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            xi_off = xi_mat.copy()
            np.fill_diagonal(xi_off, 0.0)
            adj = (xi_off > XI_THRESH).astype(np.float64)
            weight_adj = xi_off * adj
            deg = weight_adj.sum(axis=1) + 1e-12
            deg_inv_sqrt = 1.0 / np.sqrt(deg)
            l_norm = (np.eye(n_lat) - deg_inv_sqrt[:, None]
                      * weight_adj * deg_inv_sqrt[None, :])
            try:
                eigvals_l, _ = np.linalg.eigh(l_norm)
            except np.linalg.LinAlgError:
                continue
            # First nontrivial 3 eigenvalues (spatial modes)
            eigs_pool.append(eigvals_l[1:4])
        if not eigs_pool:
            continue
        eigs_array = np.stack(eigs_pool, axis=0)  # (n_seeds, 3)
        # Mean ratio of consecutive eigenvalues
        ratio_12 = float(np.mean(eigs_array[:, 1] / eigs_array[:, 0]))
        ratio_13 = float(np.mean(eigs_array[:, 2] / eigs_array[:, 0]))
        # Mean of three eigenvalues
        mean_eig = float(np.mean(eigs_array))
        # Wigner-Dyson surmise mean spacing for GUE: 32/(9π) ≈ 1.13 (mean)
        rows.append({
            "regime": reg, "N": n_lat,
            "lambda_1": float(np.mean(eigs_array[:, 0])),
            "lambda_2": float(np.mean(eigs_array[:, 1])),
            "lambda_3": float(np.mean(eigs_array[:, 2])),
            "ratio_lambda2_lambda1": ratio_12,
            "ratio_lambda3_lambda1": ratio_13,
            "mean_eig": mean_eig,
        })

    print(f"{'reg':<10} {'N':>3} {'λ_1':>10} {'λ_2':>10} {'λ_3':>10} {'λ_2/λ_1':>10} {'λ_3/λ_1':>10}")
    print("-" * 70)
    for r in rows:
        print(f"{r['regime']:<10} {r['N']:>3} "
              f"{r['lambda_1']:>10.5f} {r['lambda_2']:>10.5f} "
              f"{r['lambda_3']:>10.5f} {r['ratio_lambda2_lambda1']:>10.4f} "
              f"{r['ratio_lambda3_lambda1']:>10.4f}")
    print()
    # Test: does any combination give 4/π?
    mean_l1 = float(np.mean([r["lambda_1"] for r in rows]))
    mean_l2 = float(np.mean([r["lambda_2"] for r in rows]))
    mean_l3 = float(np.mean([r["lambda_3"] for r in rows]))
    mean_ratio_21 = float(np.mean([r["ratio_lambda2_lambda1"] for r in rows]))
    mean_ratio_31 = float(np.mean([r["ratio_lambda3_lambda1"] for r in rows]))

    print(f"Across-regime means:")
    print(f"  <λ_1> = {mean_l1:.5f}")
    print(f"  <λ_2> = {mean_l2:.5f}")
    print(f"  <λ_3> = {mean_l3:.5f}")
    print(f"  <λ_2/λ_1> = {mean_ratio_21:.4f}")
    print(f"  <λ_3/λ_1> = {mean_ratio_31:.4f}")
    print(f"  Target 4/π = {FOUR_OVER_PI:.4f}")
    print()
    # Best match
    diffs = {
        "<λ_2/λ_1>": (mean_ratio_21, abs(mean_ratio_21 - FOUR_OVER_PI)),
        "<λ_3/λ_1>": (mean_ratio_31, abs(mean_ratio_31 - FOUR_OVER_PI)),
        "<λ_3>/<λ_2>": (mean_l3 / mean_l2, abs(mean_l3 / mean_l2 - FOUR_OVER_PI)),
    }
    closest_label = min(diffs, key=lambda k: diffs[k][1])
    print(f"  Closest to 4/π: {closest_label} = {diffs[closest_label][0]:.4f} (off by {diffs[closest_label][1]:.4f})")

    return {"per_regime": rows,
            "mean_lambda1": mean_l1, "mean_lambda2": mean_l2, "mean_lambda3": mean_l3,
            "mean_ratio_21": mean_ratio_21, "mean_ratio_31": mean_ratio_31,
            "closest_to_4_over_pi": {"label": closest_label,
                                      "value": diffs[closest_label][0],
                                      "abs_diff": diffs[closest_label][1]}}


# -------------------------------------------------------------- STEP 4
def step4_xi_squared_normalisation():
    """Test if <(log Xi - <log Xi>)^2>_w_ab converges to a System-R
    rational number in continuum."""
    print()
    print("=" * 100)
    print("STEP 4: <xi^2> normalisation in System-R")
    print("=" * 100)
    rows = []
    for reg, n_lat in REGIMES:
        p = find_d1_npz(reg, REPO)
        if p is None or not p.exists(): continue
        d = np.load(p, allow_pickle=True)
        edge_arr = d["dense_cell_edge_xi_values"]
        n_seeds = min(edge_arr.shape[0], 32)
        xi_sq_pool, w_pool = [], []
        for s in range(n_seeds):
            xi_mat = edge_to_matrix(edge_arr[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            xi_off = xi_mat.copy()
            np.fill_diagonal(xi_off, 0.0)
            adj = (xi_off > XI_THRESH).astype(np.float64)
            weight_adj = xi_off * adj
            log_xi = np.where(adj > 0, np.log(np.maximum(xi_off, 1e-12)), 0.0)
            log_xi_mean = (log_xi * adj).sum() / (adj.sum() + 1e-12)
            xi_dev_sq = np.where(adj > 0, (log_xi - log_xi_mean) ** 2, 0.0)
            mask = adj > 0
            xi_sq_pool.append(xi_dev_sq[mask])
            w_pool.append(weight_adj[mask])
        if not xi_sq_pool:
            continue
        xi_sq_all = np.concatenate(xi_sq_pool)
        w_all = np.concatenate(w_pool)
        weighted_mean = float((w_all * xi_sq_all).sum() / w_all.sum())
        rows.append({"regime": reg, "N": n_lat, "xi_sq_w_mean": weighted_mean})

    print(f"{'reg':<10} {'N':>3} | {'<ξ²>_w':>10}")
    print("-" * 30)
    for r in rows:
        print(f"{r['regime']:<10} {r['N']:>3} | {r['xi_sq_w_mean']:>10.5f}")
    # Symanzik continuum extrapolation
    if len(rows) >= 4:
        N_arr = np.array([r["N"] for r in rows], dtype=float)
        y = np.array([r["xi_sq_w_mean"] for r in rows])
        X = np.column_stack([np.ones_like(N_arr), N_arr ** -2.0])
        c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
        xi_sq_inf = float(c[0])
        print(f"\nSymanzik 2-term continuum limit: <ξ²>_∞ = {xi_sq_inf:.5f}")
        # Match to System-R rationals
        candidates = {
            "1/2": 0.5, "1/3": 0.333, "1/4": 0.25, "1/5": 0.2,
            "1/π": 1.0/np.pi, "π/4": np.pi/4, "1/(2π)": 1.0/(2*np.pi),
            "γ²·100/4π·100": 100*GAMMA_SQ * (100/FOUR_OVER_PI),
            "25/(8π)": 25.0/(8*np.pi),
        }
        diffs = {label: (val, abs(val - xi_sq_inf)) for label, val in candidates.items()}
        closest = min(diffs, key=lambda k: diffs[k][1])
        print(f"Closest System-R rational: {closest} = {diffs[closest][0]:.5f} (off {diffs[closest][1]:.5f})")
    else:
        xi_sq_inf = float("nan")

    return {"per_regime": rows, "xi_sq_continuum_limit": xi_sq_inf}


def main() -> int:
    out = {"method": "kappa_t_lagrangian_4step_derivation_test"}
    out["step2_log_nonlinear"] = step2_log_nonlinear_test()
    out["step3_spectral_mode_density"] = step3_spectral_mode_density()
    out["step4_xi_squared_norm"] = step4_xi_squared_normalisation()
    out_path = REPO / "outputs" / "kappa_t_lagrangian_derivation_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
