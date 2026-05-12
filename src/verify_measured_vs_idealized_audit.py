"""Re-evaluation of all key Galerkin findings under the
operator_reading measured causal-wave coefficients
(alpha_xi = 0.900819, gamma = 0.100206, eps_sync_sq = 0.05000,
beta_pi = 0.937913, D_Omega = 0.839964) versus the
System-R-idealised rationals (alpha_xi = 9/10, gamma = 1/10,
eps_sync_sq = 1/20, beta_pi = 15/16, D_Omega = 67/80).

Tests:
  (T1) Lambda_t / Lambda_s sigma-distance under both targets
  (T2) Optimal kappa for the spatial-running Lambda
       with measured gamma^2 = 0.010041 vs idealised 1/100
  (T3) Frobenius residual recomputed under measured Lambda_struct
  (T4) Higher-order term Own1 (8 pi G running) recomputed with
       measured alpha_xi^3 instead of (9/10)^3
  (T5) Output: outputs/measured_vs_idealized_audit.json with
       all comparisons

The goal is to see whether any reported significance statement
changes when measured values are used instead of idealised
values, beyond the trivial ~0.2% offset already in the paper.
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
    D_MIN, ELL_0, EPS_D, XI_THRESH, edge_to_matrix, per_seed_galerkin)


# Two reading conventions (from Memory + h2v2_theta23 file)
IDEALIZED = {
    "name": "System-R idealised",
    "alpha_xi": 0.9,
    "gamma": 0.1,
    "eps_sync_sq": 0.05,
    "beta_pi": 15.0 / 16.0,
    "D_Omega": 67.0 / 80.0,
    "lambda_t": 0.81,            # alpha_xi^2
    "lambda_s": -0.005,           # -gamma^2/2
    "kappa_O5": 0.01,             # gamma^2
    "alpha_xi_cubed": 0.729,      # alpha_xi^3
}
MEASURED = {
    "name": "operator_reading measured",
    "alpha_xi": 0.900819,
    "gamma": 0.100206,
    "eps_sync_sq": 0.05,
    "beta_pi": 0.937913,
    "D_Omega": 0.839964,
    "lambda_t": 0.900819 ** 2,    # = 0.811475
    "lambda_s": -0.5 * 0.100206 ** 2,  # = -0.005021
    "kappa_O5": 0.100206 ** 2,    # = 0.010041
    "alpha_xi_cubed": 0.900819 ** 3,   # = 0.730991
}


def per_node_residual(g_00, g_ij, t00, t_ij, lambda_t, lambda_s, eye3):
    res00 = g_00 + lambda_t - t00
    spatial_res = (g_ij + lambda_s * eye3[None, :, :]) - t_ij
    sq = res00 ** 2 + (spatial_res ** 2).sum(axis=(1, 2))
    return np.sqrt(sq)


def t00_lift(residual, t00):
    p90_res = np.percentile(residual, 90)
    p90_t00 = np.percentile(t00, 90)
    top_res = residual >= p90_res
    top_t00 = t00 >= p90_t00
    n = len(residual)
    n_top_res, n_top_t00 = int(top_res.sum()), int(top_t00.sum())
    n_overlap = int((top_res & top_t00).sum())
    expected = n_top_res * n_top_t00 / max(n, 1)
    if expected <= 0:
        return float("nan")
    return float(n_overlap / expected)


def evaluate_at_regime(regime, n_lat, conv):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)

    blind_lam_t_per_seed = []
    blind_lam_s_per_seed = []
    res_struct_pool = []
    t00_pool = []
    res_running_pool = []  # with O5 spatial-running Lambda

    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        eye3 = prep["eye3"]

        # Per-seed blind Lambda fit (independent of convention)
        blind_lt = float(np.mean(prep["t00"] - prep["g_00_h"]))
        g_diag = np.stack([prep["g_ij_h"][:, 0, 0],
                            prep["g_ij_h"][:, 1, 1],
                            prep["g_ij_h"][:, 2, 2]], axis=1)
        t_diag = np.stack([prep["t_ij"][:, 0, 0],
                            prep["t_ij"][:, 1, 1],
                            prep["t_ij"][:, 2, 2]], axis=1)
        blind_ls = float(np.mean(t_diag - g_diag))
        blind_lam_t_per_seed.append(blind_lt)
        blind_lam_s_per_seed.append(blind_ls)

        # Struct residual under conv's lambda
        r_struct = per_node_residual(
            prep["g_00_h"], prep["g_ij_h"],
            prep["t00"], prep["t_ij"],
            conv["lambda_t"], conv["lambda_s"], eye3)
        res_struct_pool.append(np.asarray(r_struct))
        t00_pool.append(np.asarray(prep["t00"]))

        # Spatial-running Lambda (O5) under conv's kappa
        xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
        np.fill_diagonal(xi_off, 0.0)
        adj = (xi_off > XI_THRESH).astype(np.float64)
        d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
        d_mat = np.maximum(d_mat, D_MIN)
        d_sq_safe = np.where(adj > 0, d_mat ** 2, np.inf)
        weight_grad = np.where(
            adj > 0, (xi_off * adj) / (d_sq_safe + EPS_D), 0.0)
        omega_a = weight_grad.sum(axis=1)
        omega_mean = max(omega_a.mean(), 1e-9)
        factor = 1.0 + conv["kappa_O5"] * omega_a / omega_mean
        lt_a = conv["lambda_t"] * factor
        ls_a = conv["lambda_s"] * factor
        res00 = prep["g_00_h"] + lt_a - prep["t00"]
        spatial_res = (prep["g_ij_h"] + ls_a[:, None, None]
                        * eye3[None, :, :]) - prep["t_ij"]
        r_run = np.sqrt(res00 ** 2 + (spatial_res ** 2).sum(axis=(1, 2)))
        res_running_pool.append(np.asarray(r_run))

    blind_lt_mean = float(np.mean(blind_lam_t_per_seed))
    blind_lt_std = float(np.std(blind_lam_t_per_seed, ddof=1))
    blind_ls_mean = float(np.mean(blind_lam_s_per_seed))
    blind_ls_std = float(np.std(blind_lam_s_per_seed, ddof=1))

    res_struct = np.concatenate(res_struct_pool)
    res_running = np.concatenate(res_running_pool)
    t00_all = np.concatenate(t00_pool)

    return {
        "regime": regime, "N": n_lat,
        "blind_lambda_t": blind_lt_mean,
        "blind_lambda_t_std": blind_lt_std,
        "sigma_distance_lambda_t": float(
            (blind_lt_mean - conv["lambda_t"]) / max(blind_lt_std, 1e-12)),
        "blind_lambda_s": blind_ls_mean,
        "blind_lambda_s_std": blind_ls_std,
        "sigma_distance_lambda_s": float(
            (blind_ls_mean - conv["lambda_s"]) / max(blind_ls_std, 1e-12)),
        "struct_residual_mean": float(res_struct.mean()),
        "struct_residual_median": float(np.median(res_struct)),
        "struct_residual_lift_t00": t00_lift(res_struct, t00_all),
        "running_residual_mean": float(res_running.mean()),
        "running_residual_median": float(np.median(res_running)),
        "running_residual_lift_t00": t00_lift(res_running, t00_all),
    }


def main():
    print("=" * 100)
    print("Measured vs idealised: do any of our reported findings change?")
    print("=" * 100)

    table = []
    for regime, n_lat in [("P5", 50), ("P8", 84), ("P5N100", 100)]:
        for conv in (IDEALIZED, MEASURED):
            r = evaluate_at_regime(regime, n_lat, conv)
            if r is None:
                continue
            r["convention"] = conv["name"]
            r["lambda_t_target"] = conv["lambda_t"]
            r["lambda_s_target"] = conv["lambda_s"]
            r["kappa_O5"] = conv["kappa_O5"]
            table.append(r)

    print()
    print(f"{'regime':<8} {'N':>3} {'conv':<22} | "
          f"{'blind_lt':>9} {'target':>9} {'sigma_dist':>11} | "
          f"{'res_struct_med':>15} {'lift_t00':>9}")
    print("-" * 100)
    for r in table:
        print(f"{r['regime']:<8} {r['N']:>3} {r['convention']:<22} | "
              f"{r['blind_lambda_t']:>9.5f} {r['lambda_t_target']:>9.5f} "
              f"{r['sigma_distance_lambda_t']:>+11.3f} | "
              f"{r['struct_residual_median']:>15.5f} "
              f"{r['struct_residual_lift_t00']:>9.2f}")

    # Aggregate per regime: change in sigma-distance
    print()
    print("=" * 100)
    print("KEY DIFFERENCES (idealised -> measured)")
    print("=" * 100)
    for regime in ("P5", "P8", "P5N100"):
        ideal = next((r for r in table
                       if r["regime"] == regime and "idealised" in r["convention"]), None)
        meas = next((r for r in table
                      if r["regime"] == regime and "measured" in r["convention"]), None)
        if ideal is None or meas is None:
            continue
        d_sig_t = meas["sigma_distance_lambda_t"] - ideal["sigma_distance_lambda_t"]
        d_sig_s = meas["sigma_distance_lambda_s"] - ideal["sigma_distance_lambda_s"]
        d_med = (meas["struct_residual_median"]
                  - ideal["struct_residual_median"]) / max(
            ideal["struct_residual_median"], 1e-12) * 100
        d_run_med = (meas["running_residual_median"]
                       - ideal["running_residual_median"]) / max(
            ideal["running_residual_median"], 1e-12) * 100
        print(f"  {regime} (N={ideal['N']}):")
        print(f"    sigma-distance Lambda_t shift: "
              f"{ideal['sigma_distance_lambda_t']:+.3f} -> "
              f"{meas['sigma_distance_lambda_t']:+.3f} "
              f"(delta {d_sig_t:+.3f} sigma)")
        print(f"    sigma-distance Lambda_s shift: "
              f"{ideal['sigma_distance_lambda_s']:+.3f} -> "
              f"{meas['sigma_distance_lambda_s']:+.3f} "
              f"(delta {d_sig_s:+.3f} sigma)")
        print(f"    struct median residual shift: "
              f"{ideal['struct_residual_median']:.5f} -> "
              f"{meas['struct_residual_median']:.5f} "
              f"({d_med:+.2f}% relative)")
        print(f"    running median residual shift: "
              f"{ideal['running_residual_median']:.5f} -> "
              f"{meas['running_residual_median']:.5f} "
              f"({d_run_med:+.2f}% relative)")

    # Save
    out_path = REPO / "outputs" / "measured_vs_idealized_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "measured_vs_idealized_galerkin_audit",
            "schema_version": "1.0.0",
            "idealized_convention": IDEALIZED,
            "measured_convention": MEASURED,
            "per_regime_per_convention": table,
        }, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
