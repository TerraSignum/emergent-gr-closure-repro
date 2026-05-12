"""Empirical verification of derivation Step 3:

  R^Hess(a) = gamma^2 * <xi_ab^2>_w_ab + O(gamma^4).

Per node a, decompose:
  x_ab = delta_ab^2 / d_ab^2
  1 - x_ab = expected to be ~ gamma^2 * xi_ab^2 (Step 3 prediction)
  R^Hess(a) = <x_ab*(1-x_ab)>_w_ab

Test:
  (a) Mean and median of (1-x_ab) across edges
  (b) Compare to gamma^2 = 0.01 (System-R prediction)
  (c) Compute actual R^Hess(a) median per regime, compare to
      gamma^2 * <xi_ab^2>_w_ab predicted value
  (d) The ratio R^Hess_observed / R^Hess_predicted gives the
      effective normalisation constant; if ~ 4/pi, Step 4 holds.

Output: outputs/kappa_t_derivation_step3_audit.json
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
    edge_to_matrix, ELL_0, D_MIN, EPS_D, XI_THRESH)


REGIMES = [
    ("P5", 50), ("P5N64", 64), ("P5N72", 72),
    ("P5N84", 84), ("P5N100", 100),
]
GAMMA = 0.1
GAMMA_SQ = GAMMA ** 2  # 0.01
FOUR_OVER_PI = 4.0 / np.pi  # 1.2732


def main() -> int:
    print("=" * 100)
    print("Step 3 verification: R^Hess(a) ~ gamma^2 * <xi^2> + O(gamma^4)?")
    print("=" * 100)

    rows = []
    for reg, n_lat in REGIMES:
        p = find_d1_npz(reg, REPO)
        if p is None or not p.exists(): continue
        d = np.load(p, allow_pickle=True)
        edge_arr = d["dense_cell_edge_xi_values"]
        amp_arr = d["dense_cell_node_amplitude_values"]
        phase_arr = d["dense_cell_node_phase_values"]
        n_seeds = min(edge_arr.shape[0], 32)

        x_ab_pool = []     # x = delta^2/d^2
        one_minus_x_pool = []
        xi_dev_sq_pool = []  # (log Xi - log <Xi>)^2
        weight_pool = []
        r_bar_pool = []

        for s in range(n_seeds):
            xi_mat = edge_to_matrix(edge_arr[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            xi_off = xi_mat.copy()
            np.fill_diagonal(xi_off, 0.0)
            adj = (xi_off > XI_THRESH).astype(np.float64)
            weight_adj = xi_off * adj
            d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
            d_mat = np.maximum(d_mat, D_MIN)
            d_sq = d_mat * d_mat

            # Compute spectral-laplacian eigenvectors (spatial modes)
            deg = weight_adj.sum(axis=1) + 1e-12
            deg_inv_sqrt = 1.0 / np.sqrt(deg)
            l_norm = (np.eye(n_lat) - deg_inv_sqrt[:, None]
                      * weight_adj * deg_inv_sqrt[None, :])
            eigvals_l, eigvecs_l = np.linalg.eigh(l_norm)
            spatial = eigvecs_l[:, 1:4]

            spatial_diff = spatial[None, :, :] - spatial[:, None, :]
            delta_sq = (spatial_diff ** 2).sum(axis=2)

            # x_ab = delta^2 / d^2 (only on adjacent edges)
            x_ab = np.where(adj > 0, delta_sq / (d_sq + EPS_D), 0.0)
            one_minus_x = np.where(adj > 0, 1.0 - x_ab, 0.0)

            # xi_ab^2 — log-deviation; xi_ab = log(Xi_ab) - <log Xi>
            log_xi = np.where(adj > 0, np.log(np.maximum(xi_off, 1e-12)), 0.0)
            log_xi_mean = (log_xi * adj).sum() / (adj.sum() + 1e-12)
            xi_dev = np.where(adj > 0, log_xi - log_xi_mean, 0.0)

            # R^Hess(a)_trace = <x*(1-x)>_w (per node)
            xx_one_minus = x_ab * one_minus_x
            r_bar_node = (weight_adj * xx_one_minus).sum(axis=1) / (
                weight_adj.sum(axis=1) + 1e-12)

            # Pool edges where adj > 0
            mask = adj > 0
            x_ab_pool.append(x_ab[mask])
            one_minus_x_pool.append(one_minus_x[mask])
            xi_dev_sq_pool.append((xi_dev[mask]) ** 2)
            weight_pool.append(weight_adj[mask])
            r_bar_pool.append(r_bar_node)

        x_pool = np.concatenate(x_ab_pool)
        omx_pool = np.concatenate(one_minus_x_pool)
        xi_sq = np.concatenate(xi_dev_sq_pool)
        w = np.concatenate(weight_pool)
        r_bar = np.concatenate(r_bar_pool)

        # Weighted means
        weighted_mean_one_minus_x = float((w * omx_pool).sum() / w.sum())
        weighted_mean_xi_sq = float((w * xi_sq).sum() / w.sum())

        # Step 3 prediction: 1 - x = gamma^2 * xi_dev^2 (leading)
        # So <1-x>_w should be gamma^2 * <xi_dev^2>_w
        predicted_omx = GAMMA_SQ * weighted_mean_xi_sq
        ratio_observed_to_predicted = (weighted_mean_one_minus_x
                                       / max(predicted_omx, 1e-12))

        # R^Hess median per node
        r_bar_median = float(np.median(r_bar))

        # The "implicit normalisation" — does R^Hess match gamma^2*norm?
        norm_implicit = r_bar_median / GAMMA_SQ
        # Compare to 4/pi
        ratio_to_4_over_pi = norm_implicit / FOUR_OVER_PI

        rows.append({
            "regime": reg, "N": n_lat,
            "weighted_mean_1_minus_x": weighted_mean_one_minus_x,
            "weighted_mean_xi_dev_sq": weighted_mean_xi_sq,
            "predicted_1_minus_x_at_gamma_sq": predicted_omx,
            "ratio_observed_to_predicted": ratio_observed_to_predicted,
            "R_bar_Hess_median": r_bar_median,
            "implicit_norm_R_bar_over_gamma_sq": norm_implicit,
            "ratio_implicit_norm_to_4_over_pi": ratio_to_4_over_pi,
        })

    print()
    print(f"{'reg':<10} {'N':>3} | "
          f"{'<1-x>_w':>10} {'γ²·<ξ²>':>11} {'obs/pred':>10} | "
          f"{'R̄_med':>9} {'R̄/γ²':>9} {'/(4/π)':>9}")
    print("-" * 90)
    for r in rows:
        print(f"{r['regime']:<10} {r['N']:>3} | "
              f"{r['weighted_mean_1_minus_x']:>10.5f} "
              f"{r['predicted_1_minus_x_at_gamma_sq']:>11.5f} "
              f"{r['ratio_observed_to_predicted']:>10.3f} | "
              f"{r['R_bar_Hess_median']:>9.5f} "
              f"{r['implicit_norm_R_bar_over_gamma_sq']:>9.4f} "
              f"{r['ratio_implicit_norm_to_4_over_pi']:>9.4f}")

    # Mean ratios
    mean_obs_pred = float(np.mean([r["ratio_observed_to_predicted"] for r in rows]))
    mean_4pi_norm = float(np.mean([r["ratio_implicit_norm_to_4_over_pi"] for r in rows]))
    print()
    print(f"Mean obs / predicted (1-x_ab): {mean_obs_pred:.3f}")
    print(f"  -> Step 3 prediction <1-x> = gamma^2*<xi^2> holds if ratio ~1")
    print()
    print(f"Mean R^Hess_med / (gamma^2 * (4/pi)): {mean_4pi_norm:.3f}")
    print(f"  -> Step 4 prediction holds if implicit norm ~ 4/pi (i.e. ratio ~1)")

    # Verdict
    step3_holds = abs(mean_obs_pred - 1.0) < 0.5
    step4_holds = abs(mean_4pi_norm - 1.0) < 0.5
    print()
    if step3_holds and step4_holds:
        verdict = "DERIVATION_STEPS_3_AND_4_BOTH_HOLD"
    elif step3_holds:
        verdict = "STEP_3_HOLDS_STEP_4_OPEN"
    elif step4_holds:
        verdict = "STEP_4_HOLDS_STEP_3_OPEN"
    else:
        verdict = "NEITHER_STEP_HOLDS_DIRECTLY"
    print(f"VERDICT: {verdict}")

    out_path = REPO / "outputs" / "kappa_t_derivation_step3_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "kappa_t_derivation_step_3_4_lattice_verification",
            "schema_version": "1.0.0",
            "gamma_sq": GAMMA_SQ, "four_over_pi": FOUR_OVER_PI,
            "per_regime": rows,
            "mean_obs_over_predicted_step3": mean_obs_pred,
            "mean_implicit_norm_over_4_over_pi_step4": mean_4pi_norm,
            "verdict": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
