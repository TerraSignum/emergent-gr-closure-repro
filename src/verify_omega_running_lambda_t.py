"""Test whether an omega_a-running Lambda_t absorbs the R_time
median stagnation observed in the per-eigendirection audit.

Motivation: outputs/R_time_bias_source_audit.json shows
  Pearson(residual_00, omega_a) = -0.559, r^2 = 0.31
across 1720 nodes pooled over 6 regimes. omega_a is the
weighted node-degree (sum_b w_ab / d_ab^2) -- a Ricci-related
density measure. Higher omega_a nodes have systematically
more negative residual_00, suggesting an omega-running
Lambda_t component:

  Lambda_t(a; kappa) = Lambda_t_struct * (1 + kappa * (omega_a / <omega> - 1))

This is the time-time-component of the O5 spatial-running
cosmological-tensor; structurally well-motivated by System-R
(kappa = gamma^2 = 1/100 = 0.01).

Sweep kappa in {0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0} and for each
kappa, compute |R_time_median(N)| and power-law fit alpha, R^2.

A kappa with positive alpha AND R^2 >= 0.7 closes the time-time
median structurally.

Output: outputs/omega_running_lambda_t_audit.json
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
    edge_to_matrix, per_seed_galerkin, XI_THRESH, ELL_0, D_MIN, EPS_D)


REGIMES = [
    ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]

LAMBDA_T_STRUCT = 0.81
KAPPAS = [0.0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0]


def compute_omega(xi_mat, n_lat):
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    weight_adj = xi_off * adj
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.maximum(d_mat, D_MIN)
    d_sq = d_mat * d_mat
    d_sq_safe = np.where(adj > 0, d_sq, np.inf)
    weight_grad = np.where(adj > 0, weight_adj / (d_sq_safe + EPS_D), 0.0)
    return weight_grad.sum(axis=1)


def gather_per_regime():
    """For each regime, return arrays of (g_00, t00, omega) pooled across seeds."""
    pool = []
    for reg, n_lat in REGIMES:
        p = find_d1_npz(reg, REPO)
        if p is None or not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        edge_arr = d["dense_cell_edge_xi_values"]
        amp_arr = d["dense_cell_node_amplitude_values"]
        phase_arr = d["dense_cell_node_phase_values"]
        n_seeds = min(edge_arr.shape[0], 32)
        g00s, t00s, omegs = [], [], []
        for s in range(n_seeds):
            xi_mat = edge_to_matrix(edge_arr[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            psi = amp_arr[s] * np.exp(1j * phase_arr[s])
            k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
            q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
            prep = per_seed_galerkin(xi_mat, psi, k_field, q_field,
                                       n_lat, np)
            g00s.append(np.asarray(prep["g_00_h"]))
            t00s.append(np.asarray(prep["t00"]))
            omegs.append(compute_omega(xi_mat, n_lat))
        pool.append({
            "regime": reg, "N": n_lat,
            "g00": np.concatenate(g00s),
            "t00": np.concatenate(t00s),
            "omega": np.concatenate(omegs),
        })
    return pool


def power_law(N, y):
    if np.any(y <= 0) or len(N) < 3:
        return float("nan"), float("nan")
    log_N, log_y = np.log(N), np.log(y)
    slope, _ = np.polyfit(log_N, log_y, 1)
    pred = np.polyval([slope, np.polyfit(log_N, log_y, 1)[1]], log_N)
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(-slope), float(r2)


def main() -> int:
    pool = gather_per_regime()
    N_arr = np.array([r["N"] for r in pool], dtype=float)

    print("=" * 110)
    print("Omega-running Lambda_t sweep")
    print("Lambda_t(a; kappa) = 0.81 * (1 + kappa * (omega_a/<omega> - 1))")
    print("=" * 110)
    print()
    header = f"{'kappa':>7}"
    for r in pool:
        header += f" | N={int(r['N']):>3}"
    header += " | mean(|R|) | alpha | R^2"
    print(header)
    print("-" * len(header))

    summary = {}
    for kappa in KAPPAS:
        meds = []
        per_N = []
        for r in pool:
            omega_norm = r["omega"] / max(np.mean(r["omega"]), 1e-12)
            lam_t = LAMBDA_T_STRUCT * (1.0 + kappa * (omega_norm - 1.0))
            resid = r["g00"] + lam_t - r["t00"]
            med = float(np.median(np.abs(resid)))
            meds.append(med)
            per_N.append({"N": int(r["N"]), "R_time_median_abs": med})
        meds = np.array(meds)
        alpha, r2 = power_law(N_arr, meds)
        line = f"{kappa:>7.4f}"
        for m in meds:
            line += f" | {m:>5.4f}"
        line += f" | {meds.mean():>8.5f} | {alpha:>+5.2f} | {r2:>4.2f}"
        print(line)
        summary[f"kappa_{kappa}"] = {
            "kappa": kappa,
            "per_N": per_N,
            "mean_R_time_median": float(meds.mean()),
            "max_R_time_median": float(meds.max()),
            "power_law_alpha": alpha, "power_law_R_squared": r2,
        }

    # Best kappa: smallest mean residual
    best = min(summary.items(),
               key=lambda kv: kv[1]["mean_R_time_median"])
    # Best with positive alpha
    pos_alpha = [(k, v) for k, v in summary.items() if v["power_law_alpha"] > 0]
    if pos_alpha:
        best_pl = min(pos_alpha, key=lambda kv: kv[1]["mean_R_time_median"])
    else:
        best_pl = None

    print()
    print(f"BEST kappa by smallest mean |R_time_med|: {best[0]}")
    print(f"  kappa = {best[1]['kappa']}")
    print(f"  mean |R_time_med| = {best[1]['mean_R_time_median']:.6f}")
    print(f"  alpha = {best[1]['power_law_alpha']:+.3f}, R^2 = {best[1]['power_law_R_squared']:.3f}")
    if best_pl:
        print(f"\nBest kappa with POSITIVE alpha: {best_pl[0]}")
        print(f"  kappa = {best_pl[1]['kappa']}")
        print(f"  mean |R_time_med| = {best_pl[1]['mean_R_time_median']:.6f}")
        print(f"  alpha = {best_pl[1]['power_law_alpha']:+.3f}, R^2 = {best_pl[1]['power_law_R_squared']:.3f}")
    else:
        print("\nNO kappa produces positive alpha (true N-convergence).")

    out_path = REPO / "outputs" / "omega_running_lambda_t_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "omega_running_lambda_t_kappa_sweep",
            "schema_version": "1.0.0",
            "lambda_t_struct": LAMBDA_T_STRUCT,
            "kappas_swept": KAPPAS,
            "per_kappa": summary,
            "best_by_min_residual": best[0],
            "best_with_positive_alpha": best_pl[0] if best_pl else None,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
