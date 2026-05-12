"""Sweep candidate Lambda_t values against the R_time_median offset
observed in outputs/closure_median_convergence.json.

Candidates tested:
  - 0.81                (System-R structural, alpha_xi^2 = 81/100)
  - 0.811               (landings_expected.json L9 target_value)
  - 0.8122014327022157  (actuals.json h113 representative_tau_weight,
                         possible Lambda_t empirical realisation)
  - 0.8164157774914081  (user-provided alternative)
  - 0.821               (mean of best-fit Lambda_t_med across N)

For each candidate, compute |R_time_median(N)| and power-law fit.
Best candidate minimises the maximum |R_time_median(N)| AND has
the lowest power-law-fit residual.

Output: outputs/lambda_t_candidate_sweep.json
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
    ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]

CANDIDATES = {
    "system_R_idealised_0.81":            0.81,
    "L9_target_0.811":                    0.811,
    "h113_tau_0.8122014327022157":        0.8122014327022157,
    "spread_0.8120774690918408":          0.8120774690918408,
    "user_alt_0.8164157774914081":        0.8164157774914081,
    "lambda_asymp_0.8184770454425502":    0.8184770454425502,
    "w_eff_kzm_0.819":                    0.819,
    "lambda_t_best_avg_0.821":            0.821,
}


def gather(regime, n_lat):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)

    g00_pool, t00_pool = [], []
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        g00_pool.append(np.asarray(prep["g_00_h"]))
        t00_pool.append(np.asarray(prep["t00"]))
    return np.concatenate(g00_pool), np.concatenate(t00_pool)


def power_law(N, y):
    if np.any(y <= 0) or len(N) < 3:
        return float("nan"), float("nan"), float("nan")
    log_N, log_y = np.log(N), np.log(y)
    slope, intercept = np.polyfit(log_N, log_y, 1)
    pred = slope * log_N + intercept
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(-slope), float(np.exp(intercept)), float(r2)


def main() -> int:
    pool = []
    for reg, n_lat in REGIMES:
        gt = gather(reg, n_lat)
        if gt is None:
            continue
        g00, t00 = gt
        pool.append({"regime": reg, "N": n_lat, "g00": g00, "t00": t00})

    print("=" * 110)
    print("Lambda_t candidate sweep on R_time median residual")
    print("=" * 110)
    print()
    header = f"{'reg':<8} {'N':>3}"
    for c_name in CANDIDATES:
        header += f" | {c_name[:18]:>18}"
    print(header)
    print("-" * len(header))

    rows_per_cand = {c: {"med": [], "max": []} for c in CANDIDATES}
    rows_per_regime = []
    for r in pool:
        line = f"{r['regime']:<8} {r['N']:>3}"
        per_reg = {"regime": r["regime"], "N": r["N"]}
        for c_name, lam_t in CANDIDATES.items():
            res = r["g00"] + lam_t - r["t00"]
            med = float(np.median(np.abs(res)))
            mean_abs = float(np.mean(np.abs(res)))
            line += f" | {med:>18.6f}"
            rows_per_cand[c_name]["med"].append(med)
            per_reg[f"{c_name}_R_time_median"] = med
            per_reg[f"{c_name}_R_time_mean_abs"] = mean_abs
        print(line)
        rows_per_regime.append(per_reg)

    print()
    print("Power-law fit per candidate: |R_time_med(N)| ~ A * N^(-alpha)")
    print(f"{'candidate':<32} {'alpha':>10} {'A':>14} {'R^2':>8} {'max(|R|)':>12} {'mean(|R|)':>12}")
    print("-" * 100)

    candidate_summary = {}
    N_arr = np.array([r["N"] for r in pool], dtype=float)
    for c_name in CANDIDATES:
        meds = np.array(rows_per_cand[c_name]["med"])
        alpha, A, r2 = power_law(N_arr, meds)
        max_med = float(meds.max())
        mean_med = float(meds.mean())
        print(f"{c_name:<32} {alpha:>10.3f} {A:>14.6f} {r2:>8.3f} "
              f"{max_med:>12.6f} {mean_med:>12.6f}")
        candidate_summary[c_name] = {
            "lambda_t_value": CANDIDATES[c_name],
            "alpha": alpha, "A": A, "r_squared": r2,
            "max_R_time_median_across_N": max_med,
            "mean_R_time_median_across_N": mean_med,
            "per_N_values": [
                {"N": int(n), "R_time_median_abs": float(meds[i])}
                for i, n in enumerate(N_arr)
            ],
        }

    # Best candidate by smallest mean_R_time_median
    best = min(candidate_summary.items(),
               key=lambda kv: kv[1]["mean_R_time_median_across_N"])
    print()
    print(f"BEST candidate (smallest mean |R_time_med| across N): {best[0]}")
    print(f"  lambda_t = {best[1]['lambda_t_value']}")
    print(f"  mean |R_time_med| = {best[1]['mean_R_time_median_across_N']:.6f}")

    out_path = REPO / "outputs" / "lambda_t_candidate_sweep.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "lambda_t_candidate_sweep_on_R_time_median",
            "schema_version": "1.0.0",
            "candidates": CANDIDATES,
            "per_regime": rows_per_regime,
            "candidate_summary": candidate_summary,
            "best_candidate": best[0],
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
