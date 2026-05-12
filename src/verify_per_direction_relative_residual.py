"""Re-evaluate the per-direction closure under the SAME relative
Frobenius normalisation used by the rest of the P4 pipeline:

  Delta_E^per_node(a) = ||G_munu(a) + Lambda_munu - 8 pi G T_munu(a)||_F
                       / ||8 pi G T_munu(a)||_F

This is the threshold-relevant quantity (closure threshold 0.05),
NOT the absolute residual. The previous audit reported absolute
residuals which mis-stated the closure status.

Per node a, in the T-eigenbasis with Lambda_t = 0.81, Lambda_s = -0.005:

  R_00 = G_00 + Lambda_t - T_00
  R_ii = G_(ii) + Lambda_s - lambda_i  (i=1,2,3 spatial)
  R_off = ||(U^T G U)_off||_F

  ||R||_F = sqrt(R_00^2 + sum_i R_ii^2 + R_off^2)
  ||T||_F = sqrt(T_00^2 + sum_i lambda_i^2)
  Delta(a) = ||R||_F / ||T||_F

Closure: Delta_median(N) <= 0.05  AND  Delta_mean(N) <= 0.10.

Output: outputs/per_direction_relative_residual_audit.json
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
from verify_per_eigendirection_residual import (
    per_node_eigendirection_residuals)


REGIMES = [
    ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
    ("P5N128", 128),
]
LAMBDA_T = 0.81
LAMBDA_S = -0.005


def per_node_relative_delta(prep, lambda_t, lambda_s):
    """Per-node relative-Frobenius Delta_E:
       sqrt(R_00^2 + sum R_ii^2 + R_off^2) / sqrt(T_00^2 + sum lambda_i^2)
    """
    res = per_node_eigendirection_residuals(prep, lambda_t, lambda_s)
    R_time = res["R_time"]                              # (n,)
    R_diag = res["R_diag"]                              # (n, 3)
    R_off = res["R_off"]                                # (n,)
    t_eigs = res["T_eigvals"]                           # (n, 3)
    t00 = np.asarray(prep["t00"])                       # (n,)

    R_norm = np.sqrt(R_time ** 2
                      + (R_diag ** 2).sum(axis=1)
                      + R_off ** 2)
    T_norm = np.sqrt(t00 ** 2 + (t_eigs ** 2).sum(axis=1))
    delta = R_norm / np.maximum(T_norm, 1e-12)
    return delta


def gather_regime(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)
    pool = []
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        delta = per_node_relative_delta(prep, LAMBDA_T, LAMBDA_S)
        pool.append(delta)
    return np.concatenate(pool)


def power_law(N, y):
    if np.any(y <= 0) or len(N) < 3:
        return float("nan"), float("nan")
    log_N, log_y = np.log(N), np.log(y)
    slope, intercept = np.polyfit(log_N, log_y, 1)
    pred = slope * log_N + intercept
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(-slope), float(r2)


def main() -> int:
    print("=" * 110)
    print("Per-direction RELATIVE Frobenius residual (matches existing pipeline normalisation)")
    print("Closure threshold: Delta_median <= 0.05 AND Delta_mean <= 0.10")
    print("=" * 110)
    print()
    print(f"  Lambda_t = {LAMBDA_T}, Lambda_s = {LAMBDA_S}")
    print()
    print(f"{'reg':<8} {'N':>3} | "
          f"{'Delta_med':>10} {'Delta_mean':>11} {'Delta_p90':>10} {'Delta_max':>10} | "
          f"{'med <= 0.05':>11} {'mean <= 0.10':>13}")
    print("-" * 100)

    rows = []
    for reg, n_lat in REGIMES:
        d = gather_regime(reg, n_lat)
        if d is None:
            continue
        med = float(np.median(d))
        mean = float(d.mean())
        p90 = float(np.percentile(d, 90))
        dmax = float(d.max())
        med_pass = med <= 0.05
        mean_pass = mean <= 0.10
        rows.append({
            "regime": reg, "N": n_lat,
            "delta_median": med,
            "delta_mean": mean,
            "delta_p90": p90,
            "delta_max": dmax,
            "median_passes_threshold_0p05": med_pass,
            "mean_passes_threshold_0p10": mean_pass,
        })
        print(f"{reg:<8} {n_lat:>3} | "
              f"{med:>10.4f} {mean:>11.4f} {p90:>10.4f} {dmax:>10.4f} | "
              f"{'PASS' if med_pass else 'FAIL':>11} "
              f"{'PASS' if mean_pass else 'FAIL':>13}")

    # Power-law fit
    N_arr = np.array([r["N"] for r in rows], dtype=float)
    med_arr = np.array([r["delta_median"] for r in rows])
    mean_arr = np.array([r["delta_mean"] for r in rows])

    a_med, r2_med = power_law(N_arr, med_arr)
    a_mean, r2_mean = power_law(N_arr, mean_arr)

    print()
    print("Power-law fits  Delta(N) ~ N^(-alpha):")
    print(f"  Delta_median: alpha = {a_med:>+5.2f},  R^2 = {r2_med:>4.2f}")
    print(f"  Delta_mean:   alpha = {a_mean:>+5.2f},  R^2 = {r2_mean:>4.2f}")
    print()

    all_med_pass = all(r["median_passes_threshold_0p05"] for r in rows)
    all_mean_pass = all(r["mean_passes_threshold_0p10"] for r in rows)
    if all_med_pass and all_mean_pass:
        verdict = "RELATIVE_CLOSURE_HOLDS_AT_ALL_N"
    elif all_med_pass:
        verdict = "MEDIAN_CLOSURE_HOLDS_MEAN_THRESHOLD_OPEN"
    else:
        # find largest N where median passes
        passing = [r for r in rows if r["median_passes_threshold_0p05"]]
        verdict = (f"MEDIAN_CLOSURE_HOLDS_FROM_N>={min(p['N'] for p in passing)}"
                   if passing else "RELATIVE_CLOSURE_FAILS")

    print(f"OVERALL VERDICT: {verdict}")

    out_path = REPO / "outputs" / "per_direction_relative_residual_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "per_node_relative_frobenius_residual",
            "schema_version": "1.0.0",
            "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
            "thresholds": {"median_max": 0.05, "mean_max": 0.10},
            "per_regime": rows,
            "delta_median_power_law": {"alpha": a_med, "r_squared": r2_med},
            "delta_mean_power_law": {"alpha": a_mean, "r_squared": r2_mean},
            "verdict": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
