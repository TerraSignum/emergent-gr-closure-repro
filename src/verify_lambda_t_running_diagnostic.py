"""Diagnose the R_time_median stagnation observed in
outputs/closure_median_convergence.json:

  alpha(R_time_median) = -0.16,  R^2 = 0.07,  median ~ 0.018 - 0.025

While R_TF_median ~ N^(-2.22), R_off_median ~ N^(-1.45), and
R_trace_median ~ N^(-0.53) all converge, the time-time median
sits at a fixed offset. This script checks whether the offset
disappears under a regime-specific best-fit Lambda_t(N), and
whether the resulting Lambda_t(N) trend is consistent with
Lambda_t -> 0.81 (System-R structural value) at N -> infty,
or whether a Lambda_t-running of the form

  Lambda_t(N) = Lambda_t^infty + B / N^p

is implied.

Per regime, fit the best Lambda_t such that
median(G_00(a) + Lambda_t - 8 pi G T_00(a)) -> 0 over the seed pool,
then read off Lambda_t(N) and trend it.

Output: outputs/lambda_t_running_diagnostic.json
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


REGIMES_TO_TEST = [
    ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]


def gather_g00_t00(regime, n_lat):
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
        g00 = np.asarray(prep["g_00_h"])
        t00 = np.asarray(prep["t00"])
        g00_pool.append(g00)
        t00_pool.append(t00)
    return (np.concatenate(g00_pool), np.concatenate(t00_pool))


def best_lambda_t(g00, t00, eight_pi_G=1.0):
    """Find Lambda_t that puts the per-node residual median to zero:
       median(g00 + Lambda_t - 8 pi G * t00) = 0
       <=> Lambda_t = - median(g00 - 8 pi G * t00)
    """
    return float(-np.median(g00 - eight_pi_G * t00))


def best_lambda_t_mean(g00, t00, eight_pi_G=1.0):
    return float(-np.mean(g00 - eight_pi_G * t00))


def main() -> int:
    LAMBDA_T_STRUCTURAL = 0.81

    rows = []
    print("=" * 100)
    print("Lambda_t(N) running diagnostic")
    print("Equation: median(G_00(a) + Lambda_t - 8 pi G * T_00(a)) = 0  ->  best Lambda_t(N)")
    print("=" * 100)
    print()
    print(f"{'reg':<8} {'N':>3} | {'Lambda_t_med':>14} {'Lambda_t_mean':>14} "
          f"{'res @ 0.81 med':>16} {'res @ 0.81 mean':>17}")
    print("-" * 90)

    for reg, n_lat in REGIMES_TO_TEST:
        gt = gather_g00_t00(reg, n_lat)
        if gt is None:
            continue
        g00, t00 = gt
        lam_med = best_lambda_t(g00, t00)
        lam_mean = best_lambda_t_mean(g00, t00)
        res_at_struct_med = float(np.median(g00 + LAMBDA_T_STRUCTURAL - t00))
        res_at_struct_mean = float(np.mean(g00 + LAMBDA_T_STRUCTURAL - t00))
        rows.append({
            "regime": reg, "N": n_lat,
            "lambda_t_med": lam_med,
            "lambda_t_mean": lam_mean,
            "residual_at_lambda_struct_median": res_at_struct_med,
            "residual_at_lambda_struct_mean": res_at_struct_mean,
            "g00_median": float(np.median(g00)),
            "t00_median": float(np.median(t00)),
            "g00_mean": float(np.mean(g00)),
            "t00_mean": float(np.mean(t00)),
        })
        print(f"{reg:<8} {n_lat:>3} | "
              f"{lam_med:>14.6f} {lam_mean:>14.6f} "
              f"{res_at_struct_med:>16.6f} {res_at_struct_mean:>17.6f}")

    # Power-law fit: Lambda_t(N) = Lambda_inf + B * N^(-p)
    if len(rows) >= 3:
        N_arr = np.array([r["N"] for r in rows], dtype=float)
        lam_med_arr = np.array([r["lambda_t_med"] for r in rows])
        lam_mean_arr = np.array([r["lambda_t_mean"] for r in rows])

        # 2-param fit (assume Lambda_inf = 0.81 from System-R)
        # Lambda_t(N) - 0.81 = B * N^(-p)
        for label, arr in [("median-closure", lam_med_arr), ("mean-closure", lam_mean_arr)]:
            delta = arr - LAMBDA_T_STRUCTURAL
            # Fit log|delta| = log|B| - p log N (assume positive corrections; otherwise check sign)
            if np.all(np.abs(delta) > 1e-9):
                slope, intercept = np.polyfit(np.log(N_arr), np.log(np.abs(delta)), 1)
                p = -slope
                B = float(np.sign(delta[-1]) * np.exp(intercept))
                pred = LAMBDA_T_STRUCTURAL + B * N_arr ** (-p)
                resid = arr - pred
                ss_res = float((resid ** 2).sum())
                ss_tot = float(((arr - arr.mean()) ** 2).sum())
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                print(f"\n  {label}: |Lambda_t(N) - 0.81| ~ N^(-p)")
                print(f"     p = {p:.3f}, B = {B:.6f}, R^2 = {r2:.3f}")
                print(f"     Lambda_t(infty) extrapolated = {LAMBDA_T_STRUCTURAL + B * 1e6 ** (-p):.6f}")
            else:
                p, B, r2 = float("nan"), float("nan"), float("nan")
                print(f"\n  {label}: Lambda_t already at 0.81 (degenerate fit)")

    print()
    print("Interpretation:")
    print("  Lambda_t_med(N) approaching 0.81 with N -> infty signals")
    print("  consistency of the median closure with the System-R structural value.")
    print("  A nonzero Lambda_t-running term (Lambda_t = 0.81 + B/N^p)")
    print("  would be the Lambda_t-running corrective term implied by the")
    print("  R_time_median stagnation in the median-closure audit.")

    out_path = REPO / "outputs" / "lambda_t_running_diagnostic.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "lambda_t_running_per_regime",
            "schema_version": "1.0.0",
            "lambda_t_structural": LAMBDA_T_STRUCTURAL,
            "per_regime": rows,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
