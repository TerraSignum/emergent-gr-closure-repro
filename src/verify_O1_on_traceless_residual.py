"""Test whether the System-R higher-order Hessian-Ricci correction
(O1) reduces the TRACELESS spatial residual specifically.

Per the per-eigendirection decomposition
(outputs/per_eigendirection_residual_audit.json), 89% of the
mean per-node Frobenius residual at all regimes lives in the
tensorial-traceless part of the spatial diagonal block. Isotropic
Lambda_ij CANNOT absorb this contribution. The next-order
Hessian-Ricci correction
  R^(2)_ij(a) propto sum_b w_ab e_i e_j (1 - delta^2/d^2)^2
is tensorial: it can in principle reduce the traceless residual.

This audit measures the traceless and trace contributions to the
mean-square residual, BEFORE and AFTER applying the O1 correction
to G_ij, on the full canonical lattice ladder.

Output: outputs/O1_on_traceless_residual_audit.json
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
from verify_higher_order_terms_all8 import (
    LAMBDA_T, LAMBDA_S, hessian_ricci_quadratic)
from verify_per_eigendirection_residual import (
    per_node_eigendirection_residuals)


REGIMES_TO_TEST = [
    ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]


def evaluate_with_optional_O1(regime, n_lat, apply_O1):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)

    pool_R_TF, pool_R_trace, pool_R_time, pool_R_off = [], [], [], []
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        if apply_O1:
            g_00_O1, g_ij_O1 = hessian_ricci_quadratic(prep, xi_mat, n_lat)
            prep = dict(prep)  # shallow copy of view
            prep["g_00_h"] = g_00_O1
            prep["g_ij_h"] = g_ij_O1
        res = per_node_eigendirection_residuals(prep, LAMBDA_T, LAMBDA_S)
        pool_R_TF.append(res["R_TF_norm"])
        pool_R_trace.append(res["R_trace"])
        pool_R_time.append(res["R_time"])
        pool_R_off.append(res["R_off"])

    R_TF = np.concatenate(pool_R_TF)
    R_trace = np.concatenate(pool_R_trace)
    R_time = np.concatenate(pool_R_time)
    R_off = np.concatenate(pool_R_off)

    return {
        "regime": regime, "N": n_lat,
        "with_O1": bool(apply_O1),
        "R_TF_norm_mean_sq": float((R_TF ** 2).mean()),
        "R_TF_norm_median_abs": float(np.median(np.abs(R_TF))),
        "R_trace_mean_sq": float((R_trace ** 2).mean()),
        "R_trace_median_abs": float(np.median(np.abs(R_trace))),
        "R_time_mean_sq": float((R_time ** 2).mean()),
        "R_time_median_abs": float(np.median(np.abs(R_time))),
        "R_off_mean_sq": float((R_off ** 2).mean()),
        "R_off_median_abs": float(np.median(np.abs(R_off))),
    }


def main():
    print("=" * 110)
    print("Does the O1 correction (quadratic Hessian-Ricci) reduce the TRACELESS spatial residual?")
    print("=" * 110)
    print()
    print(f"{'reg':<8} {'N':>3} | {'baseline TF^2':>14} {'with O1 TF^2':>14} {'TF reduction':>14} | "
          f"{'baseline trace^2':>17} {'with O1 trace^2':>16}")
    print("-" * 100)

    results = []
    for reg, n_lat in REGIMES_TO_TEST:
        base = evaluate_with_optional_O1(reg, n_lat, apply_O1=False)
        with_O1 = evaluate_with_optional_O1(reg, n_lat, apply_O1=True)
        if base is None or with_O1 is None:
            continue
        TF_red = (with_O1["R_TF_norm_mean_sq"] - base["R_TF_norm_mean_sq"]) \
                  / max(base["R_TF_norm_mean_sq"], 1e-12) * 100
        trace_red = (with_O1["R_trace_mean_sq"] - base["R_trace_mean_sq"]) \
                     / max(base["R_trace_mean_sq"], 1e-12) * 100
        time_red = (with_O1["R_time_mean_sq"] - base["R_time_mean_sq"]) \
                    / max(base["R_time_mean_sq"], 1e-12) * 100
        results.append({
            "regime": reg, "N": n_lat,
            "baseline": base, "with_O1": with_O1,
            "TF_change_pct": TF_red,
            "trace_change_pct": trace_red,
            "time_change_pct": time_red,
        })
        print(f"{reg:<8} {n_lat:>3} | "
              f"{base['R_TF_norm_mean_sq']:>14.6f} "
              f"{with_O1['R_TF_norm_mean_sq']:>14.6f} "
              f"{TF_red:>+13.1f}% | "
              f"{base['R_trace_mean_sq']:>17.6f} "
              f"{with_O1['R_trace_mean_sq']:>16.6f}")

    print()
    print("=" * 110)
    print("VERDICT")
    print("=" * 110)
    print(f"{'regime':<8} {'N':>3} | {'TF^2 change':>13} {'trace^2 change':>15} {'time^2 change':>14}")
    for r in results:
        print(f"  {r['regime']:<8} {r['N']:>3}: "
              f"TF^2 {r['TF_change_pct']:>+8.1f}%, "
              f"trace^2 {r['trace_change_pct']:>+8.1f}%, "
              f"time^2 {r['time_change_pct']:>+8.1f}%")
    print()
    print("  Interpretation:")
    print("    TF^2 change << 0  -> O1 is reducing the traceless residual structurally.")
    print("    TF^2 change ~ 0   -> O1 doesn't help with the traceless 89% of residual.")
    print("    TF^2 change > 0   -> O1 makes the traceless WORSE (counter-productive).")

    out = REPO / "outputs" / "O1_on_traceless_residual_audit.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "method": "O1_effect_on_residual_decomposition",
            "schema_version": "1.0.0",
            "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
            "per_regime": results,
        }, f, indent=2, default=str)
    print()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
