"""Direct lattice test: does <delta^2_ab/d^2_ab>_w_ab converge
to gamma^2 * (4/pi) = 0.01273 in the N->infty limit?

This tests an alternative derivation route: instead of the
falsified expansion (1 - delta^2/d^2 ~ gamma^2 xi^2), check
if the OBSERVED value <delta^2/d^2>_w directly extrapolates
to gamma^2 * (4/pi).

If yes: we have a direct lattice identity, the Galerkin
spectral-flat / relational-metric ratio asymptotically equals
the topological projection gamma^2 * (4/pi).

If no: the gamma^2 * (4/pi) match remains empirical-only.

Output: outputs/delta_d_ratio_continuum_audit.json
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
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P5N72", 72), ("P7", 72), ("P5N84", 84), ("P8", 84),
    ("P5N100", 100), ("P6N128", 128), ("P8N128", 128),
]

GAMMA_SQ_FOUR_OVER_PI = (1.0/10.0)**2 * 4.0 / np.pi  # 0.01273


def main() -> int:
    print("=" * 100)
    print(f"Lattice continuum-limit test: <delta^2/d^2>_w -> gamma^2*(4/pi) = {GAMMA_SQ_FOUR_OVER_PI:.5f} ?")
    print("=" * 100)
    print()
    print(f"{'reg':<10} {'N':>3} | "
          f"{'<delta^2/d^2>_w':>17} {'pred (=γ²·4/π)':>16} {'rel err %':>10}")
    print("-" * 70)

    rows = []
    for reg, n_lat in REGIMES:
        p = find_d1_npz(reg, REPO)
        if p is None or not p.exists(): continue
        d = np.load(p, allow_pickle=True)
        edge_arr = d["dense_cell_edge_xi_values"]
        n_seeds = min(edge_arr.shape[0], 32)
        x_pool, w_pool = [], []
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

            deg = weight_adj.sum(axis=1) + 1e-12
            deg_inv_sqrt = 1.0 / np.sqrt(deg)
            l_norm = (np.eye(n_lat) - deg_inv_sqrt[:, None]
                      * weight_adj * deg_inv_sqrt[None, :])
            try:
                eigvals_l, eigvecs_l = np.linalg.eigh(l_norm)
            except np.linalg.LinAlgError:
                continue
            spatial = eigvecs_l[:, 1:4]
            spatial_diff = spatial[None, :, :] - spatial[:, None, :]
            delta_sq = (spatial_diff ** 2).sum(axis=2)
            x_ab = np.where(adj > 0, delta_sq / (d_sq + EPS_D), 0.0)
            mask = adj > 0
            if not np.any(np.isfinite(x_ab[mask])):
                continue
            x_pool.append(x_ab[mask])
            w_pool.append(weight_adj[mask])
        if not x_pool:
            continue
        x_all = np.concatenate(x_pool); w_all = np.concatenate(w_pool)
        finite = np.isfinite(x_all)
        x_all = x_all[finite]; w_all = w_all[finite]
        weighted_mean_x = float((w_all * x_all).sum() / w_all.sum())
        rel_err = (weighted_mean_x - GAMMA_SQ_FOUR_OVER_PI) / GAMMA_SQ_FOUR_OVER_PI * 100
        rows.append({
            "regime": reg, "N": n_lat,
            "weighted_mean_x_ab": weighted_mean_x,
            "predicted": GAMMA_SQ_FOUR_OVER_PI,
            "rel_err_percent": rel_err,
        })
        print(f"{reg:<10} {n_lat:>3} | "
              f"{weighted_mean_x:>17.5f} {GAMMA_SQ_FOUR_OVER_PI:>16.5f} {rel_err:>+10.1f}")

    # Continuum-limit fit: x(N) = x_inf + c/N^2 (Symanzik)
    if len(rows) >= 4:
        N_arr = np.array([r["N"] for r in rows], dtype=float)
        x_arr = np.array([r["weighted_mean_x_ab"] for r in rows])
        X = np.column_stack([np.ones_like(N_arr), N_arr ** -2.0])
        c, *_ = np.linalg.lstsq(X, x_arr, rcond=1e-10)
        x_inf = float(c[0])
        rel_err_inf = (x_inf - GAMMA_SQ_FOUR_OVER_PI) / GAMMA_SQ_FOUR_OVER_PI * 100
        print()
        print(f"Symanzik 2-term fit: x(N) = {x_inf:.5f} + {c[1]:+.2f}/N^2")
        print(f"Continuum limit x_inf = {x_inf:.5f}")
        print(f"Predicted gamma^2*(4/pi) = {GAMMA_SQ_FOUR_OVER_PI:.5f}")
        print(f"Relative error at x_inf vs prediction: {rel_err_inf:+.2f}%")
    else:
        x_inf = float("nan")
        rel_err_inf = float("nan")

    # Verdict
    if abs(rel_err_inf) < 5:
        verdict = "LATTICE_CONFIRMS_GAMMA_SQ_OVER_4_PI_AT_5_PERCENT"
    elif abs(rel_err_inf) < 20:
        verdict = "PARTIAL_AGREEMENT_AT_20_PERCENT"
    else:
        verdict = "NO_DIRECT_LATTICE_CONFIRMATION"
    print(f"\nVERDICT: {verdict}")

    out = {
        "method": "direct_lattice_continuum_test_delta_d_ratio_vs_gamma_sq_4_pi",
        "schema_version": "1.0.0",
        "predicted_value": GAMMA_SQ_FOUR_OVER_PI,
        "per_regime": rows,
        "symanzik_continuum_limit": x_inf,
        "rel_err_continuum_vs_prediction": rel_err_inf,
        "verdict": verdict,
    }
    out_path = REPO / "outputs" / "delta_d_ratio_continuum_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
