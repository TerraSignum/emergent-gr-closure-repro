"""Signed bulk-vs-core residual balance.

The norm-ratio observable Delta_a = ||R_a||_F / ||T_a||_F squares
everything, so it cannot expose a sign structure. But the
underlying tensor R^{mu nu}_a = G^{mu nu}_a + Lambda^{back, mu nu}_a
- 8 pi G T^{Xi, mu nu}_a is signed. By the discrete Bianchi
identity nabla_mu^{disc} G^{mu nu} -> 0, the integrated trace
sum_a tr(R^{mu nu}_a) must vanish in the continuum, with bulk
and matter-core contributions opposite in sign by conservation.

This audit:
 1. computes the signed per-node tr(R) at every regime in the
    cleaned ten-regime ladder;
 2. partitions nodes into 'core' (top heavy-tail of T_00^Xi) and
    'bulk' (the complement);
 3. reports the per-regime signed sums S_core(N) and S_bulk(N);
 4. verifies S_core + S_bulk -> 0 in the continuum and that
    sign(S_bulk) = -sign(S_core) (i.e. bulk and core are
    opposite-sign by Bianchi).

If the bulk-vs-core sign structure holds, the 'negative' content
of the unconstrained linear-in-1/N fit on the *norm* p_90, p_95
percentiles is the imprint of this signed balance: matter-core
heavy-tail saturates with positive amplitude; bulk carries the
small negative compensating amplitude; the absolute-value
percentile mixes the two, making the fit prefer a negative
extrapolation when the linear model can't follow the
super-linear bulk decay.

Output: outputs/stage6f_signed_bulk_core_balance.json
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

from stage6f_full_tensor_norm_audit import LADDER, gather_regime  # noqa: E402

OUT = REPO / "outputs" / "stage6f_signed_bulk_core_balance.json"

CORE_THRESHOLD_TAU = 0.05  # nodes with delta_full > tau are 'core'


def _signed_trace_diff(prep, lambda_t=0.81, lambda_s=-0.005):
    """tr(R) per node = (G_00 + Lambda_t - T_00) + sum_i (G_ii + Lambda_s - T_ii).

    Signs are preserved (no abs / no square)."""
    g_00 = prep["g_00_h"]
    g_ij = prep["g_ij_h"]
    t00 = prep["t00"]
    t_ij = prep["t_ij"]
    res_00 = g_00 + lambda_t - t00
    eye3 = np.eye(3)
    res_diag = (g_ij + lambda_s * eye3[None, :, :]) - t_ij
    trace_spatial = res_diag[:, 0, 0] + res_diag[:, 1, 1] + res_diag[:, 2, 2]
    return res_00 + trace_spatial  # shape (n_node,)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime, n in LADDER:
        try:
            from verify_galerkin_runner_A_hessian_ricci import per_seed_galerkin
            from _d1_npz_discovery import find_d1_npz
            from stage6f_full_tensor_norm_audit import (
                load_canonical, load_snapshots)
        except ImportError as exc:
            print(f"  import error: {exc}")
            continue
        p = find_d1_npz(regime, REPO)
        if p is None or not p.exists():
            continue
        if "snapshots" in p.name.lower():
            seeds = load_snapshots(p, n)
        else:
            seeds = load_canonical(p, n)
        signed_trace_pool = []
        delta_full_pool = []
        for xi_mat, psi, k_field, q_field in seeds:
            try:
                prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n, np)
            except Exception as exc:  # noqa: BLE001
                continue
            tr = _signed_trace_diff(prep)
            # delta_full from the standard audit, to identify core nodes
            from stage6f_full_tensor_norm_audit import (
                LAMBDA_T, LAMBDA_S, per_node_relative_delta)
            comp = per_node_relative_delta(prep, LAMBDA_T, LAMBDA_S)
            delta_full = comp["delta_full"]
            mask_finite = np.isfinite(tr) & np.isfinite(delta_full)
            signed_trace_pool.append(tr[mask_finite])
            delta_full_pool.append(delta_full[mask_finite])
        if not signed_trace_pool:
            continue
        tr = np.concatenate(signed_trace_pool)
        df = np.concatenate(delta_full_pool)
        mask_core = df > CORE_THRESHOLD_TAU
        n_node = len(tr)
        n_core = int(mask_core.sum())
        n_bulk = n_node - n_core
        s_core = float(tr[mask_core].sum()) / n_node
        s_bulk = float(tr[~mask_core].sum()) / n_node
        s_total = s_core + s_bulk
        rows.append({
            "regime": regime,
            "N": int(n),
            "n_seeds": len(seeds),
            "n_node": n_node,
            "n_core": n_core,
            "n_bulk": n_bulk,
            "core_fraction": n_core / n_node,
            "S_core_per_node": s_core,
            "S_bulk_per_node": s_bulk,
            "S_total_per_node": s_total,
            "sign_S_bulk": int(np.sign(s_bulk)),
            "sign_S_core": int(np.sign(s_core)),
            "opposite_signs": (np.sign(s_bulk) != np.sign(s_core)
                               and s_bulk != 0 and s_core != 0),
        })
        print(f"  {regime:>10s}  N={n:>4d}  "
              f"core_frac={n_core/n_node:.3f}  "
              f"S_core={s_core:+.4f}  S_bulk={s_bulk:+.4f}  "
              f"sum={s_total:+.4f}")

    out = {
        "method": "Signed bulk-vs-core trace-residual balance",
        "core_threshold_tau": CORE_THRESHOLD_TAU,
        "convention_for_sign": (
            "tr(R) = (G_00 + Lambda_t - T_00) + sum_i (G_ii + "
            "Lambda_s - T_ii); positive at matter-cores where "
            "G dominates over T after Lambda compensation."
        ),
        "per_regime": rows,
    }
    if rows:
        # cross-regime sign-balance audit
        opposite_count = sum(1 for r in rows if r["opposite_signs"])
        out["summary"] = {
            "n_regimes": len(rows),
            "regimes_with_opposite_bulk_core_signs": opposite_count,
            "fraction_opposite": opposite_count / len(rows),
            "S_total_per_node_max_abs": max(abs(r["S_total_per_node"])
                                              for r in rows),
            "S_total_per_node_min_abs": min(abs(r["S_total_per_node"])
                                              for r in rows),
        }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {OUT}")
    if rows:
        print()
        s = out["summary"]
        print(f"  {s['regimes_with_opposite_bulk_core_signs']}/"
              f"{s['n_regimes']} regimes show opposite "
              f"S_bulk vs S_core signs")
        print(f"  |S_total/node| range: "
              f"[{s['S_total_per_node_min_abs']:.4e}, "
              f"{s['S_total_per_node_max_abs']:.4e}]")


if __name__ == "__main__":
    main()
