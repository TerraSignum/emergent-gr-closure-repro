"""Identify the structural source of the R_time_median stagnation
at offset ~0.018 across all Lambda_t candidates.

Per node a:
  residual_00(a) = G_00(a) + Lambda_t_struct - T_00(a)
                 = R_bar(a)/2 + 0.81 - T_00(a)

The median |residual_00| stagnates at ~0.018 with no N-power-law for
any Lambda_t in {0.81, 0.811, 0.8122, 0.8164, 0.8185, 0.819, 0.821, 0.8114}.

Pearson correlation between residual_00(a) and per-node structural
features identifies which structural quantity systematically biases
the time-time component:

  - omega_a:    weighted node-degree (sum_b w_ab / d_ab^2)
  - r_bar_h(a): Hessian-Ricci scalar trace
  - |psi_a|^2:  matter amplitude squared
  - phase_var(a): local phase fluctuation
  - K_field(a): recombination-field row mean
  - Q_field(a): Q-field row mean
  - degree(a):  unweighted graph degree
  - xi_max(a):  max edge xi
  - xi_min(a):  min nonzero edge xi
  - xi_var(a):  variance of nonzero edges
  - boundary distance: 1 - (active edges/N) (approximate boundary proxy)

A |Pearson r| >= 0.6 with one feature  ->  candidate missing term.
A |Pearson r| < 0.3 across all  ->  bias is unstructured, not a
single missing term.

Output: outputs/R_time_bias_source_audit.json
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

LAMBDA_T = 0.81  # System-R structural reference for residual definition


def per_node_features(xi_mat, psi, k_field, q_field, n_lat):
    """Compute per-node structural feature vectors."""
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    weight_adj = xi_off * adj
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.maximum(d_mat, D_MIN)
    d_sq = d_mat * d_mat
    d_sq_safe = np.where(adj > 0, d_sq, np.inf)
    weight_grad = np.where(adj > 0, weight_adj / (d_sq_safe + EPS_D), 0.0)

    omega_a = weight_grad.sum(axis=1)
    degree = adj.sum(axis=1)
    psi_sq = (np.abs(psi) ** 2)
    phase = np.angle(psi)
    # Local phase variance: variance of phase[a] - phase[b] over neighbours
    phase_var = np.zeros(n_lat)
    for a in range(n_lat):
        nb = np.where(adj[a] > 0)[0]
        if len(nb) >= 2:
            diffs = (phase[nb] - phase[a] + np.pi) % (2 * np.pi) - np.pi
            phase_var[a] = float(diffs.var())
    k_row = (k_field * adj).sum(axis=1) / np.maximum(degree, 1)
    q_row = (q_field * adj).sum(axis=1) / np.maximum(degree, 1)
    xi_max = np.where(degree > 0, xi_off.max(axis=1), 0.0)
    xi_min = np.array([
        xi_off[a, adj[a] > 0].min() if degree[a] > 0 else 0.0
        for a in range(n_lat)
    ])
    xi_var = np.array([
        xi_off[a, adj[a] > 0].var() if degree[a] > 1 else 0.0
        for a in range(n_lat)
    ])
    boundary_proxy = 1.0 - degree / max(n_lat - 1, 1)

    return {
        "omega_a":        omega_a,
        "degree":         degree,
        "psi_sq":         psi_sq,
        "phase_var":      phase_var,
        "K_row":          k_row,
        "Q_row":          q_row,
        "xi_max":         xi_max,
        "xi_min":         xi_min,
        "xi_var":         xi_var,
        "boundary_proxy": boundary_proxy,
    }


def gather_residual_and_features():
    pool_resid = []
    pool_feats = {k: [] for k in [
        "omega_a", "degree", "psi_sq", "phase_var",
        "K_row", "Q_row", "xi_max", "xi_min", "xi_var",
        "boundary_proxy", "r_bar_h", "g_00", "t00", "N_value",
    ]}
    for reg, n_lat in REGIMES:
        p = find_d1_npz(reg, REPO)
        if p is None or not p.exists():
            continue
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
            prep = per_seed_galerkin(xi_mat, psi, k_field, q_field,
                                       n_lat, np)
            g_00 = np.asarray(prep["g_00_h"])
            t00 = np.asarray(prep["t00"])
            r_bar = np.asarray(prep["r_bar_h"])
            resid = g_00 + LAMBDA_T - t00
            feats = per_node_features(xi_mat, psi, k_field, q_field, n_lat)
            pool_resid.append(resid)
            for k, v in feats.items():
                pool_feats[k].append(v)
            pool_feats["r_bar_h"].append(r_bar)
            pool_feats["g_00"].append(g_00)
            pool_feats["t00"].append(t00)
            pool_feats["N_value"].append(np.full(n_lat, n_lat, dtype=float))

    resid_all = np.concatenate(pool_resid)
    feats_all = {k: np.concatenate(v) for k, v in pool_feats.items()}
    return resid_all, feats_all


def pearson_safe(x, y):
    mx = np.isfinite(x) & np.isfinite(y)
    if mx.sum() < 10 or np.std(x[mx]) < 1e-12 or np.std(y[mx]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[mx], y[mx])[0, 1])


def main() -> int:
    print("=" * 100)
    print("R_time_median bias source audit")
    print("Per-node Pearson correlation: residual_00(a) vs structural feature")
    print("=" * 100)

    resid, feats = gather_residual_and_features()
    print(f"\nTotal nodes pooled across 6 regimes / 4 seeds: {len(resid)}")
    print(f"Median residual:    {float(np.median(resid)):.6f}")
    print(f"Mean   residual:    {float(np.mean(resid)):.6f}")
    print(f"Std    residual:    {float(np.std(resid)):.6f}")
    print(f"Median |residual|:  {float(np.median(np.abs(resid))):.6f}")
    print()

    print(f"{'feature':<22} {'Pearson r':>11} {'r^2':>9} {'mean(feat)':>13} {'std(feat)':>13}")
    print("-" * 80)
    rows = []
    feature_keys = [k for k in feats.keys() if k not in ("g_00", "t00")]
    for k in feature_keys:
        v = feats[k]
        r = pearson_safe(resid, v)
        rows.append({
            "feature": k, "pearson_r": r,
            "r_squared": r * r if not np.isnan(r) else float("nan"),
            "mean": float(np.mean(v)) if v.size else float("nan"),
            "std": float(np.std(v)) if v.size else float("nan"),
        })
        print(f"{k:<22} {r:>+11.4f} {r * r if not np.isnan(r) else float('nan'):>9.4f} "
              f"{float(np.mean(v)):>13.5f} {float(np.std(v)):>13.5f}")

    # Top correlate
    valid = [r for r in rows if not np.isnan(r["pearson_r"])]
    valid.sort(key=lambda r: -abs(r["pearson_r"]))
    print()
    print("Top 5 |Pearson r|:")
    for r in valid[:5]:
        print(f"  {r['feature']:<22} r = {r['pearson_r']:+.4f}, r^2 = {r['r_squared']:.4f}")
    print()
    if valid and abs(valid[0]["pearson_r"]) >= 0.6:
        verdict = (f"DOMINANT_FEATURE = {valid[0]['feature']} "
                   f"(|r|={abs(valid[0]['pearson_r']):.3f})")
    elif valid and abs(valid[0]["pearson_r"]) >= 0.3:
        verdict = (f"WEAK_BUT_DETECTABLE = {valid[0]['feature']} "
                   f"(|r|={abs(valid[0]['pearson_r']):.3f})")
    else:
        verdict = "NO_STRUCTURED_SOURCE (max |r| < 0.3)"
    print(f"VERDICT: {verdict}")

    out_path = REPO / "outputs" / "R_time_bias_source_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "per_node_pearson_correlation_residual_00_vs_structural_features",
            "schema_version": "1.0.0",
            "lambda_t_reference": LAMBDA_T,
            "n_nodes_total": int(len(resid)),
            "residual_statistics": {
                "median": float(np.median(resid)),
                "mean": float(np.mean(resid)),
                "std": float(np.std(resid)),
                "median_abs": float(np.median(np.abs(resid))),
            },
            "per_feature_correlation": rows,
            "top_5_features_by_abs_r": valid[:5] if valid else [],
            "verdict": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
