"""Diagnose the factor-2 cluster bias in Lambda_t* between
small-N (9 regimes, Lambda_t* ~ 0.876) and large-N (4 regimes, ~ 0.422).

Hypothesis catalogue:
  H1: amplitude/psi scaling — |psi| means differ between clusters
  H2: edge_xi distribution differs — var(Xi) cluster-bias
  H3: K_rec / Q normalization — ff_K_seed presence vs default
  H4: regime-parameter difference — lambda_triangle, epsilon, alpha_scale
  H5: mask cut effect — t00 > 0.05 cuts more nodes in one cluster
  H6: per_seed_galerkin code-path branch — N-dependent normalization

We compute the same per-regime breakdown of:
  - mean(|psi|), mean(|psi|^2), var(|psi|)
  - mean(var(Xi)), mean(|grad psi|^2)
  - mean(K_per), mean(1-Q_per)
  - mean(T_00) across components (kin + rec)
  - mean(G_00)
  - mean fraction of nodes surviving t00>0.05 mask
across all 13 regimes, and tag the cluster.

Output: outputs/two_cluster_bias_diagnose_audit.json
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
    edge_to_matrix, per_seed_galerkin, XI_THRESH)


SMALL_CLUSTER = ["P1","P3","P4","P5","P5N64","P6","P7","P8","P5N100"]
LARGE_CLUSTER = ["P5N72","P5N84","P6N128","P8N128"]
ALL = [
    ("P1", 28), ("P3", 36), ("P4", 42), ("P5", 50), ("P5N64", 64),
    ("P6", 60), ("P7", 72), ("P8", 84), ("P5N100", 100),
    ("P5N72", 72), ("P5N84", 84), ("P6N128", 128), ("P8N128", 128),
]


def diagnose(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    keys = list(d.keys())
    has_ffK = any(k.startswith("ff_K_seed") for k in keys)
    has_ffQ = any(k.startswith("ff_Q_seed") for k in keys)

    e = d["dense_cell_edge_xi_values"]
    a = d["dense_cell_node_amplitude_values"]
    ph = d["dense_cell_node_phase_values"]
    n_seeds_used = min(e.shape[0], 32)

    # Per-seed diagnostics
    abs_psi_pool = []
    abs_psi_sq_pool = []
    var_psi_pool = []
    xi_off_pool_means = []
    var_xi_pool = []
    grad_psi_sq_pool = []
    k_per_pool = []
    one_minus_q_pool = []
    t00_kin_pool = []
    t00_rec_pool = []
    t00_pool = []
    g00_pool = []
    n_pre_mask = 0
    n_post_mask = 0

    for s in range(n_seeds_used):
        xi_mat = edge_to_matrix(e[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = a[s] * np.exp(1j*ph[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        g00 = np.asarray(prep["g_00_h"])
        t00 = np.asarray(prep["t00"])

        # Decompose t00 manually for per-component stats
        xi_off = xi_mat.copy()
        np.fill_diagonal(xi_off, 0.0)
        adj = (xi_off > XI_THRESH).astype(np.float64)
        deg = adj.sum(axis=1) + 1e-12
        K_per = (k_field*adj).sum(axis=1)/deg
        Q_per = (q_field*adj).sum(axis=1)/deg
        # t00_rec = ζ_3*Ω*(A_K*K + A_Q*(1-Q)) with ζ_3=0.5, Ω=1, A_K=1, A_Q=0.5
        t00_rec = 0.5*1.0*(1.0*K_per + 0.5*(1.0 - Q_per))
        t00_kin = t00 - t00_rec  # kinetic part = full minus rec

        # Xi-stats
        weight_adj = xi_off * adj
        xi_row_mean = weight_adj.sum(axis=1) / (adj.sum(axis=1) + 1e-12)
        var_xi_per_node = (((weight_adj - xi_row_mean[:, None])**2 * adj).sum(axis=1)
                           / (adj.sum(axis=1) + 1e-12))

        # Grad psi-stats — recover from t00 - var_xi*Z_xi/2 - var(|psi|)*kappa_xi
        # but we use per_seed_galerkin's t_munu_spectral semantics directly:
        # grad_psi_sq is the third additive piece
        amp_a = np.abs(psi)
        var_amp = (amp_a - amp_a.mean())**2

        # Use these as proxy stats (no need for full inversion):
        abs_psi_pool.append(np.abs(psi))
        abs_psi_sq_pool.append(np.abs(psi)**2)
        var_psi_pool.append(var_amp)
        var_xi_pool.append(var_xi_per_node)
        k_per_pool.append(K_per)
        one_minus_q_pool.append(1.0 - Q_per)
        xi_off_pool_means.append(np.mean(weight_adj))
        t00_kin_pool.append(t00_kin)
        t00_rec_pool.append(t00_rec)
        t00_pool.append(t00)
        g00_pool.append(g00)
        n_pre_mask += len(t00)
        mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g00)
        n_post_mask += int(np.sum(mask))

    abs_psi = np.concatenate(abs_psi_pool)
    abs_psi_sq = np.concatenate(abs_psi_sq_pool)
    var_psi = np.concatenate(var_psi_pool)
    var_xi = np.concatenate(var_xi_pool)
    K_per_all = np.concatenate(k_per_pool)
    omQ_all   = np.concatenate(one_minus_q_pool)
    t00 = np.concatenate(t00_pool)
    t00_kin = np.concatenate(t00_kin_pool)
    t00_rec = np.concatenate(t00_rec_pool)
    g00 = np.concatenate(g00_pool)
    mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g00)
    t00_m = t00[mask]; g00_m = g00[mask]; t00_kin_m = t00_kin[mask]; t00_rec_m = t00_rec[mask]

    return {
        "regime": reg, "N": int(n_lat),
        "n_seeds_used": int(n_seeds_used),
        "has_ffK_seed": bool(has_ffK),
        "has_ffQ_seed": bool(has_ffQ),
        "n_pre_mask":   int(n_pre_mask),
        "n_post_mask":  int(n_post_mask),
        "mask_survival_pct": float(n_post_mask/max(n_pre_mask,1)*100),
        "mean_abs_psi":     float(np.mean(abs_psi)),
        "mean_abs_psi_sq":  float(np.mean(abs_psi_sq)),
        "mean_var_psi":     float(np.mean(var_psi)),
        "mean_var_xi":      float(np.mean(var_xi)),
        "mean_xi_off":      float(np.mean(xi_off_pool_means)),
        "mean_K_per":       float(np.mean(K_per_all)),
        "mean_one_minus_Q": float(np.mean(omQ_all)),
        "mean_T00":         float(np.mean(t00)),
        "mean_T00_postmask":float(np.mean(t00_m) if len(t00_m)>0 else float('nan')),
        "mean_T00_kin":     float(np.mean(t00_kin_m) if len(t00_kin_m)>0 else float('nan')),
        "mean_T00_rec":     float(np.mean(t00_rec_m) if len(t00_rec_m)>0 else float('nan')),
        "mean_G00":         float(np.mean(g00_m) if len(g00_m)>0 else float('nan')),
        "lambda_t_star":    float(np.mean(t00_m - g00_m)) if len(t00_m)>0 else float('nan'),
    }


def main() -> int:
    print("="*120)
    print("Two-cluster bias diagnostic")
    print("="*120)
    rows = []
    for reg, n in ALL:
        r = diagnose(reg, n)
        if r is not None:
            r["cluster"] = "small" if reg in SMALL_CLUSTER else "large"
            rows.append(r)

    # Print key columns side by side
    cols = [
        ("regime", "<10"),
        ("N", ">4"),
        ("cluster", ">6"),
        ("ffK", ">5"),
        ("|psi|", ">8"),
        ("var(Xi)", ">9"),
        ("xi_off", ">8"),
        ("K_per", ">7"),
        ("1-Q", ">7"),
        ("T_00", ">7"),
        ("T^kin", ">7"),
        ("T^rec", ">7"),
        ("G_00", ">7"),
        ("Λt*", ">7"),
    ]
    headers = " ".join(f"{name:{fmt}}" for name, fmt in cols)
    print(headers)
    print("-"*len(headers))
    for r in rows:
        line = " ".join([
            f"{r['regime']:<10}",
            f"{r['N']:>4}",
            f"{r['cluster']:>6}",
            f"{'Y' if r['has_ffK_seed'] else 'N':>5}",
            f"{r['mean_abs_psi']:>8.4f}",
            f"{r['mean_var_xi']:>9.4f}",
            f"{r['mean_xi_off']:>8.4f}",
            f"{r['mean_K_per']:>7.4f}",
            f"{r['mean_one_minus_Q']:>7.4f}",
            f"{r['mean_T00']:>7.4f}",
            f"{r['mean_T00_kin']:>7.4f}",
            f"{r['mean_T00_rec']:>7.4f}",
            f"{r['mean_G00']:>7.4f}",
            f"{r['lambda_t_star']:>7.4f}",
        ])
        print(line)
    print()
    # Cluster-level breakdown
    for clust in ["small", "large"]:
        sub = [r for r in rows if r["cluster"] == clust]
        if not sub:
            continue
        print(f"=== {clust.upper()} cluster (n={len(sub)}) ===")
        print(f"  mean ff_K present:    {sum(r['has_ffK_seed'] for r in sub)}/{len(sub)}")
        for k in ["mean_abs_psi","mean_abs_psi_sq","mean_var_xi","mean_xi_off",
                 "mean_K_per","mean_one_minus_Q","mean_T00","mean_T00_kin",
                 "mean_T00_rec","mean_G00","lambda_t_star"]:
            v = np.array([r[k] for r in sub if not np.isnan(r[k])])
            if len(v) > 0:
                print(f"  <{k}> = {v.mean():.4f}  (std={v.std():.4f})")
        print()
    # Ratio test
    print("=== Ratio small/large for each metric ===")
    sm = [r for r in rows if r["cluster"]=="small"]
    lg = [r for r in rows if r["cluster"]=="large"]
    for k in ["mean_abs_psi","mean_abs_psi_sq","mean_var_xi","mean_xi_off",
             "mean_K_per","mean_one_minus_Q","mean_T00","mean_T00_kin",
             "mean_T00_rec","mean_G00","lambda_t_star"]:
        sv = np.array([r[k] for r in sm if not np.isnan(r[k])]).mean()
        lv = np.array([r[k] for r in lg if not np.isnan(r[k])]).mean()
        if abs(lv) > 1e-9:
            ratio = sv/lv
            print(f"  {k:<24} small/large = {sv:.4f} / {lv:.4f} = {ratio:.3f}")

    out = {
        "method": "two_cluster_bias_diagnostic",
        "schema_version": "1.0.0",
        "small_cluster_regimes": SMALL_CLUSTER,
        "large_cluster_regimes": LARGE_CLUSTER,
        "per_regime": rows,
    }
    out_path = REPO / "outputs" / "two_cluster_bias_diagnose_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
