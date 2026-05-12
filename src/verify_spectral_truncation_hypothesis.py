"""Spectral-truncation hypothesis test.

Hypothesis: The heavy-tail per-node residual nodes are
under-resolved by the fixed 3-mode spatial basis
(eigvecs[:,1:4] of the normalized graph Laplacian). If true:

  (a) The "spectral coverage" of the 3-mode basis at high-residual
      nodes is anomalously low — i.e. these nodes carry most of
      their spectral amplitude on modes k > 3.

  (b) The per-node residual should correlate positively with the
      "high-mode mass" Sum_{k>=4} v_k(a)^2, equivalently
      negatively with the low-3-mode mass Sum_{k=1..3} v_k(a)^2.

  (c) The expected scaling under truncation theory: if we extended
      the spatial basis to m modes instead of 3, the per-node
      truncation residual would scale as Sum_{k>m} v_k(a)^2.

This test is purely spectral — it uses only eigenvectors of L_norm
and the existing per-node residuals from the Galerkin Runner A.
No tensor-framework rewrite needed.

If the hypothesis fails (no correlation between truncation mass
and residual), the heavy tail has a different mechanism and the
"cluster-center spectral truncation" story does not hold.

Output: outputs/spectral_truncation_hypothesis_test.json
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
    XI_THRESH, edge_to_matrix, per_seed_galerkin)


LADDER_REGIMES = [
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]


def per_node_residual_struct(prep, xp):
    g_00 = prep["g_00_h"]
    g_ij = prep["g_ij_h"]
    t00 = prep["t00"]
    t_ij = prep["t_ij"]
    eye3 = prep["eye3"]
    LAMBDA_T = 0.81
    LAMBDA_S = -0.005
    res00 = g_00 + LAMBDA_T - t00
    spatial_res = (g_ij + LAMBDA_S * eye3[None, :, :]) - t_ij
    sq = res00 ** 2 + (spatial_res ** 2).sum(axis=(1, 2))
    return xp.sqrt(sq)


def per_node_spectral_decomposition(xi_mat, n_lat):
    xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    l_norm = (np.eye(n_lat) - (deg_inv_sqrt[:, None] * weight_adj
                                * deg_inv_sqrt[None, :]))
    eigvals, eigvecs = np.linalg.eigh(l_norm)
    # Skip k=0 (trivial DC), then v_1..v_{n_lat-1} are the
    # non-trivial spectral modes ordered by eigenvalue.
    # eigvecs[a, k] = amplitude of mode k at node a.
    # Per-node mass on each mode: |eigvecs[a,k]|^2.
    mass = eigvecs ** 2
    # For each node, fraction on each cumulative mode-set.
    # cum_mass[a, m] = sum_{k=1..m} mass[a, k] (excluding k=0)
    return eigvals, eigvecs, mass


def cumulative_mode_coverage(mass, n_modes_choices):
    """For each node, return the cumulative coverage at each
    mode-cap. mass[a,k] excludes trivial DC k=0. We sum
    mass[a, 1..m] for m in n_modes_choices."""
    n_lat = mass.shape[0]
    out = {}
    for m in n_modes_choices:
        if m + 1 > n_lat:
            continue
        cov = mass[:, 1:m+1].sum(axis=1)
        out[m] = cov  # (n_lat,) coverage by modes 1..m
    return out


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def analyse_regime(regime, n_lat):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)

    pooled = {"residual": [], "low3_coverage": [], "low6_coverage": [],
              "low9_coverage": [], "high_above_3": [], "high_above_6": [],
              "high_above_9": []}
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        residual = np.asarray(per_node_residual_struct(prep, np))

        eigvals, eigvecs, mass = per_node_spectral_decomposition(
            xi_mat, n_lat)
        cov = cumulative_mode_coverage(mass, [3, 6, 9])
        # Total non-trivial mass per node = 1 - mass[a,0] (since
        # eigenvectors are orthonormal columns -> rows sum to 1
        # for each node up to renormalisation; mass is per-node
        # squared component over modes).
        total_nontrivial = 1.0 - mass[:, 0]
        # high_above_m(a) = total_nontrivial(a) - cov[m](a)
        high_3 = total_nontrivial - cov.get(3, np.zeros(n_lat))
        high_6 = total_nontrivial - cov.get(6, np.zeros(n_lat))
        high_9 = total_nontrivial - cov.get(9, np.zeros(n_lat))

        pooled["residual"].append(residual)
        pooled["low3_coverage"].append(cov.get(3, np.zeros(n_lat)))
        pooled["low6_coverage"].append(cov.get(6, np.zeros(n_lat)))
        pooled["low9_coverage"].append(cov.get(9, np.zeros(n_lat)))
        pooled["high_above_3"].append(high_3)
        pooled["high_above_6"].append(high_6)
        pooled["high_above_9"].append(high_9)

    res = np.concatenate(pooled["residual"])
    low3 = np.concatenate(pooled["low3_coverage"])
    low6 = np.concatenate(pooled["low6_coverage"])
    low9 = np.concatenate(pooled["low9_coverage"])
    high3 = np.concatenate(pooled["high_above_3"])
    high6 = np.concatenate(pooled["high_above_6"])
    high9 = np.concatenate(pooled["high_above_9"])

    p90_res = np.percentile(res, 90)
    top_mask = res >= p90_res
    bot_mask = ~top_mask

    rec = {
        "regime": regime,
        "N": n_lat,
        "n_nodes_total": int(len(res)),
        "spearman_residual_vs_high_above_3": spearman(res, high3),
        "spearman_residual_vs_high_above_6": spearman(res, high6),
        "spearman_residual_vs_high_above_9": spearman(res, high9),
        "spearman_residual_vs_low3_coverage": spearman(res, low3),
        "top_decile_means": {
            "low3_top10": float(low3[top_mask].mean()),
            "low3_bot90": float(low3[bot_mask].mean()),
            "low3_ratio_top_over_bot": float(
                low3[top_mask].mean() / low3[bot_mask].mean()),
            "high_above_3_top10": float(high3[top_mask].mean()),
            "high_above_3_bot90": float(high3[bot_mask].mean()),
            "high_above_3_ratio_top_over_bot": float(
                high3[top_mask].mean() / high3[bot_mask].mean()),
            "high_above_6_top10_over_bot90_ratio": float(
                high6[top_mask].mean() / max(high6[bot_mask].mean(), 1e-12)),
            "high_above_9_top10_over_bot90_ratio": float(
                high9[top_mask].mean() / max(high9[bot_mask].mean(), 1e-12)),
        },
        "uniform_low3_baseline": 3.0 / (n_lat - 1),
    }
    return rec


def main():
    results = []
    print("=" * 110)
    print("Spectral truncation hypothesis: do high-residual nodes have low coverage by 3-mode basis?")
    print("=" * 110)
    print()
    print(f"{'reg':<8} {'N':>3} | {'rho res<->high>3':>17} {'high>6':>9} {'high>9':>9} | "
          f"{'top10/bot90 high>3':>20} {'high>6':>9} {'high>9':>9}")
    print("-" * 110)
    for reg, n_lat in LADDER_REGIMES:
        rec = analyse_regime(reg, n_lat)
        if rec is None:
            continue
        results.append(rec)
        td = rec["top_decile_means"]
        print(f"{reg:<8} {n_lat:>3} | "
              f"{rec['spearman_residual_vs_high_above_3']:>+.3f}{' ':12} "
              f"{rec['spearman_residual_vs_high_above_6']:>+.3f} "
              f"{rec['spearman_residual_vs_high_above_9']:>+.3f}     | "
              f"{td['high_above_3_ratio_top_over_bot']:>20.4f} "
              f"{td['high_above_6_top10_over_bot90_ratio']:>9.4f} "
              f"{td['high_above_9_top10_over_bot90_ratio']:>9.4f}")

    print()
    print("=" * 110)
    print("VERDICT")
    print("=" * 110)
    avg = lambda key: float(np.mean(
        [r[key] for r in results if not np.isnan(r[key])]))
    print(f"  Mean Spearman across regimes:")
    print(f"    residual vs high-mass-above-3:  {avg('spearman_residual_vs_high_above_3'):+.3f}  "
          f"(>0 ratifies truncation: more high-mode mass -> higher residual)")
    print(f"    residual vs high-mass-above-6:  {avg('spearman_residual_vs_high_above_6'):+.3f}")
    print(f"    residual vs high-mass-above-9:  {avg('spearman_residual_vs_high_above_9'):+.3f}")
    print(f"    residual vs low-3-mode-cov:     {avg('spearman_residual_vs_low3_coverage'):+.3f}  "
          f"(<0 ratifies truncation: low coverage -> higher residual)")

    print()
    print("  Top-decile vs bottom-90% high-above-m ratios (averaged across regimes):")
    print(f"    high_above_3 ratio: "
          f"{np.mean([r['top_decile_means']['high_above_3_ratio_top_over_bot'] for r in results]):.3f}")
    print(f"    high_above_6 ratio: "
          f"{np.mean([r['top_decile_means']['high_above_6_top10_over_bot90_ratio'] for r in results]):.3f}")
    print(f"    high_above_9 ratio: "
          f"{np.mean([r['top_decile_means']['high_above_9_top10_over_bot90_ratio'] for r in results]):.3f}")
    print()
    print("  Interpretation:")
    print("    If h>3 ratio > 1.5 and rho > 0.3 systematically, the 3-mode")
    print("    truncation IS the heavy-tail mechanism: extending to 6 or 9")
    print("    modes would proportionally reduce the high-mode mass and")
    print("    hence the truncation contribution to the per-node residual.")

    out_path = REPO / "outputs" / "spectral_truncation_hypothesis_test.json"
    out = {
        "method": "spectral_truncation_hypothesis_per_node_audit",
        "schema_version": "1.0.0",
        "structural_lambda": {"t": 0.81, "s": -0.005},
        "per_regime": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
