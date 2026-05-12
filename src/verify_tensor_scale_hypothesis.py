"""Tensor-scale hypothesis test (third try at the heavy-tail
mechanism, after boundary-region and spectral-truncation both
failed).

Hypothesis: dense cluster-center nodes have anomalously large
absolute |G_munu| and |T_munu| because the spectral-weighted
Galerkin density omega_a there inflates both. The per-node
Frobenius residual ||G + Lambda - 8 pi G T||_F is then absolutely
large at those nodes purely because both sides of the Einstein
equation are absolutely large — the RELATIVE residual
||G+Lambda-T||/max(||G||, ||T||, ||Lambda*g||) might still be
small.

If true:
  (a) ||G||_F at top-decile-residual nodes is much larger than
      bottom-90% — the absolute scale is the driver.
  (b) The relative residual at top-decile nodes is comparable to
      the bottom-90% relative residual — i.e. once we normalise
      out the tensor scale, there is no heavy tail.

If false:
  the residual is genuinely large in a relative sense too,
  and we need yet another mechanism.

Output: outputs/tensor_scale_hypothesis_test.json
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


LADDER_REGIMES = [
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]


def per_node_tensor_norms(prep, xp):
    """Return per-node ||G||_F, ||T||_F, ||G+Lambda-T||_F (struct
    Lambda)."""
    g_00 = prep["g_00_h"]
    g_ij = prep["g_ij_h"]
    t00 = prep["t00"]
    t_ij = prep["t_ij"]
    eye3 = prep["eye3"]
    LAMBDA_T = 0.81
    LAMBDA_S = -0.005

    g_norm = xp.sqrt(g_00 ** 2 + (g_ij ** 2).sum(axis=(1, 2)))
    t_norm = xp.sqrt(t00 ** 2 + (t_ij ** 2).sum(axis=(1, 2)))

    res00 = g_00 + LAMBDA_T - t00
    spatial_res = (g_ij + LAMBDA_S * eye3[None, :, :]) - t_ij
    res_norm = xp.sqrt(res00 ** 2 + (spatial_res ** 2).sum(axis=(1, 2)))

    rel_norm = res_norm / xp.maximum(
        xp.maximum(g_norm, t_norm), 1e-12)
    return g_norm, t_norm, res_norm, rel_norm


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

    pooled = {"residual": [], "g_norm": [], "t_norm": [],
              "rel_residual": []}
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        g_norm, t_norm, res_norm, rel_norm = per_node_tensor_norms(prep, np)
        pooled["residual"].append(np.asarray(res_norm))
        pooled["g_norm"].append(np.asarray(g_norm))
        pooled["t_norm"].append(np.asarray(t_norm))
        pooled["rel_residual"].append(np.asarray(rel_norm))

    res = np.concatenate(pooled["residual"])
    gn = np.concatenate(pooled["g_norm"])
    tn = np.concatenate(pooled["t_norm"])
    rel = np.concatenate(pooled["rel_residual"])

    p90_res = np.percentile(res, 90)
    top_mask = res >= p90_res
    bot_mask = ~top_mask

    return {
        "regime": regime,
        "N": n_lat,
        "n_nodes_total": int(len(res)),
        "spearman_residual_vs_g_norm": spearman(res, gn),
        "spearman_residual_vs_t_norm": spearman(res, tn),
        "spearman_residual_vs_rel_residual": spearman(res, rel),
        "top_decile_means": {
            "g_norm_top10": float(gn[top_mask].mean()),
            "g_norm_bot90": float(gn[bot_mask].mean()),
            "g_norm_ratio": float(gn[top_mask].mean() / gn[bot_mask].mean()),
            "t_norm_top10": float(tn[top_mask].mean()),
            "t_norm_bot90": float(tn[bot_mask].mean()),
            "t_norm_ratio": float(tn[top_mask].mean() / tn[bot_mask].mean()),
            "rel_residual_top10": float(rel[top_mask].mean()),
            "rel_residual_bot90": float(rel[bot_mask].mean()),
            "rel_residual_ratio": float(
                rel[top_mask].mean() / max(rel[bot_mask].mean(), 1e-12)),
        },
        "regime_means": {
            "absolute_residual_mean": float(res.mean()),
            "absolute_residual_median": float(np.median(res)),
            "relative_residual_mean": float(rel.mean()),
            "relative_residual_median": float(np.median(rel)),
            "g_norm_mean": float(gn.mean()),
            "t_norm_mean": float(tn.mean()),
        },
    }


def main():
    results = []
    print("=" * 110)
    print("Tensor-scale hypothesis: do high-residual nodes have large absolute |G|, |T| but small relative residual?")
    print("=" * 110)
    print()
    print(f"{'reg':<8} {'N':>3} | {'rho res<->|G|':>14} {'res<->|T|':>10} {'res<->rel':>10} | "
          f"{'top10/bot90 |G|':>16} {'|T| ratio':>10} {'rel-res ratio':>14} | "
          f"{'rel-res mean':>13} {'rel-res med':>12}")
    print("-" * 130)
    for reg, n_lat in LADDER_REGIMES:
        rec = analyse_regime(reg, n_lat)
        if rec is None:
            continue
        results.append(rec)
        td = rec["top_decile_means"]
        rm = rec["regime_means"]
        print(f"{reg:<8} {n_lat:>3} | "
              f"{rec['spearman_residual_vs_g_norm']:>+.3f}{' ':9} "
              f"{rec['spearman_residual_vs_t_norm']:>+.3f}{' ':3} "
              f"{rec['spearman_residual_vs_rel_residual']:>+.3f}{' ':3} | "
              f"{td['g_norm_ratio']:>16.3f} "
              f"{td['t_norm_ratio']:>10.3f} "
              f"{td['rel_residual_ratio']:>14.3f} | "
              f"{rm['relative_residual_mean']:>13.4f} "
              f"{rm['relative_residual_median']:>12.4f}")

    print()
    print("=" * 110)
    print("VERDICT")
    print("=" * 110)
    avg = lambda key: float(np.mean(
        [r[key] for r in results if not np.isnan(r[key])]))
    print(f"  Mean Spearman across regimes:")
    print(f"    residual vs |G|_F:   {avg('spearman_residual_vs_g_norm'):+.3f}")
    print(f"    residual vs |T|_F:   {avg('spearman_residual_vs_t_norm'):+.3f}")
    print(f"    residual vs rel-res: {avg('spearman_residual_vs_rel_residual'):+.3f}")
    print()
    print("  Top-decile/bot90 ratios (averaged):")
    print(f"    |G|_F:  {np.mean([r['top_decile_means']['g_norm_ratio'] for r in results]):.3f}")
    print(f"    |T|_F:  {np.mean([r['top_decile_means']['t_norm_ratio'] for r in results]):.3f}")
    print(f"    rel-residual: {np.mean([r['top_decile_means']['rel_residual_ratio'] for r in results]):.3f}")
    print()
    print("  Relative-residual ladder convergence:")
    print(f"  {'reg':<8} {'N':>3} {'rel mean':>10} {'rel median':>11} {'abs mean':>10} {'abs median':>11}")
    for r in results:
        rm = r["regime_means"]
        print(f"  {r['regime']:<8} {r['N']:>3} {rm['relative_residual_mean']:>10.4f} "
              f"{rm['relative_residual_median']:>11.4f} "
              f"{rm['absolute_residual_mean']:>10.4f} "
              f"{rm['absolute_residual_median']:>11.4f}")

    out_path = REPO / "outputs" / "tensor_scale_hypothesis_test.json"
    out = {
        "method": "tensor_scale_hypothesis_per_node_audit",
        "schema_version": "1.0.0",
        "structural_lambda": {"t": 0.81, "s": -0.005},
        "per_regime": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
