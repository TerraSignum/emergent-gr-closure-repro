"""Mechanistic audit of the heavy-tail mechanism in the per-node
4x4 Galerkin Frobenius residual.

The earlier "top-tertile-by-spectral-amplitude classifies boundary"
hypothesis did NOT hold on the data: the supposedly-boundary
subset's mean residual was not systematically larger than the
interior subset's mean. So calling the heavy tail a
"boundary-region effect" was hand-wavy.

This audit tests three concrete, measurable hypotheses for what
the top-decile-residual nodes actually are:

  H_degree    : low graph degree (few Xi-neighbors above
                threshold)  → noisy discrete derivatives
  H_amp_small : small |psi| (carrier amplitude near zero)
                → inflation in 1/(omega + epsilon) inside the
                gradient construction
  H_omega_small : small omega_a (small per-node
                spectral-weighted gradient density)
                → same 1/omega inflation mechanism

For each regime + seed, we compute per-node:
  - residual (struct Lambda)
  - graph_degree
  - amp = |psi|
  - omega_a (the Galerkin density per node)

Then for each candidate predictor X in {degree, amp, omega_a},
we compute:
  - Spearman rank correlation between X and residual
  - mean of X within the top-decile-residual subset vs the
    bottom-90% subset

If a single predictor cleanly separates the top decile, we have
the physical mechanism. If none does, the heavy tail is
intrinsic to the discrete construction (random outliers) and
must be explained as such.

Output: outputs/heavy_tail_mechanism_audit.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Block CuPy.
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


def spearman(x, y):
    """Spearman rank correlation."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def per_node_features(xi_mat, psi, n_lat):
    """Compute per-node features: degree, |psi|, omega_a."""
    xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    degree = adj.sum(axis=1)

    weight_adj = xi_off * adj
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.maximum(d_mat, D_MIN)
    d_sq = d_mat * d_mat
    d_sq_safe = np.where(adj > 0, d_sq, np.inf)
    weight_grad = np.where(adj > 0, weight_adj / (d_sq_safe + EPS_D), 0.0)
    omega_a = weight_grad.sum(axis=1)

    psi_safe = np.where(np.isfinite(psi.real) & np.isfinite(psi.imag),
                         psi, 0.0 + 0.0j)
    amp = np.abs(psi_safe)

    return {
        "degree": degree,
        "amp": amp,
        "omega_a": omega_a,
    }


def analyse_regime(regime, n_lat):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)

    pooled = {"residual": [], "degree": [], "amp": [], "omega_a": []}
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        residual = np.asarray(per_node_residual_struct(prep, np))
        feats = per_node_features(xi_mat, psi, n_lat)

        pooled["residual"].append(residual)
        pooled["degree"].append(feats["degree"])
        pooled["amp"].append(feats["amp"])
        pooled["omega_a"].append(feats["omega_a"])

    res = np.concatenate(pooled["residual"])
    deg = np.concatenate(pooled["degree"])
    amp = np.concatenate(pooled["amp"])
    omega = np.concatenate(pooled["omega_a"])

    # Top-decile mask.
    p90_res = np.percentile(res, 90)
    top_mask = res >= p90_res
    bot_mask = ~top_mask

    rec = {
        "regime": regime,
        "N": n_lat,
        "n_seeds": n_seeds,
        "n_nodes_total": int(len(res)),
        "spearman_residual_vs_degree": spearman(res, -deg),  # negate so high res ~ low degree
        "spearman_residual_vs_amp_inv": spearman(res, -amp),  # high res ~ low amp
        "spearman_residual_vs_omega_inv": spearman(res, -omega),
        "top_decile_vs_bottom_90pct": {
            "degree": {
                "top10_mean": float(deg[top_mask].mean()),
                "bot90_mean": float(deg[bot_mask].mean()),
                "ratio": float(deg[top_mask].mean() / deg[bot_mask].mean()),
            },
            "amp": {
                "top10_mean": float(amp[top_mask].mean()),
                "bot90_mean": float(amp[bot_mask].mean()),
                "ratio": float(amp[top_mask].mean() / amp[bot_mask].mean()),
            },
            "omega_a": {
                "top10_mean": float(omega[top_mask].mean()),
                "bot90_mean": float(omega[bot_mask].mean()),
                "ratio": float(omega[top_mask].mean() / omega[bot_mask].mean()),
            },
        },
    }
    return rec


def main():
    results = []
    print("=" * 100)
    print("Heavy-tail mechanism audit: which per-node feature predicts the top-decile?")
    print("=" * 100)
    print()
    print(f"{'reg':<8} {'N':>3} | {'spearman:res~-deg':>17} {'res~-amp':>10} "
          f"{'res~-omega':>11} | {'top10/bot90 deg ratio':>22} "
          f"{'top10/bot90 amp ratio':>22} {'top10/bot90 omega':>18}")
    print("-" * 130)
    for reg, n_lat in LADDER_REGIMES:
        rec = analyse_regime(reg, n_lat)
        if rec is None:
            continue
        results.append(rec)
        td = rec["top_decile_vs_bottom_90pct"]
        print(f"{reg:<8} {n_lat:>3} | "
              f"{rec['spearman_residual_vs_degree']:>+.3f}{' ':10} "
              f"{rec['spearman_residual_vs_amp_inv']:>+.3f}{' ':3} "
              f"{rec['spearman_residual_vs_omega_inv']:>+.3f}{' ':5} | "
              f"{td['degree']['ratio']:>22.3f} "
              f"{td['amp']['ratio']:>22.3f} "
              f"{td['omega_a']['ratio']:>18.3f}")

    # Verdict.
    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    avg = lambda key: float(np.mean([r[key] for r in results]))
    print(f"  Mean Spearman across regimes:")
    print(f"    residual ~ -degree:      {avg('spearman_residual_vs_degree'):+.3f}  (>0 means low degree -> high res)")
    print(f"    residual ~ -|psi|:       {avg('spearman_residual_vs_amp_inv'):+.3f}  (>0 means small amp -> high res)")
    print(f"    residual ~ -omega_a:     {avg('spearman_residual_vs_omega_inv'):+.3f}  (>0 means small omega -> high res)")

    out_path = REPO / "outputs" / "heavy_tail_mechanism_audit.json"
    out = {
        "method": "heavy_tail_mechanism_per_node_predictor_audit",
        "schema_version": "1.0.0",
        "predictors_tested": ["graph_degree", "abs_psi", "omega_a_galerkin_density"],
        "per_regime": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
