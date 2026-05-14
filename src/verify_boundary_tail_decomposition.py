"""Boundary-tail decomposition of the per-node 4x4 Galerkin
Frobenius residual.

Hypothesis: the persistent gap between mean and median residual
(ratio ~2.5-3 at all N >= 28) is a finite-lattice geometric
artefact of nodes near the embedding-graph boundary, not a
structural deviation from the closure prediction. If this is
true, then:

  (a) trimming the top decile of nodes brings the trimmed mean
      close to the median (both below 0.05);
  (b) classifying nodes as "boundary" vs "interior" by their
      amplitude on the highest-eigenvalue spectral modes (which
      are localised at boundary cells in the spectral-Laplacian
      construction) shows the heavy-tail nodes are predominantly
      boundary nodes;
  (c) the boundary-fraction itself scales as ~1/N (boundary cells
      = O(N^(d-1)) on a d-dimensional lattice, total = O(N^d)),
      so the boundary contribution to the mean scales as
      <residual>_boundary * (boundary_fraction) -> 0 as N -> inf;
  (d) the interior-only mean is already at or below the closure
      threshold for N >= 60, and tracks the median much more
      tightly.

This script tests (a)-(d) on the existing P0..P8 + P5N64 + P5N100
NPZ files. No new lattice runs are needed.

Output: outputs/boundary_tail_decomposition.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Block CuPy (GPU contention with other processes).
class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, name):
        raise ImportError("cupy disabled for this audit")

sys.meta_path.insert(0, _BlockCupy())

from _d1_npz_discovery import find_d1_npz
from verify_galerkin_runner_A_hessian_ricci import (
    edge_to_matrix, per_seed_galerkin)


LADDER_REGIMES = [
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]

CLOSURE_THRESHOLD = 0.05


def per_node_frobenius(prep, xp):
    """Recompute the per-node 4x4 residual under the System-R
    structural Lambda_munu, returning (n_lat,) array of
    per-node residuals."""
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


def boundary_score_per_node(xi_off, n_lat, xp):
    """Return per-node 'boundary score': amplitude on the
    highest-eigenvalue spectral modes of the lattice graph
    Laplacian. Boundary cells localise on the high-frequency
    modes; bulk cells distribute across many modes.

    We use the top-3 eigenvectors and define
       boundary_score(a) = sum_{k in top3} eigvec_k(a)^2
    so that nodes with high amplitude on boundary modes get a
    high score, and nodes well-mixed across the spectrum stay
    near the uniform value 3/n_lat.
    """
    XI_THRESH = 0.1
    adj = (xi_off > XI_THRESH).astype(xp.float64)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12
    deg_inv_sqrt = 1.0 / xp.sqrt(deg)
    l_norm = (xp.eye(n_lat) - (deg_inv_sqrt[:, None]
                                * weight_adj
                                * deg_inv_sqrt[None, :]))
    eigvals, eigvecs = xp.linalg.eigh(l_norm)
    # Top-3 eigenvalues.
    top3 = eigvecs[:, -3:]  # (n_lat, 3)
    score = (top3 ** 2).sum(axis=1)
    return score


def trimmed_mean(values, frac=0.10):
    """Drop the top `frac` fraction of values, then average."""
    arr = np.asarray(values).ravel()
    n_drop = int(np.ceil(frac * len(arr)))
    if n_drop >= len(arr):
        return float("nan")
    return float(np.mean(np.sort(arr)[:-n_drop])) if n_drop > 0 else float(np.mean(arr))


def analyse_regime(regime, n_lat, xp):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)
    pooled_residual = []
    pooled_boundary = []
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, xp)

        # Per-node residual (using struct Lambda).
        residual = np.asarray(per_node_frobenius(prep, xp))

        # Boundary score on same xi.
        xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
        np.fill_diagonal(xi_off, 0.0)
        score = np.asarray(boundary_score_per_node(xi_off, n_lat, xp))

        pooled_residual.append(residual)
        pooled_boundary.append(score)

    residual = np.concatenate(pooled_residual)
    score = np.concatenate(pooled_boundary)

    # Quantiles + statistics.
    full_mean = float(np.mean(residual))
    full_median = float(np.median(residual))
    p75 = float(np.percentile(residual, 75))
    p90 = float(np.percentile(residual, 90))
    p95 = float(np.percentile(residual, 95))
    p99 = float(np.percentile(residual, 99))
    tmean_10 = trimmed_mean(residual, frac=0.10)
    tmean_20 = trimmed_mean(residual, frac=0.20)

    # Boundary classification: top tertile of boundary score is
    # "boundary"; bottom two-thirds is "interior".
    cutoff = float(np.percentile(score, 100 * 2 / 3))
    boundary_mask = score >= cutoff
    interior_mask = ~boundary_mask
    boundary_mean = float(np.mean(residual[boundary_mask])) \
        if boundary_mask.any() else float("nan")
    interior_mean = float(np.mean(residual[interior_mask])) \
        if interior_mask.any() else float("nan")
    boundary_median = float(np.median(residual[boundary_mask])) \
        if boundary_mask.any() else float("nan")
    interior_median = float(np.median(residual[interior_mask])) \
        if interior_mask.any() else float("nan")

    # Boundary fraction.
    boundary_fraction = float(boundary_mask.mean())

    # Top-decile residuals.
    top_decile_mask = residual >= np.percentile(residual, 90)
    top_decile_overlap = float(
        (top_decile_mask & boundary_mask).sum() / top_decile_mask.sum()
    ) if top_decile_mask.any() else float("nan")

    return {
        "regime": regime,
        "N": n_lat,
        "n_seeds": n_seeds,
        "n_nodes_total": int(len(residual)),
        "full_mean": full_mean,
        "full_median": full_median,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "trimmed_mean_top10pct_dropped": tmean_10,
        "trimmed_mean_top20pct_dropped": tmean_20,
        "boundary_classification": {
            "definition": (
                "top tertile of nodes by amplitude on top-3 "
                "highest-eigenvalue spectral modes"),
            "boundary_fraction": boundary_fraction,
            "boundary_mean": boundary_mean,
            "boundary_median": boundary_median,
            "interior_mean": interior_mean,
            "interior_median": interior_median,
            "top_decile_overlap_with_boundary": top_decile_overlap,
        },
    }


def main():
    import numpy as np
    xp = np
    t0 = time.time()
    results = []
    for regime, n_lat in LADDER_REGIMES:
        rec = analyse_regime(regime, n_lat, xp)
        if rec is None:
            continue
        results.append(rec)
        b = rec["boundary_classification"]
        print(f"{regime:<8} N={n_lat:>3} | "
              f"mean={rec['full_mean']:.4f} med={rec['full_median']:.4f} | "
              f"trim10={rec['trimmed_mean_top10pct_dropped']:.4f} "
              f"trim20={rec['trimmed_mean_top20pct_dropped']:.4f} | "
              f"bnd_mean={b['boundary_mean']:.4f} "
              f"int_mean={b['interior_mean']:.4f} | "
              f"top10%-bnd-overlap={b['top_decile_overlap_with_boundary']:.2f}",
              flush=True)

    # Test the 1/N scaling of boundary-fraction times boundary-residual.
    print("\n--- 1/N scaling test for boundary contribution to mean ---")
    print(f"{'N':>3} {'bnd_frac':>9} {'bnd_residual':>14} "
          f"{'bnd_contrib_to_mean':>20} {'expected ~ 1/N':>16}")
    for r in results:
        b = r["boundary_classification"]
        contrib = b["boundary_fraction"] * b["boundary_mean"]
        exp_1_over_n = 1.0 / r["N"]
        print(f"{r['N']:>3} {b['boundary_fraction']:>9.3f} "
              f"{b['boundary_mean']:>14.4f} "
              f"{contrib:>20.4f} {exp_1_over_n:>16.4f}")

    # Power-law fit on interior_mean vs N (within-regime P5
    # subladder) compared to full_mean.
    p5_set = [r for r in results if r["regime"] in ("P5", "P5N64", "P5N100")]
    p5_set = sorted(p5_set, key=lambda r: r["N"])
    if len(p5_set) >= 2:
        ns = np.array([r["N"] for r in p5_set])
        full_means = np.array([r["full_mean"] for r in p5_set])
        int_means = np.array(
            [r["boundary_classification"]["interior_mean"] for r in p5_set]
        )
        s_full, i_full = np.polyfit(np.log(ns), np.log(full_means), 1)
        s_int, i_int = np.polyfit(np.log(ns), np.log(int_means), 1)
        print(f"\n--- Within-regime P5 fits (full_mean vs interior_mean) ---")
        print(f"  full mean: alpha = {-s_full:.3f}, "
              f"intercept exp({i_full:.3f}) = {np.exp(i_full):.3f}")
        print(f"  interior mean: alpha = {-s_int:.3f}, "
              f"intercept exp({i_int:.3f}) = {np.exp(i_int):.3f}")
        # Crossover at 0.05.
        n_full = np.exp((np.log(0.05) - i_full) / s_full)
        n_int = np.exp((np.log(0.05) - i_int) / s_int)
        print(f"  full mean crosses 0.05 at N = {n_full:.0f}")
        print(f"  interior mean crosses 0.05 at N = {n_int:.0f}")

    out_path = REPO / "outputs" / "boundary_tail_decomposition.json"
    out = {
        "method": "boundary_tail_decomposition_per_node_residual",
        "schema_version": "1.0.0",
        "closure_threshold": CLOSURE_THRESHOLD,
        "structural_lambda": {"t": 0.81, "s": -0.005},
        "per_regime": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}; elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
