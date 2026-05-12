"""Integrate P5N200 (snapshot file) into Candidate B + IR-average test.

The P5N200 lattice run produced snapshots (n_seeds=2, n_snap=5, n_lat=200);
not a standard D1 NPZ. We use the LAST snapshot per seed as the converged
ground state and run per_seed_galerkin -> Candidate B (Lambda_t = T_00^rec).

Output: outputs/p5n200_candidate_B_audit.json
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

from verify_galerkin_runner_A_hessian_ricci import per_seed_galerkin

P5N200_SNAP = (REPO.parent / "results_d1_p5n200" / "P5N200.snapshots.npz")


def main() -> int:
    print("=" * 100)
    print("P5N200 integration: Candidate B + IR-average reconstruction")
    print("=" * 100)
    if not P5N200_SNAP.exists():
        print(f"ERROR: snapshot file not found at {P5N200_SNAP}")
        return 1
    d = np.load(P5N200_SNAP, allow_pickle=True)
    print(f"  snapshot file: {P5N200_SNAP}")
    print(f"  edge_xi_snapshots: {d['edge_xi_snapshots'].shape}")
    print(f"  psi_real_snapshots: {d['psi_real_snapshots'].shape}")
    n_seeds = int(d["n_seeds"][0]) if hasattr(d["n_seeds"], '__len__') else int(d["n_seeds"])
    n_lat   = int(d["n_lat"][0])   if hasattr(d["n_lat"],   '__len__') else int(d["n_lat"])
    print(f"  n_seeds={n_seeds}, n_lat={n_lat}")

    # Take LAST snapshot per seed
    edge_last = d["edge_xi_snapshots"][:, -1, :, :]    # (n_seeds, n_lat, n_lat)
    psi_re    = d["psi_real_snapshots"][:, -1, :]      # (n_seeds, n_lat)
    psi_im    = d["psi_imag_snapshots"][:, -1, :]      # (n_seeds, n_lat)

    g_pool, t_pool, trec_pool = [], [], []
    for s in range(n_seeds):
        xi_mat = edge_last[s].copy()
        np.fill_diagonal(xi_mat, 1.0)
        psi = psi_re[s] + 1j*psi_im[s]
        # Snapshot file has no ff_K/Q seed fields -> use defaults
        k_field = np.full((n_lat, n_lat), 0.55)
        q_field = np.full((n_lat, n_lat), 0.45)
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        # Decompose T_00 -> T_00^kin, T_00^rec
        adj = (xi_mat - np.eye(n_lat) > 0.6).astype(np.float64)
        deg = adj.sum(axis=1) + 1e-12
        K_per = (k_field * adj).sum(axis=1) / deg
        Q_per = (q_field * adj).sum(axis=1) / deg
        trec = 0.5 * 1.0 * (1.0 * K_per + 0.5 * (1.0 - Q_per))   # ζ_3=0.5, Ω=1, A_K=1, A_Q=0.5
        g_pool.append(np.asarray(prep["g_00_h"]))
        t_pool.append(np.asarray(prep["t00"]))
        trec_pool.append(trec)

    g00 = np.concatenate(g_pool)
    t00 = np.concatenate(t_pool)
    trec = np.concatenate(trec_pool)
    mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g00) & np.isfinite(trec)
    g00 = g00[mask]; t00 = t00[mask]; trec = trec[mask]
    print(f"  n_nodes after mask: {len(t00)}")

    # Critical Test: T_00^rec / T_00 ratio
    ratio = trec / t00
    print(f"\n  T_00^rec / T_00 ratio: median = {np.median(ratio):.4f}, std = {np.std(ratio):.4f}, CV = {np.std(ratio)/np.mean(ratio)*100:.2f}%")
    print(f"  expected: regime-invariant ~ 0.97 (spread 0.045 across 13 prior regimes)")

    # Candidate B residual: |G_00 + T_00^rec - T_00| / |T_00|
    cand_B = np.abs(g00 + trec - t00) / np.maximum(np.abs(t00), 1e-9)
    print(f"\n  Candidate B residual: median = {np.median(cand_B):.4f}, mean = {np.mean(cand_B):.4f}, p90 = {np.percentile(cand_B, 90):.4f}")
    print(f"  DoD thresholds:        median <= 0.05, mean <= 0.10")
    dod_med = np.median(cand_B) <= 0.05
    dod_mean = np.mean(cand_B) <= 0.10
    print(f"  median pass: {dod_med}, mean pass: {dod_mean}")

    # IR-average comparison
    print(f"\n  IR-test:")
    print(f"  mean(T_00)     = {np.mean(t00):.4f}")
    print(f"  mean(G_00)     = {np.mean(g00):.4f}")
    print(f"  mean(T_00^rec) = {np.mean(trec):.4f}")
    print(f"  mean(T-G)      = {np.mean(t00-g00):.4f}")
    print(f"  Reference Lambda_eff (H3c v9) = 0.314")
    print(f"  rel-offset <T_00^rec> vs 0.314 = {(np.mean(trec)-0.314)/0.314*100:+.1f}%")

    # P5-Sequence within-regime convergence
    print(f"\n  P5-Sequence Lambda_t cluster (memory: large-N forms tighter band):")
    print(f"    P5N72  : <T_00^rec> = 0.349")
    print(f"    P5N84  : <T_00^rec> = 0.312")
    print(f"    P5N100 : <T_00^rec> = 0.463")
    print(f"    P5N128*: not available")
    print(f"    P5N200 : <T_00^rec> = {np.mean(trec):.4f}  <-- new point")

    out = {
        "method": "p5n200_candidate_B_with_decomposition",
        "schema_version": "1.0.0",
        "regime": "P5N200",
        "N": int(n_lat),
        "n_seeds_used": int(n_seeds),
        "n_nodes": int(len(t00)),
        "T00_rec_over_T00": {
            "median": float(np.median(ratio)),
            "mean":   float(np.mean(ratio)),
            "std":    float(np.std(ratio)),
            "CV_pct": float(np.std(ratio)/np.mean(ratio)*100),
        },
        "candidate_B_residual": {
            "median": float(np.median(cand_B)),
            "mean":   float(np.mean(cand_B)),
            "p90":    float(np.percentile(cand_B, 90)),
            "dod_median_pass": bool(dod_med),
            "dod_mean_pass":   bool(dod_mean),
        },
        "absolute_means": {
            "T00":    float(np.mean(t00)),
            "G00":    float(np.mean(g00)),
            "T00rec": float(np.mean(trec)),
            "T_minus_G": float(np.mean(t00-g00)),
        },
        "Lambda_eff_reference": 0.314,
        "rel_offset_T00rec_pct": float((np.mean(trec)-0.314)/0.314*100),
    }
    out_path = REPO / "outputs" / "p5n200_candidate_B_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
