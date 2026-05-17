r"""Thorough BKT-style vortex-content diagnostic for the post-inversion
regime, with vectorised triangle enumeration + spatial pair correlation.

Hypothesis: N_inv ~ 600 is a BKT-style vortex-antivortex-unbinding
transition in the carrier topology:
  N <= N_inv: bound vortex-antivortex pairs (low density, short pair-
              correlation length)
  N >  N_inv: unbound free vortex plasma (rising density, long pair-
              correlation length)

Method per snapshot:
  1. Skeleton triangle list via fully-vectorised enumeration
     (no Python triple loop): for each edge (i,j) compute common
     neighbours k = A[i] & A[j], emit (i,j,k) with i<j<k.
  2. Per-node phase phi from psi_real + psi_imag.
  3. Per-triangle winding integer (mod 2pi) via phase differences.
  4. Vortex coordinates: centroid in the 2-3 Fiedler-eigenvector
     embedding of the weighted Laplacian.
  5. Pair-correlation function: distance to nearest opposite-sign
     vortex; mean distance proxies "bound pair length".

Aggregated per snapshot:
  - n_v, n_a, total_tri
  - vortex_density = (n_v + n_a) / total_tri
  - vortex_imbalance = |n_v - n_a| / (n_v + n_a)
  - mean_pair_distance, median_pair_distance (in Fiedler-embedding units)

Aggregated per run: trajectory of all metrics over iter.
Plus headline: did the trajectory cross a BKT-like
transition (vortex density growth + pair distance increase)?
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parent
OUTPUTS = REPO / "outputs"

TAU = 0.10


def phase_from_psi(psi_real: np.ndarray, psi_imag: np.ndarray) -> np.ndarray:
    return np.arctan2(psi_imag, psi_real)


def enumerate_skeleton_triangles(xi: np.ndarray, tau: float = TAU,
                                    max_triangles: int = 200000) -> np.ndarray:
    """Vectorised full enumeration of triangles in the skeleton-thresholded
    graph. Returns (M, 3) array with i < j < k, capped at max_triangles."""
    adj = (np.abs(xi - np.diag(np.diag(xi))) > tau)
    np.fill_diagonal(adj, False)
    n = adj.shape[0]

    # Build edge list (i < j) via upper-triangle nonzeros
    iu, ju = np.where(np.triu(adj, k=1))
    triangles_chunks = []
    total = 0
    for idx in range(iu.size):
        i, j = int(iu[idx]), int(ju[idx])
        # Common neighbours k with k > j (to enforce i < j < k)
        common = adj[i] & adj[j]
        common[: j + 1] = False  # require k > j
        ks = np.flatnonzero(common)
        if ks.size == 0:
            continue
        chunk = np.empty((ks.size, 3), dtype=np.int64)
        chunk[:, 0] = i
        chunk[:, 1] = j
        chunk[:, 2] = ks
        triangles_chunks.append(chunk)
        total += ks.size
        if total >= max_triangles:
            break
    if not triangles_chunks:
        return np.zeros((0, 3), dtype=np.int64)
    triangles = np.concatenate(triangles_chunks, axis=0)
    if triangles.shape[0] > max_triangles:
        rng = np.random.default_rng(42)
        idx = rng.choice(triangles.shape[0], size=max_triangles, replace=False)
        triangles = triangles[idx]
    return triangles


def vortex_signs_on_triangles(phase: np.ndarray,
                                triangles: np.ndarray
                                ) -> np.ndarray:
    """Returns per-triangle winding integer (-1, 0, +1, possibly ±2)."""
    if triangles.shape[0] == 0:
        return np.zeros(0, dtype=np.int32)
    i_idx = triangles[:, 0]
    j_idx = triangles[:, 1]
    k_idx = triangles[:, 2]
    d_ij = ((phase[j_idx] - phase[i_idx] + math.pi) % (2 * math.pi)) - math.pi
    d_jk = ((phase[k_idx] - phase[j_idx] + math.pi) % (2 * math.pi)) - math.pi
    d_ki = ((phase[i_idx] - phase[k_idx] + math.pi) % (2 * math.pi)) - math.pi
    total = d_ij + d_jk + d_ki
    winding = np.round(total / (2 * math.pi)).astype(np.int32)
    return winding


def fiedler_embedding(xi: np.ndarray, n_dims: int = 3) -> np.ndarray:
    """Return per-node coordinates in the top-{n_dims}-eigenvector space
    of the weighted normalised Laplacian (Fiedler embedding)."""
    n = xi.shape[0]
    w = xi.copy()
    np.fill_diagonal(w, 0.0)
    deg = np.maximum(w.sum(axis=1), 1e-12)
    d_inv = 1.0 / np.sqrt(deg)
    L = np.eye(n) - (d_inv[:, None] * w * d_inv[None, :])
    L = 0.5 * (L + L.T)
    try:
        # Compute lowest n_dims+1 eigenvectors via dense eigh
        eigs, vecs = np.linalg.eigh(L)
        # Skip the constant lambda_1=0 eigenvector at index 0
        return vecs[:, 1: 1 + n_dims]
    except np.linalg.LinAlgError:
        return np.zeros((n, n_dims))


def triangle_centroids(embedding: np.ndarray,
                        triangles: np.ndarray) -> np.ndarray:
    """Per-triangle centroid in embedding space; shape (M, n_dims)."""
    if triangles.shape[0] == 0:
        return np.zeros((0, embedding.shape[1]))
    coords = embedding[triangles]  # (M, 3, n_dims)
    return coords.mean(axis=1)


def pair_distance_stats(centroids: np.ndarray,
                         signs: np.ndarray) -> dict:
    """Compute distance from each (anti)vortex to its nearest
    opposite-sign (anti)vortex; aggregate stats.

    Bound pairs: small mean / median distance.
    Free vortices: large mean / median distance.
    """
    v_mask = signs > 0
    a_mask = signs < 0
    n_v = int(v_mask.sum())
    n_a = int(a_mask.sum())
    if n_v == 0 or n_a == 0:
        return {"n_v": n_v, "n_a": n_a,
                "mean_pair_distance": float("nan"),
                "median_pair_distance": float("nan"),
                "min_pair_distance": float("nan")}
    v_coords = centroids[v_mask]
    a_coords = centroids[a_mask]
    # For each vortex, distance to nearest antivortex
    # Vectorised: for each vortex i, ||v_i - a_j|| for all j, take min
    nn_dists = []
    for v in v_coords:
        d = np.linalg.norm(a_coords - v, axis=1)
        nn_dists.append(float(d.min()))
    return {
        "n_v": n_v,
        "n_a": n_a,
        "mean_pair_distance": float(np.mean(nn_dists)),
        "median_pair_distance": float(np.median(nn_dists)),
        "min_pair_distance": float(np.min(nn_dists)),
        "max_pair_distance": float(np.max(nn_dists)),
    }


def per_snapshot_full_vortex_audit(xi: np.ndarray,
                                       psi_real: np.ndarray,
                                       psi_imag: np.ndarray) -> dict:
    phase = phase_from_psi(psi_real, psi_imag)
    triangles = enumerate_skeleton_triangles(xi)
    if triangles.shape[0] == 0:
        return {
            "n_triangles": 0,
            "n_v": 0, "n_a": 0,
            "vortex_density": 0.0, "vortex_imbalance": 0.0,
            "mean_pair_distance": float("nan"),
            "median_pair_distance": float("nan"),
        }
    signs = vortex_signs_on_triangles(phase, triangles)
    n_v = int((signs > 0).sum())
    n_a = int((signs < 0).sum())
    n_tri = int(triangles.shape[0])
    vortex_density = (n_v + n_a) / max(n_tri, 1)
    vortex_imbalance = abs(n_v - n_a) / max(n_v + n_a, 1)
    # Fiedler embedding + pair-distance stats
    embedding = fiedler_embedding(xi, n_dims=3)
    centroids = triangle_centroids(embedding, triangles)
    pair_stats = pair_distance_stats(centroids, signs)
    return {
        "n_triangles": n_tri,
        "n_v": n_v,
        "n_a": n_a,
        "vortex_density": vortex_density,
        "vortex_imbalance": vortex_imbalance,
        **pair_stats,
    }


def analyze_run(npz_path: Path, label: str) -> dict:
    print(f"\n=== {label} ({npz_path.name}) ===")
    d = np.load(npz_path, allow_pickle=True)
    xi_snaps = d["edge_xi_snapshots"]
    psi_real = d["psi_real_snapshots"]
    psi_imag = d["psi_imag_snapshots"]
    steps = d["snapshot_steps"]
    n_seeds, n_snaps = xi_snaps.shape[:2]
    n_lat = int(xi_snaps.shape[2])
    print(f"  {n_seeds} seeds × {n_snaps} snapshots, N={n_lat}")

    per_step = []
    for snap_idx in range(n_snaps):
        per_seed = []
        for s in range(n_seeds):
            xi = xi_snaps[s, snap_idx].astype(np.float64)
            pr = psi_real[s, snap_idx].astype(np.float64)
            pi = psi_imag[s, snap_idx].astype(np.float64)
            per_seed.append(per_snapshot_full_vortex_audit(xi, pr, pi))
        # Average across seeds (skip NaN entries)
        avg = {}
        for k in per_seed[0]:
            vals = []
            for r in per_seed:
                v = r[k]
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    continue
                vals.append(v)
            avg[k] = float(np.mean(vals)) if vals else float("nan")
        avg["step"] = int(steps[snap_idx])
        per_step.append(avg)
        if snap_idx % 5 == 0:
            print(f"    iter {avg['step']:>4}: "
                  f"n_tri={avg['n_triangles']:>7.0f} "
                  f"n_v={avg['n_v']:>5.0f} n_a={avg['n_a']:>5.0f} "
                  f"density={avg['vortex_density']:>7.4f} "
                  f"pair_med={avg['median_pair_distance']:>9.5f}")
    return {"label": label, "N": n_lat, "n_seeds": n_seeds, "per_step": per_step}


def headline(run_result: dict) -> dict:
    per_step = run_result["per_step"]
    first = per_step[0]
    last = per_step[-1]
    # Find first iter where density > 1.5x baseline (BKT-unbinding signature)
    base = max(first["vortex_density"], 1e-6)
    crossing_iter = None
    for r in per_step:
        if r["vortex_density"] > 1.5 * base:
            crossing_iter = r["step"]
            break
    return {
        "delta_density": last["vortex_density"] - first["vortex_density"],
        "delta_imbalance": last["vortex_imbalance"] - first["vortex_imbalance"],
        "delta_pair_distance": (last["median_pair_distance"] -
                                  first["median_pair_distance"])
                                if not (math.isnan(first.get("median_pair_distance", float("nan"))) or
                                          math.isnan(last.get("median_pair_distance", float("nan")))) else float("nan"),
        "crossing_iter_1_5x_density": crossing_iter,
        "first_density": first["vortex_density"],
        "last_density": last["vortex_density"],
        "first_pair_distance": first.get("median_pair_distance", float("nan")),
        "last_pair_distance": last.get("median_pair_distance", float("nan")),
    }


def main():
    print("=" * 78)
    print("Thorough BKT-style vortex-content diagnostic for post-inversion")
    print("=" * 78)
    runs = [
        ("results_d1_p5n1024_longiter_3seeds/P5N1024.snapshots.npz",
         "N=1024 post-inv, iter=0..1000, 3 seeds (snap every 25)"),
        ("results_d1_p5n700_longiter_2seeds/P5N700.snapshots.npz",
         "N=700 post-inv, iter=0..2000, 2 seeds (snap every 50)"),
    ]
    all_results = {}
    for rel, label in runs:
        path = (REPO.parent / rel).resolve()
        if not path.is_file():
            print(f"  [skip] {label}: {path} missing")
            continue
        try:
            res = analyze_run(path, label)
        except Exception as e:
            print(f"  ERROR analysing {label}: {e}")
            continue
        all_results[label] = res
        h = headline(res)
        res["headline"] = h
        print(f"\n  HEADLINE:")
        print(f"    delta vortex_density:        {h['delta_density']:+.5f}")
        print(f"    delta vortex_imbalance:      {h['delta_imbalance']:+.5f}")
        print(f"    delta pair_distance (median): {h['delta_pair_distance']:+.5f}")
        print(f"    first density:               {h['first_density']:.5f}")
        print(f"    last density:                {h['last_density']:.5f}")
        if h["crossing_iter_1_5x_density"] is not None:
            print(f"    1.5x density crossing at iter {h['crossing_iter_1_5x_density']}")
        else:
            print(f"    no 1.5x density crossing observed")
        # BKT prediction check
        bkt_signature_density = h["delta_density"] > 0.01
        bkt_signature_pair_dist = (not math.isnan(h["delta_pair_distance"])
                                     and h["delta_pair_distance"] > 0.001)
        print(f"    BKT signature: density growth > 1pp = {bkt_signature_density}, "
              f"pair-dist growth > 0.001 = {bkt_signature_pair_dist}")

    out_path = OUTPUTS / "analyze_post_inversion_vortex_content.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str),
                          encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
