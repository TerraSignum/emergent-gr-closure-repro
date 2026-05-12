"""Lemma B Phase-2 Step 3d: skeleton-diameter scaling
(Friedman-random vs geometric-Cayley discriminator).

Step 3c diagnosed the tau=0.10 skeleton as STRUCTURED_RANDOM
(bulk inside Kesten-McKay support, isolated-eigenvalue count
grows linearly in N). The Cayley-graph-vs-Friedman question
is resolved by graph-theoretic geometry, not just spectrum:

  - Friedman regime (random d-regular):
      diameter(A_skel)  ~  log_d(N)
    Small-world, dense expansion.

  - Geometric / Cayley regime on a finite group of dim k:
      diameter(A_skel)  ~  N^(1/k)
    Larger diameter due to geometric structure.

The diameter scaling is a clean discriminator. Two ladder
points already distinguish: at N=50 vs N=512, the ratio of
diameters is log(512)/log(50) = 1.59 for Friedman, vs
(512/50)^(1/k) = 10.24^(1/k) for Cayley-of-dim-k, which is
3.20 (k=2), 2.17 (k=3), 1.78 (k=4), 1.59 (k≈5.0).

Additional diagnostics per snapshot:
  - skeleton girth (shortest cycle length); Cayley graphs have
    specific girths depending on the group generators.
  - degree-sequence variance; Cayley graphs are regular (CV=0),
    Friedman random is near-regular (CV ~ 1/sqrt(d)), SBM is
    bimodal.
  - skeleton connectivity (fraction of vertices in giant
    connected component).

Output: outputs/verify_lemma_B_skeleton_diameter.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import (
    shortest_path,
    connected_components,
)

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_skeleton_diameter.json"

LADDER = [
    ("P5",     50,  "results_d1_fix17/d1_p5.npz",            "xi_seedK"),
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz", "edge_xi_snapshots"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz", "edge_xi_snapshots"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz", "edge_xi_snapshots"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz", "edge_xi_snapshots"),
    ("P5N128", 128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz", "edge_xi_snapshots"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz", "edge_xi_snapshots"),
    ("P5N256", 256, "results_d1_p5n256_12seeds/P5N256.snapshots.npz", "edge_xi_snapshots"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz", "edge_xi_snapshots"),
    ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz", "edge_xi_snapshots"),
]

TAU_SKEL = 0.10
MAX_SEEDS_PER_REGIME = 8  # diameter is O(N^2) per source, expensive at large N


def load_all_xi(npz_path: Path, hint: str) -> list[np.ndarray]:
    if not npz_path.exists():
        return []
    z = np.load(npz_path, allow_pickle=True)
    matrices: list[np.ndarray] = []
    if hint == "edge_xi_snapshots" and "edge_xi_snapshots" in z.files:
        snaps = np.asarray(z["edge_xi_snapshots"])
        last = snaps.shape[1] - 1
        for s in range(snaps.shape[0]):
            xi = np.asarray(snaps[s, last], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            matrices.append(xi)
        return matrices
    if hint == "xi_seedK":
        n_seeds = sum(1 for k in z.files if k.startswith("xi_seed"))
        for s in range(n_seeds):
            key = f"xi_seed{s}"
            if key not in z.files:
                continue
            xi = np.asarray(z[key], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            matrices.append(xi)
        return matrices
    return matrices


def skeleton_diameter_diagnostics(xi: np.ndarray, tau: float
                                  ) -> dict[str, Any] | None:
    """Compute diameter, connectivity, degree statistics on
    the tau-skeleton."""
    n = xi.shape[0]
    a_skel = ((xi - np.eye(n)) > tau).astype(np.int8)
    np.fill_diagonal(a_skel, 0)
    a_skel = ((a_skel + a_skel.T) > 0).astype(np.int8)
    # Sparse representation for csgraph
    sparse = sp.csr_matrix(a_skel)
    # Connectivity
    n_comp, labels = connected_components(sparse, directed=False)
    # Giant component size
    sizes = np.bincount(labels)
    giant_frac = float(sizes.max() / n)
    # Shortest paths (BFS, unweighted) — for diameter, only on
    # the giant component
    if giant_frac < 1.0:
        # restrict to giant component
        giant_idx = np.where(labels == labels[sizes.argmax()])[0]
        sub = sparse[giant_idx, :][:, giant_idx]
    else:
        sub = sparse
    # diameter via shortest_path (returns NxN distance matrix)
    try:
        dist = shortest_path(sub, directed=False, unweighted=True)
    except Exception:
        return None
    finite_mask = np.isfinite(dist)
    if not finite_mask.any():
        return None
    finite_vals = dist[finite_mask]
    diameter = float(finite_vals.max())
    mean_dist = float(finite_vals.mean())
    # degree sequence
    deg = a_skel.sum(axis=1).astype(float)
    deg_mean = float(deg.mean())
    deg_cv = float(deg.std(ddof=1) / max(deg_mean, 1e-12)) if deg.size > 1 else 0.0
    # girth: BFS-based shortest cycle through each vertex
    girth = _girth_estimate(a_skel)
    return {
        "n_components": int(n_comp),
        "giant_component_fraction": giant_frac,
        "diameter": diameter,
        "mean_path_length": mean_dist,
        "skel_degree_mean": deg_mean,
        "skel_degree_CV": deg_cv,
        "girth": girth,
    }


def _girth_estimate(a: np.ndarray, max_check: int = 50) -> float:
    """Estimate girth (shortest cycle length) by BFS from up to
    max_check randomly-selected vertices. Returns the minimum
    over the sample. Returns inf if no cycle found.
    """
    n = a.shape[0]
    rng = np.random.default_rng(0)
    sample = rng.choice(n, size=min(max_check, n), replace=False)
    best = float("inf")
    for start in sample:
        # BFS, track parent to detect cycles
        dist = -np.ones(n, dtype=int)
        parent = -np.ones(n, dtype=int)
        dist[start] = 0
        frontier = [start]
        while frontier:
            next_frontier = []
            for u in frontier:
                nbrs = np.where(a[u, :] > 0)[0]
                for v in nbrs:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        next_frontier.append(v)
                    elif v != parent[u]:
                        # cycle detected: length = dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < best:
                            best = cycle_len
            frontier = next_frontier
        if best <= 3:
            break  # cannot improve below triangle
    return float(best) if np.isfinite(best) else -1.0


def audit_regime(regime, n_lat, rel, hint):
    npz = REPO_ROOT / rel
    xis = load_all_xi(npz, hint)
    if not xis:
        return {"regime": regime, "N": n_lat, "n_seeds_loaded": 0,
                "status": "SNAPSHOT_NOT_AVAILABLE"}
    # Throttle seeds at large N
    xis = xis[:MAX_SEEDS_PER_REGIME]
    diags = [skeleton_diameter_diagnostics(xi, TAU_SKEL) for xi in xis]
    diags = [d for d in diags if d is not None]
    if not diags:
        return {"regime": regime, "N": n_lat, "n_seeds_loaded": len(xis),
                "status": "ALL_SEEDS_FAILED"}
    out = {"regime": regime, "N": n_lat,
           "n_seeds_used": len(diags),
           "status": "OK"}
    for key in ("diameter", "mean_path_length", "giant_component_fraction",
                "skel_degree_mean", "skel_degree_CV", "girth"):
        vals = [d[key] for d in diags if np.isfinite(d[key]) or d[key] >= 0]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def fit_diameter_scaling(per_regime: list[dict[str, Any]]
                         ) -> dict[str, Any]:
    """Fit diameter(N) to two competing models:
      - log_d:  diam = c + a * log(N) / log(d)  (Friedman)
      - power:  diam = c * N^alpha               (geometric Cayley)
    AICc-ranked.
    """
    valid = [(r["N"], r["diameter_mean"], r.get("skel_degree_mean_mean", 12))
             for r in per_regime if r.get("status") == "OK"
             and r.get("diameter_mean") is not None]
    if len(valid) < 3:
        return {"verdict": "INSUFFICIENT_DATA"}
    n_arr = np.array([v[0] for v in valid], dtype=float)
    y = np.array([v[1] for v in valid], dtype=float)
    n_pts = len(valid)

    def aicc(sse, k):
        return n_pts * np.log(sse / n_pts + 1e-30) + 2 * k + 2 * k * (k + 1) / max(n_pts - k - 1, 1)

    # log model: y = c + a * log(N) (any base works; absorb into a)
    A = np.column_stack([np.ones_like(n_arr), np.log(n_arr)])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    c_log, a_log = float(sol[0]), float(sol[1])
    y_pred = c_log + a_log * np.log(n_arr)
    sse_log = float(((y - y_pred) ** 2).sum())
    # power model: log(y) = log(c) + alpha * log(N)
    if np.all(y > 0):
        A = np.column_stack([np.ones_like(n_arr), np.log(n_arr)])
        sol, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
        c_pow, alpha_pow = float(np.exp(sol[0])), float(sol[1])
        y_pred = c_pow * n_arr ** alpha_pow
        sse_pow = float(((y - y_pred) ** 2).sum())
    else:
        c_pow, alpha_pow, sse_pow = float("nan"), float("nan"), float("inf")

    aicc_log = aicc(sse_log, 2)
    aicc_pow = aicc(sse_pow, 2) if np.isfinite(sse_pow) else float("inf")

    preferred = "log" if aicc_log < aicc_pow else "power"

    # Interpretation: log -> Friedman; power with alpha > 0.3 -> Cayley
    if preferred == "log":
        interpretation = ("FRIEDMAN_LIKE — diameter scales "
                          "logarithmically with N (small-world)")
    elif alpha_pow > 0.3:
        interpretation = (f"CAYLEY_GEOMETRIC — diameter scales as "
                          f"N^{alpha_pow:.3f}, geometric structure "
                          f"with intrinsic dim ~ {1.0/alpha_pow:.2f}")
    else:
        interpretation = ("WEAK_POWER — preferred power-law but with "
                          "small exponent; quasi-logarithmic")

    return {
        "n_points": n_pts,
        "log_model": {"c": c_log, "a": a_log, "AICc": aicc_log,
                      "SSE": sse_log},
        "power_model": {"c": c_pow, "alpha": alpha_pow,
                        "AICc": aicc_pow, "SSE": sse_pow},
        "preferred": preferred,
        "interpretation": interpretation,
    }


def main():
    per_regime = [audit_regime(*row) for row in LADDER]
    diameter_fit = fit_diameter_scaling(per_regime)
    out = {
        "headline": ("Lemma B Phase-2 Step 3d: skeleton-diameter "
                     "scaling and geometric-vs-random discriminator. "
                     "Friedman-random graphs have diam ~ log_d(N); "
                     "geometric Cayley graphs have diam ~ N^(1/k)."),
        "method": (
            "For each tau=0.10 skeleton snapshot: compute graph "
            "diameter, mean path length, giant-component fraction, "
            "girth, and degree-sequence CV. Per regime: cross-seed "
            "mean. N-scaling fit (log vs power) with AICc selection."),
        "tau_skeleton": TAU_SKEL,
        "max_seeds_per_regime": MAX_SEEDS_PER_REGIME,
        "per_regime": per_regime,
        "diameter_scaling_fit": diameter_fit,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(per_regime, diameter_fit)
    return 0


def print_summary(per_regime, fit):
    print("=" * 78)
    print("Lemma B Phase-2 Step 3d: Skeleton diameter & geometry")
    print("=" * 78)
    print(f"{'Regime':<8} {'N':>4} {'seeds':>6} "
          f"{'diam':>7} {'mean_path':>10} {'giant':>7} "
          f"{'d_skel':>7} {'CV(d)':>7} {'girth':>6}")
    print("-" * 78)
    for r in per_regime:
        if r["status"] != "OK":
            print(f"{r['regime']:<8} {r['N']:>4}  {r['status']}")
            continue
        print(f"{r['regime']:<8} {r['N']:>4} "
              f"{r['n_seeds_used']:>6} "
              f"{r['diameter_mean']:>7.2f} "
              f"{r['mean_path_length_mean']:>10.3f} "
              f"{r['giant_component_fraction_mean']:>7.3f} "
              f"{r['skel_degree_mean_mean']:>7.2f} "
              f"{r['skel_degree_CV_mean']:>7.3f} "
              f"{r['girth_mean']:>6.2f}")
    print()
    if fit.get("verdict") == "INSUFFICIENT_DATA":
        print("Diameter scaling fit: INSUFFICIENT_DATA")
        return
    print("Diameter scaling fit:")
    print(f"  log model:   diam = {fit['log_model']['c']:.3f} + "
          f"{fit['log_model']['a']:.3f} * log(N), "
          f"AICc = {fit['log_model']['AICc']:.2f}")
    print(f"  power model: diam = {fit['power_model']['c']:.3f} * "
          f"N^{fit['power_model']['alpha']:.3f}, "
          f"AICc = {fit['power_model']['AICc']:.2f}")
    print(f"  Preferred: {fit['preferred']}")
    print(f"  Interpretation: {fit['interpretation']}")


if __name__ == "__main__":
    raise SystemExit(main())
