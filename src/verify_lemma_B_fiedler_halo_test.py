"""Lemma B U-loop: is the Fiedler-active set the halo around
the matter-core cusp?

User intuition (T-loop follow-up): "Matter-Core-Umgebung" =
halo? In the corpus's anisotropic-source companion paper, the
matter-core is the NFW-cusp (top-1% / top-5% by t_00 or
row-variance), and the halo is the radial neighbourhood of
the cusp where R_00 has the characteristic positive
contribution.

This audit tests: for each node, compute the graph-distance
to the nearest matter-core cusp node. If Fiedler-top-30%
nodes are at:
  - distance 0: they ARE the cusp (= matter-core itself)
  - distance medium: they are the halo (= matter-core
    neighbourhood)
  - distance far: they are pure bulk
the verdict identifies which.

We use the corpus-validated row-variance(Xi) proxy for
matter-core (AUC 0.83-0.90 in 8/8 LORO folds). C99 = top-1%
by row-variance is the cusp; the halo is the ring at
distance 1-3 from the cusp; bulk is distance >= 4.

Output: outputs/verify_lemma_B_fiedler_halo_test.json
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import deque

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_fiedler_halo_test.json"

LADDER = [
    ("P5",     50,  "results_d1_fix17/d1_p5.npz",                       "xi_seedK"),
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz",     "edge_xi_snapshots"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz",     "edge_xi_snapshots"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz",     "edge_xi_snapshots"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N128", 128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz",  "edge_xi_snapshots"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz",    "edge_xi_snapshots"),
    ("P5N256", 256, "results_d1_p5n256_12seeds/P5N256.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz",   "edge_xi_snapshots"),
]

# Use tau=0.10 skeleton (corpus-standard structural-skeleton threshold)
# for the BFS adjacency, matching the prop:expander_skeleton identification.
TAU_SKEL = 0.10


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


def bfs_dist_to_targets(adj: np.ndarray, targets: list[int]) -> np.ndarray:
    """Unweighted BFS distance from every node to the nearest target."""
    n = adj.shape[0]
    dist = np.full(n, -1, dtype=int)
    q = deque()
    for t in targets:
        dist[t] = 0
        q.append(t)
    while q:
        u = q.popleft()
        for v in np.where(adj[u] > 0)[0]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(int(v))
    return dist


def per_snapshot(xi: np.ndarray) -> dict | None:
    n = xi.shape[0]
    w = xi - np.eye(n)
    w = np.maximum(w, 0.0)
    deg_sum = w.sum(axis=1)
    if np.any(deg_sum <= 1e-12):
        return None

    # Row-variance(Xi) = corpus-validated matter-core proxy
    deg_var = np.array([np.var(w[i, :]) for i in range(n)])
    # Cusp = C99 = top-1% by row-variance
    k_cusp = max(1, int(np.round(0.01 * n)))
    cusp_idx = list(np.argsort(-deg_var)[:k_cusp].tolist())

    # Fiedler vector
    d_inv_sqrt = 1.0 / np.sqrt(deg_sum)
    norm = w * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    lap = np.eye(n) - norm
    lap = 0.5 * (lap + lap.T)
    eigvals, eigvecs = np.linalg.eigh(lap)
    fiedler = np.abs(eigvecs[:, 1])
    k_fiedler = max(3, int(np.round(0.30 * n)))
    fiedler_idx = set(np.argsort(-fiedler)[:k_fiedler].tolist())

    # Adjacency at tau=0.10 (corpus-standard skeleton)
    adj = (w > TAU_SKEL).astype(int)
    np.fill_diagonal(adj, 0)
    adj = ((adj + adj.T) > 0).astype(int)

    # BFS distance from cusp
    dist = bfs_dist_to_targets(adj, cusp_idx)
    # Some nodes may be unreachable from cusp (dist=-1); treat as inf
    dist_finite = dist.copy()
    dist_finite[dist < 0] = -1  # placeholder
    valid_mask = dist >= 0

    # Mean distance for: Fiedler-top30%, random-30%, all nodes
    if not valid_mask.any():
        return None

    fiedler_dist = np.array([dist[i] for i in fiedler_idx
                             if dist[i] >= 0])
    if fiedler_dist.size == 0:
        return None
    rng = np.random.default_rng(42)
    n_repeats = 20
    rand_means = []
    for _ in range(n_repeats):
        idx = rng.choice(n, size=k_fiedler, replace=False)
        d_r = np.array([dist[i] for i in idx if dist[i] >= 0])
        if d_r.size > 0:
            rand_means.append(float(d_r.mean()))
    rand_mean = float(np.mean(rand_means)) if rand_means else float("nan")

    all_dist = dist[valid_mask].astype(float)
    return {
        "n_nodes": n,
        "n_reachable": int(valid_mask.sum()),
        "k_cusp": k_cusp,
        "k_fiedler_30pct": k_fiedler,
        # Mean distance to cusp for the Fiedler-30%, all nodes, random-30%
        "fiedler_mean_dist_to_cusp": float(fiedler_dist.mean()),
        "fiedler_median_dist_to_cusp": float(np.median(fiedler_dist)),
        "fiedler_in_cusp_count": int(np.sum(fiedler_dist == 0)),
        "fiedler_in_halo_count": int(np.sum((fiedler_dist >= 1)
                                            & (fiedler_dist <= 2))),
        "fiedler_in_bulk_count": int(np.sum(fiedler_dist >= 3)),
        "global_mean_dist_to_cusp": float(all_dist.mean()),
        "global_max_dist_to_cusp": float(all_dist.max()),
        "random_30pct_mean_dist_to_cusp": rand_mean,
    }


def audit_regime(regime, n_lat, rel, hint):
    npz = REPO_ROOT / rel
    xis = load_all_xi(npz, hint)
    if not xis:
        return {"regime": regime, "N": n_lat,
                "status": "SNAPSHOT_NOT_AVAILABLE"}
    diags = [per_snapshot(xi) for xi in xis]
    diags = [d for d in diags if d is not None]
    if not diags:
        return {"regime": regime, "N": n_lat,
                "status": "ALL_DEGENERATE"}
    out = {"regime": regime, "N": n_lat,
           "n_seeds": len(diags),
           "status": "OK"}
    keys = ["fiedler_mean_dist_to_cusp",
            "fiedler_median_dist_to_cusp",
            "fiedler_in_cusp_count", "fiedler_in_halo_count",
            "fiedler_in_bulk_count",
            "global_mean_dist_to_cusp", "global_max_dist_to_cusp",
            "random_30pct_mean_dist_to_cusp",
            "k_fiedler_30pct"]
    for key in keys:
        vals = [d[key] for d in diags if d[key] is not None]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
    # Fraction in halo
    if out.get("k_fiedler_30pct_mean", 0) > 0:
        out["fiedler_halo_fraction"] = (
            out.get("fiedler_in_halo_count_mean", 0)
            / out["k_fiedler_30pct_mean"])
        out["fiedler_cusp_fraction"] = (
            out.get("fiedler_in_cusp_count_mean", 0)
            / out["k_fiedler_30pct_mean"])
        out["fiedler_bulk_fraction"] = (
            out.get("fiedler_in_bulk_count_mean", 0)
            / out["k_fiedler_30pct_mean"])
    return out


def main():
    per_regime = [audit_regime(*row) for row in LADDER]
    out = {
        "headline": ("Lemma B U-loop: Fiedler-active set distance "
                     "to matter-core cusp (corpus-validated proxy: "
                     "C99 = top-1% by row-variance). Tests halo "
                     "hypothesis."),
        "per_regime": per_regime,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(per_regime)
    return 0


def print_summary(per_regime):
    print("=" * 110)
    print("Lemma B U-loop: Fiedler-active set distance to matter-core cusp (halo test)")
    print("=" * 110)
    print(f"{'Regime':<8} {'N':>4} {'seeds':>6} "
          f"{'Fied_mean':>10} {'rand_mean':>10} {'global':>8} "
          f"{'f_cusp%':>9} {'f_halo%':>9} {'f_bulk%':>9}")
    print("-" * 110)
    for r in per_regime:
        if r["status"] != "OK":
            continue
        print(f"{r['regime']:<8} {r['N']:>4} {r['n_seeds']:>6} "
              f"{r.get('fiedler_mean_dist_to_cusp_mean', 0):>10.3f} "
              f"{r.get('random_30pct_mean_dist_to_cusp_mean', 0):>10.3f} "
              f"{r.get('global_mean_dist_to_cusp_mean', 0):>8.3f} "
              f"{100*r.get('fiedler_cusp_fraction', 0):>9.2f} "
              f"{100*r.get('fiedler_halo_fraction', 0):>9.2f} "
              f"{100*r.get('fiedler_bulk_fraction', 0):>9.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
