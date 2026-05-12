"""Persistent homology with Vietoris-Rips filtration on the Xi-graph.

For each LAT-canonical regime (P5, P5N64, P5N100, P5N300):
  1. Build distance matrix d_ij = -log(max(Xi_ij, eps)) clipped at floor.
     This matches the manuscript metric d_ij = -ELL_0 * log(Xi_ij)
     up to a global rescaling.
  2. Sort edges by distance d.
  3. Sweep filtration parameter eps from 0 to max distance.
  4. Track Betti-0 (connected components) via union-find as edges
     are added in order. Birth = 0 for each isolated node;
     death = filtration value at which the component merges.
  5. Track cycle rank H_1 = |E| - |V| + |components| at each step.
     Birth of 1-cycle = filtration value at which the cycle closes
     (an edge connects two nodes in the same component).
  6. Track triangle count (3-cliques) at each filtration step.
  7. Generate persistence diagrams and compute total persistence,
     mean lifetime, max lifetime per dimension.

Compare across N: do the persistence diagrams scale, or are there
N-stable persistence features?

Compare matter-core vs background sub-filtration: are the
persistence diagrams of the top-decile-T_00 induced subgraph
distinguishable from random subgraphs?
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

from verify_galerkin_runner_A_hessian_ricci import (
    edge_to_matrix, per_seed_galerkin)

LAT_LADDER = [
    ("P5",      50,  "results_d1_fix17/d1_p5.npz",     "d1"),
    ("P5N64",   64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz",  "snap"),
    ("P5N100", 100,  "results_d1_p5n100_24seeds/P5N100.snapshots.npz","snap"),
    ("P5N300", 300,  "results_d1_p5n300_12seeds/P5N300.snapshots.npz", "snap"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz",  "snap"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz",  "snap"),
]


def load_seed_payload(rel_path, n_lat, kind, max_seeds=32):
    fp = REPO.parent / rel_path
    if not fp.exists():
        return []
    z = np.load(fp, allow_pickle=True)
    out = []
    if kind == "snap":
        snaps = z["edge_xi_snapshots"]
        psi_re = z["psi_real_snapshots"]
        psi_im = z["psi_imag_snapshots"]
        last_idx = snaps.shape[1] - 1
        ns = min(int(snaps.shape[0]), max_seeds)
        has_kq = "k_snapshots" in z.files and "q_snapshots" in z.files
        for s in range(ns):
            xi_mat = np.asarray(snaps[s, last_idx], dtype=float)
            psi = (np.asarray(psi_re[s, last_idx], dtype=float)
                   + 1j * np.asarray(psi_im[s, last_idx], dtype=float))
            if has_kq:
                k_field = np.asarray(z["k_snapshots"][s, last_idx],
                                      dtype=float)
                q_field = np.asarray(z["q_snapshots"][s, last_idx],
                                      dtype=float)
            else:
                k_field = np.full((n_lat, n_lat), 0.55)
                q_field = np.full((n_lat, n_lat), 0.45)
            out.append((xi_mat, psi, k_field, q_field))
        return out
    edge_arr = z["dense_cell_edge_xi_values"]
    amp_arr = z["dense_cell_node_amplitude_values"]
    phase_arr = z["dense_cell_node_phase_values"]
    ns = min(int(edge_arr.shape[0]), max_seeds)
    for s in range(ns):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = np.asarray(z[f"ff_K_seed{s}"], dtype=float)
        q_field = np.asarray(z[f"ff_Q_seed{s}"], dtype=float)
        out.append((xi_mat, psi, k_field, q_field))
    return out


def xi_to_distance(xi_mat, eps=1e-6):
    """d_ij = -log(max(Xi_ij, eps)). Symmetric; diagonal forced to 0."""
    x = np.array(xi_mat, dtype=float)
    np.fill_diagonal(x, 1.0)
    x = np.where(np.isfinite(x), x, eps)
    x = np.clip(x, eps, 1.0)
    x = 0.5 * (x + x.T)  # symmetrise
    d = -np.log(x)
    np.fill_diagonal(d, 0.0)
    return d


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_components -= 1
        return True


def vr_filtration_h0_h1(d_mat, n_steps=80, n_subsample_edges=None,
                          mask=None):
    """Vietoris-Rips H_0/H_1 persistence.

    Sort edges by distance (within mask if provided). For each edge
    addition, update union-find. Edge that connects two different
    components -> 0-cycle death (component merge). Edge that connects
    same component -> 1-cycle birth.

    Returns:
      persistence_pairs_H0: list of (birth=0, death=eps_at_merge)
      persistence_pairs_H1: list of (birth=eps_at_close, death=inf)
        (open at end of filtration)
      curve_H0: list of (eps, n_components)
      curve_H1: list of (eps, n_cycles)
      curve_triangles: list of (eps, triangle_count)  triangles closed
        when third edge added making (i,j,k) all-pairwise connected.
    """
    n = d_mat.shape[0]
    if mask is None:
        mask = np.ones(n, dtype=bool)
    idx = np.where(mask)[0]
    n_sub = idx.size
    if n_sub < 2:
        return [], [], [], [], []
    sub_idx = {int(v): k for k, v in enumerate(idx)}
    # All edges within mask
    edges = []
    for ii in range(n_sub):
        i = idx[ii]
        for jj in range(ii + 1, n_sub):
            j = idx[jj]
            edges.append((float(d_mat[i, j]), int(ii), int(jj)))
    edges.sort()
    if n_subsample_edges and len(edges) > n_subsample_edges:
        # sample uniformly to control compute cost
        step = max(1, len(edges) // n_subsample_edges)
        edges = edges[::step]
    uf = UnionFind(n_sub)
    # Adjacency tracking for triangle count
    adj = [set() for _ in range(n_sub)]
    n_components = n_sub
    n_cycles = 0
    n_edges = 0
    n_triangles = 0
    persistence_H0 = []
    persistence_H1 = []
    curve_H0 = []
    curve_H1 = []
    curve_tri = []
    for eps, a, b in edges:
        # Triangle count: triangles formed by adding this edge =
        # |adj[a] ∩ adj[b]|
        common = adj[a] & adj[b]
        new_triangles = len(common)
        n_triangles += new_triangles
        # Add edge
        adj[a].add(b)
        adj[b].add(a)
        n_edges += 1
        if uf.union(a, b):
            n_components -= 1
            # 0-cycle death: a smaller component merged into a
            # larger one at filtration time eps
            persistence_H0.append((0.0, float(eps)))
        else:
            # 1-cycle birth: closing a cycle inside a component
            persistence_H1.append((float(eps), float("inf")))
            n_cycles += 1
        curve_H0.append((float(eps), n_components))
        curve_H1.append((float(eps), n_cycles))
        curve_tri.append((float(eps), n_triangles))
    return persistence_H0, persistence_H1, curve_H0, curve_H1, curve_tri


def persistence_summary(pH, dim):
    """Total persistence, mean lifetime, max lifetime, count."""
    if not pH:
        return {"count": 0, "total": 0.0, "mean": 0.0, "max": 0.0}
    if dim == 0:
        # H_0: birth=0, death=eps_merge
        lifetimes = [d - b for b, d in pH if math.isfinite(d)]
    else:
        # H_1: birth=eps_close, death=inf -> lifetime=inf-birth
        # Use birth time as the "depth" (smaller birth = persists longer)
        lifetimes = [b for b, _ in pH]
    if not lifetimes:
        return {"count": 0, "total": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "count": len(lifetimes),
        "total": float(sum(lifetimes)),
        "mean": float(np.mean(lifetimes)),
        "max": float(np.max(lifetimes)),
        "median": float(np.median(lifetimes)),
    }


def downsample_curve(curve, n_points=20):
    if len(curve) <= n_points:
        return curve
    step = max(1, len(curve) // n_points)
    return curve[::step]


def main() -> int:
    print("=" * 78)
    print("Vietoris-Rips persistent homology on Xi-graph (LAT-canonical)")
    print("=" * 78)

    bundle = {"per_regime": {}}
    for reg, n_lat, rel, kind in LAT_LADDER:
        print(f"\n[{reg}, N={n_lat}]")
        payload = load_seed_payload(rel, n_lat, kind, max_seeds=2)
        if not payload:
            continue

        per_seed = []
        for sidx, (xi_mat, psi, kf, qf) in enumerate(payload):
            np.fill_diagonal(xi_mat, 1.0)
            d_mat = xi_to_distance(xi_mat)
            try:
                prep = per_seed_galerkin(xi_mat, psi, kf, qf, n_lat, np)
            except Exception as e:
                print(f"  seed {sidx} galerkin failed: {e}")
                continue
            T = np.asarray(prep["t00"])
            thr_T = float(np.quantile(T, 0.90))
            is_core = T >= thr_T

            # Subsample edges for cost control at large N
            n_subsample = None if n_lat <= 100 else 5000

            # Full graph PH
            pH0, pH1, c0, c1, ctri = vr_filtration_h0_h1(
                d_mat, n_subsample_edges=n_subsample)
            sH0 = persistence_summary(pH0, 0)
            sH1 = persistence_summary(pH1, 1)

            # Matter-core induced subgraph PH
            pH0_core, pH1_core, c0_core, c1_core, ctri_core = \
                vr_filtration_h0_h1(d_mat, n_subsample_edges=n_subsample,
                                      mask=is_core)
            sH0_core = persistence_summary(pH0_core, 0)
            sH1_core = persistence_summary(pH1_core, 1)

            # Random subset of same size as core, for null
            rng = np.random.default_rng(seed=100 + sidx)
            n_core = int(is_core.sum())
            null_h1_counts = []
            null_h0_max = []
            for _ in range(20):
                rand_idx = rng.choice(n_lat, n_core, replace=False)
                rand_mask = np.zeros(n_lat, dtype=bool)
                rand_mask[rand_idx] = True
                _, pH1_r, _, _, _ = vr_filtration_h0_h1(
                    d_mat, n_subsample_edges=n_subsample, mask=rand_mask)
                null_h1_counts.append(len(pH1_r))

            per_seed.append({
                "seed": sidx,
                "summary_H0_full": sH0,
                "summary_H1_full": sH1,
                "summary_H0_core": sH0_core,
                "summary_H1_core": sH1_core,
                "n_triangles_full_max": ctri[-1][1] if ctri else 0,
                "n_triangles_core_max": ctri_core[-1][1] if ctri_core else 0,
                "n_core": n_core,
                "null_h1_count_mean": float(np.mean(null_h1_counts))
                                       if null_h1_counts else 0.0,
                "null_h1_count_q95": float(np.quantile(null_h1_counts, 0.95))
                                       if null_h1_counts else 0.0,
                # Compact curves for plotting
                "curve_H0_full": downsample_curve(c0, 30),
                "curve_H1_full": downsample_curve(c1, 30),
                "curve_triangles_full": downsample_curve(ctri, 30),
            })
            print(f"  seed{sidx}: H_0 full count={sH0['count']}, "
                  f"max_lifetime={sH0['max']:.3f}; "
                  f"H_1 full count={sH1['count']}, "
                  f"mean_birth={sH1['mean']:.3f}; "
                  f"triangles_full={ctri[-1][1] if ctri else 0}")
            print(f"          H_0 core count={sH0_core['count']}, "
                  f"H_1 core count={sH1_core['count']}, "
                  f"triangles_core={ctri_core[-1][1] if ctri_core else 0}")
            print(f"          null H_1 (random subset same size) "
                  f"mean={per_seed[-1]['null_h1_count_mean']:.1f}, "
                  f"95%={per_seed[-1]['null_h1_count_q95']:.1f}")

        if not per_seed:
            continue

        # Aggregate
        agg = {
            "n_lat": n_lat, "n_seeds": len(per_seed),
            "H0_full_max_lifetime_mean": float(np.mean(
                [r["summary_H0_full"]["max"] for r in per_seed])),
            "H0_full_total_mean": float(np.mean(
                [r["summary_H0_full"]["total"] for r in per_seed])),
            "H1_full_count_mean": float(np.mean(
                [r["summary_H1_full"]["count"] for r in per_seed])),
            "H1_full_mean_birth_mean": float(np.mean(
                [r["summary_H1_full"]["mean"] for r in per_seed])),
            "H1_core_count_mean": float(np.mean(
                [r["summary_H1_core"]["count"] for r in per_seed])),
            "null_h1_count_mean_mean": float(np.mean(
                [r["null_h1_count_mean"] for r in per_seed])),
            "null_h1_count_q95_mean": float(np.mean(
                [r["null_h1_count_q95"] for r in per_seed])),
            "n_triangles_full_max_mean": float(np.mean(
                [r["n_triangles_full_max"] for r in per_seed])),
            "n_triangles_core_max_mean": float(np.mean(
                [r["n_triangles_core_max"] for r in per_seed])),
            "per_seed": per_seed,
        }
        bundle["per_regime"][reg] = agg

    # Synthesis
    print()
    print("=" * 78)
    print("Cross-regime persistent homology synthesis")
    print("=" * 78)
    print(f"{'reg':>8} {'N':>4} {'H1_full':>9} {'H1_core':>9} "
          f"{'null_H1':>9} {'lift':>9} {'tri_full':>10} "
          f"{'tri_core':>10} {'H0_max_life':>13}")
    for reg, n_lat, _, _ in LAT_LADDER:
        if reg not in bundle["per_regime"]:
            continue
        a = bundle["per_regime"][reg]
        lift_h1 = (a["H1_core_count_mean"]
                    - a["null_h1_count_mean_mean"])
        print(f"{reg:>8} {n_lat:>4} {a['H1_full_count_mean']:>9.1f} "
              f"{a['H1_core_count_mean']:>9.1f} "
              f"{a['null_h1_count_mean_mean']:>9.1f} "
              f"{lift_h1:>+9.1f} "
              f"{a['n_triangles_full_max_mean']:>10.0f} "
              f"{a['n_triangles_core_max_mean']:>10.0f} "
              f"{a['H0_full_max_lifetime_mean']:>13.3f}")

    out = REPO / "outputs" / "xi_persistent_homology.json"
    out.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
