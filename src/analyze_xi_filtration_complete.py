"""Complete VR persistent-homology + triangle-filtration analysis on
the Xi-graph for LAT-canonical regimes.

Two combined diagnostics, both derived from the Vietoris-Rips
filtration of d_ij = -log Xi_ij:

A. Vietoris-Rips H_0/H_1 persistence diagrams.
   The 0-cycle persistence captures component lifetimes (how long
   does each cluster exist before merging into a larger one).
   The 1-cycle persistence captures cycle birth times (when the
   filtration parameter eps reaches the value at which a cycle
   closes). We track the FULL persistence diagram, not just
   counts at saturation; the diagnostic value is the lifetime
   distribution itself.

B. Triangle (2-simplex) filtration.
   For each filtration value eps, count triangles (i,j,k) where all
   three pairwise edges have d_pair <= eps. The triangle-density
   curve T(eps) tells us at what Xi-coupling scale the graph
   transitions from forest-like (no triangles) to clique-like
   (triangle-rich). This is the proper 2-skeleton lift of the
   1-graph H_1 diagnostic.

For each diagnostic we report:
  - the curve as a function of physical filtration eps
  - a finite-truncation snapshot at eps* corresponding to Xi=0.5
    (matches the XI_THRESH used in the basic clustering analysis)
  - matter-core induced subgraph values vs full graph
  - random-subset null distribution at matched core size
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

XI_THRESH = 0.5
EPS_STAR = -math.log(XI_THRESH)  # = 0.6931 (truncation matching basic clustering)
# Filtration grid in eps space; covers Xi in [0.05, 1.0]
EPS_GRID = np.linspace(0.0, -math.log(0.05), 30)  # [0, 3.0]


LAT_LADDER = [
    ("P5",      50,  "results_d1_fix17/d1_p5.npz",                 "d1"),
    ("P5N64",   64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz",   "snap"),
    ("P5N100", 100,  "results_d1_p5n100_24seeds/P5N100.snapshots.npz", "snap"),
    ("P5N128", 128,  "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz","snap"),
    ("P5N200", 200,  "results_d1_p5n200_8seeds/P5N200.snapshots.npz",  "snap"),
    ("P5N300", 300,  "results_d1_p5n300_12seeds/P5N300.snapshots.npz",         "snap"),
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


def xi_to_distance(xi_mat, eps_floor=1e-6):
    x = np.array(xi_mat, dtype=float)
    np.fill_diagonal(x, 1.0)
    x = np.where(np.isfinite(x), x, eps_floor)
    x = np.clip(x, eps_floor, 1.0)
    x = 0.5 * (x + x.T)
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


def vr_persistence_with_truncation(d_mat, mask, eps_max,
                                    eps_grid):
    """Run VR filtration up to eps_max. Return:
      - (b, d) pairs for H_0 (b=0, d <= eps_max if merged else inf)
      - (b, d) pairs for H_1 with d=inf (open at end)
      - curves: list of (eps, n_components, n_cycles, n_triangles)
        sampled at eps_grid points within [0, eps_max].
    """
    n = d_mat.shape[0]
    idx = np.where(mask)[0]
    n_sub = idx.size
    if n_sub < 2:
        return [], [], [], 0
    # All edges within mask, sorted
    edges = []
    for ii in range(n_sub):
        i = idx[ii]
        for jj in range(ii + 1, n_sub):
            j = idx[jj]
            edges.append((float(d_mat[i, j]), int(ii), int(jj)))
    edges.sort()

    uf = UnionFind(n_sub)
    adj = [set() for _ in range(n_sub)]
    n_components = n_sub
    n_cycles = 0
    n_triangles = 0
    persistence_H0 = []
    persistence_H1 = []
    grid_idx = 0
    grid_curve = []

    for eps, a, b in edges:
        if eps > eps_max:
            break
        # Sample grid points up to current eps
        while (grid_idx < len(eps_grid)
                and eps_grid[grid_idx] < eps):
            grid_curve.append((float(eps_grid[grid_idx]),
                               int(n_components),
                               int(n_cycles),
                               int(n_triangles)))
            grid_idx += 1
        # Triangle count from this edge: |adj[a] ∩ adj[b]|
        common = adj[a] & adj[b]
        new_triangles = len(common)
        n_triangles += new_triangles
        adj[a].add(b)
        adj[b].add(a)
        if uf.union(a, b):
            n_components -= 1
            persistence_H0.append((0.0, float(eps)))
        else:
            persistence_H1.append((float(eps), float("inf")))
            n_cycles += 1

    # Fill remaining grid points (after last edge or eps_max)
    while grid_idx < len(eps_grid):
        if eps_grid[grid_idx] > eps_max:
            break
        grid_curve.append((float(eps_grid[grid_idx]),
                           int(n_components),
                           int(n_cycles),
                           int(n_triangles)))
        grid_idx += 1

    # Add infinite-life H_0 for components still open at eps_max
    for _ in range(n_components - 1):
        persistence_H0.append((0.0, float("inf")))

    return persistence_H0, persistence_H1, grid_curve, n_sub


def summary(pH, dim, eps_max):
    """Summarise persistence pairs:
       - count finite + infinite
       - mean lifetime (finite only)
       - max finite lifetime
       - 'persistence' = sum of (death - birth) for finite features
    """
    if not pH:
        return {"count": 0, "count_finite": 0, "count_infinite": 0,
                "mean_lifetime": 0.0, "max_lifetime": 0.0,
                "total_persistence": 0.0,
                "mean_birth": 0.0, "max_birth": 0.0}
    finite = [(b, d) for b, d in pH if math.isfinite(d)]
    infinite = [(b, d) for b, d in pH if not math.isfinite(d)]
    if dim == 0:
        # H_0: birth=0; lifetime = death
        lifes = [d for _, d in finite]
    else:
        # H_1: birth time; lifetime = eps_max - birth (truncated)
        lifes = [eps_max - b for b, _ in pH if math.isfinite(b)]
    births = [b for b, _ in pH if math.isfinite(b)]
    return {
        "count": len(pH),
        "count_finite": len(finite),
        "count_infinite": len(infinite),
        "mean_lifetime": float(np.mean(lifes)) if lifes else 0.0,
        "max_lifetime": float(np.max(lifes)) if lifes else 0.0,
        "total_persistence": float(np.sum(lifes)) if lifes else 0.0,
        "mean_birth": float(np.mean(births)) if births else 0.0,
        "max_birth": float(np.max(births)) if births else 0.0,
    }


def main() -> int:
    print("=" * 78)
    print("VR persistent-homology + triangle-filtration on Xi-graph")
    print("=" * 78)
    print(f"  Filtration eps = -log Xi; truncated at eps* = "
          f"{EPS_STAR:.3f} (Xi >= {XI_THRESH})")
    print(f"  Grid: {len(EPS_GRID)} points from 0 to "
          f"{EPS_GRID[-1]:.2f}")
    print()

    bundle = {
        "method": "VR-PH + triangle filtration on Xi-graph "
                  "with finite truncation",
        "eps_star_threshold": float(EPS_STAR),
        "eps_grid": EPS_GRID.tolist(),
        "per_regime": {},
    }

    for reg, n_lat, rel, kind in LAT_LADDER:
        print(f"\n[{reg}, N={n_lat}]")
        # Cap seeds for tractability of persistent-homology
        # filtration. Bumped P5N256 to 8 seeds to verify the
        # H_1_core anomaly first reported on 2 seeds.
        if n_lat >= 300:
            n_cap = 2
        else:
            n_cap = 8
        payload = load_seed_payload(rel, n_lat, kind, max_seeds=n_cap)
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
            G = np.asarray(prep["g_00_h"])
            thr_T = float(np.quantile(T, 0.90))
            is_core = T >= thr_T
            n_core = int(is_core.sum())

            mask_full = np.ones(n_lat, dtype=bool)

            # FULL graph PH up to eps_star (= XI_THRESH match) AND
            # extended grid up to log(1/0.05) for diagnostic curves
            pH0_full, pH1_full, curve_full, _ = \
                vr_persistence_with_truncation(d_mat, mask_full,
                                                  eps_max=EPS_GRID[-1],
                                                  eps_grid=EPS_GRID)
            # Truncated at eps_star (matches XI_THRESH=0.5)
            pH0_full_star, pH1_full_star, curve_full_star, _ = \
                vr_persistence_with_truncation(d_mat, mask_full,
                                                  eps_max=EPS_STAR,
                                                  eps_grid=EPS_GRID)

            sH0_full_star = summary(pH0_full_star, 0, EPS_STAR)
            sH1_full_star = summary(pH1_full_star, 1, EPS_STAR)

            # Matter-core sub-PH at eps_star
            pH0_core, pH1_core, curve_core, _ = \
                vr_persistence_with_truncation(d_mat, is_core,
                                                  eps_max=EPS_STAR,
                                                  eps_grid=EPS_GRID)
            sH0_core = summary(pH0_core, 0, EPS_STAR)
            sH1_core = summary(pH1_core, 1, EPS_STAR)

            # Random-subset null at eps_star
            rng = np.random.default_rng(seed=200 + sidx)
            null_h1_counts = []
            null_tri_counts = []
            for _ in range(15):
                rand_idx = rng.choice(n_lat, n_core, replace=False)
                rand_mask = np.zeros(n_lat, dtype=bool)
                rand_mask[rand_idx] = True
                _, pH1_r, curve_r, _ = vr_persistence_with_truncation(
                    d_mat, rand_mask, eps_max=EPS_STAR,
                    eps_grid=EPS_GRID)
                null_h1_counts.append(len(pH1_r))
                null_tri_counts.append(curve_r[-1][3]
                                        if curve_r else 0)

            tri_at_star_full = curve_full_star[-1][3] if curve_full_star else 0
            tri_at_star_core = curve_core[-1][3] if curve_core else 0

            per_seed.append({
                "seed": sidx,
                "n_core": n_core,
                # Full graph
                "H0_full_star": sH0_full_star,
                "H1_full_star": sH1_full_star,
                "tri_full_star": tri_at_star_full,
                # Core
                "H0_core_star": sH0_core,
                "H1_core_star": sH1_core,
                "tri_core_star": tri_at_star_core,
                # Null
                "null_h1_count_mean": float(np.mean(null_h1_counts))
                                       if null_h1_counts else 0.0,
                "null_h1_count_q95": float(np.quantile(null_h1_counts, 0.95))
                                       if null_h1_counts else 0.0,
                "null_tri_mean": float(np.mean(null_tri_counts))
                                  if null_tri_counts else 0.0,
                "null_tri_q95": float(np.quantile(null_tri_counts, 0.95))
                                  if null_tri_counts else 0.0,
                # Filtration curves (component, cycle, triangle counts)
                "curve_full_extended": curve_full,
            })
            print(f"  seed{sidx}: n_core={n_core}")
            print(f"    Full graph @ eps*={EPS_STAR:.3f}: "
                  f"H_0={sH0_full_star['count_infinite']} (inf-pers), "
                  f"H_1={sH1_full_star['count']}, "
                  f"tri={tri_at_star_full}")
            print(f"    Core graph @ eps*: "
                  f"H_0={sH0_core['count_infinite']} (inf-pers), "
                  f"H_1={sH1_core['count']}, "
                  f"tri={tri_at_star_core}")
            print(f"    Null H_1 mean={per_seed[-1]['null_h1_count_mean']:.1f}, "
                  f"q95={per_seed[-1]['null_h1_count_q95']:.1f}")
            print(f"    Null tri mean={per_seed[-1]['null_tri_mean']:.1f}, "
                  f"q95={per_seed[-1]['null_tri_q95']:.1f}")
            # H_1 birth-time distribution within full graph extended
            if pH1_full:
                births = [b for b, _ in pH1_full
                          if math.isfinite(b)]
                if births:
                    print(f"    Full-graph H_1 birth-time mean: "
                          f"{float(np.mean(births)):.3f}, "
                          f"min={min(births):.3f}, "
                          f"max={max(births):.3f}")

        if not per_seed:
            continue

        agg = {
            "n_lat": n_lat, "n_seeds": len(per_seed),
            "H1_full_star_count_mean": float(np.mean(
                [r["H1_full_star"]["count"] for r in per_seed])),
            "H1_core_star_count_mean": float(np.mean(
                [r["H1_core_star"]["count"] for r in per_seed])),
            "null_h1_count_mean_mean": float(np.mean(
                [r["null_h1_count_mean"] for r in per_seed])),
            "null_h1_count_q95_mean": float(np.mean(
                [r["null_h1_count_q95"] for r in per_seed])),
            "tri_full_star_mean": float(np.mean(
                [r["tri_full_star"] for r in per_seed])),
            "tri_core_star_mean": float(np.mean(
                [r["tri_core_star"] for r in per_seed])),
            "null_tri_mean_mean": float(np.mean(
                [r["null_tri_mean"] for r in per_seed])),
            "null_tri_q95_mean": float(np.mean(
                [r["null_tri_q95"] for r in per_seed])),
            "per_seed": per_seed,
        }
        bundle["per_regime"][reg] = agg

    # Cross-regime synthesis
    print()
    print("=" * 78)
    print(f"Cross-regime: persistence + triangulation @ eps* = "
          f"{EPS_STAR:.3f} (Xi >= {XI_THRESH})")
    print("=" * 78)
    print(f"{'reg':>8} {'N':>4} {'H1_full':>8} {'H1_core':>8} "
          f"{'null_H1':>9} {'lift':>7} {'tri_full':>9} "
          f"{'tri_core':>9} {'null_tri':>9} {'tri_lift':>10}")
    for reg, n_lat, _, _ in LAT_LADDER:
        if reg not in bundle["per_regime"]:
            continue
        a = bundle["per_regime"][reg]
        h1_lift = a["H1_core_star_count_mean"] - a["null_h1_count_mean_mean"]
        tri_lift = a["tri_core_star_mean"] - a["null_tri_mean_mean"]
        print(f"{reg:>8} {n_lat:>4} {a['H1_full_star_count_mean']:>8.1f} "
              f"{a['H1_core_star_count_mean']:>8.1f} "
              f"{a['null_h1_count_mean_mean']:>9.1f} "
              f"{h1_lift:>+7.1f} "
              f"{a['tri_full_star_mean']:>9.1f} "
              f"{a['tri_core_star_mean']:>9.1f} "
              f"{a['null_tri_mean_mean']:>9.1f} "
              f"{tri_lift:>+10.1f}")

    out = REPO / "outputs" / "xi_filtration_complete.json"
    out.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
