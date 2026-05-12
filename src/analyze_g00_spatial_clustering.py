"""Proper graph-based spatial clustering diagnostics for G_00 vs T_00
on within-P5 LAT-canonical regimes.

Replaces the previous 1D-modular np.roll adjacency (which was a
placeholder) with the natural Xi-graph adjacency:
  node i, j are neighbours iff Xi[i, j] > XI_THRESH

Diagnostics computed per regime:
  1. Xi-graph degree distribution
  2. Mean degree of matter-core nodes vs background
  3. Connected components of matter-core subgraph
  4. Largest-connected-component (GCC) fraction of core nodes
  5. Clustering coefficient C = 3 * triangles / triples
  6. Mean shortest-path-length on matter-core subgraph
  7. Spearman rho(T_00, degree)
  8. Spearman rho(G_00, degree)
  9. epsilon_in_GCC vs epsilon_outside_GCC

If matter is genuinely localized in spatially-extended structures,
we should see:
  - GCC fraction substantially > 0.10 (random-core baseline)
  - Mean degree(core) > Mean degree(background)
  - Clustering coefficient > random
  - epsilon_in_GCC > epsilon_outside (curvature concentrates in
    extended matter structures)
"""
from __future__ import annotations
import json
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


def build_xi_adjacency(xi_mat, threshold=XI_THRESH):
    """Symmetric adjacency from off-diagonal Xi-thresholding."""
    x = np.array(xi_mat, dtype=float)
    np.fill_diagonal(x, 0.0)
    x = np.where(np.isfinite(x), x, 0.0)
    # Symmetrise (already should be, but enforce)
    x = 0.5 * (x + x.T)
    adj = (x > threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


def connected_components(adj, mask):
    """BFS connected components on subgraph induced by `mask`.
    Returns list of components (each a list of node indices)."""
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []
    for start in range(n):
        if not mask[start] or visited[start]:
            continue
        comp = [start]
        visited[start] = True
        queue = [start]
        while queue:
            v = queue.pop(0)
            for u in np.where(adj[v] > 0)[0]:
                if mask[u] and not visited[u]:
                    visited[u] = True
                    comp.append(int(u))
                    queue.append(int(u))
        components.append(comp)
    return components


def clustering_coefficient(adj, mask=None):
    """Average clustering coefficient over nodes in `mask`
    (or all nodes if None)."""
    n = adj.shape[0]
    if mask is None:
        mask = np.ones(n, dtype=bool)
    coeffs = []
    for v in range(n):
        if not mask[v]:
            continue
        nbrs = np.where(adj[v] > 0)[0]
        k = nbrs.size
        if k < 2:
            continue
        sub = adj[np.ix_(nbrs, nbrs)]
        triangles = float(np.sum(sub) / 2)
        possible = k * (k - 1) / 2
        coeffs.append(triangles / possible if possible > 0 else 0.0)
    return float(np.mean(coeffs)) if coeffs else 0.0


def random_subset_baseline(adj, frac, n_trials=200, seed=0):
    """Baseline GCC fraction and clustering for random subset of size
    `frac * n` nodes."""
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    k = int(round(frac * n))
    if k < 2:
        return {"gcc_frac": 0.0, "clustering": 0.0,
                "mean_degree": 0.0}
    gcc_fracs, clust_arr, deg_arr = [], [], []
    for _ in range(n_trials):
        idx = rng.choice(n, k, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        comps = connected_components(adj, mask)
        if comps:
            largest = max(len(c) for c in comps)
            gcc_fracs.append(largest / k)
        else:
            gcc_fracs.append(0.0)
        clust_arr.append(clustering_coefficient(adj, mask))
        # mean degree of subset within subset
        sub = adj[np.ix_(idx, idx)]
        deg_arr.append(float(np.mean(sub.sum(axis=1))))
    return {
        "gcc_frac": float(np.mean(gcc_fracs)),
        "gcc_frac_std": float(np.std(gcc_fracs)),
        "clustering": float(np.mean(clust_arr)),
        "mean_degree": float(np.mean(deg_arr)),
    }


def main() -> int:
    print("=" * 78)
    print("Spatial clustering diagnostics, Xi-graph adjacency, LAT only")
    print("=" * 78)

    bundle = {"per_regime": {}}
    eps_GCC_vs_outside = []

    for reg, n_lat, rel, kind in LAT_LADDER:
        print(f"\n[{reg}, N={n_lat}]")
        payload = load_seed_payload(rel, n_lat, kind, max_seeds=32)
        if not payload:
            continue

        # Per-seed: build adjacency, compute G/T, then aggregate
        per_seed_records = []
        for seed_idx, (xi_mat, psi, kf, qf) in enumerate(payload):
            np.fill_diagonal(xi_mat, 1.0)
            try:
                prep = per_seed_galerkin(xi_mat, psi, kf, qf, n_lat, np)
            except Exception as e:
                print(f"  seed {seed_idx} galerkin failed: {e}")
                continue
            G = np.asarray(prep["g_00_h"])
            T = np.asarray(prep["t00"])
            adj = build_xi_adjacency(xi_mat, XI_THRESH)
            deg = adj.sum(axis=1)
            mean_deg = float(deg.mean())
            edge_density = float(adj.sum() / (n_lat * (n_lat - 1)))
            # Matter-core: top 10% T_00
            thr_T = float(np.quantile(T, 0.90))
            is_core = T >= thr_T
            n_core = int(is_core.sum())
            # Degree statistics
            deg_core = float(deg[is_core].mean()) if n_core > 0 else 0.0
            deg_back = float(deg[~is_core].mean()) if (n_lat - n_core) > 0 else 0.0
            # Connected components of core
            comps = connected_components(adj, is_core)
            comp_sizes = sorted([len(c) for c in comps], reverse=True)
            gcc_size = comp_sizes[0] if comp_sizes else 0
            gcc_frac = gcc_size / n_core if n_core > 0 else 0.0
            n_components = len(comps)
            # Clustering coefficient
            clust_core = clustering_coefficient(adj, is_core)
            clust_all = clustering_coefficient(adj)
            # Spearman rho(T, deg) and rho(G, deg)
            rT = np.argsort(np.argsort(T))
            rG = np.argsort(np.argsort(G))
            rD = np.argsort(np.argsort(deg))
            rho_T_deg = float(np.corrcoef(rT, rD)[0, 1])
            rho_G_deg = float(np.corrcoef(rG, rD)[0, 1])
            # epsilon in GCC vs outside-GCC
            gcc_nodes = comps[0] if comps else []
            in_gcc = np.zeros(n_lat, dtype=bool)
            in_gcc[gcc_nodes] = True
            G_gcc = G[in_gcc]
            T_gcc = T[in_gcc]
            G_out = G[~in_gcc]
            T_out = T[~in_gcc]
            if G_gcc.size > 1 and G_out.size > 1:
                eps_in = ((float(np.median(G_gcc))
                            - float(np.median(G_out)))
                          / (float(np.median(T_gcc))
                              - float(np.median(T_out)))) \
                          if abs(float(np.median(T_gcc))
                                 - float(np.median(T_out))) > 1e-9 \
                          else float("nan")
            else:
                eps_in = float("nan")

            per_seed_records.append({
                "seed": seed_idx,
                "edge_density": edge_density,
                "mean_deg": mean_deg,
                "n_core": n_core,
                "deg_core": deg_core,
                "deg_back": deg_back,
                "deg_ratio_core_back": deg_core / deg_back if deg_back > 0
                                        else float("nan"),
                "n_components": n_components,
                "gcc_size": gcc_size,
                "gcc_frac": gcc_frac,
                "comp_sizes_top5": comp_sizes[:5],
                "clust_core": clust_core,
                "clust_all": clust_all,
                "rho_T_deg": rho_T_deg,
                "rho_G_deg": rho_G_deg,
                "eps_GCC_vs_outside": eps_in,
            })

        # Random-subset baseline (1 representative seed for cost)
        if per_seed_records:
            xi_first = payload[0][0]
            np.fill_diagonal(xi_first, 1.0)
            adj_first = build_xi_adjacency(xi_first, XI_THRESH)
            baseline = random_subset_baseline(adj_first, frac=0.10,
                                                n_trials=100, seed=42)
        else:
            baseline = {"gcc_frac": float("nan"),
                        "clustering": float("nan"),
                        "mean_degree": float("nan")}

        # Aggregate
        if per_seed_records:
            avg = {
                "n_lat": n_lat,
                "n_seeds": len(per_seed_records),
                "edge_density_mean": float(np.mean(
                    [r["edge_density"] for r in per_seed_records])),
                "mean_deg_mean": float(np.mean(
                    [r["mean_deg"] for r in per_seed_records])),
                "deg_core_mean": float(np.mean(
                    [r["deg_core"] for r in per_seed_records])),
                "deg_back_mean": float(np.mean(
                    [r["deg_back"] for r in per_seed_records])),
                "deg_ratio_core_back_mean": float(np.mean(
                    [r["deg_ratio_core_back"] for r in per_seed_records
                     if np.isfinite(r["deg_ratio_core_back"])])),
                "n_components_mean": float(np.mean(
                    [r["n_components"] for r in per_seed_records])),
                "gcc_frac_mean": float(np.mean(
                    [r["gcc_frac"] for r in per_seed_records])),
                "clust_core_mean": float(np.mean(
                    [r["clust_core"] for r in per_seed_records])),
                "clust_all_mean": float(np.mean(
                    [r["clust_all"] for r in per_seed_records])),
                "rho_T_deg_mean": float(np.mean(
                    [r["rho_T_deg"] for r in per_seed_records])),
                "rho_G_deg_mean": float(np.mean(
                    [r["rho_G_deg"] for r in per_seed_records])),
                "eps_GCC_vs_outside_mean": float(np.mean(
                    [r["eps_GCC_vs_outside"] for r in per_seed_records
                     if np.isfinite(r["eps_GCC_vs_outside"])])),
                "baseline_gcc_frac": baseline["gcc_frac"],
                "baseline_clust": baseline["clustering"],
                "baseline_mean_deg": baseline["mean_degree"],
                "per_seed_records": per_seed_records,
            }
            bundle["per_regime"][reg] = avg

            print(f"  Edge density (Xi > {XI_THRESH}):  "
                  f"{avg['edge_density_mean']:.4f}")
            print(f"  Mean degree:                  {avg['mean_deg_mean']:.2f}")
            print(f"  Mean deg(core):               {avg['deg_core_mean']:.2f}")
            print(f"  Mean deg(back):               {avg['deg_back_mean']:.2f}")
            print(f"  Ratio deg(core)/deg(back):   "
                  f"{avg['deg_ratio_core_back_mean']:.3f}")
            print(f"  Connected components of core: "
                  f"{avg['n_components_mean']:.1f}")
            print(f"  GCC fraction (core in GCC):   "
                  f"{avg['gcc_frac_mean']:.3f}  vs baseline "
                  f"{avg['baseline_gcc_frac']:.3f}")
            print(f"  Clustering coeff (core):      "
                  f"{avg['clust_core_mean']:.4f}  vs all "
                  f"{avg['clust_all_mean']:.4f}  baseline "
                  f"{avg['baseline_clust']:.4f}")
            print(f"  Spearman rho(T_00, degree):   "
                  f"{avg['rho_T_deg_mean']:+.3f}")
            print(f"  Spearman rho(G_00, degree):   "
                  f"{avg['rho_G_deg_mean']:+.3f}")
            print(f"  epsilon_GCC_vs_outside:       "
                  f"{avg['eps_GCC_vs_outside_mean']:+.4f}")

            eps_GCC_vs_outside.append((n_lat, avg["eps_GCC_vs_outside_mean"]))

    # Cross-regime trend
    print()
    print("=" * 78)
    print("Cross-regime trend")
    print("=" * 78)
    rows = bundle["per_regime"]
    print(f"{'reg':>8} {'N':>4} {'GCC frac':>10} {'baseline':>10} "
          f"{'lift':>10} {'deg ratio':>11} {'rho(T,d)':>10} {'eps_GCC':>10}")
    for reg, n_lat, _, _ in LAT_LADDER:
        if reg not in rows:
            continue
        r = rows[reg]
        gcc_lift = r["gcc_frac_mean"] - r["baseline_gcc_frac"]
        print(f"{reg:>8} {n_lat:>4} {r['gcc_frac_mean']:>10.3f} "
              f"{r['baseline_gcc_frac']:>10.3f} "
              f"{gcc_lift:>+10.3f} "
              f"{r['deg_ratio_core_back_mean']:>11.3f} "
              f"{r['rho_T_deg_mean']:>+10.3f} "
              f"{r['eps_GCC_vs_outside_mean']:>+10.4f}")

    # Synthesis
    print()
    print("=" * 78)
    print("Synthesis")
    print("=" * 78)
    valid = [r for r in rows.values()]
    if valid:
        gcc_lifts = [r["gcc_frac_mean"] - r["baseline_gcc_frac"]
                     for r in valid]
        deg_ratios = [r["deg_ratio_core_back_mean"] for r in valid
                      if np.isfinite(r["deg_ratio_core_back_mean"])]
        rho_Ts = [r["rho_T_deg_mean"] for r in valid]
        eps_gccs = [r["eps_GCC_vs_outside_mean"] for r in valid
                    if np.isfinite(r["eps_GCC_vs_outside_mean"])]
        print(f"  Mean GCC-frac lift over baseline: "
              f"{float(np.mean(gcc_lifts)):+.3f} "
              f"(std {float(np.std(gcc_lifts)):.3f})")
        print(f"  Mean deg(core)/deg(back) ratio:   "
              f"{float(np.mean(deg_ratios)):.3f}")
        print(f"  Mean Spearman rho(T_00, degree):  "
              f"{float(np.mean(rho_Ts)):+.3f}")
        if eps_gccs:
            print(f"  Mean eps_GCC vs outside:          "
                  f"{float(np.mean(eps_gccs)):+.4f}")
        print()
        if float(np.mean(gcc_lifts)) > 0.20:
            print("  -> Matter-core nodes substantially MORE clustered than")
            print("     random; matter-curvature emergence is localized in")
            print("     extended structures (giant connected component).")
        elif float(np.mean(gcc_lifts)) > 0.05:
            print("  -> Mild clustering of matter-core nodes above random.")
        else:
            print("  -> No clustering signal: matter-core distribution")
            print("     consistent with random in the Xi-graph.")
        if float(np.mean(rho_Ts)) > 0.30:
            print(f"  -> Strong Spearman correlation rho(T_00, degree)="
                  f"{float(np.mean(rho_Ts)):+.2f}: matter concentrates")
            print("     where the Xi-graph is dense (high-degree nodes).")

    out = REPO / "outputs" / "g00_spatial_clustering.json"
    out.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
