"""Extended graph-theoretic cluster diagnostics on the Xi-graph,
LAT-canonical regimes only (P5, P5N64, P5N100, P5N300).

Diagnostics added beyond the basic GCC/clustering of the previous
analysis:

  1. Spectral gap lambda_2 - lambda_1 of the unnormalised Laplacian
     on the matter-core subgraph (related to the transport-operator
     spectral gap in the causal-wave equation).
  2. Assortativity (Pearson correlation of degrees across edges)
     -- measures whether high-degree connects to high-degree.
  3. Modularity of the matter-core community against random partition
     -- whether the matter-core forms a real community.
  4. Betti numbers via cycle rank (H_0 components, H_1 cycles) on
     matter-core subgraph.
  5. Edge expansion = boundary(core)/|core| -- isoperimetric quantity,
     related to graph-Laplacian eigenvalues.
  6. Eigenvector centrality of matter-core vs background.
  7. Mean shortest-path length and diameter of GCC.
  8. Algebraic connectivity (lambda_2 of normalised Laplacian).
  9. Triangle density correlation with T_00.
 10. Cycle-rank growth scaling with N.

Each diagnostic has implications for either the causal-wave
transport operator (P2) or the matter-curvature coupling (P4).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import deque

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


def build_xi_adjacency(xi_mat, threshold=XI_THRESH):
    x = np.array(xi_mat, dtype=float)
    np.fill_diagonal(x, 0.0)
    x = np.where(np.isfinite(x), x, 0.0)
    x = 0.5 * (x + x.T)
    adj = (x > threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


def connected_components(adj, mask):
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []
    for start in range(n):
        if not mask[start] or visited[start]:
            continue
        comp = [start]
        visited[start] = True
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in np.where(adj[v] > 0)[0]:
                if mask[u] and not visited[u]:
                    visited[u] = True
                    comp.append(int(u))
                    queue.append(int(u))
        components.append(comp)
    return components


def laplacian_spectral_gap(adj, normalised=True):
    """Return lambda_2 of L (or L_norm) -- algebraic connectivity."""
    n = adj.shape[0]
    deg = adj.sum(axis=1).astype(float)
    if normalised:
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg + 1e-12), 0.0)
        L = (np.eye(n) - (deg_inv_sqrt[:, None] * adj
                           * deg_inv_sqrt[None, :]))
    else:
        L = np.diag(deg) - adj.astype(float)
    eig = np.linalg.eigvalsh(L)
    eig = np.sort(eig)
    if eig.size < 2:
        return 0.0
    return float(eig[1])


def assortativity(adj):
    """Pearson correlation of degrees across edges."""
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    edges = np.array(np.where(np.triu(adj, k=1) > 0)).T
    if edges.shape[0] < 2:
        return 0.0
    d_left = deg[edges[:, 0]]
    d_right = deg[edges[:, 1]]
    if np.std(d_left) < 1e-9 or np.std(d_right) < 1e-9:
        return 0.0
    return float(np.corrcoef(d_left, d_right)[0, 1])


def edge_expansion(adj, mask):
    """Number of edges leaving mask divided by |mask|."""
    n_core = int(mask.sum())
    if n_core == 0 or n_core == adj.shape[0]:
        return 0.0
    boundary = 0
    for i in np.where(mask)[0]:
        for j in np.where(adj[i] > 0)[0]:
            if not mask[j]:
                boundary += 1
    return boundary / n_core


def cycle_rank(adj, mask):
    """H_1 ~ #edges - #nodes + #components on subgraph."""
    n_in = int(mask.sum())
    if n_in < 2:
        return 0
    sub = adj[np.ix_(np.where(mask)[0], np.where(mask)[0])]
    n_edges = int(sub.sum() / 2)
    comps = connected_components(adj, mask)
    return n_edges - n_in + len(comps)


def eigenvector_centrality(adj, max_iter=300, tol=1e-9):
    """Power iteration for principal eigenvector."""
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0)
    v = np.ones(n) / np.sqrt(n)
    for _ in range(max_iter):
        Av = adj @ v
        norm = float(np.linalg.norm(Av))
        if norm < 1e-12:
            return np.zeros(n)
        v_new = Av / norm
        if float(np.linalg.norm(v_new - v)) < tol:
            v = v_new
            break
        v = v_new
    return v


def shortest_path_lengths_gcc(adj, gcc_nodes):
    """Mean and max shortest-path-length within GCC via BFS."""
    if len(gcc_nodes) < 2:
        return 0.0, 0
    idx = {v: i for i, v in enumerate(gcc_nodes)}
    n_g = len(gcc_nodes)
    distances = []
    diameters = []
    for src in gcc_nodes:
        dist = {src: 0}
        queue = deque([src])
        while queue:
            v = queue.popleft()
            for u in np.where(adj[v] > 0)[0]:
                if int(u) in idx and int(u) not in dist:
                    dist[int(u)] = dist[v] + 1
                    queue.append(int(u))
        if dist:
            d_vals = list(dist.values())
            distances.extend(d_vals)
            diameters.append(max(d_vals))
    mean_d = float(np.mean(distances)) if distances else 0.0
    diam = int(max(diameters)) if diameters else 0
    return mean_d, diam


def modularity(adj, partition):
    """Newman-Girvan modularity with a partition (label per node)."""
    m = adj.sum() / 2.0
    if m < 1:
        return 0.0
    deg = adj.sum(axis=1)
    n = adj.shape[0]
    Q = 0.0
    for c in set(partition):
        idx = np.where(partition == c)[0]
        sub = adj[np.ix_(idx, idx)]
        e_cc = sub.sum() / 2.0
        a_c = deg[idx].sum() / (2 * m)
        Q += e_cc / m - a_c ** 2
    return float(Q)


def triangle_count_per_node(adj):
    """Count triangles each node is part of."""
    A2 = adj @ adj
    return np.array([int(A2[i, i] * adj[i, i] + 0)
                     if False else int((adj[i] @ A2[i]) / 2)
                     for i in range(adj.shape[0])], dtype=int)


def main() -> int:
    print("=" * 78)
    print("Extended cluster diagnostics on Xi-graph (LAT-canonical only)")
    print("=" * 78)

    bundle = {"per_regime": {}}

    for reg, n_lat, rel, kind in LAT_LADDER:
        print(f"\n[{reg}, N={n_lat}]")
        payload = load_seed_payload(rel, n_lat, kind, max_seeds=32)
        if not payload:
            continue

        per_seed = []
        for sidx, (xi_mat, psi, kf, qf) in enumerate(payload):
            np.fill_diagonal(xi_mat, 1.0)
            try:
                prep = per_seed_galerkin(xi_mat, psi, kf, qf, n_lat, np)
            except Exception as e:
                print(f"  seed {sidx} galerkin failed: {e}")
                continue
            G = np.asarray(prep["g_00_h"])
            T = np.asarray(prep["t00"])
            adj = build_xi_adjacency(xi_mat, XI_THRESH)
            deg = adj.sum(axis=1)
            n_edges = int(adj.sum() / 2)
            edge_density = float(n_edges / (n_lat * (n_lat - 1) / 2))

            # Matter-core
            thr_T = float(np.quantile(T, 0.90))
            is_core = T >= thr_T
            n_core = int(is_core.sum())

            # 1. Spectral gap (algebraic connectivity)
            try:
                lambda2_full = laplacian_spectral_gap(adj, normalised=True)
            except Exception:
                lambda2_full = float("nan")
            # On matter-core subgraph
            core_idx = np.where(is_core)[0]
            if core_idx.size >= 3:
                sub_adj = adj[np.ix_(core_idx, core_idx)]
                try:
                    lambda2_core = laplacian_spectral_gap(sub_adj,
                                                            normalised=True)
                except Exception:
                    lambda2_core = float("nan")
            else:
                lambda2_core = float("nan")

            # 2. Assortativity
            r_assort = assortativity(adj)

            # 3. Modularity using matter-core vs background partition
            partition = is_core.astype(int)
            Q_core = modularity(adj, partition)

            # 4. Betti numbers (H_0 = components, H_1 = cycles) on core
            comps = connected_components(adj, is_core)
            H0_core = len(comps)
            H1_core = cycle_rank(adj, is_core)
            comp_sizes = sorted([len(c) for c in comps], reverse=True)
            gcc_size = comp_sizes[0] if comp_sizes else 0

            # 5. Edge expansion of matter-core
            h_core = edge_expansion(adj, is_core)

            # 6. Eigenvector centrality
            ec = eigenvector_centrality(adj)
            ec_core = float(np.mean(ec[is_core])) if n_core > 0 else 0.0
            ec_back = float(np.mean(ec[~is_core])) if (n_lat - n_core) > 0 else 0.0
            # Spearman rho(ec, T_00)
            rEC = np.argsort(np.argsort(ec))
            rT = np.argsort(np.argsort(T))
            rho_ec_T = float(np.corrcoef(rEC, rT)[0, 1])
            rG = np.argsort(np.argsort(G))
            rho_ec_G = float(np.corrcoef(rEC, rG)[0, 1])

            # 7. Mean shortest-path on GCC
            if comps:
                gcc = comps[0]
                mean_path, diam = shortest_path_lengths_gcc(adj, gcc)
            else:
                mean_path, diam = 0.0, 0

            # 8. Triangle density correlation with T_00
            tri_per = triangle_count_per_node(adj)
            if np.std(tri_per) > 1e-9 and np.std(T) > 1e-9:
                rTri = np.argsort(np.argsort(tri_per))
                rho_tri_T = float(np.corrcoef(rTri, rT)[0, 1])
                rho_tri_G = float(np.corrcoef(rTri, rG)[0, 1])
            else:
                rho_tri_T = rho_tri_G = float("nan")

            per_seed.append({
                "seed": sidx,
                "edge_density": edge_density,
                "n_edges": n_edges,
                "lambda2_full": lambda2_full,
                "lambda2_core": lambda2_core,
                "assortativity": r_assort,
                "modularity_core": Q_core,
                "H0_core": H0_core,
                "H1_core": H1_core,
                "gcc_size": gcc_size,
                "edge_expansion_core": h_core,
                "ec_core": ec_core,
                "ec_back": ec_back,
                "ec_ratio_core_back": ec_core / ec_back if ec_back > 0
                                       else float("nan"),
                "rho_ec_T": rho_ec_T,
                "rho_ec_G": rho_ec_G,
                "mean_path_gcc": mean_path,
                "diameter_gcc": diam,
                "rho_tri_T": rho_tri_T,
                "rho_tri_G": rho_tri_G,
            })

        if not per_seed:
            continue

        # Aggregate
        keys_to_avg = ["edge_density", "n_edges",
                        "lambda2_full", "lambda2_core",
                        "assortativity", "modularity_core",
                        "H0_core", "H1_core", "gcc_size",
                        "edge_expansion_core", "ec_core", "ec_back",
                        "ec_ratio_core_back", "rho_ec_T", "rho_ec_G",
                        "mean_path_gcc", "diameter_gcc",
                        "rho_tri_T", "rho_tri_G"]
        agg = {"n_lat": n_lat, "n_seeds": len(per_seed)}
        for k in keys_to_avg:
            vals = [r[k] for r in per_seed
                    if isinstance(r[k], (int, float))
                    and np.isfinite(r[k])]
            agg[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg["per_seed"] = per_seed

        bundle["per_regime"][reg] = agg
        # Print summary
        print(f"  edge_density       = {agg['edge_density_mean']:.4f}, "
              f"n_edges = {agg['n_edges_mean']:.0f}")
        print(f"  algebraic_conn     full L_norm: lambda_2 = "
              f"{agg['lambda2_full_mean']:.4f}")
        print(f"  algebraic_conn     core subgraph: lambda_2 = "
              f"{agg['lambda2_core_mean']:.4f}")
        print(f"  assortativity      r = {agg['assortativity_mean']:+.3f}")
        print(f"  modularity (core)  Q = {agg['modularity_core_mean']:+.3f}")
        print(f"  Betti H_0 (core)   = {agg['H0_core_mean']:.1f}")
        print(f"  Betti H_1 (core)   = {agg['H1_core_mean']:.1f}")
        print(f"  edge expansion h   = {agg['edge_expansion_core_mean']:.3f}")
        print(f"  ec(core)/ec(back)  = {agg['ec_ratio_core_back_mean']:.2f}")
        print(f"  rho(ec, T_00)      = {agg['rho_ec_T_mean']:+.3f}")
        print(f"  rho(ec, G_00)      = {agg['rho_ec_G_mean']:+.3f}")
        print(f"  rho(triangles, T)  = {agg['rho_tri_T_mean']:+.3f}")
        print(f"  rho(triangles, G)  = {agg['rho_tri_G_mean']:+.3f}")
        print(f"  GCC mean path      = {agg['mean_path_gcc_mean']:.2f}")
        print(f"  GCC diameter       = {agg['diameter_gcc_mean']:.1f}")

    # Cross-regime trend
    print()
    print("=" * 78)
    print("Cross-regime trend table")
    print("=" * 78)
    print(f"{'reg':>8} {'N':>4} {'lam2_full':>9} {'lam2_core':>9} "
          f"{'assort':>7} {'Q':>7} {'h_exp':>7} "
          f"{'rho(ec,T)':>10} {'rho(ec,G)':>10}")
    for reg, n_lat, _, _ in LAT_LADDER:
        if reg not in bundle["per_regime"]:
            continue
        a = bundle["per_regime"][reg]
        print(f"{reg:>8} {n_lat:>4} "
              f"{a['lambda2_full_mean']:>9.4f} "
              f"{a['lambda2_core_mean']:>9.4f} "
              f"{a['assortativity_mean']:>+7.3f} "
              f"{a['modularity_core_mean']:>+7.3f} "
              f"{a['edge_expansion_core_mean']:>7.3f} "
              f"{a['rho_ec_T_mean']:>+10.3f} "
              f"{a['rho_ec_G_mean']:>+10.3f}")

    # Synthesis
    print()
    print("=" * 78)
    print("Synthesis: which diagnostics are physically informative")
    print("=" * 78)
    rs = list(bundle["per_regime"].values())
    if rs:
        Ns = np.array([r["n_lat"] for r in rs])
        lam_full = np.array([r["lambda2_full_mean"] for r in rs])
        lam_core = np.array([r["lambda2_core_mean"] for r in rs])
        assort = np.array([r["assortativity_mean"] for r in rs])
        mod = np.array([r["modularity_core_mean"] for r in rs])
        exp_h = np.array([r["edge_expansion_core_mean"] for r in rs])
        rho_ec_T = np.array([r["rho_ec_T_mean"] for r in rs])
        rho_ec_G = np.array([r["rho_ec_G_mean"] for r in rs])

        print()
        print("  Summary statistics across N=50,64,100,300:")
        print(f"    lambda_2 (full Laplacian):  mean = {lam_full.mean():.4f}, "
              f"range [{lam_full.min():.4f}, {lam_full.max():.4f}]")
        print(f"    lambda_2 (core subgraph):   mean = {lam_core.mean():.4f}, "
              f"range [{lam_core.min():.4f}, {lam_core.max():.4f}]")
        print(f"    assortativity:              mean = {assort.mean():+.3f}, "
              f"range [{assort.min():+.3f}, {assort.max():+.3f}]")
        print(f"    modularity (core):          mean = {mod.mean():+.3f}, "
              f"range [{mod.min():+.3f}, {mod.max():+.3f}]")
        print(f"    edge expansion h:           mean = {exp_h.mean():.3f}, "
              f"range [{exp_h.min():.3f}, {exp_h.max():.3f}]")
        print(f"    rho(eigvec_centrality, T):  mean = {rho_ec_T.mean():+.3f}, "
              f"range [{rho_ec_T.min():+.3f}, {rho_ec_T.max():+.3f}]")
        print(f"    rho(eigvec_centrality, G):  mean = {rho_ec_G.mean():+.3f}, "
              f"range [{rho_ec_G.min():+.3f}, {rho_ec_G.max():+.3f}]")

    out = REPO / "outputs" / "extended_cluster_diagnostics.json"
    out.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
