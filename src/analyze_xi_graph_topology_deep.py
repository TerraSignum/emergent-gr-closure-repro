"""Deeper graph-topological analysis on the Xi-graph for P4
manuscript section. Three blocks:

  Block A: Eigenvector-centrality (EC) interpretation
    - Bootstrap CI on Spearman rho(EC, T_00) and rho(EC, G_00)
    - Per-seed variation (not just regime mean)
    - Compare EC ranking vs degree ranking
    - Compute null distribution (rho when T is randomly permuted)
    - Decay of EC mass: how concentrated is the centrality?

  Block B: Degree-concentration deep dive
    - Per-decile mean degree (not just top-10% vs rest)
    - Bootstrap CI on degree ratio
    - Null test: random T-assignment baseline
    - Tail-index: how peaked is the degree distribution at top?

  Block C: Betti-H_1 = 0 verification
    - Per-seed verification (every seed individually)
    - Compare to background-subgraph H_1
    - Compare to random-subgraph H_1 of same size
    - Edge density of core subgraph
    - Tree-vs-forest decomposition diagnostics
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


def cycle_rank(adj, mask):
    n_in = int(mask.sum())
    if n_in < 2:
        return 0
    sub = adj[np.ix_(np.where(mask)[0], np.where(mask)[0])]
    n_edges = int(sub.sum() / 2)
    comps = connected_components(adj, mask)
    return n_edges - n_in + len(comps)


def eigenvector_centrality(adj, max_iter=500, tol=1e-10):
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
            return v_new
        v = v_new
    return v


def spearman_rho(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_rho(x, y, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rho = spearman_rho(x[idx], y[idx])
        if np.isfinite(rho):
            out.append(rho)
    arr = np.asarray(out)
    return arr


def null_rho(x, y, n_perm=2000, seed=1):
    """Null distribution by permuting y."""
    rng = np.random.default_rng(seed)
    n = len(x)
    out = []
    for _ in range(n_perm):
        idx = rng.permutation(n)
        rho = spearman_rho(x, y[idx])
        if np.isfinite(rho):
            out.append(rho)
    return np.asarray(out)


def per_decile_degree(deg, T):
    """Mean degree per T_00 decile."""
    qs = np.quantile(T, np.linspace(0, 1, 11))
    out = []
    for i in range(10):
        if i < 9:
            mask = (T >= qs[i]) & (T < qs[i+1])
        else:
            mask = (T >= qs[i])
        if mask.sum() > 0:
            out.append({
                "decile": i+1,
                "n": int(mask.sum()),
                "T_med": float(np.median(T[mask])),
                "deg_mean": float(np.mean(deg[mask])),
                "deg_std": float(np.std(deg[mask])),
            })
    return out


def random_subset_betti(adj, k, n_trials=200, seed=2):
    """For random k-node subsets, compute H_1 distribution."""
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    if k < 2 or k > n:
        return np.zeros(n_trials)
    h1s = []
    for _ in range(n_trials):
        idx = rng.choice(n, k, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        h1s.append(cycle_rank(adj, mask))
    return np.asarray(h1s)


def main() -> int:
    print("=" * 78)
    print("Deep Xi-graph topology analysis for P4 manuscript section")
    print("=" * 78)

    bundle = {"per_regime": {}}

    for reg, n_lat, rel, kind in LAT_LADDER:
        print(f"\n[{reg}, N={n_lat}]")
        payload = load_seed_payload(rel, n_lat, kind, max_seeds=32)
        if not payload:
            continue

        per_seed_records = []
        for sidx, (xi_mat, psi, kf, qf) in enumerate(payload):
            np.fill_diagonal(xi_mat, 1.0)
            try:
                prep = per_seed_galerkin(xi_mat, psi, kf, qf, n_lat, np)
            except Exception as e:
                print(f"  seed {sidx} failed: {e}")
                continue
            G = np.asarray(prep["g_00_h"])
            T = np.asarray(prep["t00"])
            adj = build_xi_adjacency(xi_mat, XI_THRESH)
            deg = adj.sum(axis=1)
            ec = eigenvector_centrality(adj)

            # ----- Block A: Eigenvector centrality -----
            rho_ec_T = spearman_rho(ec, T)
            rho_ec_G = spearman_rho(ec, G)
            # Bootstrap CI
            boot_T = bootstrap_rho(ec, T, n_boot=500, seed=10 + sidx)
            boot_G = bootstrap_rho(ec, G, n_boot=500, seed=20 + sidx)
            # Null distribution by permutation
            null_T = null_rho(ec, T, n_perm=1000, seed=30 + sidx)
            null_G = null_rho(ec, G, n_perm=1000, seed=40 + sidx)
            # p-value: fraction of |null| >= |observed|
            p_T = float(np.mean(np.abs(null_T) >= abs(rho_ec_T)))
            p_G = float(np.mean(np.abs(null_G) >= abs(rho_ec_G)))
            # EC mass concentration: top-10% EC carries what fraction of total?
            ec_top10 = float(np.sum(np.sort(ec)[-max(1, n_lat // 10):])) \
                        / max(float(np.sum(ec)), 1e-12)
            # Degree-rank vs EC-rank correlation
            rho_deg_ec = spearman_rho(deg, ec)

            # ----- Block B: Degree concentration -----
            decile_data = per_decile_degree(deg, T)
            # Top decile and bottom decile
            top_dec = decile_data[-1] if decile_data else None
            bot_dec = decile_data[0] if decile_data else None
            if top_dec and bot_dec and bot_dec["deg_mean"] > 0:
                deg_ratio = top_dec["deg_mean"] / bot_dec["deg_mean"]
            else:
                deg_ratio = float("nan")
            # Bootstrap on deg_top10 / deg_back
            rng = np.random.default_rng(50 + sidx)
            boot_ratio = []
            n = T.size
            for _ in range(500):
                idx = rng.integers(0, n, n)
                Tb = T[idx]
                Db = deg[idx]
                thr = np.quantile(Tb, 0.90)
                core = Tb >= thr
                if core.sum() < 3 or (~core).sum() < 3:
                    continue
                d_c = float(np.mean(Db[core]))
                d_b = float(np.mean(Db[~core]))
                if d_b > 0:
                    boot_ratio.append(d_c / d_b)
            boot_ratio = np.asarray(boot_ratio)
            # Null: random T-permutation
            null_ratios = []
            for _ in range(1000):
                Tperm = rng.permutation(T)
                thr = np.quantile(Tperm, 0.90)
                core = Tperm >= thr
                if core.sum() < 3 or (~core).sum() < 3:
                    continue
                d_c = float(np.mean(deg[core]))
                d_b = float(np.mean(deg[~core]))
                if d_b > 0:
                    null_ratios.append(d_c / d_b)
            null_ratios = np.asarray(null_ratios)

            # ----- Block C: Betti H_1 = 0 verification -----
            thr_T = np.quantile(T, 0.90)
            is_core = T >= thr_T
            n_core = int(is_core.sum())
            comps_core = connected_components(adj, is_core)
            h0_core = len(comps_core)
            h1_core = cycle_rank(adj, is_core)
            # Edge density of core subgraph
            sub = adj[np.ix_(np.where(is_core)[0],
                              np.where(is_core)[0])]
            sub_edges = int(sub.sum() / 2)
            sub_density = (sub_edges / max(1, n_core * (n_core - 1) / 2)) \
                          if n_core > 1 else 0.0
            # Background H_1
            h1_back = cycle_rank(adj, ~is_core)
            n_back = int((~is_core).sum())
            # Random-subset H_1 distribution (matched core size)
            null_h1 = random_subset_betti(adj, n_core, n_trials=200,
                                            seed=60 + sidx)

            # Tree decomposition: comp sizes
            comp_sizes = sorted([len(c) for c in comps_core], reverse=True)

            per_seed_records.append({
                "seed": sidx,
                "n_lat": n_lat,
                # Block A
                "rho_ec_T": rho_ec_T, "rho_ec_G": rho_ec_G,
                "boot_T_q025": float(np.quantile(boot_T, 0.025)) if boot_T.size else float("nan"),
                "boot_T_q975": float(np.quantile(boot_T, 0.975)) if boot_T.size else float("nan"),
                "boot_G_q025": float(np.quantile(boot_G, 0.025)) if boot_G.size else float("nan"),
                "boot_G_q975": float(np.quantile(boot_G, 0.975)) if boot_G.size else float("nan"),
                "null_T_q975": float(np.quantile(np.abs(null_T), 0.975)) if null_T.size else float("nan"),
                "null_G_q975": float(np.quantile(np.abs(null_G), 0.975)) if null_G.size else float("nan"),
                "p_T": p_T, "p_G": p_G,
                "ec_top10_mass_fraction": ec_top10,
                "rho_deg_ec": rho_deg_ec,
                # Block B
                "deg_ratio_top_bot_decile": deg_ratio,
                "boot_ratio_q025": float(np.quantile(boot_ratio, 0.025)) if boot_ratio.size else float("nan"),
                "boot_ratio_median": float(np.median(boot_ratio)) if boot_ratio.size else float("nan"),
                "boot_ratio_q975": float(np.quantile(boot_ratio, 0.975)) if boot_ratio.size else float("nan"),
                "null_ratio_q975": float(np.quantile(null_ratios, 0.975)) if null_ratios.size else float("nan"),
                "decile_data": decile_data,
                # Block C
                "h0_core": h0_core, "h1_core": h1_core,
                "h1_back": h1_back,
                "sub_density_core": sub_density,
                "comp_sizes_top5": comp_sizes[:5],
                "null_h1_mean": float(null_h1.mean()) if null_h1.size else float("nan"),
                "null_h1_q95": float(np.quantile(null_h1, 0.95)) if null_h1.size else float("nan"),
                "null_h1_max": int(null_h1.max()) if null_h1.size else 0,
                "n_core": n_core, "n_back": n_back,
            })

        if not per_seed_records:
            continue

        # Aggregate across seeds
        def avg(key):
            vals = [r[key] for r in per_seed_records
                    if isinstance(r[key], (int, float))
                    and np.isfinite(r[key])]
            return float(np.mean(vals)) if vals else float("nan")

        agg = {
            "n_lat": n_lat,
            "n_seeds": len(per_seed_records),
            "rho_ec_T_mean": avg("rho_ec_T"),
            "rho_ec_T_min": float(np.min([r["rho_ec_T"] for r in per_seed_records])),
            "rho_ec_T_max": float(np.max([r["rho_ec_T"] for r in per_seed_records])),
            "rho_ec_G_mean": avg("rho_ec_G"),
            "rho_ec_G_min": float(np.min([r["rho_ec_G"] for r in per_seed_records])),
            "rho_ec_G_max": float(np.max([r["rho_ec_G"] for r in per_seed_records])),
            "boot_T_q025_mean": avg("boot_T_q025"),
            "boot_T_q975_mean": avg("boot_T_q975"),
            "boot_G_q025_mean": avg("boot_G_q025"),
            "boot_G_q975_mean": avg("boot_G_q975"),
            "p_T_max": float(np.max([r["p_T"] for r in per_seed_records])),
            "p_G_max": float(np.max([r["p_G"] for r in per_seed_records])),
            "ec_top10_mass_fraction_mean": avg("ec_top10_mass_fraction"),
            "rho_deg_ec_mean": avg("rho_deg_ec"),
            "deg_ratio_mean": avg("deg_ratio_top_bot_decile"),
            "boot_ratio_q025_mean": avg("boot_ratio_q025"),
            "boot_ratio_median_mean": avg("boot_ratio_median"),
            "boot_ratio_q975_mean": avg("boot_ratio_q975"),
            "null_ratio_q975_mean": avg("null_ratio_q975"),
            "h0_core_mean": avg("h0_core"),
            "h1_core_max": int(np.max([r["h1_core"] for r in per_seed_records])),
            "h1_core_all_zero": bool(all(r["h1_core"] == 0
                                          for r in per_seed_records)),
            "h1_back_mean": avg("h1_back"),
            "sub_density_core_mean": avg("sub_density_core"),
            "null_h1_q95_mean": avg("null_h1_q95"),
            "null_h1_mean_mean": avg("null_h1_mean"),
            "per_seed": per_seed_records,
        }
        bundle["per_regime"][reg] = agg

        # Print summary per regime
        print(f"  --- Block A: Eigenvector centrality ---")
        print(f"    rho_Spearman(EC, T_00) = {agg['rho_ec_T_mean']:+.3f} "
              f"(seeds: [{agg['rho_ec_T_min']:+.3f}, "
              f"{agg['rho_ec_T_max']:+.3f}])")
        print(f"    rho_Spearman(EC, G_00) = {agg['rho_ec_G_mean']:+.3f} "
              f"(seeds: [{agg['rho_ec_G_min']:+.3f}, "
              f"{agg['rho_ec_G_max']:+.3f}])")
        print(f"    Bootstrap 95% CI rho(EC,T): "
              f"[{agg['boot_T_q025_mean']:+.3f}, "
              f"{agg['boot_T_q975_mean']:+.3f}]")
        print(f"    Bootstrap 95% CI rho(EC,G): "
              f"[{agg['boot_G_q025_mean']:+.3f}, "
              f"{agg['boot_G_q975_mean']:+.3f}]")
        print(f"    Permutation p_T_max = {agg['p_T_max']:.4f}, "
              f"p_G_max = {agg['p_G_max']:.4f}")
        print(f"    EC top-10% mass fraction: "
              f"{agg['ec_top10_mass_fraction_mean']:.3f}")
        print(f"    rho(deg, EC) = {agg['rho_deg_ec_mean']:+.3f}")
        print(f"  --- Block B: Degree concentration ---")
        print(f"    deg(top10)/deg(bot10) = {agg['deg_ratio_mean']:.2f}")
        print(f"    Bootstrap 95% CI on deg_core/deg_back: "
              f"[{agg['boot_ratio_q025_mean']:.2f}, "
              f"{agg['boot_ratio_q975_mean']:.2f}]")
        print(f"    Null (perm) 97.5%: {agg['null_ratio_q975_mean']:.2f}")
        print(f"  --- Block C: Betti H_1 verification ---")
        print(f"    H_0 core = {agg['h0_core_mean']:.1f}, "
              f"H_1 core max across seeds = {agg['h1_core_max']}")
        print(f"    H_1 background = {agg['h1_back_mean']:.1f}")
        print(f"    All seeds H_1=0: {agg['h1_core_all_zero']}")
        print(f"    Core subgraph edge density = "
              f"{agg['sub_density_core_mean']:.4f}")
        print(f"    Null H_1 (random subgraph same size): mean = "
              f"{agg['null_h1_mean_mean']:.2f}, 95% = "
              f"{agg['null_h1_q95_mean']:.1f}")

    # Cross-regime synthesis
    print()
    print("=" * 78)
    print("Cross-regime synthesis")
    print("=" * 78)
    rs = list(bundle["per_regime"].values())
    if rs:
        print()
        print("Block A: rho(EC, T_00), rho(EC, G_00)")
        for r in rs:
            print(f"  N={r['n_lat']:>4}: rho(EC,T)={r['rho_ec_T_mean']:+.3f} "
                  f"95%CI [{r['boot_T_q025_mean']:+.3f}, "
                  f"{r['boot_T_q975_mean']:+.3f}], "
                  f"rho(EC,G)={r['rho_ec_G_mean']:+.3f} 95%CI "
                  f"[{r['boot_G_q025_mean']:+.3f}, "
                  f"{r['boot_G_q975_mean']:+.3f}]")
        print()
        print("Block B: deg ratio (top vs bottom decile of T_00)")
        for r in rs:
            print(f"  N={r['n_lat']:>4}: ratio = {r['deg_ratio_mean']:.2f} "
                  f"(boot 95% [{r['boot_ratio_q025_mean']:.2f}, "
                  f"{r['boot_ratio_q975_mean']:.2f}], "
                  f"null 97.5% < {r['null_ratio_q975_mean']:.2f})")
        print()
        print("Block C: Betti H_1 of matter-core subgraph")
        for r in rs:
            print(f"  N={r['n_lat']:>4}: max H_1 across seeds = "
                  f"{r['h1_core_max']}, "
                  f"all_seeds_H1=0: {r['h1_core_all_zero']}, "
                  f"null_H_1 mean={r['null_h1_mean_mean']:.2f}")

    out = REPO / "outputs" / "xi_graph_topology_deep.json"
    out.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
