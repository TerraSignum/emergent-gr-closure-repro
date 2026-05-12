"""P5N512 topology extension: Euler characteristic and H_1 cycle
rank on the snapshot-NPZ ladder.

Closes the 'discrete-graph topology of T_00, G_00 reported on
only 4 regimes' open item by extending the topological-observable
audit to N=512 (12 seeds, 13 snapshots/seed). On the per-snapshot
final-time Xi-graph the script reports

  V          = number of nodes (= N)
  E(eps)     = number of off-diagonal edges with Xi_ij > eps
  triangles(eps) = closed 3-cliques on the eps-thresholded graph
  beta_0(eps) = number of connected components (union-find)
  beta_1(eps) = E - V + beta_0   (cycle rank, 1-skeleton)
  chi(eps)   = V - E + triangles  (2-complex Euler char)

at three structural thresholds
  eps in {p95, p99, p99.5}
to match the matter-core hierarchy of the parent paper.

Output: outputs/verify_p5n512_topology_extension.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

REPO = Path(__file__).resolve().parents[1]
REPO_ROOT = REPO.parent

LADDER = [
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz"),
    ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz"),
]


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> bool:
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


def topology_at_threshold(xi: np.ndarray, eps: float):
    """Compute (E, triangles, beta_0, beta_1, chi) at threshold eps."""
    n = xi.shape[0]
    mask = (xi > eps)
    np.fill_diagonal(mask, False)
    mask = np.triu(mask)
    edges_iu, edges_jv = np.nonzero(mask)
    n_edges = int(edges_iu.size)
    uf = UnionFind(n)
    for i, j in zip(edges_iu, edges_jv):
        uf.union(int(i), int(j))
    beta_0 = int(uf.n_components)
    beta_1 = n_edges - n + beta_0
    # Triangle count via adjacency-set intersection on thresholded
    # graph (symmetric adjacency)
    sym_mask = (xi > eps)
    np.fill_diagonal(sym_mask, False)
    sym_mask = sym_mask | sym_mask.T
    if n_edges <= 0:
        n_tri = 0
    else:
        # For N=512 the triple loop is O(E*deg_max). Use adjacency
        # matrix powering: tr(A^3) / 6 = number of triangles. A is
        # boolean-cast to float for matmul.
        a_f = sym_mask.astype(np.float32)
        a_sq = a_f @ a_f
        n_tri = int(np.einsum("ij,ji->", a_sq, a_f) // 6)
    chi = n - n_edges + n_tri
    return {
        "n_edges": n_edges,
        "n_triangles": n_tri,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "chi": chi,
    }


def percentile_eps(xi: np.ndarray, p: float) -> float:
    off = xi[~np.eye(xi.shape[0], dtype=bool)]
    return float(np.percentile(off, p))


def main() -> int:
    rows = []
    for label, n_lat, sub in LADDER:
        path = REPO_ROOT / sub
        if not path.exists():
            print(f"  skip {label}: {path} missing")
            continue
        z = np.load(path, allow_pickle=True)
        snaps = z["edge_xi_snapshots"]
        last_idx = snaps.shape[1] - 1
        n_seeds = int(snaps.shape[0])
        per_seed = []
        for s in range(n_seeds):
            xi = np.asarray(snaps[s, last_idx], dtype=float)
            xi = 0.5 * (xi + xi.T)
            eps_p95 = percentile_eps(xi, 95.0)
            eps_p99 = percentile_eps(xi, 99.0)
            eps_p995 = percentile_eps(xi, 99.5)
            t95 = topology_at_threshold(xi, eps_p95)
            t99 = topology_at_threshold(xi, eps_p99)
            t995 = topology_at_threshold(xi, eps_p995)
            per_seed.append({
                "seed": s,
                "eps_p95": eps_p95,
                "eps_p99": eps_p99,
                "eps_p99_5": eps_p995,
                "p95": t95,
                "p99": t99,
                "p99_5": t995,
            })

        def _agg(key, sub):
            vals = [r[key][sub] for r in per_seed]
            arr = np.array(vals, dtype=float)
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

        agg = {
            f"{p}_{q}": _agg(p, q)
            for p in ("p95", "p99", "p99_5")
            for q in ("n_edges", "n_triangles",
                      "beta_0", "beta_1", "chi")
        }

        row = {
            "regime_label": label,
            "N": n_lat,
            "n_seeds": n_seeds,
            "agg": agg,
            "per_seed": per_seed,
        }
        rows.append(row)
        print(f"  {label} N={n_lat} seeds={n_seeds}")
        for cut in ("p95", "p99", "p99_5"):
            ed = agg[f"{cut}_n_edges"]["mean"]
            tr = agg[f"{cut}_n_triangles"]["mean"]
            b0 = agg[f"{cut}_beta_0"]["mean"]
            b1 = agg[f"{cut}_beta_1"]["mean"]
            chi = agg[f"{cut}_chi"]["mean"]
            print(f"    {cut}: E={ed:.0f} tri={tr:.0f} "
                  f"beta_0={b0:.1f} beta_1={b1:.1f} "
                  f"chi={chi:.1f}")

    # Cross-regime trend: does beta_1 stay zero asymptotically?
    n_arr = np.array([r["N"] for r in rows], dtype=float)
    summary = {}
    for cut in ("p95", "p99", "p99_5"):
        b1_arr = np.array(
            [r["agg"][f"{cut}_beta_1"]["mean"] for r in rows])
        chi_arr = np.array(
            [r["agg"][f"{cut}_chi"]["mean"] for r in rows])
        # b1/N normalisation
        b1_per_N = b1_arr / n_arr
        chi_per_N = chi_arr / n_arr
        if len(n_arr) >= 2:
            slope_b1, _ = np.polyfit(np.log(n_arr),
                                      np.log(np.maximum(b1_arr, 1.0)), 1)
        else:
            slope_b1 = float("nan")
        summary[cut] = {
            "beta_1_per_N_at_largest_N":
                float(b1_per_N[-1]) if len(b1_per_N) else None,
            "chi_per_N_at_largest_N":
                float(chi_per_N[-1]) if len(chi_per_N) else None,
            "log_slope_beta_1_in_log_N": float(slope_b1),
            "beta_1_grows_at_least_linearly_in_N":
                bool(slope_b1 >= 0.95) if math.isfinite(
                    slope_b1) else None,
            "verdict_h1_zero_at_continuum":
                bool(b1_per_N[-1] < 0.5) if len(b1_per_N) else None,
        }

    bundle = {
        "method": "verify_p5n512_topology_extension",
        "schema_version": "1.0.0",
        "ladder": [r["regime_label"] for r in rows],
        "rows": rows,
        "cross_regime_summary": summary,
    }
    out = REPO / "outputs" / "verify_p5n512_topology_extension.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    print()
    print("=" * 70)
    print("Cross-regime trend (largest-N readouts)")
    print("=" * 70)
    for cut, s in summary.items():
        print(f"  {cut}: beta_1/N = "
              f"{s['beta_1_per_N_at_largest_N']:.4f}  "
              f"chi/N = {s['chi_per_N_at_largest_N']:.4f}  "
              f"log-slope beta_1 = {s['log_slope_beta_1_in_log_N']:.2f}  "
              f"H_1 zero at continuum: "
              f"{s['verdict_h1_zero_at_continuum']}")
    print(f"  saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
