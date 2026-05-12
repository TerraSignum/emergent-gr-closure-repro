"""Multi-N M3-violation audit on the within-P5 ladder.

Extends the single-regime `audit_M3_violations.py` to the
within-$P_5$ ladder $N\in\{50,64,72,84,100,128,200,300\}$ via the
bundled snapshot NPZ payloads. Reports per-N statistics for
the four definitions:

  raw_count_rate, significant_count_rate (slack > 0.05),
  penalty_residual_L2, penalty_residual_max (sup-norm).

The sup-norm trend is the load-bearing diagnostic for the open
question of whether the L^infty triangle slack converges to zero
in the within-P5 continuum limit, complementing the previously
reported L^2 trend.

Output: outputs/audit_M3_violations_multi_N.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from audit_M3_violations import m3_violation_rate
from verify_galerkin_runner_A_hessian_ricci import edge_to_matrix

PARENT = REPO.parent

LADDER = [
    ("P5",     50, "results_d1_fix17/d1_p5.npz",                 "d1"),
    ("P5N64",  64, "results_d1_p5n64_24seeds/P5N64.snapshots.npz",  "snap"),
    ("P5N72",  72, "results_d1_p5n72_24seeds/P5N72.snapshots.npz",  "snap"),
    ("P5N84",  84, "results_d1_p5n84_24seeds/P5N84.snapshots.npz",  "snap"),
    ("P5N100",100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz","snap"),
    ("P5N128",128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz","snap"),
    ("P5N200",200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz", "snap"),
    ("P5N300",300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz",        "snap"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz",  "snap"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz",  "snap"),
]


def load_xi_seeds(rel_path: str, kind: str, n_lat: int):
    """Yield per-seed Xi matrices from either d1 NPZ or snapshot NPZ."""
    fp = PARENT / rel_path
    if not fp.exists():
        return []
    z = np.load(fp, allow_pickle=True)
    seeds = []
    if kind == "snap":
        snaps = z["edge_xi_snapshots"]
        last = snaps.shape[1] - 1
        ns = int(snaps.shape[0])
        for s in range(ns):
            xi = np.asarray(snaps[s, last], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            seeds.append(xi)
    elif kind == "d1" and "dense_cell_edge_xi_values" in z.files:
        edge = z["dense_cell_edge_xi_values"]
        ns = int(edge.shape[0])
        for s in range(ns):
            xi = edge_to_matrix(edge[s], n_lat).astype(float)
            np.fill_diagonal(xi, 1.0)
            seeds.append(xi)
    return seeds


def main():
    print("=" * 78)
    print("Multi-N M3-violation audit on the within-P5 ladder")
    print("=" * 78)
    print()
    print(f"  {'regime':<8} {'N':>4} {'n_seeds':>8} "
          f"{'raw_pct':>10} {'sig_pct':>10} "
          f"{'L2':>10} {'sup':>10}")
    print("-" * 78)

    rows = []
    for regime, n_lat, rel_path, kind in LADDER:
        xis = load_xi_seeds(rel_path, kind, n_lat)
        if not xis:
            print(f"  {regime:<8} {n_lat:>4} -- file missing: {rel_path}")
            continue
        per_seed = []
        for xi in xis:
            res = m3_violation_rate(xi, tol=1e-12)
            per_seed.append(res)
        rate_raw = np.mean([r["raw_count_rate"] for r in per_seed])
        rate_sig = np.mean([r["significant_count_rate"] for r in per_seed])
        L2 = np.mean([r["penalty_residual_L2"] for r in per_seed])
        sup = np.mean([r["penalty_residual_max"] for r in per_seed])
        std_L2 = np.std([r["penalty_residual_L2"] for r in per_seed])
        std_sup = np.std([r["penalty_residual_max"] for r in per_seed])
        print(f"  {regime:<8} {n_lat:>4} {len(per_seed):>8} "
              f"{rate_raw*100:>9.3f}% {rate_sig*100:>9.3f}% "
              f"{L2:>10.4f} {sup:>10.4f}")
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds": len(per_seed),
            "raw_count_rate": float(rate_raw),
            "significant_count_rate": float(rate_sig),
            "penalty_residual_L2_mean": float(L2),
            "penalty_residual_L2_std": float(std_L2),
            "penalty_residual_sup_mean": float(sup),
            "penalty_residual_sup_std": float(std_sup),
        })

    bundle = {
        "method": "M3_multi_N_within_P5",
        "ladder": rows,
        "definitions": {
            "raw_count_rate": "fraction of off-diagonal triples with Xi_ij * Xi_jk > Xi_ik + tol",
            "significant_count_rate": "same, with absolute slack > 0.05",
            "penalty_residual_L2": "sqrt(sum max(0, slack)^2 / n^3) -- gradient-flow penalty",
            "penalty_residual_sup": "max slack value across all valid triples",
        },
    }
    out = REPO / "outputs" / "audit_M3_violations_multi_N.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
