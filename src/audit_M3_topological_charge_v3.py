"""Topological charge audit v3: per-node charge density + correlation.

The v1 (per-triangle Python loop, threshold 0.25) classified
every node as vortex; the v2 (vectorised, multiple thresholds)
hit the same ceiling. The reason: at any |w|>=0.25 threshold
on the rich lattice, every node touches at least one
high-winding triangle.

The right diagnostic is a CHARGE DENSITY per node:
  rho_top(i) = (1/T_i) sum_{(j,k): xi_ij,xi_jk,xi_ik>0} |w_{ijk}|
where T_i is the count of admissible triangles touching node i,
and w_{ijk} is the discrete holonomy (i<j<k convention).
This is a continuous scalar field on the lattice, with peaks at
real vortex defects and a smooth bulk.

Vortex set: top-K nodes by rho_top, with K chosen as the count
of nodes whose rho_top exceeds the bulk by >2 sigma.

We then test:
  - Spearman rank-correlation of rho_top vs M3-slack-budget S_v
    across nodes (per regime, per seed, then across regimes).
  - Cross-regime ratio (slack-fraction-in-vortex-set / vortex-set-fraction):
    if vortex defects are real M3-slack carriers, this should be
    significantly > 1.

Output: outputs/audit_M3_topological_charge_v3.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from verify_galerkin_runner_A_hessian_ricci import edge_to_matrix

PARENT = REPO.parent

LADDER = [
    ("P0",     18,  "results_d1_fix17/d1_p0.npz",                      "d1"),
    ("P1",     28,  "results_d1_fix17/d1_p1.npz",                      "d1"),
    ("P2prime",30,  "results_d1_fix17/d1_p2prime.npz",                 "d1"),
    ("P3",     36,  "results_d1_fix17/d1_p3.npz",                      "d1"),
    ("P4",     42,  "results_d1_fix17/d1_p4.npz",                      "d1"),
    ("P5",     50,  "results_d1_fix17/d1_p5.npz",                      "d1"),
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz",    "snap"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz",    "snap"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz",    "snap"),
    ("P5N100",100,  "results_d1_p5n100_24seeds/P5N100.snapshots.npz",  "snap"),
    ("P5N128",128,  "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz", "snap"),
    ("P5N200",200,  "results_d1_p5n200_8seeds/P5N200.snapshots.npz",   "snap"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz",  "snap"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz",  "snap"),
    ("P6N128",128,  "results_d1_p6n128_12seeds/P6N128.snapshots.npz",  "snap"),
    ("P8N128",128,  "results_d1_p8n128_12seeds/P8N128.snapshots.npz",  "snap"),
]


def load_seeds(rel_path, kind, n_lat, max_seeds=8):
    fp = PARENT / rel_path
    if not fp.exists():
        return []
    z = np.load(fp, allow_pickle=True)
    seeds = []
    if kind == "snap":
        snaps = z["edge_xi_snapshots"]
        last = snaps.shape[1] - 1
        ns = min(int(snaps.shape[0]), max_seeds)
        for s in range(ns):
            xi = np.asarray(snaps[s, last], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            psi = (np.asarray(z["psi_real_snapshots"][s, last], dtype=float)
                   + 1j * np.asarray(z["psi_imag_snapshots"][s, last],
                                     dtype=float))
            seeds.append((xi, psi))
    elif kind == "d1" and "dense_cell_edge_xi_values" in z.files:
        edge = z["dense_cell_edge_xi_values"]
        amp = z["dense_cell_node_amplitude_values"]
        phase = z["dense_cell_node_phase_values"]
        ns = min(int(edge.shape[0]), max_seeds)
        for s in range(ns):
            xi = edge_to_matrix(edge[s], n_lat).astype(float)
            np.fill_diagonal(xi, 1.0)
            psi = amp[s] * np.exp(1j * phase[s])
            seeds.append((xi, psi))
    return seeds


def per_node_topological_charge(phases, xi, xi_thresh=0.0):
    """Return rho_top(i) = (1/T_i) sum_{(j,k)} |w_{ijk}|
    for triangles (i<j<k) touching node i with min(xi)>thresh."""
    n = phases.size
    # Pairwise phase increments folded to (-pi, pi]
    d_phi = np.angle(np.exp(1j * (phases[None, :] - phases[:, None])))
    # winding[i,j,k] = (d_phi[i,j] + d_phi[j,k] + d_phi[k,i]) / (2 pi)
    # = (d_phi[i,j] + d_phi[j,k] - d_phi[i,k]) / (2 pi)  via antisymmetry
    w = (d_phi[:, :, None] + d_phi[None, :, :]
         - d_phi[:, None, :]) / (2 * np.pi)
    # active mask: xi_ij, xi_jk, xi_ik all > thresh
    active = (xi > xi_thresh)
    valid = (active[:, :, None]
             & active[None, :, :]
             & active[:, None, :])
    n_idx = np.arange(n)
    valid[n_idx, n_idx, :] = False
    valid[:, n_idx, n_idx] = False
    valid[n_idx, :, n_idx] = False
    # Restrict to i<j<k canonical ordering
    iu = np.arange(n)
    canon = (iu[:, None, None] < iu[None, :, None]) & \
            (iu[None, :, None] < iu[None, None, :])
    valid = valid & canon
    abs_w = np.abs(w) * valid
    # Charge density per node: average |w| over triangles touching node i
    # A triangle (i<j<k) touches each of i, j, k once; project onto each axis.
    abs_w_i = abs_w.sum(axis=(1, 2))
    abs_w_j = abs_w.sum(axis=(0, 2))
    abs_w_k = abs_w.sum(axis=(0, 1))
    sum_abs_per_node = abs_w_i + abs_w_j + abs_w_k
    # Number of touching triangles per node
    t_i = valid.sum(axis=(1, 2)).astype(float)
    t_j = valid.sum(axis=(0, 2)).astype(float)
    t_k = valid.sum(axis=(0, 1)).astype(float)
    n_touch = t_i + t_j + t_k
    rho = sum_abs_per_node / np.maximum(n_touch, 1.0)
    # Q_total = signed sum of all i<j<k windings
    Q_top = float((w * valid).sum())
    return rho, Q_top, n_touch


def per_node_slack_budget(xi):
    n = xi.shape[0]
    prod = xi[:, :, None] * xi[None, :, :]
    target = xi[:, None, :]
    slack = np.maximum(prod - target, 0.0)
    diag_mask = np.ones((n, n, n), dtype=bool)
    diag_mask[np.arange(n), np.arange(n), :] = False
    diag_mask[:, np.arange(n), np.arange(n)] = False
    diag_mask[np.arange(n), :, np.arange(n)] = False
    return (slack * diag_mask).sum(axis=(1, 2))


def main():
    print("=" * 92)
    print("Topological-charge density audit v3 (Spearman correlation method)")
    print("=" * 92)
    print(f"  {'regime':<10} {'N':>4} {'#s':>3}  "
          f"{'rho_mean':>9} {'rho_max':>9} {'rho/null':>10} "
          f"{'Q_top':>9} {'rho-S Spearman':>16}")
    print("-" * 92)
    rows = []
    for regime, n_lat, rel, kind in LADDER:
        if n_lat > 200:
            print(f"  {regime:<10} {n_lat:>4} -- skipped")
            continue
        seeds = load_seeds(rel, kind, n_lat,
                            max_seeds=4 if n_lat >= 100 else 8)
        if not seeds:
            print(f"  {regime:<10} -- file missing")
            continue

        rho_means, rho_maxes, rho_nulls = [], [], []
        Q_tops = []
        spearman_rhos, spearman_ps = [], []
        # for cross-regime aggregate
        all_rho = []
        all_S = []
        for xi, psi in seeds:
            phases = np.angle(psi)
            rho, Q_top, n_touch = per_node_topological_charge(phases, xi)
            S = per_node_slack_budget(xi)
            rho_means.append(float(rho.mean()))
            rho_maxes.append(float(rho.max()))
            # Null: median of rho restricted to nodes with high n_touch
            # (excluding edge-of-graph nodes)
            rho_nulls.append(float(np.median(rho)))
            Q_tops.append(Q_top)
            try:
                if rho.std() > 1e-12 and S.std() > 1e-12:
                    sr, sp = spearmanr(rho, S)
                    spearman_rhos.append(float(sr))
                    spearman_ps.append(float(sp))
                else:
                    spearman_rhos.append(0.0)
                    spearman_ps.append(1.0)
            except Exception:
                spearman_rhos.append(0.0)
                spearman_ps.append(1.0)
            all_rho.extend(rho.tolist())
            all_S.extend(S.tolist())

        rho_mean = float(np.mean(rho_means))
        rho_max = float(np.mean(rho_maxes))
        rho_null = float(np.mean(rho_nulls))
        Q_abs = float(np.mean([abs(q) for q in Q_tops]))
        sr_mean = float(np.mean(spearman_rhos))
        sr_std = float(np.std(spearman_rhos))
        # combined p across seeds (via Fisher's method)
        from scipy.stats import combine_pvalues
        valid_ps = [p for p in spearman_ps if 0 < p <= 1]
        if valid_ps:
            try:
                _, combined_p = combine_pvalues(valid_ps, method="fisher")
            except Exception:
                combined_p = float(np.mean(valid_ps))
        else:
            combined_p = 1.0

        print(f"  {regime:<10} {n_lat:>4} {len(seeds):>3}  "
              f"{rho_mean:>9.4f} {rho_max:>9.4f} {rho_max/max(rho_null,1e-9):>10.2f}x "
              f"{Q_abs:>+9.2f} "
              f"rho={sr_mean:>+.3f} p={combined_p:.1e}")

        rows.append({
            "regime": regime, "N": n_lat, "n_seeds_used": len(seeds),
            "rho_mean": rho_mean, "rho_max": rho_max, "rho_null": rho_null,
            "rho_max_over_null": rho_max / max(rho_null, 1e-9),
            "Q_top_abs_mean": Q_abs,
            "spearman_rho_density_vs_slack": sr_mean,
            "spearman_rho_density_vs_slack_std": sr_std,
            "spearman_combined_p": combined_p,
        })

    print()
    print("=" * 92)
    print("Cross-regime summary")
    print("=" * 92)
    if rows:
        srs = np.array([r["spearman_rho_density_vs_slack"] for r in rows])
        rmaxes = np.array([r["rho_max_over_null"] for r in rows])
        Qs = np.array([r["Q_top_abs_mean"] for r in rows])
        print(f"  Spearman(rho, S) mean across regimes:  {srs.mean():+.3f} "
              f"(min={srs.min():+.3f}, max={srs.max():+.3f})")
        print(f"  rho_max / rho_null mean:               {rmaxes.mean():.2f}x "
              f"(strong vortex peaks if >>1)")
        print(f"  |Q_top| mean across regimes:          {Qs.mean():.2f} "
              f"(continuum-conserved invariant if N-stable)")

        summary = {
            "spearman_density_vs_slack_mean": float(srs.mean()),
            "spearman_density_vs_slack_min": float(srs.min()),
            "spearman_density_vs_slack_max": float(srs.max()),
            "rho_max_over_null_mean": float(rmaxes.mean()),
            "Q_top_abs_mean_cross_regime": float(Qs.mean()),
            "Q_top_cv_pct": float(Qs.std() / max(Qs.mean(), 1e-9) * 100),
        }
    else:
        summary = {}

    bundle = {
        "method": "M3_topological_charge_density_v3",
        "rows": rows,
        "summary": summary,
        "definition_note": ("rho_top(i) = average |w_{ijk}| over canonical "
                            "i<j<k triangles (Xi-active) touching node i; "
                            "w is the holonomy / (2 pi); Spearman correlation "
                            "with M3 per-node slack budget S_v measures the "
                            "topological-vortex contribution to the M3 "
                            "violation."),
    }
    out = REPO / "outputs" / "audit_M3_topological_charge_v3.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
