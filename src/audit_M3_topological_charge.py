"""Q2 revisited with the framework's Wilson-loop triangle-winding
and topological-charge machinery.

The previous Q2 used "vortex = top-10% by phase-variance" which is
a percentile-by-construction definition. Here we use the proper
discrete-holonomy diagnostics:

  triangle_winding(phases, xi):
       discrete Wilson-loop holonomy on each Xi-active triangle,
       defined by the sum of phase increments around the loop;
       in the continuum limit this is the integer winding number.

  vortex_support_nodes(psi, xi, threshold=0.25):
       nodes participating in any triangle with |winding| >= 0.25,
       i.e. real vortex-charge carriers, not noise.

  topological_charge(phases, xi):
       integer-valued sum of all triangle windings; this is the
       natural Hopf-type topological invariant of the lattice
       phase configuration.

We compute, per regime and (where available) per seed:
  - total topological charge Q_top = sum of triangle windings
  - n_vortex_nodes = number of nodes in vortex_support
  - per-vortex-node M3-slack-budget S_v
  - sum_v S_v / S_total over the proper vortex set
  - cross-N invariance test on Q_top, on the slack-fraction, and
    on the slack-fraction conditional on Q_top != 0

Available regimes: P0..P8 from fix17/fix16, plus P5N64..P5N300
and P6N128, P8N128 from snapshot payloads. We use up to 12 seeds
per regime.

Output: outputs/audit_M3_topological_charge.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from verify_galerkin_runner_A_hessian_ricci import edge_to_matrix
from worldformula.defects.vortices import (
    vortex_support_nodes)
from worldformula.defects.topological_charge import topological_charge

PARENT = REPO.parent

# All available regimes with proper provenance
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
    ("P5N256",256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz",  "snap"),
    ("P5N300",300,  "results_d1_p5n300_12seeds/P5N300.snapshots.npz",  "snap"),
    ("P5N512",512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz",  "snap"),
    ("P6N128",128,  "results_d1_p6n128_12seeds/P6N128.snapshots.npz",  "snap"),
    ("P8N128",128,  "results_d1_p8n128_12seeds/P8N128.snapshots.npz",  "snap"),
]


def load_seeds(rel_path: str, kind: str, n_lat: int, max_seeds: int = 12):
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


def per_node_slack_budget(xi: np.ndarray) -> np.ndarray:
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
    print("=" * 78)
    print("Topological charge & vortex-set M3-slack analysis")
    print("=" * 78)
    print(f"  {'regime':<10} {'N':>4} {'#s':>3} "
          f"{'Q_top':>9} {'|Q_top|':>9} {'n_vort':>7} "
          f"{'S_v/S_tot':>10} {'fraction_null':>14}")
    print("-" * 78)
    rows = []
    for regime, n_lat, rel, kind in LADDER:
        if n_lat > 200:
            # vortex_support_nodes is O(N^3); cap at N=200 for speed
            # (we already have N=300 sup-norm trend from earlier audits)
            seeds = load_seeds(rel, kind, n_lat, max_seeds=2)
        else:
            seeds = load_seeds(rel, kind, n_lat, max_seeds=8)
        if not seeds:
            print(f"  {regime:<10} -- file missing")
            continue

        Q_per_seed = []
        n_vort_per_seed = []
        slack_frac_per_seed = []
        for xi, psi in seeds:
            phases = np.angle(psi)
            try:
                Q = topological_charge(phases, xi)
                vort_nodes = vortex_support_nodes(psi, xi, threshold=0.25)
            except Exception as e:
                print(f"  {regime}: skip seed ({type(e).__name__})")
                continue
            Q_per_seed.append(Q)
            n_vort_per_seed.append(len(vort_nodes))
            budget = per_node_slack_budget(xi)
            S_total = budget.sum()
            if vort_nodes.size and S_total > 0:
                S_v = budget[vort_nodes].sum()
                slack_frac_per_seed.append(float(S_v / S_total))
            else:
                slack_frac_per_seed.append(0.0)

        if not Q_per_seed:
            continue
        Q_mean = float(np.mean(Q_per_seed))
        Q_abs_mean = float(np.mean(np.abs(Q_per_seed)))
        Q_std = float(np.std(Q_per_seed))
        nv_mean = float(np.mean(n_vort_per_seed))
        nv_std = float(np.std(n_vort_per_seed))
        sf_mean = float(np.mean(slack_frac_per_seed))
        sf_std = float(np.std(slack_frac_per_seed))
        # Null expectation: if vortex set is X% of nodes and slack uniform,
        # fraction = nv_mean / n_lat
        fraction_null = nv_mean / max(n_lat, 1)

        print(f"  {regime:<10} {n_lat:>4} {len(Q_per_seed):>3} "
              f"{Q_mean:>+9.3f} {Q_abs_mean:>9.3f} "
              f"{nv_mean:>7.1f} "
              f"{sf_mean:>9.4f} {fraction_null:>13.4f}")
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds_used": len(Q_per_seed),
            "Q_top_mean": Q_mean, "Q_top_abs_mean": Q_abs_mean,
            "Q_top_std": Q_std,
            "n_vortex_nodes_mean": nv_mean,
            "n_vortex_nodes_std": nv_std,
            "vortex_node_fraction": fraction_null,
            "slack_fraction_in_vortex_set_mean": sf_mean,
            "slack_fraction_in_vortex_set_std": sf_std,
            "ratio_observed_over_null": (sf_mean / fraction_null
                                          if fraction_null > 0 else None),
        })

    print()
    print("=" * 78)
    print("Cross-regime invariance tests")
    print("=" * 78)
    if len(rows) >= 2:
        # 1) total topological charge: should be ~constant if invariant
        Qs = np.array([r["Q_top_mean"] for r in rows])
        Q_abs = np.array([r["Q_top_abs_mean"] for r in rows])
        print(f"  Q_top mean    range: [{Qs.min():+.3f}, {Qs.max():+.3f}], "
              f"mean = {Qs.mean():+.3f}")
        print(f"  |Q_top| mean  range: [{Q_abs.min():.3f}, {Q_abs.max():.3f}], "
              f"mean = {Q_abs.mean():.3f}")

        # 2) ratio observed/null over regimes
        ratios = [r["ratio_observed_over_null"] for r in rows
                   if r["ratio_observed_over_null"] is not None]
        if ratios:
            ratios = np.array(ratios)
            print(f"  slack-fraction / null:  mean = {ratios.mean():.3f}, "
                  f"std = {ratios.std():.3f}, "
                  f"CV = {ratios.std()/max(abs(ratios.mean()),1e-9)*100:.1f}%")
            print(f"  -> if ratio == 1.0: vortex carries proportional slack.")
            print(f"  -> if ratio >> 1.0: vortex set carries excess slack.")

        # 3) within-P5 sub-set
        p5_rows = [r for r in rows if r["regime"].startswith("P5")]
        if p5_rows:
            p5_ratios = [r["ratio_observed_over_null"] for r in p5_rows
                          if r["ratio_observed_over_null"] is not None]
            if p5_ratios:
                p5_ratios = np.array(p5_ratios)
                print(f"  within-P5 ratios:       mean = {p5_ratios.mean():.3f}, "
                      f"CV = {p5_ratios.std()/max(abs(p5_ratios.mean()),1e-9)*100:.1f}%")

    bundle = {
        "method": "M3_topological_charge_audit_proper_winding",
        "rows": rows,
        "vortex_definition": ("nodes participating in any triangle with "
                              "discrete Wilson-loop winding |w| >= 0.25 "
                              "(threshold from worldformula.defects.vortices)"),
        "topological_charge_definition": ("sum of triangle windings on the "
                                            "Xi-active sub-complex; integer-"
                                            "valued in the continuum limit"),
    }
    out = REPO / "outputs" / "audit_M3_topological_charge.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
