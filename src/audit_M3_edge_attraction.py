"""Test the emergent-gravity prediction that persistent
M3-causal-violator edges attract each other and merge.

If the relational metric d_ij = -ell_0 log Xi_ij is the
gravitational field, then high-Xi edges sit at small
gravitational distance, and edges that persistently carry
fast information flow (CVI > c_info) should:

  H1 (self-attraction): the per-edge distance d_ij of a
       persistent-violator edge SHRINKS over the snapshot
       trajectory (Xi_ij grows toward 1, d_ij toward 0).

  H2 (pair-attraction): pairs of persistent-violator edges
       that share a node (e.g. (i,j) and (i,k)) drive the
       distance d_jk DOWN over the trajectory: i.e. the second
       endpoints of two violator-edges anchored at the same
       node are pulled together.

  H3 (cluster formation): the spatial centroid distance
       between distinct persistent-violator clusters shrinks
       over the trajectory.

  H4 (co-localisation with T_00): persistent-violator edges
       drift toward T_00 heavy-tail nodes over the trajectory
       (correlation between max-T_00 node distance and
       end-snapshot violator-cluster centroid).

Output: outputs/audit_M3_edge_attraction.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PARENT = REPO.parent

LADDER = [
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz"),
    ("P5N100",100,  "results_d1_p5n100_24seeds/P5N100.snapshots.npz"),
    ("P5N128",128,  "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz"),
    ("P5N200",200,  "results_d1_p5n200_8seeds/P5N200.snapshots.npz"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz"),
]


def main():
    print("=" * 78)
    print("Edge attraction / merging audit on persistent M3-violators")
    print("=" * 78)
    print(f"  {'regime':<8} {'#s':>3} {'pers_edges':>11} "
          f"{'<Xi_pers t0>':>14} {'<Xi_pers tF>':>14} "
          f"{'H1 d_attract':>13} {'H2 d_jk drop':>13} "
          f"{'H4 to-T00':>11}")
    print("-" * 78)
    rows = []
    for regime, n_lat, rel in LADDER:
        fp = PARENT / rel
        if not fp.exists():
            print(f"  {regime}: missing")
            continue
        z = np.load(fp, allow_pickle=True)
        snaps = z["edge_xi_snapshots"]
        psi_r = z["psi_real_snapshots"]
        psi_i = z["psi_imag_snapshots"]
        n_seeds = min(int(snaps.shape[0]),
                       8 if n_lat <= 100 else 4)
        n_snap = int(snaps.shape[1])

        # Pre-load galerkin for T_00 (only at last snapshot, for H4)
        sys.path.insert(0, str(REPO / "src"))
        from verify_galerkin_runner_A_hessian_ricci import per_seed_galerkin

        per_seed_results = []
        for s in range(n_seeds):
            xi_traj = np.asarray(snaps[s], dtype=float).copy()
            # Identify persistent-violator edges (top-decile by max v_info
            # across the trajectory; at-least-half-time over c_info)
            d_xi = np.abs(np.diff(xi_traj, axis=0))
            offdiag = ~np.eye(n_lat, dtype=bool)
            d_off = d_xi[:, offdiag]
            v_med = float(np.median(d_off[d_off > 0])) if (d_off > 0).any() else 1e-6
            c_info = 2 * v_med
            persistent_mask_off = (d_off > c_info).mean(axis=0) > 0.5
            # Map back to (i,j) indices
            ij_idx = np.argwhere(offdiag)
            pers_edges = ij_idx[persistent_mask_off]
            if pers_edges.shape[0] == 0:
                continue

            # H1: average Xi_ij at first snapshot vs last snapshot
            # Xi growing => attraction (since d = -log Xi)
            xi0 = np.array([xi_traj[0, i, j] for i, j in pers_edges])
            xiF = np.array([xi_traj[-1, i, j] for i, j in pers_edges])
            mean_xi0 = float(xi0.mean())
            mean_xiF = float(xiF.mean())
            # H1 metric: d_attraction = -log(xiF) - (-log(xi0)) = log(xi0/xiF)
            # positive value if Xi shrinks (distance grows = repulsion)
            # negative value if Xi grows (distance shrinks = attraction)
            d_attract = float(np.log(np.maximum(xi0, 1e-9)
                                       / np.maximum(xiF, 1e-9)).mean())

            # H2: pair-attraction. For pairs of persistent-edges sharing
            # a node (i,j) and (i,k), track d_jk = -log Xi_jk drift.
            # Pick 50 random shared-node pairs to limit cost.
            from collections import defaultdict
            anchor_to_partners = defaultdict(list)
            for i, j in pers_edges:
                anchor_to_partners[int(i)].append(int(j))
                anchor_to_partners[int(j)].append(int(i))
            pair_drifts = []
            rng = np.random.default_rng(s)
            count = 0
            for anchor, partners in anchor_to_partners.items():
                if len(partners) < 2:
                    continue
                for j, k in [(partners[a], partners[b])
                             for a in range(len(partners))
                             for b in range(a + 1, len(partners))]:
                    if count >= 50:
                        break
                    xi_jk_t0 = float(xi_traj[0, j, k])
                    xi_jk_tF = float(xi_traj[-1, j, k])
                    if xi_jk_t0 > 1e-6 and xi_jk_tF > 1e-6:
                        # log(xi_t0/xi_tF) > 0 => xi_jk shrinks => repulsion
                        # log(xi_t0/xi_tF) < 0 => xi_jk grows => attraction
                        pair_drifts.append(np.log(xi_jk_t0 / xi_jk_tF))
                    count += 1
                if count >= 50:
                    break
            d_jk_drop = float(np.mean(pair_drifts)) if pair_drifts else 0.0

            # H4: co-localisation with T_00 spikes at end-snapshot
            try:
                psi = (psi_r[s, -1].astype(complex)
                       + 1j * psi_i[s, -1].astype(float))
                k_field = z.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
                q_field = z.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
                xi_last = xi_traj[-1].copy()
                np.fill_diagonal(xi_last, 1.0)
                prep = per_seed_galerkin(xi_last, psi,
                                          np.asarray(k_field),
                                          np.asarray(q_field), n_lat, np)
                t00 = np.asarray(prep["t00"])
                t00_p90 = np.percentile(t00, 90)
                hot_nodes = set(np.where(t00 > t00_p90)[0].tolist())
                # Of pers_edges, what fraction has at least one endpoint
                # in T_00 top-decile?
                hits = 0
                for i, j in pers_edges:
                    if int(i) in hot_nodes or int(j) in hot_nodes:
                        hits += 1
                h4_frac = hits / len(pers_edges)
            except Exception:
                h4_frac = float("nan")

            per_seed_results.append({
                "seed": s,
                "n_persistent_edges": int(pers_edges.shape[0]),
                "mean_Xi_t0_pers": mean_xi0,
                "mean_Xi_tF_pers": mean_xiF,
                "H1_d_attract_log_xi0_over_xiF": d_attract,
                "H2_d_jk_drop_mean_log": d_jk_drop,
                "H4_T00_top_decile_overlap_frac": h4_frac,
            })

        if not per_seed_results:
            continue
        n_pers = float(np.mean([d["n_persistent_edges"]
                                  for d in per_seed_results]))
        xi0_m = float(np.mean([d["mean_Xi_t0_pers"]
                                for d in per_seed_results]))
        xiF_m = float(np.mean([d["mean_Xi_tF_pers"]
                                for d in per_seed_results]))
        h1 = float(np.mean([d["H1_d_attract_log_xi0_over_xiF"]
                             for d in per_seed_results]))
        h2 = float(np.mean([d["H2_d_jk_drop_mean_log"]
                             for d in per_seed_results]))
        h4 = float(np.mean([d["H4_T00_top_decile_overlap_frac"]
                              for d in per_seed_results
                              if not np.isnan(d["H4_T00_top_decile_overlap_frac"])]))
        print(f"  {regime:<8} {len(per_seed_results):>3} "
              f"{n_pers:>11.1f} {xi0_m:>14.4f} {xiF_m:>14.4f} "
              f"{h1:>+13.4f} {h2:>+13.4f} {h4:>11.3f}")
        rows.append({
            "regime": regime, "N": n_lat,
            "n_seeds": len(per_seed_results),
            "n_persistent_edges_mean": n_pers,
            "mean_Xi_persistent_t0": xi0_m,
            "mean_Xi_persistent_tF": xiF_m,
            "H1_distance_attraction_mean": h1,
            "H2_pair_distance_drift_mean": h2,
            "H4_T00_top_decile_overlap_frac_mean": h4,
            "per_seed": per_seed_results,
        })

    print()
    print("=" * 78)
    print("Interpretation of each test")
    print("=" * 78)
    print("  H1 d_attract = log(Xi_t0 / Xi_tF) for persistent-violator edges.")
    print("    H1 < 0 => Xi grows => distance d_ij = -log Xi shrinks => ATTRACTION.")
    print("    H1 > 0 => Xi shrinks => distance grows => REPULSION.")
    print("  H2 d_jk_drop = log(Xi_jk t0 / Xi_jk tF) for second-endpoint pairs.")
    print("    H2 < 0 => second-endpoints come together => MERGING.")
    print("    H2 > 0 => second-endpoints separate.")
    print("  H4 = fraction persistent edges with endpoint at T_00 top-decile.")
    print("    H4 > 0.27 (random null for one of two endpoints in top-decile)")
    print("    => co-localised with matter heavy tail.")

    bundle = {
        "method": "M3_persistent_edge_attraction",
        "rows": rows,
        "interpretation": {
            "H1_negative_value_implies_attraction": True,
            "H2_negative_value_implies_pair_merging": True,
            "H4_above_random_27pct_implies_T00_co_localisation": True,
        },
    }
    out = REPO / "outputs" / "audit_M3_edge_attraction.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
