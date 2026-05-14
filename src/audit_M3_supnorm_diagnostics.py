"""Multi-N L^infty M3-violation diagnostics.

Bundles three complementary tests of the open question
``does sup_ijk delta_ijk converge to zero on the within-P5
ladder?'' (the L^2 mean and significant-count rates do
converge; the sup-norm stays in [0.27, 0.33] across N=50..300):

  P1 (localization):   identify the top-K worst-case triples
                        (i,j,k) per regime, record their distance
                        statistics, T_00 ranks, and persistence
                        across seeds.

  P2 (conditional):    re-evaluate sup_ijk delta_ijk on
                        admissible-triple subsets (min-distance
                        threshold, no-defect-crossing, T_00
                        below-median).

  P7 (heavy-tail):     measure correlation between worst-case
                        triple location and T_00 heavy-tail
                        cluster (matter-localized hypothesis).
                        If high correlation, the persistent sup
                        is matter-localized and the
                        epsilon-quasimetric reading is
                        structural rather than numerical.

Output: outputs/audit_M3_supnorm_diagnostics.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from verify_galerkin_runner_A_hessian_ricci import (
    edge_to_matrix, per_seed_galerkin)

PARENT = REPO.parent

LADDER = [
    ("P5",     50, "results_d1_fix17/d1_p5.npz",                     "d1"),
    ("P5N64",  64, "results_d1_p5n64_24seeds/P5N64.snapshots.npz",   "snap"),
    ("P5N72",  72, "results_d1_p5n72_24seeds/P5N72.snapshots.npz",   "snap"),
    ("P5N84",  84, "results_d1_p5n84_24seeds/P5N84.snapshots.npz",   "snap"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz", "snap"),
    ("P5N128", 128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz", "snap"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz",  "snap"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz",         "snap"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz",  "snap"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz",  "snap"),
]


def load_xi_psi_kq_seed(rel_path: str, kind: str, n_lat: int, seed: int):
    fp = PARENT / rel_path
    if not fp.exists():
        return None
    z = np.load(fp, allow_pickle=True)
    if kind == "snap":
        snaps = z["edge_xi_snapshots"]
        last = snaps.shape[1] - 1
        if seed >= snaps.shape[0]:
            return None
        xi = np.asarray(snaps[seed, last], dtype=float).copy()
        np.fill_diagonal(xi, 1.0)
        psi = (np.asarray(z["psi_real_snapshots"][seed, last], dtype=float)
               + 1j * np.asarray(z["psi_imag_snapshots"][seed, last], dtype=float))
        k_field = z.get(f"ff_K_seed{seed}", np.full((n_lat, n_lat), 0.55))
        q_field = z.get(f"ff_Q_seed{seed}", np.full((n_lat, n_lat), 0.45))
    elif kind == "d1" and "dense_cell_edge_xi_values" in z.files:
        edge = z["dense_cell_edge_xi_values"]
        if seed >= edge.shape[0]:
            return None
        xi = edge_to_matrix(edge[seed], n_lat).astype(float)
        np.fill_diagonal(xi, 1.0)
        amp = z["dense_cell_node_amplitude_values"][seed]
        phase = z["dense_cell_node_phase_values"][seed]
        psi = amp * np.exp(1j * phase)
        k_field = z.get(f"ff_K_seed{seed}", np.full((n_lat, n_lat), 0.55))
        q_field = z.get(f"ff_Q_seed{seed}", np.full((n_lat, n_lat), 0.45))
    else:
        return None
    return xi, psi, np.asarray(k_field), np.asarray(q_field)


def topk_worst_triples(xi: np.ndarray, top_k: int = 20):
    """Return [(slack, i, j, k), ...] sorted descending."""
    n = xi.shape[0]
    prod = xi[:, :, None] * xi[None, :, :]
    target = xi[:, None, :]
    slack = prod - target
    diag_mask = np.ones((n, n, n), dtype=bool)
    diag_mask[np.arange(n), np.arange(n), :] = False
    diag_mask[:, np.arange(n), np.arange(n)] = False
    diag_mask[np.arange(n), :, np.arange(n)] = False
    slack_masked = np.where(diag_mask, slack, -1.0)
    flat = slack_masked.ravel()
    idx_sorted = np.argpartition(flat, -top_k)[-top_k:]
    idx_sorted = idx_sorted[np.argsort(flat[idx_sorted])[::-1]]
    out = []
    for k_flat in idx_sorted:
        s = float(flat[k_flat])
        if s <= 0:
            continue
        i, j, k = np.unravel_index(int(k_flat), slack.shape)
        out.append((s, int(i), int(j), int(k)))
    return out


def conditional_sup_subsets(xi: np.ndarray, t00: np.ndarray):
    """Return per-subset sup-slack:
       (a) all triples
       (b) min-distance > 0.4 quantile (drop near-degenerate)
       (c) T_00 below median for all three indices (avoid heavy-tail
           cluster crossings)
    """
    n = xi.shape[0]
    prod = xi[:, :, None] * xi[None, :, :]
    target = xi[:, None, :]
    slack = prod - target
    diag_mask = np.ones((n, n, n), dtype=bool)
    diag_mask[np.arange(n), np.arange(n), :] = False
    diag_mask[:, np.arange(n), np.arange(n)] = False
    diag_mask[np.arange(n), :, np.arange(n)] = False

    sup_all = float(np.maximum(slack * diag_mask, 0.0).max())

    # Subset (b): drop triples where any pair has Xi > 0.95 (near-equal).
    # Broadcast each pairwise mask explicitly to (n,n,n).
    near_thresh = 0.95
    pair_ij_low = (xi <= near_thresh)[:, :, None]   # (n,n,1) → broadcast to (n,n,n)
    pair_jk_low = (xi <= near_thresh)[None, :, :]   # (1,n,n)
    pair_ik_low = (xi <= near_thresh)[:, None, :]   # (n,1,n)
    pair_low_all = (np.broadcast_to(pair_ij_low, (n, n, n))
                    & np.broadcast_to(pair_jk_low, (n, n, n))
                    & np.broadcast_to(pair_ik_low, (n, n, n)))
    valid_b = diag_mask & pair_low_all
    sup_b = float(np.maximum(slack * valid_b, 0.0).max()) if valid_b.any() else 0.0

    # Subset (c): T_00 below median for all 3 indices
    t00_med = float(np.median(t00))
    low_t00 = t00 < t00_med
    low_mask = (low_t00[:, None, None] & low_t00[None, :, None]
                & low_t00[None, None, :])
    valid_c = diag_mask & low_mask
    sup_c = float(np.maximum(slack * valid_c, 0.0).max()) if valid_c.any() else 0.0

    # Subset (d): T_00 above 90th percentile — heavy-tail cluster
    t00_p90 = float(np.percentile(t00, 90))
    high_t00 = t00 > t00_p90
    high_mask_t = (high_t00[:, None, None] | high_t00[None, :, None]
                   | high_t00[None, None, :])
    valid_d = diag_mask & high_mask_t
    sup_d = float(np.maximum(slack * valid_d, 0.0).max()) if valid_d.any() else 0.0

    return {
        "sup_all": sup_all,
        "sup_no_near_unity_pair": sup_b,
        "sup_t00_below_median": sup_c,
        "sup_t00_above_p90_overlap": sup_d,
    }


def heavy_tail_correlation(top_triples, t00):
    """How often do the top-K worst-case triple indices land in
    the T_00 top-decile (heavy-tail cluster)?"""
    if not top_triples:
        return {}
    t00_p90 = float(np.percentile(t00, 90))
    high = t00 > t00_p90
    in_heavy = 0
    total = 0
    for s, i, j, k in top_triples:
        in_heavy += int(bool(high[i] or high[j] or high[k]))
        total += 1
    return {
        "top_K": total,
        "in_heavy_tail_count": in_heavy,
        "fraction": in_heavy / total if total else 0.0,
        "expected_random": 1 - 0.9**3,  # P(at least one of 3 in top-decile)
    }


def main():
    print("=" * 78)
    print("Multi-N L^infty M3-violation diagnostics: P1+P2+P7")
    print("=" * 78)

    rows = []
    for regime, n_lat, rel, kind in LADDER:
        # Use seed 0 for diagnostics (consistent across regimes)
        payload = load_xi_psi_kq_seed(rel, kind, n_lat, seed=0)
        if payload is None:
            print(f"  {regime:<8} N={n_lat}: file missing")
            continue
        xi, psi, k_field, q_field = payload
        np.fill_diagonal(xi, 1.0)

        # T_00 via per-seed Galerkin
        try:
            prep = per_seed_galerkin(xi.copy(), psi, k_field, q_field, n_lat, np)
            t00 = np.asarray(prep["t00"])
        except Exception as e:
            print(f"  {regime:<8} galerkin failed: {e}")
            continue

        # P1: top-K worst-case triples
        top_triples = topk_worst_triples(xi, top_k=20)

        # P2: conditional sup
        cond = conditional_sup_subsets(xi, t00)

        # P7: heavy-tail correlation
        ht = heavy_tail_correlation(top_triples, t00)

        # Triple-coordinate persistence: how often do the top-K triples
        # share an index pair across worst-case rankings?
        nodes_in_top = set()
        for s, i, j, k in top_triples[:10]:
            nodes_in_top.add(i); nodes_in_top.add(j); nodes_in_top.add(k)

        # Localization metric: fraction of unique indices vs total
        # (top-10 triples have at most 30 unique indices; if all 30
        # unique, violations are scattered; if e.g. 10 unique, they
        # cluster at a small set of nodes)
        n_unique = len(nodes_in_top)
        max_possible = min(n_lat, 30)
        loc_index = 1.0 - (n_unique - 3) / (max_possible - 3) if max_possible > 3 else 0.0

        sup_med = np.median([t[0] for t in top_triples[:10]]) if top_triples else 0.0

        row = {
            "regime": regime, "N": n_lat,
            "sup_seed0": top_triples[0][0] if top_triples else 0.0,
            "top10_median_slack": float(sup_med),
            "n_unique_indices_top10": n_unique,
            "localization_index": float(loc_index),
            "conditional_sup": cond,
            "heavy_tail_correlation": ht,
            "top5_triples": [
                {"slack": s, "i": i, "j": j, "k": k}
                for s, i, j, k in top_triples[:5]
            ],
        }
        rows.append(row)

        print(f"\n--- {regime} N={n_lat} ---")
        print(f"  sup_all          = {cond['sup_all']:.4f}")
        print(f"  sup excl Xi>0.95 = {cond['sup_no_near_unity_pair']:.4f}")
        print(f"  sup T_00<median  = {cond['sup_t00_below_median']:.4f}")
        print(f"  sup T_00 hot pt  = {cond['sup_t00_above_p90_overlap']:.4f}")
        print(f"  top-10 unique idx= {n_unique}/{max_possible} (loc_idx={loc_index:.3f})")
        print(f"  heavy-tail corr  = {ht['fraction']:.3%} (random {ht['expected_random']:.1%})")

    bundle = {
        "method": "M3_supnorm_diagnostics_P1P2P7",
        "rows": rows,
        "interpretation": {
            "P1_localization": ("If localization_index > 0.5 across N, "
                                "violations cluster at few nodes (topological)."),
            "P2_conditional": ("If sup_no_near_unity_pair << sup_all, "
                              "near-equal Xi pairs drive the sup; numerical."),
            "P7_heavy_tail": ("If heavy_tail_correlation.fraction >> random "
                             "expectation (~27%), worst-case triples cluster "
                             "in matter-localized region; structural."),
        },
    }
    out = REPO / "outputs" / "audit_M3_supnorm_diagnostics.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
