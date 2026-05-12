"""Autoloop 5-stage resonance audit.

Solves the 5 user-points raised 2026-05-02 about 3-body materialization
resonance, in one comprehensive audit script:

S1 — Sparse subgraph (top-K-activity instead of threshold-based persistence)
     Sweep K in {N, 2N, 4N, 8N} top-active edges and report graph
     density of type-3/2/1 components per K.

S2 — 3-body resonance LIFT at sparse K
     For sparsest K where 1-edges, 2-chains, 3-cycles separately exist:
     measure resonance-node fraction (n3>=1 AND n2>=1 AND n1>=1)
     vs independent baseline; asym in resonance-triangles vs non-resonance.

S3 — Emergent-time 3-body transition (1->2->3 over snapshots)
     For each edge, track persistence-status across snapshots; categorize
     transition events: isolated-1 -> in-2-chain -> in-3-cycle.
     Report rate per regime; correlate with end-state asym.

S4 — Spiral backreaction at triangle-formation events
     For each triangle that FORMS during the trajectory (last edge
     activates last), measure psi-phase drift at the 3 corner nodes
     between (formation_t-1) and formation_t. Test if drift sign/magnitude
     correlates with triangle phase-class (PPN/PNN).

S5 — +1 traveling: track end-only-persistent isolated edges (top-K
     definition), see if they drift spatially over snapshots toward
     2-chain attachment points.

Output: outputs/audit_autoloop_5stage_resonance.json
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
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
    ("P5N256",256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz"),
    ("P5N300",300,  "results_d1_p5n300_12seeds/P5N300.snapshots.npz"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz"),
]


def topk_persistent_edges(d_xi, n_lat, K):
    """Top-K persistent edges by trajectory-mean |Delta Xi|.
    d_xi shape: (n_snap-1, N, N).
    Returns set of (i,j) with i<j, len = K (or fewer if not enough).
    """
    activity = d_xi.mean(axis=0)
    np.fill_diagonal(activity, 0)
    # Take upper triangle only
    iu, ju = np.triu_indices(n_lat, k=1)
    vals = activity[iu, ju]
    if vals.size == 0:
        return set()
    # Top-K
    K_eff = min(K, vals.size)
    top_idx = np.argpartition(-vals, K_eff - 1)[:K_eff]
    edges = set()
    for idx in top_idx:
        a, b = int(iu[idx]), int(ju[idx])
        edges.add((min(a, b), max(a, b)))
    return edges


def classify_subgraph(edge_set, n_lat):
    """Find 3-cycles in edge_set, then classify each edge as type3/2/1.
    Type-3: in any triangle.
    Type-2: not type-3, but has at least one persistent neighbor at
            either endpoint.
    Type-1: isolated.
    Returns: triangles, type3/type2/type1 dicts, node_n3/n2/n1 arrays.
    """
    adj = defaultdict(set)
    for a, b in edge_set:
        adj[a].add(b)
        adj[b].add(a)
    triangles = []
    tri_edges = set()
    for (i, j) in edge_set:
        common = adj[i] & adj[j]
        for k in common:
            if k <= j:
                continue
            triangles.append((i, j, k))
            tri_edges.add((min(i, j), max(i, j)))
            tri_edges.add((min(j, k), max(j, k)))
            tri_edges.add((min(i, k), max(i, k)))
    type3 = {e: e in tri_edges for e in edge_set}
    type1 = {}
    type2 = {}
    for (a, b) in edge_set:
        nbrs_a = adj[a] - {b}
        nbrs_b = adj[b] - {a}
        if type3[(a, b)]:
            type1[(a, b)] = False
            type2[(a, b)] = False
        elif not nbrs_a and not nbrs_b:
            type1[(a, b)] = True
            type2[(a, b)] = False
        else:
            type1[(a, b)] = False
            type2[(a, b)] = True
    node_n3 = np.zeros(n_lat, dtype=int)
    node_n2 = np.zeros(n_lat, dtype=int)
    node_n1 = np.zeros(n_lat, dtype=int)
    for (a, b) in edge_set:
        if type3[(a, b)]:
            node_n3[a] += 1
            node_n3[b] += 1
        elif type2[(a, b)]:
            node_n2[a] += 1
            node_n2[b] += 1
        elif type1[(a, b)]:
            node_n1[a] += 1
            node_n1[b] += 1
    return triangles, type3, type2, type1, node_n3, node_n2, node_n1, adj


def asym_for_triangles(triangles, psi_last):
    if not triangles:
        return float("nan"), 0, 0
    phi = np.angle(psi_last)
    n_PPN = 0
    n_PNN = 0
    for i, j, k in triangles:
        d_ij = np.angle(np.exp(1j * (phi[j] - phi[i])))
        d_jk = np.angle(np.exp(1j * (phi[k] - phi[j])))
        d_ki = np.angle(np.exp(1j * (phi[i] - phi[k])))
        if (abs(d_ij) < 1e-9 or abs(d_jk) < 1e-9
            or abs(d_ki) < 1e-9):
            continue
        n_pos = ((1 if d_ij > 0 else 0) + (1 if d_jk > 0 else 0)
                  + (1 if d_ki > 0 else 0))
        if n_pos == 2:
            n_PPN += 1
        elif n_pos == 1:
            n_PNN += 1
    if n_PPN + n_PNN == 0:
        return float("nan"), n_PPN, n_PNN
    return (n_PPN - n_PNN) / (n_PPN + n_PNN), n_PPN, n_PNN


def stage_1_2(xi_traj, psi_last, n_lat):
    """Stage 1+2: top-K subgraph density profile + 3-body LIFT + asym.
    Sweeps K to find a regime where type1/2/3 separate."""
    d_xi = np.abs(np.diff(xi_traj, axis=0))
    K_values = [n_lat, 2 * n_lat, 4 * n_lat, 8 * n_lat]
    out = {}
    for K in K_values:
        edges = topk_persistent_edges(d_xi, n_lat, K)
        if not edges:
            continue
        (triangles, type3, type2, type1, node_n3, node_n2, node_n1,
          adj) = classify_subgraph(edges, n_lat)
        n_t3 = sum(type3.values())
        n_t2 = sum(type2.values())
        n_t1 = sum(type1.values())
        # Resonance-node = node with n3>=1 AND n2>=1 AND n1>=1
        res_mask = (node_n3 >= 1) & (node_n2 >= 1) & (node_n1 >= 1)
        n_res = int(res_mask.sum())
        frac_res = n_res / n_lat
        p_n3 = float((node_n3 >= 1).mean())
        p_n2 = float((node_n2 >= 1).mean())
        p_n1 = float((node_n1 >= 1).mean())
        indep = p_n3 * p_n2 * p_n1
        lift = (frac_res / indep) if indep > 1e-12 else float("nan")
        # Asym in resonance triangles vs non-resonance
        res_tri = [t for t in triangles
                    if res_mask[t[0]] and res_mask[t[1]]
                      and res_mask[t[2]]]
        non_tri = [t for t in triangles
                    if not (res_mask[t[0]] or res_mask[t[1]]
                              or res_mask[t[2]])]
        asym_res, _, _ = asym_for_triangles(res_tri, psi_last)
        asym_non, _, _ = asym_for_triangles(non_tri, psi_last)
        asym_all, _, _ = asym_for_triangles(triangles, psi_last)
        out[f"K{K}"] = {
            "K": K,
            "n_type3": n_t3, "n_type2": n_t2, "n_type1": n_t1,
            "n_triangles": len(triangles),
            "frac_resonance_nodes": frac_res,
            "indep_baseline": indep,
            "lift_3body": lift,
            "asym_all": asym_all,
            "asym_resonance": asym_res,
            "asym_non_resonance": asym_non,
            "n_resonance_triangles": len(res_tri),
            "n_non_resonance_triangles": len(non_tri),
        }
    return out


def stage_3_emergent_time_transitions(xi_traj, n_lat, K_factor=4):
    """Stage 3: track edge classification changes over snapshots.
    Per snapshot, take top-K edges by INSTANTANEOUS Xi; compute
    type3/2/1 status. Edge transitions: 1 -> 2 -> 3 (or any direction).
    Returns transition counts per type."""
    K = K_factor * n_lat
    n_snap = xi_traj.shape[0]
    # Per-snapshot edge classification
    edge_status = {}  # edge -> list of types per snapshot ('1','2','3', or None)
    # We track only edges that appear in TOP-K at any snapshot
    all_edges = set()
    types_per_snap = []
    for t in range(n_snap):
        xi_t = xi_traj[t].copy()
        np.fill_diagonal(xi_t, 0)
        iu, ju = np.triu_indices(n_lat, k=1)
        vals = xi_t[iu, ju]
        if vals.size == 0:
            types_per_snap.append({})
            continue
        K_eff = min(K, vals.size)
        top_idx = np.argpartition(-vals, K_eff - 1)[:K_eff]
        edges = set((int(iu[i]), int(ju[i])) for i in top_idx)
        all_edges |= edges
        triangles, type3, type2, type1, _, _, _, _ = classify_subgraph(
            edges, n_lat)
        snap_type = {}
        for e in edges:
            if type3.get(e):
                snap_type[e] = 3
            elif type2.get(e):
                snap_type[e] = 2
            elif type1.get(e):
                snap_type[e] = 1
        types_per_snap.append(snap_type)
    # Count transitions per edge
    transitions = defaultdict(int)  # (from, to) -> count
    for e in all_edges:
        prev = None
        for t in range(n_snap):
            cur = types_per_snap[t].get(e)
            if prev is not None and cur is not None and prev != cur:
                transitions[(prev, cur)] += 1
            prev = cur
    return {
        "K": K,
        "n_unique_edges_tracked": len(all_edges),
        "transitions_1_to_2": transitions[(1, 2)],
        "transitions_2_to_1": transitions[(2, 1)],
        "transitions_2_to_3": transitions[(2, 3)],
        "transitions_3_to_2": transitions[(3, 2)],
        "transitions_1_to_3": transitions[(1, 3)],
        "transitions_3_to_1": transitions[(3, 1)],
        "net_to_3": (transitions[(1, 3)] + transitions[(2, 3)]
                       - transitions[(3, 1)] - transitions[(3, 2)]),
        "net_to_1": (transitions[(2, 1)] + transitions[(3, 1)]
                       - transitions[(1, 2)] - transitions[(1, 3)]),
    }


def stage_4_spiral_backreaction(xi_traj, psi_real_traj, psi_imag_traj,
                                  n_lat, K_factor=4):
    """Stage 4: for each end-state triangle that FORMS during the
    trajectory (becomes type-3 in the last snapshot, was not before),
    measure psi-phase change at the 3 corners between snapshot-1 and
    last snapshot."""
    K = K_factor * n_lat
    n_snap = xi_traj.shape[0]
    if n_snap < 2:
        return {"n_formation_events": 0,
                "phi_drift_PPN": float("nan"),
                "phi_drift_PNN": float("nan")}
    # End-state classification
    xi_T = xi_traj[-1].copy()
    np.fill_diagonal(xi_T, 0)
    iu, ju = np.triu_indices(n_lat, k=1)
    vals = xi_T[iu, ju]
    K_eff = min(K, vals.size)
    top_idx = np.argpartition(-vals, K_eff - 1)[:K_eff]
    edges_T = set((int(iu[i]), int(ju[i])) for i in top_idx)
    triangles_T, _, _, _, _, _, _, _ = classify_subgraph(edges_T, n_lat)
    # Pre-state classification (snapshot 1)
    xi_pre = xi_traj[1].copy() if n_snap > 1 else xi_traj[0].copy()
    np.fill_diagonal(xi_pre, 0)
    vals_pre = xi_pre[iu, ju]
    top_pre = np.argpartition(-vals_pre, K_eff - 1)[:K_eff]
    edges_pre = set((int(iu[i]), int(ju[i])) for i in top_pre)
    # Triangle is "formation event" if all three edges in T but not all in pre
    formation_events = []
    for (i, j, k) in triangles_T:
        e_ij = (min(i, j), max(i, j))
        e_jk = (min(j, k), max(j, k))
        e_ik = (min(i, k), max(i, k))
        in_pre = (e_ij in edges_pre) + (e_jk in edges_pre) + (e_ik in edges_pre)
        if in_pre < 3:
            formation_events.append((i, j, k))
    if not formation_events:
        return {"n_formation_events": 0,
                "phi_drift_PPN": float("nan"),
                "phi_drift_PNN": float("nan")}
    psi_pre = psi_real_traj[1] + 1j * psi_imag_traj[1] if n_snap > 1 \
              else psi_real_traj[0] + 1j * psi_imag_traj[0]
    psi_T = psi_real_traj[-1] + 1j * psi_imag_traj[-1]
    phi_pre = np.angle(psi_pre)
    phi_T = np.angle(psi_T)
    drift = np.angle(np.exp(1j * (phi_T - phi_pre)))  # node-wise
    # Categorize formation events by triangle phase class at end-state
    drift_by_class = {"PPN": [], "PNN": []}
    for (i, j, k) in formation_events:
        d_ij = np.angle(np.exp(1j * (phi_T[j] - phi_T[i])))
        d_jk = np.angle(np.exp(1j * (phi_T[k] - phi_T[j])))
        d_ki = np.angle(np.exp(1j * (phi_T[i] - phi_T[k])))
        if abs(d_ij) < 1e-9 or abs(d_jk) < 1e-9 or abs(d_ki) < 1e-9:
            continue
        n_pos = ((1 if d_ij > 0 else 0) + (1 if d_jk > 0 else 0)
                  + (1 if d_ki > 0 else 0))
        # Phi drift at 3 corners (sign of sum)
        d_total = drift[i] + drift[j] + drift[k]
        if n_pos == 2:
            drift_by_class["PPN"].append(d_total)
        elif n_pos == 1:
            drift_by_class["PNN"].append(d_total)
    return {
        "n_formation_events": len(formation_events),
        "n_PPN_events": len(drift_by_class["PPN"]),
        "n_PNN_events": len(drift_by_class["PNN"]),
        "phi_drift_PPN": (float(np.mean(drift_by_class["PPN"]))
                            if drift_by_class["PPN"] else float("nan")),
        "phi_drift_PNN": (float(np.mean(drift_by_class["PNN"]))
                            if drift_by_class["PNN"] else float("nan")),
    }


def stage_5_traveling_isolates(xi_traj, n_lat, K_factor=2):
    """Stage 5: track 'isolated' edges (type-1 in some snapshots) and
    measure how their lattice-position changes over time. Compute
    'travel distance' as max(node-index) - min(node-index) across
    snapshots for the same edge identity (we don't track specific edges
    by ID since edges can swap; instead we track the TYPE-1 set's
    spatial spread)."""
    K = K_factor * n_lat
    n_snap = xi_traj.shape[0]
    type1_positions = []  # list of arrays of node-position-of-type1-edges
    for t in range(n_snap):
        xi_t = xi_traj[t].copy()
        np.fill_diagonal(xi_t, 0)
        iu, ju = np.triu_indices(n_lat, k=1)
        vals = xi_t[iu, ju]
        K_eff = min(K, vals.size)
        if K_eff <= 0:
            type1_positions.append(np.array([]))
            continue
        top_idx = np.argpartition(-vals, K_eff - 1)[:K_eff]
        edges = set((int(iu[i]), int(ju[i])) for i in top_idx)
        _, type3, type2, type1, _, _, _, _ = classify_subgraph(edges, n_lat)
        positions = [(a + b) / 2.0 for (a, b) in edges if type1.get((a, b))]
        type1_positions.append(np.array(positions))
    n_type1_per_snap = [len(p) for p in type1_positions]
    # Travel: difference of mean type1-position between first and last
    if not type1_positions[0].size or not type1_positions[-1].size:
        travel = float("nan")
    else:
        travel = abs(float(np.mean(type1_positions[-1]))
                      - float(np.mean(type1_positions[0])))
    # Spread: std of type1-position averaged over snapshots
    spreads = [float(p.std()) for p in type1_positions if p.size > 1]
    avg_spread = (float(np.mean(spreads)) if spreads else float("nan"))
    return {
        "K": K,
        "n_type1_per_snap_mean": float(np.mean(n_type1_per_snap)),
        "n_type1_per_snap_max": int(max(n_type1_per_snap)),
        "type1_mean_position_first": (float(np.mean(type1_positions[0]))
                                          if type1_positions[0].size else float("nan")),
        "type1_mean_position_last": (float(np.mean(type1_positions[-1]))
                                         if type1_positions[-1].size else float("nan")),
        "travel_lattice_units": travel,
        "spread_avg": avg_spread,
    }


def main():
    print("=" * 80)
    print("Autoloop 5-stage resonance audit")
    print("=" * 80)
    print()
    rows = []
    for regime, n_lat, rel in LADDER:
        fp = PARENT / rel
        if not fp.exists():
            continue
        z = np.load(fp, allow_pickle=True)
        snaps = z["edge_xi_snapshots"]
        psi_r = z["psi_real_snapshots"]
        psi_i = z["psi_imag_snapshots"]
        n_seeds = int(snaps.shape[0])
        per_seed = []
        for s in range(n_seeds):
            xi_traj = np.asarray(snaps[s], dtype=float).copy()
            psi_last = (psi_r[s, -1].astype(float)
                        + 1j * psi_i[s, -1].astype(float))
            s12 = stage_1_2(xi_traj, psi_last, n_lat)
            s3 = stage_3_emergent_time_transitions(xi_traj, n_lat)
            s4 = stage_4_spiral_backreaction(xi_traj,
                                                 psi_r[s], psi_i[s], n_lat)
            s5 = stage_5_traveling_isolates(xi_traj, n_lat)
            per_seed.append({"seed": s, "S12": s12, "S3": s3,
                              "S4": s4, "S5": s5})
        if not per_seed:
            continue
        # Aggregate selectively
        # Stage 1+2 best-K (where lift is highest)
        K_vals = [n_lat, 2 * n_lat, 4 * n_lat, 8 * n_lat]
        s12_summary = {}
        for K in K_vals:
            key = f"K{K}"
            n_t1 = np.mean([d["S12"].get(key, {}).get("n_type1", 0)
                              for d in per_seed])
            n_t2 = np.mean([d["S12"].get(key, {}).get("n_type2", 0)
                              for d in per_seed])
            n_t3 = np.mean([d["S12"].get(key, {}).get("n_type3", 0)
                              for d in per_seed])
            lift = np.mean([d["S12"].get(key, {}).get("lift_3body", np.nan)
                              for d in per_seed
                              if not np.isnan(d["S12"].get(key, {}).get(
                                  "lift_3body", np.nan))])
            asym_res = np.mean([
                d["S12"].get(key, {}).get("asym_resonance", np.nan)
                for d in per_seed
                if not np.isnan(d["S12"].get(key, {}).get("asym_resonance", np.nan))])
            asym_non = np.mean([
                d["S12"].get(key, {}).get("asym_non_resonance", np.nan)
                for d in per_seed
                if not np.isnan(d["S12"].get(key, {}).get("asym_non_resonance", np.nan))])
            asym_all = np.mean([
                d["S12"].get(key, {}).get("asym_all", np.nan)
                for d in per_seed
                if not np.isnan(d["S12"].get(key, {}).get("asym_all", np.nan))])
            s12_summary[key] = {
                "n_type1": float(n_t1), "n_type2": float(n_t2),
                "n_type3": float(n_t3),
                "lift_3body": float(lift) if not np.isnan(lift) else float("nan"),
                "asym_resonance": float(asym_res) if not np.isnan(asym_res) else float("nan"),
                "asym_non_resonance": float(asym_non) if not np.isnan(asym_non) else float("nan"),
                "asym_all": float(asym_all) if not np.isnan(asym_all) else float("nan"),
            }
        s3_summary = {
            "transitions_1_to_2": np.mean([d["S3"]["transitions_1_to_2"]
                                             for d in per_seed]),
            "transitions_2_to_3": np.mean([d["S3"]["transitions_2_to_3"]
                                             for d in per_seed]),
            "transitions_3_to_2": np.mean([d["S3"]["transitions_3_to_2"]
                                             for d in per_seed]),
            "transitions_2_to_1": np.mean([d["S3"]["transitions_2_to_1"]
                                             for d in per_seed]),
            "net_to_3": np.mean([d["S3"]["net_to_3"] for d in per_seed]),
            "net_to_1": np.mean([d["S3"]["net_to_1"] for d in per_seed]),
        }
        s4_summary = {
            "n_formation_events": np.mean([d["S4"]["n_formation_events"]
                                              for d in per_seed]),
            "phi_drift_PPN": np.nanmean([d["S4"]["phi_drift_PPN"]
                                            for d in per_seed]),
            "phi_drift_PNN": np.nanmean([d["S4"]["phi_drift_PNN"]
                                            for d in per_seed]),
        }
        s5_summary = {
            "n_type1_per_snap_mean": np.mean([d["S5"]["n_type1_per_snap_mean"]
                                                  for d in per_seed]),
            "travel_lattice_units": np.nanmean([d["S5"]["travel_lattice_units"]
                                                    for d in per_seed]),
            "spread_avg": np.nanmean([d["S5"]["spread_avg"] for d in per_seed]),
        }
        # Print
        print(f"--- {regime} N={n_lat} (n_seeds={len(per_seed)}) ---")
        print(f"  S1+S2 (top-K subgraph):")
        for K in K_vals:
            r = s12_summary[f"K{K}"]
            print(f"    K={K:>4d}: type[3,2,1]=[{r['n_type3']:>5.0f}, "
                  f"{r['n_type2']:>4.0f}, {r['n_type1']:>3.0f}]  "
                  f"lift={r['lift_3body']:.2f}  "
                  f"asym(res/all/non)={r['asym_resonance']:+.4f}/"
                  f"{r['asym_all']:+.4f}/{r['asym_non_resonance']:+.4f}")
        print(f"  S3 (em-time transitions/seed): "
              f"1->2={s3_summary['transitions_1_to_2']:.1f}, "
              f"2->3={s3_summary['transitions_2_to_3']:.1f}, "
              f"3->2={s3_summary['transitions_3_to_2']:.1f}, "
              f"net_to_3={s3_summary['net_to_3']:+.1f}")
        print(f"  S4 (formation events): "
              f"n={s4_summary['n_formation_events']:.0f}, "
              f"phi_drift PPN={s4_summary['phi_drift_PPN']:+.4f}, "
              f"PNN={s4_summary['phi_drift_PNN']:+.4f}")
        print(f"  S5 (traveling type1): "
              f"n_t1/snap={s5_summary['n_type1_per_snap_mean']:.1f}, "
              f"travel={s5_summary['travel_lattice_units']:.2f}, "
              f"spread={s5_summary['spread_avg']:.2f}")
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds": len(per_seed),
            "S12": s12_summary, "S3": s3_summary,
            "S4": s4_summary, "S5": s5_summary,
        })
    print()
    print("=" * 80)
    print("Cross-regime synthesis")
    print("=" * 80)
    if rows:
        # Best-K analysis
        for K in [rows[0]["N"], 2 * rows[0]["N"], 4 * rows[0]["N"], 8 * rows[0]["N"]]:
            pass  # variable K per regime, can't aggregate trivially
        # Per multiplicity-of-N: K = N case
        lift_KN = [r["S12"][f"K{r['N']}"]["lift_3body"] for r in rows
                    if not np.isnan(r["S12"][f"K{r['N']}"]["lift_3body"])]
        if lift_KN:
            arr = np.array(lift_KN)
            print(f"  S2 lift_3body @ K=N (sparsest): "
                  f"{arr.mean():.2f} +/- {arr.std():.2f}")
        asym_res_KN = [r["S12"][f"K{r['N']}"]["asym_resonance"]
                          for r in rows
                          if not np.isnan(r["S12"][f"K{r['N']}"]["asym_resonance"])]
        if asym_res_KN:
            arr = np.array(asym_res_KN)
            print(f"  S2 asym_resonance @ K=N: "
                  f"{arr.mean():+.5f} +/- {arr.std():.5f}")
        # S3 net-to-3 trend
        net3 = np.array([r["S3"]["net_to_3"] for r in rows])
        print(f"  S3 net_to_3 cross-regime mean: {net3.mean():+.2f} "
              f"+/- {net3.std():.2f}")
        # S4 phi drift PPN-PNN difference
        ppn_drift = np.array([r["S4"]["phi_drift_PPN"] for r in rows
                                if not np.isnan(r["S4"]["phi_drift_PPN"])])
        pnn_drift = np.array([r["S4"]["phi_drift_PNN"] for r in rows
                                if not np.isnan(r["S4"]["phi_drift_PNN"])])
        if ppn_drift.size and pnn_drift.size:
            print(f"  S4 phi-drift PPN-PNN diff: "
                  f"{ppn_drift.mean() - pnn_drift.mean():+.5f}")
        # S5 traveling
        travel = np.array([r["S5"]["travel_lattice_units"] for r in rows
                              if not np.isnan(r["S5"]["travel_lattice_units"])])
        if travel.size:
            print(f"  S5 type1 travel cross-regime: "
                  f"{travel.mean():.2f} +/- {travel.std():.2f} lattice units")

    bundle = {
        "method": "autoloop_5stage_resonance",
        "rows": rows,
    }
    out = REPO / "outputs" / "audit_autoloop_5stage_resonance.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
