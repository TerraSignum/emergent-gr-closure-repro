"""Event-driven materialization audit: four readings of the
3-wave + 2-wave (+ 1-wave calibrator) hypothesis.

Background: Stufes A/B/C falsified bulk-statistic claims (T2 generic,
K_vortex K-Q-driven, no mass asymmetry up to N=300). The user's pivot:
materialization is event-driven, not bulk. Specifically a coincidence of
three structures - triangle (3 waves), edge-chain (2 waves), and a
single-wave calibrator whose spin sets the matter-state.

Four readings tested in parallel on the 8-regime ladder (P5N64..P5N300):

L1 (mode-coincidence at node):
  For each node i, count "modes" = number of persistent edges incident
  on i. High-mode nodes (5+) are coincidence sites. Test if high-mode
  nodes have distinctive |psi|^2, K-Q signature, or spatial clustering.

L2 (graph: triangle + tail):
  For each persistent triangle (i,j,k), check for attached 2-edge tails
  at any corner: (i,l) and (l,m) both persistent with m not in {i,j,k}.
  Such "5-edge triangle-with-tail" structures: count, mass-proxy of all
  involved nodes, stability statistics. Compare to triangles with no tail.

L3 (vortex-cluster collision):
  Identify vortex cores at end-snapshot: nodes where local ψ phase
  varies fastest (|grad phi| maxima). Cluster vortices spatially via
  KMeans. Test if 3-vortex clusters + 2-vortex clusters collide
  (overlap node sets). Outcome: phase-mass asymmetry at collision nodes.

L4 (triangle formation event + calibrator spin):
  Track edge-activation events (Xi_ij crosses c_info from below). A
  triangle FORMATION event = 3 edges of a triangle (i,j,k) all activate
  within Delta_t = 2 snapshots. At formation moment, find the nearest
  persistent edge (l,m) outside the triangle = calibrator. Spin =
  sign(arg(psi_m * conj(psi_l))). Group triangles by calibrator spin;
  test if outcome statistics (Pi3 binding, |psi|^2 at triangle, longevity)
  depend on spin sign.

L5 (Matter-vs-Antimatter mixed/homogeneous interpretation):
  User hypothesis: Antimatter = "3/3" homogeneous-sign triangles
  (classes PPP and NNN), Matter = "3/2+1" mixed-sign triangles (PPN
  and PNN). Compute n(matter)/n(antimatter) ratio per regime. Compare
  to binomial expectation (75:25 = 3:1). Also compare mass-proxy of
  the matter class vs antimatter class (test: do mixed triangles have
  different |psi|^2 than homogeneous triangles?).

Outputs: outputs/audit_event_driven_materialization.json
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


def find_persistent_triangles(pers_edges):
    edge_set = set()
    for a, b in pers_edges:
        a, b = int(a), int(b)
        if a == b:
            continue
        edge_set.add((min(a, b), max(a, b)))
    adj = defaultdict(set)
    for a, b in edge_set:
        adj[a].add(b)
        adj[b].add(a)
    triangles = []
    for (i, j) in edge_set:
        common = adj[i] & adj[j]
        for k in common:
            if k <= j:
                continue
            triangles.append((i, j, k))
    return triangles, edge_set, adj


def per_seed_persistent(xi_traj, n_lat):
    d_xi = np.abs(np.diff(xi_traj, axis=0))
    offdiag = ~np.eye(n_lat, dtype=bool)
    d_off = d_xi[:, offdiag]
    v_med = (float(np.median(d_off[d_off > 0]))
              if (d_off > 0).any() else 1e-6)
    c_info = 2 * v_med
    persistent_mask_off = (d_off > c_info).mean(axis=0) > 0.5
    ij_idx = np.argwhere(offdiag)
    pers_edges = ij_idx[persistent_mask_off]
    return pers_edges, c_info, d_xi


def l1_mode_coincidence(pers_edges, psi_last, k_last, q_last, n_lat):
    """L1: count persistent-edge degree per node; report mass-proxy
    of high-mode nodes vs low-mode nodes."""
    deg = np.zeros(n_lat, dtype=int)
    for i, j in pers_edges:
        deg[int(i)] += 1
        deg[int(j)] += 1
    psi_abs2 = np.real(psi_last) ** 2 + np.imag(psi_last) ** 2
    if k_last.ndim == 2:
        k_node = k_last.mean(axis=1)
        q_node = q_last.mean(axis=1)
    else:
        k_node, q_node = k_last, q_last
    dca = k_node - q_node
    # 5-mode threshold = nodes with degree >= 5
    high_mask = deg >= 5
    low_mask = deg < 5
    n_high = int(high_mask.sum())
    n_low = int(low_mask.sum())
    if n_high == 0 or n_low == 0:
        return {
            "n_high_mode_nodes": n_high, "n_low_mode_nodes": n_low,
            "mass_high": float("nan"), "mass_low": float("nan"),
            "dca_high": float("nan"), "dca_low": float("nan"),
            "mass_ratio_high_low": float("nan"),
        }
    return {
        "n_high_mode_nodes": n_high, "n_low_mode_nodes": n_low,
        "mass_high": float(psi_abs2[high_mask].mean()),
        "mass_low": float(psi_abs2[low_mask].mean()),
        "dca_high": float(dca[high_mask].mean()),
        "dca_low": float(dca[low_mask].mean()),
        "mass_ratio_high_low": float(psi_abs2[high_mask].mean()
                                       / max(psi_abs2[low_mask].mean(), 1e-12)),
    }


def l2_triangle_with_tail(triangles, adj, psi_last, xi_traj,
                            n_max=400):
    """L2: count triangles with 2-edge tail attached at any corner.
    Compare mass-proxy and Pi3 of with-tail vs without-tail."""
    if not triangles:
        return {
            "n_with_tail": 0, "n_without_tail": 0,
            "mass_with": float("nan"), "mass_without": float("nan"),
            "Pi3_with": float("nan"), "Pi3_without": float("nan"),
        }
    psi_abs2 = np.real(psi_last) ** 2 + np.imag(psi_last) ** 2
    xi_last = xi_traj[-1]
    with_tail_data = []
    without_tail_data = []
    sample = (triangles
              if len(triangles) <= n_max
              else [triangles[i] for i in np.random.default_rng(0).choice(
                  len(triangles), n_max, replace=False)])
    for i, j, k in sample:
        # Check for 2-edge tail at any corner
        has_tail = False
        for corner in (i, j, k):
            for l_ in adj[corner]:
                if l_ in (i, j, k):
                    continue
                # l_ must be persistent neighbor of corner (already in adj)
                # need a 2nd edge from l_ to some m_ not in (i,j,k,l_)
                for m_ in adj[l_]:
                    if m_ in (i, j, k, l_):
                        continue
                    has_tail = True
                    break
                if has_tail:
                    break
            if has_tail:
                break
        m_proxy = (psi_abs2[i] + psi_abs2[j] + psi_abs2[k]) / 3.0
        pi3 = xi_last[i, j] * xi_last[j, k] * xi_last[k, i]
        if has_tail:
            with_tail_data.append((m_proxy, pi3))
        else:
            without_tail_data.append((m_proxy, pi3))
    n_with = len(with_tail_data)
    n_without = len(without_tail_data)
    mass_with = (float(np.mean([d[0] for d in with_tail_data]))
                 if with_tail_data else float("nan"))
    mass_without = (float(np.mean([d[0] for d in without_tail_data]))
                    if without_tail_data else float("nan"))
    pi3_with = (float(np.mean([d[1] for d in with_tail_data]))
                if with_tail_data else float("nan"))
    pi3_without = (float(np.mean([d[1] for d in without_tail_data]))
                   if without_tail_data else float("nan"))
    return {
        "n_with_tail": n_with, "n_without_tail": n_without,
        "frac_with_tail": n_with / max(n_with + n_without, 1),
        "mass_with": mass_with, "mass_without": mass_without,
        "Pi3_with": pi3_with, "Pi3_without": pi3_without,
        "mass_ratio_with_without": (mass_with / mass_without
                                      if mass_without and mass_without > 0
                                      else float("nan")),
    }


def l3_vortex_cluster_collisions(psi_last, n_lat, k_top=15):
    """L3: cluster top-K nodes by |grad phi| as vortex cores. Cluster
    them with simple distance-based grouping; identify 3-clusters and
    2-clusters; check for spatial overlap (collisions)."""
    phi = np.angle(psi_last)
    # 1d phase gradient on the lattice: |delta phi| with periodic neighbor
    grad = np.abs(np.diff(phi, append=phi[0]))
    # Reduce to (-pi, pi]
    grad = np.minimum(grad, 2 * np.pi - grad)
    # Top-K vortex candidates
    top_idx = np.argsort(grad)[-k_top:]
    if len(top_idx) < 5:
        return {
            "n_vortex_cores": int(len(top_idx)),
            "n_3clusters": 0, "n_2clusters": 0,
            "n_collisions": 0, "collision_mass": float("nan"),
        }
    # Cluster by 1D position adjacency on the lattice (mod n_lat)
    sorted_pos = np.sort(top_idx)
    clusters = []
    cur = [int(sorted_pos[0])]
    for p in sorted_pos[1:]:
        if p - cur[-1] <= max(2, n_lat // 30):
            cur.append(int(p))
        else:
            clusters.append(cur)
            cur = [int(p)]
    clusters.append(cur)
    n_3 = sum(1 for c in clusters if len(c) >= 3)
    n_2 = sum(1 for c in clusters if len(c) == 2)
    # Collisions: pair-wise spatial closeness between a 3-cluster and 2-cluster
    psi_abs2 = np.real(psi_last) ** 2 + np.imag(psi_last) ** 2
    n_coll = 0
    coll_masses = []
    for c3 in [c for c in clusters if len(c) >= 3]:
        for c2 in [c for c in clusters if len(c) == 2]:
            # min distance on the ring
            d_min = min((min((p3 - p2) % n_lat,
                              (p2 - p3) % n_lat))
                        for p3 in c3 for p2 in c2)
            if d_min <= max(2, n_lat // 25):
                n_coll += 1
                # mass at all collision-zone nodes
                zone = list(set(c3) | set(c2))
                coll_masses.append(float(psi_abs2[zone].mean()))
    return {
        "n_vortex_cores": int(len(top_idx)),
        "n_3clusters": int(n_3), "n_2clusters": int(n_2),
        "n_collisions": int(n_coll),
        "collision_mass": (float(np.mean(coll_masses))
                            if coll_masses else float("nan")),
    }


def l5_matter_vs_antimatter(triangles, psi_last, xi_traj, n_max=2000):
    """L5: classify each persistent triangle as Matter (mixed-sign 3/2+1)
    or Antimatter (homogeneous-sign 3/3). Compare frequencies + mass-proxy.
    User hypothesis: Antimatter = (+++) or (---), Matter = (++-) or (+--).
    """
    if not triangles:
        return {
            "n_matter": 0, "n_antimatter": 0,
            "matter_over_antimatter": float("nan"),
            "mass_matter": float("nan"), "mass_antimatter": float("nan"),
            "Pi3_matter": float("nan"), "Pi3_antimatter": float("nan"),
            "mass_ratio_matter_over_antimatter": float("nan"),
        }
    sample = (triangles
              if len(triangles) <= n_max
              else [triangles[i] for i in np.random.default_rng(0).choice(
                    len(triangles), n_max, replace=False)])
    phi = np.angle(psi_last)
    psi_abs2 = np.real(psi_last) ** 2 + np.imag(psi_last) ** 2
    xi_last = xi_traj[-1]
    matter_data, antimatter_data = [], []
    for i, j, k in sample:
        d_ij = np.angle(np.exp(1j * (phi[j] - phi[i])))
        d_jk = np.angle(np.exp(1j * (phi[k] - phi[j])))
        d_ki = np.angle(np.exp(1j * (phi[i] - phi[k])))
        s = (int(np.sign(d_ij)), int(np.sign(d_jk)), int(np.sign(d_ki)))
        if 0 in s:
            continue
        n_pos = sum(1 for x in s if x > 0)
        homogeneous = (n_pos == 3 or n_pos == 0)
        m_proxy = (psi_abs2[i] + psi_abs2[j] + psi_abs2[k]) / 3.0
        pi3 = xi_last[i, j] * xi_last[j, k] * xi_last[k, i]
        if homogeneous:
            antimatter_data.append((m_proxy, pi3))
        else:
            matter_data.append((m_proxy, pi3))
    n_m, n_a = len(matter_data), len(antimatter_data)
    mass_m = (float(np.mean([d[0] for d in matter_data]))
              if matter_data else float("nan"))
    mass_a = (float(np.mean([d[0] for d in antimatter_data]))
              if antimatter_data else float("nan"))
    pi3_m = (float(np.mean([d[1] for d in matter_data]))
             if matter_data else float("nan"))
    pi3_a = (float(np.mean([d[1] for d in antimatter_data]))
             if antimatter_data else float("nan"))
    return {
        "n_matter": n_m, "n_antimatter": n_a,
        "matter_over_antimatter": (n_m / n_a if n_a > 0 else float("nan")),
        "mass_matter": mass_m, "mass_antimatter": mass_a,
        "Pi3_matter": pi3_m, "Pi3_antimatter": pi3_a,
        "mass_ratio_matter_over_antimatter": (
            mass_m / mass_a if mass_a and mass_a > 0 else float("nan")),
    }


def l4_triangle_formation_calibrator(xi_traj, psi_last, n_lat,
                                      c_info, max_events=200):
    """L4: track triangle formation events along trajectory + calibrator-spin.
    For each formation event (3 edges of a triangle activate within
    delta_t=2 snapshots), find nearest persistent calibrator-edge and
    its spin. Group outcomes by spin sign."""
    n_snap = xi_traj.shape[0]
    if n_snap < 4:
        return {"n_events": 0,
                "outcome_mass_pos_spin": float("nan"),
                "outcome_mass_neg_spin": float("nan"),
                "n_pos_spin": 0, "n_neg_spin": 0}
    # Edge activation: per edge, the first snapshot t* where d_xi(t*)
    # crosses c_info from below (in the lookahead diff from snapshot t*).
    d_xi = np.abs(np.diff(xi_traj, axis=0))   # (n_snap-1, N, N)
    # Per-edge first activation time
    crossed = (d_xi > c_info)
    activation_t = np.full((n_lat, n_lat), n_snap, dtype=int)
    # find earliest t with True
    has_crossing = crossed.any(axis=0)
    first = np.argmax(crossed, axis=0)
    activation_t[has_crossing] = first[has_crossing]
    # Final persistent mask (for forming a triangle)
    offdiag = ~np.eye(n_lat, dtype=bool)
    d_off = d_xi[:, offdiag]
    persistent_mask_off = (d_off > c_info).mean(axis=0) > 0.5
    ij_idx = np.argwhere(offdiag)
    pers_edges_set = set()
    for k_e, (i, j) in enumerate(ij_idx):
        if persistent_mask_off[k_e]:
            pers_edges_set.add((int(min(i, j)), int(max(i, j))))
    # For each persistent triangle, compute formation time = max of 3 edge
    # activation times; window = max - min activation t. Event qualifies
    # if window <= 2 snapshots.
    pos_outcomes = []
    neg_outcomes = []
    psi_abs2 = np.real(psi_last) ** 2 + np.imag(psi_last) ** 2
    phi = np.angle(psi_last)
    pers_edges_list = list(pers_edges_set)
    if len(pers_edges_list) == 0:
        return {"n_events": 0,
                "outcome_mass_pos_spin": float("nan"),
                "outcome_mass_neg_spin": float("nan"),
                "n_pos_spin": 0, "n_neg_spin": 0}
    adj = defaultdict(set)
    for a, b in pers_edges_set:
        adj[a].add(b)
        adj[b].add(a)
    triangles = []
    for (i, j) in pers_edges_set:
        common = adj[i] & adj[j]
        for k in common:
            if k <= j:
                continue
            triangles.append((i, j, k))
    if len(triangles) > max_events:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(triangles), max_events, replace=False)
        triangles = [triangles[i] for i in idx]
    pers_edges_arr = np.array(pers_edges_list)
    for i, j, k in triangles:
        t_ij = activation_t[i, j]
        t_jk = activation_t[j, k]
        t_ki = activation_t[k, i]
        formation_window = max(t_ij, t_jk, t_ki) - min(t_ij, t_jk, t_ki)
        if formation_window > 2:
            continue
        # Find nearest calibrator: a persistent edge (l,m) NOT in the
        # triangle, choose the one whose lattice index is closest to
        # the triangle centroid.
        centroid = (i + j + k) / 3.0
        d2 = ((pers_edges_arr[:, 0] + pers_edges_arr[:, 1]) / 2.0
              - centroid) ** 2
        # Mask out triangle edges
        for (a, b) in [(min(i, j), max(i, j)),
                        (min(j, k), max(j, k)),
                        (min(i, k), max(i, k))]:
            mask = ((pers_edges_arr[:, 0] == a)
                     & (pers_edges_arr[:, 1] == b))
            d2[mask] = np.inf
        if not np.isfinite(d2).any():
            continue
        cal_idx = int(np.argmin(d2))
        l_, m_ = pers_edges_arr[cal_idx]
        spin = np.angle(np.exp(1j * (phi[m_] - phi[l_])))
        m_proxy = (psi_abs2[i] + psi_abs2[j] + psi_abs2[k]) / 3.0
        if spin > 0:
            pos_outcomes.append(m_proxy)
        elif spin < 0:
            neg_outcomes.append(m_proxy)
    return {
        "n_events": int(len(pos_outcomes) + len(neg_outcomes)),
        "n_pos_spin": int(len(pos_outcomes)),
        "n_neg_spin": int(len(neg_outcomes)),
        "outcome_mass_pos_spin": (float(np.mean(pos_outcomes))
                                    if pos_outcomes else float("nan")),
        "outcome_mass_neg_spin": (float(np.mean(neg_outcomes))
                                    if neg_outcomes else float("nan")),
        "spin_outcome_ratio": (float(np.mean(pos_outcomes)
                                       / np.mean(neg_outcomes))
                                if (pos_outcomes and neg_outcomes
                                     and np.mean(neg_outcomes) > 0)
                                else float("nan")),
    }


def main():
    print("=" * 80)
    print("Event-driven materialization audit (L1, L2, L3, L4)")
    print("=" * 80)
    print()
    rows = []
    for regime, n_lat, rel in LADDER:
        fp = PARENT / rel
        if not fp.exists():
            print(f"{regime}: missing")
            continue
        z = np.load(fp, allow_pickle=True)
        snaps = z["edge_xi_snapshots"]
        psi_r = z["psi_real_snapshots"]
        psi_i = z["psi_imag_snapshots"]
        k_snaps = z["k_snapshots"]
        q_snaps = z["q_snapshots"]
        n_seeds = int(snaps.shape[0])  # use ALL available seeds
        per_seed = []
        for s in range(n_seeds):
            xi_traj = np.asarray(snaps[s], dtype=float).copy()
            psi_last = (psi_r[s, -1].astype(float)
                        + 1j * psi_i[s, -1].astype(float))
            k_last = np.asarray(k_snaps[s, -1], dtype=float)
            q_last = np.asarray(q_snaps[s, -1], dtype=float)
            pers_edges, c_info, _ = per_seed_persistent(xi_traj, n_lat)
            triangles, _, adj = find_persistent_triangles(pers_edges)
            l1 = l1_mode_coincidence(pers_edges, psi_last, k_last,
                                       q_last, n_lat)
            l2 = l2_triangle_with_tail(triangles, adj,
                                          psi_last, xi_traj)
            l5 = l5_matter_vs_antimatter(triangles, psi_last, xi_traj)
            l3 = l3_vortex_cluster_collisions(psi_last, n_lat)
            l4 = l4_triangle_formation_calibrator(
                xi_traj, psi_last, n_lat, c_info)
            per_seed.append({"seed": s,
                             "L1": l1, "L2": l2, "L3": l3,
                             "L4": l4, "L5": l5})
        if not per_seed:
            continue
        # Aggregate
        def mn(grp, key):
            vals = []
            for d in per_seed:
                v = d.get(grp, {}).get(key)
                if v is None:
                    continue
                if isinstance(v, float) and np.isnan(v):
                    continue
                vals.append(v)
            return float(np.mean(vals)) if vals else float("nan")
        l1_ratio = mn("L1", "mass_ratio_high_low")
        l2_ratio = mn("L2", "mass_ratio_with_without")
        l2_frac = mn("L2", "frac_with_tail")
        l3_n_coll = mn("L3", "n_collisions")
        l3_n3 = mn("L3", "n_3clusters")
        l3_n2 = mn("L3", "n_2clusters")
        l3_mass = mn("L3", "collision_mass")
        l4_n = mn("L4", "n_events")
        l4_pos = mn("L4", "outcome_mass_pos_spin")
        l4_neg = mn("L4", "outcome_mass_neg_spin")
        l4_ratio = mn("L4", "spin_outcome_ratio")
        l4_npos = mn("L4", "n_pos_spin")
        l4_nneg = mn("L4", "n_neg_spin")
        l5_ratio_count = mn("L5", "matter_over_antimatter")
        l5_ratio_mass = mn("L5", "mass_ratio_matter_over_antimatter")
        l5_n_matter = mn("L5", "n_matter")
        l5_n_anti = mn("L5", "n_antimatter")
        print(f"--- {regime} N={n_lat} (n_seeds={len(per_seed)}) ---")
        print(f"  L1 (mode-coincidence): mass_ratio_high/low = {l1_ratio:.5f}  "
              f"(=1 means no signal)")
        print(f"  L2 (triangle+tail):  frac_with_tail={l2_frac:.3f}  "
              f"mass_ratio_with/without = {l2_ratio:.5f}")
        print(f"  L3 (vortex collisions): n3={l3_n3:.1f} n2={l3_n2:.1f} "
              f"n_coll={l3_n_coll:.2f} mass={l3_mass:.4f}")
        print(f"  L4 (formation+spin): events={l4_n:.0f} "
              f"(+spin={l4_npos:.0f}, -spin={l4_nneg:.0f}) "
              f"mass(+)/mass(-) = {l4_ratio:.5f}")
        print(f"  L5 (matter/antimatter): n_M/n_A = {l5_ratio_count:.4f} "
              f"(binom 3.0; n_M={l5_n_matter:.0f}, n_A={l5_n_anti:.0f})  "
              f"mass(M)/mass(A) = {l5_ratio_mass:.5f}")
        rows.append({
            "regime": regime, "N": n_lat,
            "n_seeds": len(per_seed),
            "L1_mass_ratio_high_low_mean": l1_ratio,
            "L2_frac_with_tail_mean": l2_frac,
            "L2_mass_ratio_with_without_mean": l2_ratio,
            "L3_n_3clusters_mean": l3_n3,
            "L3_n_2clusters_mean": l3_n2,
            "L3_n_collisions_mean": l3_n_coll,
            "L3_collision_mass_mean": l3_mass,
            "L4_n_events_mean": l4_n,
            "L4_n_pos_spin_mean": l4_npos,
            "L4_n_neg_spin_mean": l4_nneg,
            "L4_outcome_mass_pos_spin": l4_pos,
            "L4_outcome_mass_neg_spin": l4_neg,
            "L4_spin_outcome_ratio_mean": l4_ratio,
            "L5_n_matter_mean": l5_n_matter,
            "L5_n_antimatter_mean": l5_n_anti,
            "L5_ratio_count_matter_over_antimatter": l5_ratio_count,
            "L5_ratio_mass_matter_over_antimatter": l5_ratio_mass,
            "per_seed": per_seed,
        })
    print()
    print("=" * 80)
    print("Cross-regime synthesis (L1, L2, L3, L4)")
    print("=" * 80)
    if rows:
        for tag, key in [
            ("L1 mass_ratio_high/low (=1: no signal)",
             "L1_mass_ratio_high_low_mean"),
            ("L2 mass_ratio_with/without_tail (=1: no signal)",
             "L2_mass_ratio_with_without_mean"),
            ("L2 frac_with_tail",
             "L2_frac_with_tail_mean"),
            ("L3 n_collisions per seed",
             "L3_n_collisions_mean"),
            ("L4 spin_outcome_ratio (=1: no signal)",
             "L4_spin_outcome_ratio_mean"),
            ("L4 n_events per seed",
             "L4_n_events_mean"),
            ("L5 n_matter/n_antimatter (binom 3.0)",
             "L5_ratio_count_matter_over_antimatter"),
            ("L5 mass_M/mass_A (=1: no signal)",
             "L5_ratio_mass_matter_over_antimatter"),
        ]:
            arr = np.array([r[key] for r in rows
                            if not np.isnan(r[key])])
            if arr.size:
                sig = (abs(arr.mean() - 1.0) / arr.std()
                        if "ratio" in tag and arr.std() > 1e-12
                        else float("nan"))
                print(f"  {tag}: {arr.mean():.5f} +/- {arr.std():.5f}"
                      + (f"   (deviation from 1 = {sig:.2f}σ)"
                          if not np.isnan(sig) else ""))

    bundle = {
        "method": "event_driven_materialization_L1_L2_L3_L4",
        "rows": rows,
    }
    out = REPO / "outputs" / "audit_event_driven_materialization.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
