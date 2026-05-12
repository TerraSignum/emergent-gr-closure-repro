"""3-body resonance audit: spatial colocation of 3-edge / 2-edge / 1-edge
persistent structures.

User hypothesis 2026-05-02: materialization is not Mexican-hat phase
transition but 3-body resonance. A 3-wave (triangle) + 2-wave (open
2-edge chain) + 1-wave (isolated single edge) must spatially coincide
for the +1 to "calibrate" the 3/2 pair (Saturn-moon analogy).

Test design:
  1. For each persistent edge classify by local context:
       Type-3: edge is part of at least one persistent triangle
       Type-2: edge is in a 2-chain (shares node with another persistent
               edge whose other end is NOT triangle-connected back)
       Type-1: edge is isolated (no persistent-edge neighbors at either
               endpoint)
     (An edge can carry multiple type-flags via different incidences.)

  2. Per node count incident edges of each type: n3(i), n2(i), n1(i).
     Then "resonance node" = node with n3>=1 AND n2>=1 AND n1>=1.

  3. Compute:
       - frac_resonance = n_resonance_nodes / N
       - independent baseline = P(n3>0) * P(n2>0) * P(n1>0)
       - LIFT = frac_resonance / independent_baseline
     If LIFT >> 1, structures CO-LOCATE beyond random chance.

  4. Per-class asym test: for triangles where ALL THREE corners are
     resonance-nodes (full 3/2/1 calibration), measure asym(PPN-PNN).
     Compare to triangles where NO corner is resonance-node (no
     calibration). Hypothesis: asym signal lives in
     resonance-calibrated triangles.

  5. Also report spatial spiral-backreaction proxy: at each resonance
     node, compute local phase winding number around it (over its
     persistent-edge neighbors). Check correlation with asym sign.

Output: outputs/audit_3body_resonance.json
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


def classify_edges(pers_edges, n_lat):
    """For each persistent edge, identify whether it is type-3/2/1.
    Returns:
      adj: dict node -> set of neighbors
      tri_edges: set of (a,b) edges that participate in a triangle
      type3, type2, type1: dict edge->bool
      triangles: list of (i,j,k) triangles
      node_n3, node_n2, node_n1: arrays of incident counts per node
    """
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
    type3 = {e: (e in tri_edges) for e in edge_set}
    # Type-2: not type-3, but has at least one persistent-edge neighbor
    # that itself is not in the same triangle as e
    type2 = {}
    for (a, b) in edge_set:
        if type3[(a, b)]:
            type2[(a, b)] = False
            continue
        nbrs_a = adj[a] - {b}
        nbrs_b = adj[b] - {a}
        type2[(a, b)] = bool(nbrs_a or nbrs_b)
    type1 = {}
    for (a, b) in edge_set:
        nbrs_a = adj[a] - {b}
        nbrs_b = adj[b] - {a}
        type1[(a, b)] = (not type3[(a, b)] and not nbrs_a and not nbrs_b)
    # Per-node counts
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
    return (adj, triangles, edge_set, type3, type2, type1,
             node_n3, node_n2, node_n1)


def asym_in_set(triangles, psi_last):
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
        n_pos = ((1 if d_ij > 0 else 0)
                  + (1 if d_jk > 0 else 0)
                  + (1 if d_ki > 0 else 0))
        if n_pos == 2:
            n_PPN += 1
        elif n_pos == 1:
            n_PNN += 1
    if n_PPN + n_PNN == 0:
        return float("nan"), n_PPN, n_PNN
    return (n_PPN - n_PNN) / (n_PPN + n_PNN), n_PPN, n_PNN


def main():
    print("=" * 80)
    print("3-body resonance audit: 3-edge + 2-edge + 1-edge spatial colocation")
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
            d_xi = np.abs(np.diff(xi_traj, axis=0))
            offdiag = ~np.eye(n_lat, dtype=bool)
            d_off = d_xi[:, offdiag]
            v_med = (float(np.median(d_off[d_off > 0]))
                      if (d_off > 0).any() else 1e-6)
            c_info = 2 * v_med
            persistent_mask_off = (d_off > c_info).mean(axis=0) > 0.5
            ij_idx = np.argwhere(offdiag)
            pers_edges = ij_idx[persistent_mask_off]
            (adj, triangles, edge_set, type3, type2, type1,
              node_n3, node_n2, node_n1) = classify_edges(
                pers_edges, n_lat)
            if not triangles:
                continue
            # Per-node resonance flag
            res_node = ((node_n3 >= 1) & (node_n2 >= 1) & (node_n1 >= 1))
            n_res = int(res_node.sum())
            frac_res = n_res / n_lat
            # Independent baseline
            p_n3 = float((node_n3 >= 1).mean())
            p_n2 = float((node_n2 >= 1).mean())
            p_n1 = float((node_n1 >= 1).mean())
            indep = p_n3 * p_n2 * p_n1
            lift = (frac_res / indep) if indep > 1e-12 else float("nan")
            # Asym in resonance-calibrated vs non-resonance triangles
            res_tri = [t for t in triangles
                        if res_node[t[0]] and res_node[t[1]]
                          and res_node[t[2]]]
            non_tri = [t for t in triangles
                        if not (res_node[t[0]] or res_node[t[1]]
                                or res_node[t[2]])]
            asym_res, n_ppn_res, n_pnn_res = asym_in_set(
                res_tri, psi_last)
            asym_non, n_ppn_non, n_pnn_non = asym_in_set(
                non_tri, psi_last)
            asym_all, n_ppn_all, n_pnn_all = asym_in_set(
                triangles, psi_last)
            per_seed.append({
                "seed": s,
                "n_pers_edges": len(edge_set),
                "n_triangles": len(triangles),
                "n_type3_edges": int(sum(type3.values())),
                "n_type2_edges": int(sum(type2.values())),
                "n_type1_edges": int(sum(type1.values())),
                "n_resonance_nodes": n_res,
                "frac_resonance_nodes": frac_res,
                "p_n3_node": p_n3, "p_n2_node": p_n2, "p_n1_node": p_n1,
                "indep_baseline": indep,
                "lift_3body": lift,
                "n_resonance_triangles": len(res_tri),
                "n_non_resonance_triangles": len(non_tri),
                "asym_all": asym_all,
                "asym_resonance": asym_res,
                "asym_non_resonance": asym_non,
            })
        if not per_seed:
            continue
        def mn(key, default=float("nan")):
            vals = [d[key] for d in per_seed
                     if not (isinstance(d.get(key), float) and np.isnan(d[key]))]
            return float(np.mean(vals)) if vals else default
        def std(key):
            vals = [d[key] for d in per_seed
                     if not (isinstance(d.get(key), float) and np.isnan(d[key]))]
            return (float(np.std(vals))
                     if vals else float("nan"))
        lift_m = mn("lift_3body")
        frac_m = mn("frac_resonance_nodes")
        asym_all_m = mn("asym_all")
        asym_res_m = mn("asym_resonance")
        asym_non_m = mn("asym_non_resonance")
        n_res_tri_m = mn("n_resonance_triangles")
        n_non_tri_m = mn("n_non_resonance_triangles")
        unc_res = std("asym_resonance") / np.sqrt(len(per_seed))
        unc_non = std("asym_non_resonance") / np.sqrt(len(per_seed))
        print(f"--- {regime} N={n_lat} (n_seeds={len(per_seed)}) ---")
        print(f"  type3 edges = {mn('n_type3_edges'):>5.0f}, "
              f"type2 = {mn('n_type2_edges'):>5.0f}, "
              f"type1 = {mn('n_type1_edges'):>4.0f}")
        print(f"  frac_resonance_nodes = {frac_m:.3f}, "
              f"independent baseline = {mn('indep_baseline'):.3f}, "
              f"LIFT = {lift_m:.3f}")
        print(f"  asym_all      = {asym_all_m:+.5f}")
        print(f"  asym_resonance= {asym_res_m:+.5f} +/- {unc_res:.5f}  "
              f"(n_res_tri={n_res_tri_m:.0f})")
        print(f"  asym_non_res  = {asym_non_m:+.5f} +/- {unc_non:.5f}  "
              f"(n_non_tri={n_non_tri_m:.0f})")
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds": len(per_seed),
            "lift_3body_mean": lift_m,
            "frac_resonance_nodes_mean": frac_m,
            "asym_all_mean": asym_all_m,
            "asym_resonance_mean": asym_res_m,
            "asym_resonance_unc": unc_res,
            "asym_non_resonance_mean": asym_non_m,
            "asym_non_resonance_unc": unc_non,
            "n_resonance_triangles_mean": n_res_tri_m,
            "n_non_resonance_triangles_mean": n_non_tri_m,
            "per_seed": per_seed,
        })
    print()
    print("=" * 80)
    print("Cross-regime synthesis")
    print("=" * 80)
    if rows:
        N_arr = np.array([r["N"] for r in rows], dtype=float)
        lift_arr = np.array([r["lift_3body_mean"] for r in rows])
        a_res = np.array([r["asym_resonance_mean"] for r in rows
                            if not np.isnan(r["asym_resonance_mean"])])
        a_non = np.array([r["asym_non_resonance_mean"] for r in rows
                            if not np.isnan(r["asym_non_resonance_mean"])])
        a_all = np.array([r["asym_all_mean"] for r in rows
                            if not np.isnan(r["asym_all_mean"])])
        print(f"  3-body LIFT cross-regime = {lift_arr.mean():.3f} "
              f"+/- {lift_arr.std():.3f}")
        print(f"  asym_all cross-regime         = {a_all.mean():+.5f} "
              f"+/- {a_all.std():.5f}")
        if a_res.size:
            print(f"  asym_resonance cross-regime   = {a_res.mean():+.5f} "
                  f"+/- {a_res.std():.5f}")
        if a_non.size:
            print(f"  asym_non_resonance cross-regime= {a_non.mean():+.5f} "
                  f"+/- {a_non.std():.5f}")
        # N-trend on resonance asym
        if a_res.size >= 3:
            unc_arr = np.array([r["asym_resonance_unc"] for r in rows
                                  if not np.isnan(r["asym_resonance_mean"])])
            N_a = np.array([r["N"] for r in rows
                              if not np.isnan(r["asym_resonance_mean"])],
                              dtype=float)
            x = 1.0 / N_a
            w = 1.0 / np.maximum(unc_arr, 1e-6) ** 2
            A = np.column_stack([np.ones_like(x), x])
            try:
                AtWA = A.T @ (w[:, None] * A)
                AtWy = A.T @ (w * a_res)
                coef = np.linalg.solve(AtWA, AtWy)
                a_inf, b = coef
                cov = np.linalg.inv(AtWA)
                a_unc = float(np.sqrt(cov[0, 0]))
                print(f"  Symanzik fit asym_resonance(N) = a + b/N:")
                print(f"    a_inf = {a_inf:+.5f} +/- {a_unc:.5f}, "
                      f"b = {b:+.4f}")
                print(f"    target -pi/200 = {-np.pi/200:+.5f}, "
                      f"diff = {a_inf - (-np.pi/200):+.5f}")
            except np.linalg.LinAlgError:
                pass

    bundle = {
        "method": "3body_resonance_audit_3plus2plus1",
        "rows": rows,
    }
    out = REPO / "outputs" / "audit_3body_resonance.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
