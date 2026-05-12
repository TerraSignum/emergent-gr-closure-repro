"""Triangle-closure audit: bi-edge to tri-edge bound-state transition.

Hypothesis (User 2026-05-02): the step from QFT pair-vacuum to a baryon-
type bound state is the framework's bi-to-tri move: two persistent edges
sharing a node already attract their second-endpoints (H2 audit, edge-
attraction), and the closing third edge forms a bound 3-cycle. If
persistent-violator triangles tighten over the trajectory, this is the
Lattice-level analog of pair-to-triplet transition that gives matter.

Per regime/seed:
  T1 (existence):  count persistent triangles -- 3-cycles (i,j,k) where
        ALL THREE edges (i,j), (j,k), (k,i) cross v_info > c_info >50%
        time. Compare to a Bernoulli-null expectation
        n_tri_null = C(N,3) * p_pers^3 with p_pers = persistent-edge fraction.

  T2 (binding gain): track Pi_3(t) = <Xi_ij Xi_jk Xi_ki>_persistent_triangles.
        Compute log(Pi_3_tF / Pi_3_t0). If positive, triangle Xi-product
        grows over the trajectory -- 3-body binding gain.

  T3 (triangle vs edge gain): geometric ratio
        R_31 = <Pi_3^(1/3)> / <Xi_ij>_persistent_edges
        Compare R_31_t0 to R_31_tF. R_31_tF > R_31_t0 means the triangle
        (cube-rooted, dimension-matched) becomes tighter than the edge mean.

  T4 (M3 slack inside persistent triangles): for each triangle compute
        S_M3 = max over the three triple permutations of
        max(Xi_a*Xi_b - Xi_c, 0)
        track <S_M3>_t0 vs <S_M3>_tF. If S_M3 grows, M3-violation IS
        the binding observable; this is the structural sup-violation.

  T5 (System-R coefficient match): compare end-snapshot
        Pi_3_tF^(1/3) to algebraic candidates (alpha_xi^2 * beta_pi),
        (alpha_xi * beta_pi^2), (gamma * beta_pi), etc.

Output: outputs/audit_M3_triangle_closure.json
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
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz"),
]

# System-R algebraic constants (rational numbers from the framework)
ALPHA_XI = 9 / 10           # 0.9
BETA_PI  = 15 / 16          # 0.9375
GAMMA    = 1 / 10           # 0.1
D_R      = 67 / 80          # 0.8375
EPS2     = 1 / 20           # 0.05


def find_persistent_triangles(pers_edges: np.ndarray, n_lat: int,
                              sample_for_traj: int = 500):
    """Identify 3-cycles (i,j,k) all of whose 3 edges are in pers_edges.
    Returns: full triangle count (n_total), and a uniformly-sampled
    subset of up to sample_for_traj triangles for trajectory tracking.
    Bernoulli null returned alongside.
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
    # Full enumeration of persistent triangles (uncapped)
    all_triangles = []
    for (i, j) in edge_set:
        common = adj[i] & adj[j]
        for k in common:
            if k <= j:
                continue
            all_triangles.append((i, j, k))
    n_total = len(all_triangles)
    # Uniform sample for trajectory cost
    if n_total > sample_for_traj:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_total, size=sample_for_traj, replace=False)
        sample = [all_triangles[i] for i in idx]
    else:
        sample = list(all_triangles)
    n_pers = len(edge_set)
    n_off = n_lat * (n_lat - 1) // 2
    p_pers = n_pers / max(n_off, 1)
    n_tri_null = (n_lat * (n_lat - 1) * (n_lat - 2) / 6.0) * (p_pers ** 3)
    return n_total, sample, p_pers, n_tri_null


def main():
    print("=" * 78)
    print("Triangle-closure audit: bi-edge -> tri-edge bound-state transition")
    print("=" * 78)
    print(f"  System-R rationals: alpha_xi={ALPHA_XI}, beta_pi={BETA_PI}, "
          f"gamma={GAMMA}, D={D_R}, eps^2={EPS2}")
    print()
    header = (
        f"  {'reg':<7} {'#s':>2} "
        f"{'n_pers':>7} {'n_tri':>7} {'n_null':>8} "
        f"{'Pi3_t0':>8} {'Pi3_tF':>8} "
        f"{'T2_gain':>8} {'T3_R31_t0':>10} {'T3_R31_tF':>10} "
        f"{'T4_S_t0':>9} {'T4_S_tF':>9}"
    )
    print(header)
    print("-" * len(header))
    rows = []
    for regime, n_lat, rel in LADDER:
        fp = PARENT / rel
        if not fp.exists():
            print(f"  {regime}: missing")
            continue
        z = np.load(fp, allow_pickle=True)
        snaps = z["edge_xi_snapshots"]
        n_seeds = min(int(snaps.shape[0]),
                      8 if n_lat <= 100 else 4)
        n_snap = int(snaps.shape[1])

        per_seed = []
        for s in range(n_seeds):
            xi_traj = np.asarray(snaps[s], dtype=float).copy()
            d_xi = np.abs(np.diff(xi_traj, axis=0))
            offdiag = ~np.eye(n_lat, dtype=bool)
            d_off = d_xi[:, offdiag]
            v_med = (float(np.median(d_off[d_off > 0]))
                     if (d_off > 0).any() else 1e-6)
            c_info = 2 * v_med
            persistent_mask_off = (d_off > c_info).mean(axis=0) > 0.5
            ij_idx = np.argwhere(offdiag)
            pers_edges = ij_idx[persistent_mask_off]
            if pers_edges.shape[0] == 0:
                continue
            # Persistent triangles (full enumeration, sampled subset for traj)
            n_total, triangles, p_pers, n_null = find_persistent_triangles(
                pers_edges, n_lat, sample_for_traj=500)
            if not triangles:
                per_seed.append({
                    "seed": s,
                    "n_pers": int(pers_edges.shape[0]),
                    "n_pers_triangles_total": 0,
                    "n_null": float(n_null),
                    "p_pers": p_pers,
                })
                continue

            # T2 + T3 + T4 across the trajectory
            tri_arr = np.array(triangles, dtype=int)  # (T,3)
            i_arr, j_arr, k_arr = tri_arr[:, 0], tri_arr[:, 1], tri_arr[:, 2]
            # Track at t=0 and t=last
            xi0 = xi_traj[0]
            xiF = xi_traj[-1]
            xi_ij_0 = xi0[i_arr, j_arr]
            xi_jk_0 = xi0[j_arr, k_arr]
            xi_ki_0 = xi0[k_arr, i_arr]
            xi_ij_F = xiF[i_arr, j_arr]
            xi_jk_F = xiF[j_arr, k_arr]
            xi_ki_F = xiF[k_arr, i_arr]
            pi3_t0 = (xi_ij_0 * xi_jk_0 * xi_ki_0)
            pi3_tF = (xi_ij_F * xi_jk_F * xi_ki_F)
            mean_pi3_t0 = float(pi3_t0.mean())
            mean_pi3_tF = float(pi3_tF.mean())
            t2_gain = (
                float(np.log(max(mean_pi3_tF, 1e-30)
                              / max(mean_pi3_t0, 1e-30))))
            # T3 cube-root tri vs persistent edge mean
            pers_xi_t0 = np.array([xi_traj[0, i, j] for i, j in pers_edges])
            pers_xi_tF = np.array([xi_traj[-1, i, j] for i, j in pers_edges])
            edge_mean_t0 = float(pers_xi_t0.mean())
            edge_mean_tF = float(pers_xi_tF.mean())
            cube_t0 = float((pi3_t0 ** (1 / 3.0)).mean())
            cube_tF = float((pi3_tF ** (1 / 3.0)).mean())
            r31_t0 = cube_t0 / max(edge_mean_t0, 1e-9)
            r31_tF = cube_tF / max(edge_mean_tF, 1e-9)
            # T4: M3 slack inside the triangle (max over 3 perms)
            def m3_slack(a, b, c):
                # max over (a*b - c, b*c - a, c*a - b, 0)
                return np.maximum(np.maximum(a * b - c, b * c - a),
                                  np.maximum(c * a - b, 0.0))
            slack_t0 = m3_slack(xi_ij_0, xi_jk_0, xi_ki_0)
            slack_tF = m3_slack(xi_ij_F, xi_jk_F, xi_ki_F)
            t4_t0 = float(slack_t0.mean())
            t4_tF = float(slack_tF.mean())
            per_seed.append({
                "seed": s,
                "n_pers": int(pers_edges.shape[0]),
                "n_pers_triangles_total": int(n_total),
                "n_pers_triangles_sampled": int(len(triangles)),
                "n_null": float(n_null),
                "p_pers": p_pers,
                "edge_mean_t0": edge_mean_t0,
                "edge_mean_tF": edge_mean_tF,
                "Pi3_t0": mean_pi3_t0,
                "Pi3_tF": mean_pi3_tF,
                "T2_log_Pi3_gain": t2_gain,
                "T3_R31_t0": r31_t0,
                "T3_R31_tF": r31_tF,
                "T4_slack_t0": t4_t0,
                "T4_slack_tF": t4_tF,
                "Pi3_tF_cuberoot": cube_tF,
            })

        if not per_seed or all(d.get("n_pers_triangles_total", 0) == 0
                                for d in per_seed):
            print(f"  {regime:<7}: no persistent triangles")
            continue
        # Aggregate (only over seeds with triangles)
        good = [d for d in per_seed
                if d.get("n_pers_triangles_total", 0) > 0]
        n_tri_mean = float(np.mean([d["n_pers_triangles_total"]
                                      for d in good]))
        n_null_mean = float(np.mean([d["n_null"] for d in good]))
        pi3_t0_m = float(np.mean([d["Pi3_t0"] for d in good]))
        pi3_tF_m = float(np.mean([d["Pi3_tF"] for d in good]))
        t2_gain_m = float(np.mean([d["T2_log_Pi3_gain"] for d in good]))
        r31_t0_m = float(np.mean([d["T3_R31_t0"] for d in good]))
        r31_tF_m = float(np.mean([d["T3_R31_tF"] for d in good]))
        t4_t0_m = float(np.mean([d["T4_slack_t0"] for d in good]))
        t4_tF_m = float(np.mean([d["T4_slack_tF"] for d in good]))
        cube_tF_m = float(np.mean([d["Pi3_tF_cuberoot"] for d in good]))
        n_pers_m = float(np.mean([d["n_pers"] for d in good]))
        print(f"  {regime:<7} {len(good):>2} "
              f"{n_pers_m:>7.0f} {n_tri_mean:>7.0f} {n_null_mean:>8.0f} "
              f"{pi3_t0_m:>8.5f} {pi3_tF_m:>8.5f} "
              f"{t2_gain_m:>+8.4f} {r31_t0_m:>10.4f} {r31_tF_m:>10.4f} "
              f"{t4_t0_m:>9.4f} {t4_tF_m:>9.4f}")
        rows.append({
            "regime": regime, "N": n_lat,
            "n_seeds_with_triangles": len(good),
            "n_persistent_edges_mean": n_pers_m,
            "n_persistent_triangles_mean": n_tri_mean,
            "n_triangles_bernoulli_null_mean": n_null_mean,
            "T1_lift_over_null": (n_tri_mean
                                   / max(n_null_mean, 1e-9)),
            "Pi3_t0_mean": pi3_t0_m,
            "Pi3_tF_mean": pi3_tF_m,
            "T2_log_Pi3_gain_mean": t2_gain_m,
            "T3_R31_t0_mean": r31_t0_m,
            "T3_R31_tF_mean": r31_tF_m,
            "T4_M3_slack_t0_mean": t4_t0_m,
            "T4_M3_slack_tF_mean": t4_tF_m,
            "Pi3_tF_cuberoot_mean": cube_tF_m,
            "per_seed": per_seed,
        })

    # Cross-regime System-R candidate-match (T5)
    print()
    print("=" * 78)
    print("T5: Cross-regime cube-root triangle-Xi vs algebraic candidates")
    print("=" * 78)
    candidates = {
        "alpha_xi^2 * beta_pi": ALPHA_XI ** 2 * BETA_PI,
        "alpha_xi * beta_pi^2": ALPHA_XI * BETA_PI ** 2,
        "alpha_xi * beta_pi * gamma": ALPHA_XI * BETA_PI * GAMMA,
        "alpha_xi^3": ALPHA_XI ** 3,
        "beta_pi^3": BETA_PI ** 3,
        "D * gamma * beta_pi": D_R * GAMMA * BETA_PI,
        "alpha_xi * eps^2": ALPHA_XI * EPS2,
        "gamma^3": GAMMA ** 3,
        "alpha_xi^2 * gamma": ALPHA_XI ** 2 * GAMMA,
    }
    if rows:
        cube_vals = np.array([r["Pi3_tF_cuberoot_mean"] for r in rows])
        cube_mean = float(cube_vals.mean())
        cube_std = float(cube_vals.std())
        print(f"  Mean cube-root <Pi3>^(1/3) cross-regime = "
              f"{cube_mean:.4f} +/- {cube_std:.4f}")
        print(f"  Per regime:")
        for r in rows:
            print(f"    {r['regime']:<7} cube={r['Pi3_tF_cuberoot_mean']:.4f}")
        print(f"  Algebraic candidates:")
        for name, val in candidates.items():
            print(f"    {name:<26} = {val:.4f} "
                  f"(diff={cube_mean - val:+.4f})")
    print()
    print("=" * 78)
    print("Interpretation")
    print("=" * 78)
    print("  T1 lift: persistent-triangle count over Bernoulli null.")
    print("    T1 >> 1 => persistent edges cluster into triangles, NOT random pairs.")
    print("  T2 log(Pi3 gain): Xi-triple-product growth over trajectory.")
    print("    T2 > 0 => triangle Xi-product GROWS => 3-body binding gain.")
    print("  T3 R31: cube-root triangle Xi vs persistent edge mean.")
    print("    R31 > 1 => triangle is tighter (geometric mean) than the average edge.")
    print("  T4 M3 slack inside triangle:")
    print("    T4 > 0 means the triangle is M3-violating internally.")
    print("    T4_tF > T4_t0 => binding violation INCREASES => bound state grows.")

    bundle = {
        "method": "M3_triangle_bi_to_tri_closure",
        "system_R": {
            "alpha_xi": ALPHA_XI, "beta_pi": BETA_PI, "gamma": GAMMA,
            "D": D_R, "eps_squared": EPS2,
        },
        "candidates": candidates,
        "rows": rows,
        "interpretation": {
            "T1_lift_over_1_implies_clustered_triangles": True,
            "T2_positive_implies_3body_binding_gain": True,
            "T3_R31_above_1_implies_triangle_tighter_than_edge": True,
            "T4_increase_implies_M3_slack_is_binding_observable": True,
        },
    }
    out = REPO / "outputs" / "audit_M3_triangle_closure.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
