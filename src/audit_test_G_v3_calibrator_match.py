"""Test G_v3: '+1 primary defect calibration' via 4-cycle Wilson-loop match.

User hypothesis 2026-05-02: the +1 (calibrator-edge) is primary defect.
Materialization outcome (matter vs antimatter) depends on whether the
4-cycle closure of triangle + calibrator is geometrically calibrated.

Pure-3 doesn't exist in any subgraph (K=N to dense), so we redefine:
For each persistent triangle (i,j,k) with at least one attached extra
edge (l,m), pick the STRONGEST extra edge (highest Xi) as 'primary
calibrator'. Test the calibration condition via:

  4-cycle Wilson loop: phi_loop = phi_to_calibrator + phi_around_triangle
  - If phi_loop ≡ 0 mod 2pi: calibrated (matter)
  - If phi_loop != 0: uncalibrated (antimatter)

Bin triangles by |phi_loop| into low (calibrated) vs high (uncalibrated)
and compute PPN-PNN asymmetry per bin.

Also compute geometric-distance-mismatch: does the calibrator's d-value
match what would be predicted by triangle inequality extension?

P_G3v3:
  - asym(low |phi_loop|) ≠ asym(high |phi_loop|)
  - matter-class lives in low-|phi_loop|, antimatter in high

Output: outputs/audit_test_G_v3_calibrator_match.json
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


def triangle_phase_class(triangle, phi):
    i, j, k = triangle
    d_ij = np.angle(np.exp(1j * (phi[j] - phi[i])))
    d_jk = np.angle(np.exp(1j * (phi[k] - phi[j])))
    d_ki = np.angle(np.exp(1j * (phi[i] - phi[k])))
    if abs(d_ij) < 1e-9 or abs(d_jk) < 1e-9 or abs(d_ki) < 1e-9:
        return "ambiguous"
    n_pos = ((1 if d_ij > 0 else 0) + (1 if d_jk > 0 else 0)
              + (1 if d_ki > 0 else 0))
    if n_pos == 3:
        return "PPP"
    if n_pos == 2:
        return "PPN"
    if n_pos == 1:
        return "PNN"
    return "NNN"


def primary_calibrator(triangle, edge_set, adj, xi_last):
    """Find the strongest non-triangle persistent edge attached to any
    triangle corner. Returns (corner, far_node, xi_value, d_value) or
    None if no calibrator exists."""
    i, j, k = triangle
    candidates = []
    for corner in (i, j, k):
        for nbr in adj[corner]:
            if nbr in (i, j, k):
                continue
            edge = (min(corner, nbr), max(corner, nbr))
            if edge not in edge_set:
                continue
            xi_val = xi_last[corner, nbr]
            candidates.append((corner, nbr, xi_val))
    if not candidates:
        return None
    # Pick highest Xi (strongest binding)
    candidates.sort(key=lambda x: -x[2])
    corner, nbr, xi_val = candidates[0]
    d_val = -np.log(max(xi_val, 1e-9))
    return corner, nbr, float(xi_val), float(d_val)


def four_cycle_wilson_phi(triangle, calibrator_far, phi):
    """4-cycle: corner -> calibrator_far -> corner -> j -> k -> back.
    Compute total enclosed phase via Wilson loop on the modified
    4-cycle.

    Actually: for the 4-cycle (corner, calibrator_far, j, k) — but
    calibrator_far is not connected to j and k by triangle edges.
    Simpler: use the 'extension' loop = 3-edge triangle plus the
    calibrator edge as a tail. The 'loop closure' would require
    calibrator_far -> some triangle corner. Take the L1 distance
    in phase space.

    Define phi_loop = (phi[corner] - phi[calibrator_far])
                      + (sum of triangle principal phases)
                    = how much phi_calibrator_far deviates from
                      the triangle's enclosed-phase.
    """
    i, j, k = triangle
    d_ij = np.angle(np.exp(1j * (phi[j] - phi[i])))
    d_jk = np.angle(np.exp(1j * (phi[k] - phi[j])))
    d_ki = np.angle(np.exp(1j * (phi[i] - phi[k])))
    triangle_phase = d_ij + d_jk + d_ki  # principal sum
    cal_phase = np.angle(np.exp(1j * (phi[calibrator_far]
                                         - phi[triangle[0]])))
    return float(triangle_phase + cal_phase)


def main():
    print("=" * 80)
    print("Test G_v3: 4-cycle Wilson-loop calibration via primary +1")
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
            xi_last = xi_traj[-1].copy()
            psi_last = (psi_r[s, -1].astype(float)
                        + 1j * psi_i[s, -1].astype(float))
            phi = np.angle(psi_last)
            d_xi = np.abs(np.diff(xi_traj, axis=0))
            offdiag = ~np.eye(n_lat, dtype=bool)
            d_off = d_xi[:, offdiag]
            v_med = (float(np.median(d_off[d_off > 0]))
                      if (d_off > 0).any() else 1e-6)
            c_info = 2 * v_med
            persistent_mask_off = (d_off > c_info).mean(axis=0) > 0.5
            ij_idx = np.argwhere(offdiag)
            pers_edges = ij_idx[persistent_mask_off]
            triangles, edge_set, adj = find_persistent_triangles(
                pers_edges)
            if not triangles:
                continue
            # For each triangle, find primary calibrator + compute
            # 4-cycle phase + classify triangle
            triangle_data = []
            for t in triangles:
                cal = primary_calibrator(t, edge_set, adj, xi_last)
                if cal is None:
                    continue
                corner, far_node, xi_cal, d_cal = cal
                phi_loop = four_cycle_wilson_phi(t, far_node, phi)
                pclass = triangle_phase_class(t, phi)
                # Geometric-distance match: does d_cal "fit" expected
                # value from triangle geometry? Compare to mean d in triangle
                d_ij_t = -np.log(max(xi_last[t[0], t[1]], 1e-9))
                d_jk_t = -np.log(max(xi_last[t[1], t[2]], 1e-9))
                d_ki_t = -np.log(max(xi_last[t[2], t[0]], 1e-9))
                d_mean = (d_ij_t + d_jk_t + d_ki_t) / 3.0
                d_mismatch = abs(d_cal - d_mean) / max(d_mean, 1e-6)
                triangle_data.append({
                    "phi_loop": phi_loop,
                    "abs_phi_loop": abs(phi_loop),
                    "pclass": pclass,
                    "d_mismatch": d_mismatch,
                })
            if not triangle_data:
                continue
            # Bin by |phi_loop|: low (calibrated) vs high (uncalibrated)
            phi_arr = np.array([d["abs_phi_loop"]
                                  for d in triangle_data])
            median_phi = float(np.median(phi_arr))
            calib_classes = defaultdict(int)
            uncalib_classes = defaultdict(int)
            for d in triangle_data:
                if d["abs_phi_loop"] < median_phi:
                    calib_classes[d["pclass"]] += 1
                else:
                    uncalib_classes[d["pclass"]] += 1
            def asym(cls):
                p = cls.get("PPN", 0)
                n = cls.get("PNN", 0)
                if p + n == 0:
                    return float("nan")
                return (p - n) / (p + n)
            asym_calib = asym(calib_classes)
            asym_uncalib = asym(uncalib_classes)
            # Bin by d_mismatch
            d_arr = np.array([d["d_mismatch"] for d in triangle_data])
            median_d = float(np.median(d_arr))
            d_calib_classes = defaultdict(int)
            d_uncalib_classes = defaultdict(int)
            for d in triangle_data:
                if d["d_mismatch"] < median_d:
                    d_calib_classes[d["pclass"]] += 1
                else:
                    d_uncalib_classes[d["pclass"]] += 1
            asym_d_calib = asym(d_calib_classes)
            asym_d_uncalib = asym(d_uncalib_classes)
            # Joint bin: both phi_loop AND d_mismatch low
            joint_calib_classes = defaultdict(int)
            joint_uncalib_classes = defaultdict(int)
            for d in triangle_data:
                if (d["abs_phi_loop"] < median_phi
                    and d["d_mismatch"] < median_d):
                    joint_calib_classes[d["pclass"]] += 1
                elif (d["abs_phi_loop"] >= median_phi
                       and d["d_mismatch"] >= median_d):
                    joint_uncalib_classes[d["pclass"]] += 1
            asym_joint_calib = asym(joint_calib_classes)
            asym_joint_uncalib = asym(joint_uncalib_classes)
            # All triangles
            all_classes = defaultdict(int)
            for d in triangle_data:
                all_classes[d["pclass"]] += 1
            asym_all = asym(all_classes)
            per_seed.append({
                "seed": s,
                "n_triangles_with_cal": len(triangle_data),
                "asym_phi_calib": asym_calib,
                "asym_phi_uncalib": asym_uncalib,
                "asym_d_calib": asym_d_calib,
                "asym_d_uncalib": asym_d_uncalib,
                "asym_joint_calib": asym_joint_calib,
                "asym_joint_uncalib": asym_joint_uncalib,
                "asym_all": asym_all,
                "median_phi_loop": median_phi,
                "median_d_mismatch": median_d,
                "n_calib": sum(calib_classes.values()),
                "n_uncalib": sum(uncalib_classes.values()),
            })
        if not per_seed:
            continue
        def mn(key):
            vals = [d[key] for d in per_seed
                     if not (isinstance(d[key], float)
                              and np.isnan(d[key]))]
            return float(np.mean(vals)) if vals else float("nan")
        def std_unc(key):
            vals = [d[key] for d in per_seed
                     if not (isinstance(d[key], float)
                              and np.isnan(d[key]))]
            if len(vals) < 2:
                return float("nan")
            return float(np.std(vals)) / np.sqrt(len(vals))
        a_phi_calib = mn("asym_phi_calib")
        u_phi_calib = std_unc("asym_phi_calib")
        a_phi_uncalib = mn("asym_phi_uncalib")
        u_phi_uncalib = std_unc("asym_phi_uncalib")
        a_d_calib = mn("asym_d_calib")
        a_d_uncalib = mn("asym_d_uncalib")
        a_joint_calib = mn("asym_joint_calib")
        u_joint_calib = std_unc("asym_joint_calib")
        a_joint_uncalib = mn("asym_joint_uncalib")
        u_joint_uncalib = std_unc("asym_joint_uncalib")
        a_all = mn("asym_all")
        n_tri = mn("n_triangles_with_cal")
        print(f"--- {regime} N={n_lat} (n_seeds={len(per_seed)}, "
              f"n_tri/seed={n_tri:.0f}) ---")
        print(f"  Phi-loop calib:    asym = {a_phi_calib:+.4f} "
              f"+/- {u_phi_calib:.4f}")
        print(f"  Phi-loop uncalib:  asym = {a_phi_uncalib:+.4f} "
              f"+/- {u_phi_uncalib:.4f}")
        print(f"  Phi diff (c-u):    {a_phi_calib - a_phi_uncalib:+.4f}")
        print(f"  D-mismatch calib:  asym = {a_d_calib:+.4f}")
        print(f"  D-mismatch uncal:  asym = {a_d_uncalib:+.4f}")
        print(f"  Joint calib:       asym = {a_joint_calib:+.4f} "
              f"+/- {u_joint_calib:.4f}")
        print(f"  Joint uncalib:     asym = {a_joint_uncalib:+.4f} "
              f"+/- {u_joint_uncalib:.4f}")
        print(f"  Joint diff (c-u):  {a_joint_calib - a_joint_uncalib:+.4f}")
        print(f"  asym(all):         {a_all:+.4f}")
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds": len(per_seed),
            "n_triangles_per_seed_mean": n_tri,
            "asym_phi_calib_mean": a_phi_calib,
            "asym_phi_uncalib_mean": a_phi_uncalib,
            "phi_diff_calib_minus_uncalib": a_phi_calib - a_phi_uncalib,
            "asym_d_calib_mean": a_d_calib,
            "asym_d_uncalib_mean": a_d_uncalib,
            "asym_joint_calib_mean": a_joint_calib,
            "asym_joint_uncalib_mean": a_joint_uncalib,
            "joint_diff_calib_minus_uncalib": a_joint_calib - a_joint_uncalib,
            "asym_all_mean": a_all,
            "per_seed": per_seed,
        })
    print()
    print("=" * 80)
    print("Cross-regime synthesis")
    print("=" * 80)
    if rows:
        for tag, key in [
            ("phi_loop_calib", "asym_phi_calib_mean"),
            ("phi_loop_uncalib", "asym_phi_uncalib_mean"),
            ("phi_diff (calib-uncalib)", "phi_diff_calib_minus_uncalib"),
            ("d_mismatch_calib", "asym_d_calib_mean"),
            ("d_mismatch_uncalib", "asym_d_uncalib_mean"),
            ("joint_calib", "asym_joint_calib_mean"),
            ("joint_uncalib", "asym_joint_uncalib_mean"),
            ("joint diff (c-u)", "joint_diff_calib_minus_uncalib"),
        ]:
            vals = [r[key] for r in rows
                     if not np.isnan(r[key])]
            if vals:
                arr = np.array(vals)
                unc = arr.std() / np.sqrt(len(arr))
                sig = abs(arr.mean()) / max(unc, 1e-9)
                print(f"  {tag:<28} = {arr.mean():+.5f} +/- {unc:.5f}  "
                      f"({sig:.2f}σ vs 0)")

    bundle = {
        "method": "test_G_v3_calibrator_4cycle_wilson_match",
        "rows": rows,
    }
    out = REPO / "outputs" / "audit_test_G_v3_calibrator_match.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
