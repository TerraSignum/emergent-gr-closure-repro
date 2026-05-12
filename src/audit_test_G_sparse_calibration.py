"""Test G_sparse: '+1 calibration' on top-K sparse persistent-edge subgraph.

Companion to audit_test_G_calibration_hypothesis.py. The dense version
saw frac_with_tail=1.0 cross-regime (every triangle has tails). To test
the user's pure-3 vs 3+1 calibration hypothesis directly, we need a
SPARSE subgraph where pure-3 triangles actually exist.

Sparse selection: top-K by trajectory-mean |Delta Xi|, sweep K in
{N, 1.5N, 2N, 3N}. At K=N we expect ~few triangles and some pure-3.

For each K, compute:
  - n_pure3, n_3plus1, n_3plus2, n_3plus_more counts
  - per-class asym(PPN-PNN), n_class
  - geometry signature mean per class (perimeter, log-area, Pi3, mean|psi|^2)
  - mismatch_to_pure3 for 3+1 triangles
  - asym in calibrated (mismatch < median) vs uncalibrated subset

Test predictions:
  P_G1: asym(calibrated 3+1) > asym(uncalibrated 3+1) (matter vs antimatter)
  P_G3: asym(pure-3) ≈ 0 (pre-calibration neutral)
  P_G_geom: |signature(pure-3) - signature(3+1, calibrated)| smaller than
            |signature(pure-3) - signature(3+1, uncalibrated)|

Output: outputs/audit_test_G_sparse_calibration.json
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

K_FACTORS = [1.0, 1.5, 2.0, 3.0]


def topk_persistent_edges(d_xi, n_lat, K):
    activity = d_xi.mean(axis=0)
    np.fill_diagonal(activity, 0)
    iu, ju = np.triu_indices(n_lat, k=1)
    vals = activity[iu, ju]
    if vals.size == 0:
        return set()
    K_eff = min(K, vals.size)
    top_idx = np.argpartition(-vals, K_eff - 1)[:K_eff]
    edges = set()
    for idx in top_idx:
        a, b = int(iu[idx]), int(ju[idx])
        edges.add((min(a, b), max(a, b)))
    return edges


def find_triangles(edge_set):
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
    return triangles, adj


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


def gravity_signature(triangle, xi_last, psi_abs2):
    i, j, k = triangle
    d_ij = -np.log(max(xi_last[i, j], 1e-9))
    d_jk = -np.log(max(xi_last[j, k], 1e-9))
    d_ki = -np.log(max(xi_last[k, i], 1e-9))
    perim = d_ij + d_jk + d_ki
    s = perim / 2
    factor = max((s - d_ij) * (s - d_jk) * (s - d_ki), 1e-30)
    log_area = 0.5 * np.log(s * factor)
    pi3 = xi_last[i, j] * xi_last[j, k] * xi_last[k, i]
    mean_psi2 = (psi_abs2[i] + psi_abs2[j] + psi_abs2[k]) / 3.0
    return np.array([perim, log_area, pi3, mean_psi2])


def asym_from_classes(class_dict):
    n_PPN = class_dict.get("PPN", 0)
    n_PNN = class_dict.get("PNN", 0)
    if n_PPN + n_PNN == 0:
        return float("nan"), 0
    return (n_PPN - n_PNN) / (n_PPN + n_PNN), n_PPN + n_PNN


def per_seed_test_G_sparse(xi_traj, psi_last, n_lat, K_factor):
    K = int(K_factor * n_lat)
    d_xi = np.abs(np.diff(xi_traj, axis=0))
    edges = topk_persistent_edges(d_xi, n_lat, K)
    triangles, adj = find_triangles(edges)
    if not triangles:
        return None
    psi_abs2 = np.real(psi_last) ** 2 + np.imag(psi_last) ** 2
    phi = np.angle(psi_last)
    xi_last = xi_traj[-1].copy()
    # Per-triangle data
    tri_data = []
    for t in triangles:
        i, j, k = t
        deg = [len(adj[c]) for c in (i, j, k)]
        extras = sum(d - 2 for d in deg)
        sig = gravity_signature(t, xi_last, psi_abs2)
        pclass = triangle_phase_class(t, phi)
        tri_data.append((t, sig, pclass, extras))
    # Group by extras
    pure3 = [d for d in tri_data if d[3] == 0]
    plus1 = [d for d in tri_data if d[3] == 1]
    plus2 = [d for d in tri_data if d[3] == 2]
    plus_more = [d for d in tri_data if d[3] >= 3]
    # Class counts per group
    classes_pure = defaultdict(int)
    for _, _, pc, _ in pure3:
        classes_pure[pc] += 1
    classes_plus1 = defaultdict(int)
    for _, _, pc, _ in plus1:
        classes_plus1[pc] += 1
    classes_plus2 = defaultdict(int)
    for _, _, pc, _ in plus2:
        classes_plus2[pc] += 1
    classes_plus_more = defaultdict(int)
    for _, _, pc, _ in plus_more:
        classes_plus_more[pc] += 1
    # Asym per group
    asym_pure, n_pure_class = asym_from_classes(classes_pure)
    asym_plus1, n_plus1_class = asym_from_classes(classes_plus1)
    asym_plus2, n_plus2_class = asym_from_classes(classes_plus2)
    asym_plus_more, n_plus_more_class = asym_from_classes(
        classes_plus_more)
    # Mean signature per group
    sig_pure_mean = (np.mean([d[1] for d in pure3], axis=0)
                       if pure3 else None)
    sig_plus1_mean = (np.mean([d[1] for d in plus1], axis=0)
                        if plus1 else None)
    sig_plus2_mean = (np.mean([d[1] for d in plus2], axis=0)
                        if plus2 else None)
    sig_pure_std = (np.std([d[1] for d in pure3], axis=0)
                     if len(pure3) > 1 else None)
    # Mismatch of 3+1 from pure-3 baseline
    if pure3 and plus1:
        std_safe = (sig_pure_std
                     if sig_pure_std is not None
                     else np.ones_like(sig_pure_mean))
        std_safe = np.where(std_safe < 1e-9, 1.0, std_safe)
        mismatches_plus1 = np.array([
            float(np.linalg.norm((d[1] - sig_pure_mean) / std_safe))
            for d in plus1])
        median_mm = float(np.median(mismatches_plus1))
        calib_classes = defaultdict(int)
        uncalib_classes = defaultdict(int)
        for d, mm in zip(plus1, mismatches_plus1):
            if mm < median_mm:
                calib_classes[d[2]] += 1
            else:
                uncalib_classes[d[2]] += 1
        asym_calib, n_calib = asym_from_classes(calib_classes)
        asym_uncalib, n_uncalib = asym_from_classes(uncalib_classes)
    else:
        asym_calib, asym_uncalib = float("nan"), float("nan")
        n_calib, n_uncalib = 0, 0
    # Geometric mismatch test G_geom
    geo_diff_calib_to_pure = float("nan")
    geo_diff_uncalib_to_pure = float("nan")
    if pure3 and plus1 and sig_pure_mean is not None:
        # Compute median signature of calibrated and uncalibrated 3+1
        std_safe = (sig_pure_std
                     if sig_pure_std is not None
                     else np.ones_like(sig_pure_mean))
        std_safe = np.where(std_safe < 1e-9, 1.0, std_safe)
        mismatches_plus1 = np.array([
            float(np.linalg.norm((d[1] - sig_pure_mean) / std_safe))
            for d in plus1])
        median_mm = float(np.median(mismatches_plus1))
        sigs_calib = [d[1] for d, mm in zip(plus1, mismatches_plus1)
                       if mm < median_mm]
        sigs_uncalib = [d[1] for d, mm in zip(plus1, mismatches_plus1)
                         if mm >= median_mm]
        if sigs_calib:
            sig_calib_mean = np.mean(sigs_calib, axis=0)
            geo_diff_calib_to_pure = float(np.linalg.norm(
                (sig_calib_mean - sig_pure_mean) / std_safe))
        if sigs_uncalib:
            sig_uncalib_mean = np.mean(sigs_uncalib, axis=0)
            geo_diff_uncalib_to_pure = float(np.linalg.norm(
                (sig_uncalib_mean - sig_pure_mean) / std_safe))
    return {
        "K_factor": K_factor, "K": K,
        "n_total_triangles": len(triangles),
        "n_pure3": len(pure3), "n_3plus1": len(plus1),
        "n_3plus2": len(plus2), "n_3plus_more": len(plus_more),
        "asym_pure3": asym_pure, "n_class_pure3": n_pure_class,
        "asym_3plus1": asym_plus1, "n_class_3plus1": n_plus1_class,
        "asym_3plus2": asym_plus2, "n_class_3plus2": n_plus2_class,
        "asym_3plus_more": asym_plus_more,
        "n_class_3plus_more": n_plus_more_class,
        "asym_calibrated": asym_calib, "n_calib": n_calib,
        "asym_uncalibrated": asym_uncalib, "n_uncalib": n_uncalib,
        "geo_diff_calib_to_pure": geo_diff_calib_to_pure,
        "geo_diff_uncalib_to_pure": geo_diff_uncalib_to_pure,
    }


def main():
    print("=" * 80)
    print("Test G_sparse: pure-3 vs 3+1 calibration on top-K subgraph")
    print("=" * 80)
    print(f"K factors: {K_FACTORS}")
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
        per_K = {}
        for K_f in K_FACTORS:
            per_seed = []
            for s in range(n_seeds):
                xi_traj = np.asarray(snaps[s], dtype=float).copy()
                psi_last = (psi_r[s, -1].astype(float)
                            + 1j * psi_i[s, -1].astype(float))
                stats = per_seed_test_G_sparse(xi_traj, psi_last,
                                                  n_lat, K_f)
                if stats is not None:
                    per_seed.append(stats)
            per_K[K_f] = per_seed
        if not any(per_K.values()):
            continue
        print(f"--- {regime} N={n_lat} ---")
        regime_data = {"regime": regime, "N": n_lat,
                        "n_seeds": n_seeds}
        for K_f in K_FACTORS:
            per_seed = per_K[K_f]
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
            n_pure = mn("n_pure3")
            n_p1 = mn("n_3plus1")
            n_p2 = mn("n_3plus2")
            n_pm = mn("n_3plus_more")
            a_pure = mn("asym_pure3")
            u_pure = std_unc("asym_pure3")
            a_p1 = mn("asym_3plus1")
            u_p1 = std_unc("asym_3plus1")
            a_p2 = mn("asym_3plus2")
            u_p2 = std_unc("asym_3plus2")
            a_calib = mn("asym_calibrated")
            u_calib = std_unc("asym_calibrated")
            a_uncal = mn("asym_uncalibrated")
            u_uncal = std_unc("asym_uncalibrated")
            geo_calib = mn("geo_diff_calib_to_pure")
            geo_uncal = mn("geo_diff_uncalib_to_pure")
            print(f"  K={K_f}*N: counts pure={n_pure:.1f} 3+1={n_p1:.1f} "
                  f"3+2={n_p2:.1f} more={n_pm:.1f}")
            print(f"          asym pure={a_pure:+.4f}+-{u_pure:.4f}, "
                  f"3+1={a_p1:+.4f}+-{u_p1:.4f}, "
                  f"3+2={a_p2:+.4f}+-{u_p2:.4f}")
            print(f"          asym calib={a_calib:+.4f}+-{u_calib:.4f}, "
                  f"uncalib={a_uncal:+.4f}+-{u_uncal:.4f}, "
                  f"diff={a_calib - a_uncal:+.4f}")
            print(f"          geo_diff calib_to_pure={geo_calib:.3f}, "
                  f"uncalib_to_pure={geo_uncal:.3f}")
            regime_data[f"K{K_f}"] = {
                "n_pure3_mean": n_pure, "n_3plus1_mean": n_p1,
                "n_3plus2_mean": n_p2, "n_3plus_more_mean": n_pm,
                "asym_pure3_mean": a_pure, "asym_pure3_unc": u_pure,
                "asym_3plus1_mean": a_p1, "asym_3plus1_unc": u_p1,
                "asym_3plus2_mean": a_p2, "asym_3plus2_unc": u_p2,
                "asym_calibrated_mean": a_calib,
                "asym_calibrated_unc": u_calib,
                "asym_uncalibrated_mean": a_uncal,
                "asym_uncalibrated_unc": u_uncal,
                "calib_minus_uncalib": a_calib - a_uncal,
                "geo_diff_calib_to_pure_mean": geo_calib,
                "geo_diff_uncalib_to_pure_mean": geo_uncal,
                "per_seed": per_seed,
            }
        rows.append(regime_data)
    print()
    print("=" * 80)
    print("Cross-regime synthesis per K factor")
    print("=" * 80)
    if rows:
        for K_f in K_FACTORS:
            print(f"\n--- K = {K_f} * N ---")
            for tag, key in [
                ("asym_pure3", "asym_pure3_mean"),
                ("asym_3plus1", "asym_3plus1_mean"),
                ("asym_calibrated", "asym_calibrated_mean"),
                ("asym_uncalibrated", "asym_uncalibrated_mean"),
                ("calib_minus_uncalib", "calib_minus_uncalib"),
                ("geo_diff_calib_to_pure",
                 "geo_diff_calib_to_pure_mean"),
                ("geo_diff_uncalib_to_pure",
                 "geo_diff_uncalib_to_pure_mean"),
            ]:
                vals = [r[f"K{K_f}"][key]
                          for r in rows
                          if f"K{K_f}" in r
                          and not np.isnan(r[f"K{K_f}"][key])]
                if vals:
                    arr = np.array(vals)
                    sigma_vs_0 = (abs(arr.mean())
                                    / (arr.std() / np.sqrt(len(arr)))
                                    if arr.std() > 1e-12 else 0)
                    print(f"  {tag:<25} = {arr.mean():+.5f} +/- "
                          f"{arr.std()/np.sqrt(len(arr)):.5f}  "
                          f"({sigma_vs_0:.2f}σ vs 0)")
    bundle = {
        "method": "test_G_sparse_calibration_top_k_subgraph",
        "K_factors": K_FACTORS,
        "rows": rows,
    }
    out = REPO / "outputs" / "audit_test_G_sparse_calibration.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
