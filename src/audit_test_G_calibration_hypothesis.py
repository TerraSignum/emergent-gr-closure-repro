"""Test G: '+1 as primary defect, calibration condition' hypothesis.

User hypothesis 2026-05-02 (formulated after Test F):
The +1 (single attached calibrator-edge) is the PRIMARY defect that
breaks the local geometry. A persistent triangle (3-cycle) emerges
either as MATTER or ANTIMATTER depending on whether the resulting
geometry of the "3=2+1" structure (triangle + attached calibrator)
matches the geometry that pure "3" alone would have.

Concretely:
  - 'Pure 3' triangles: all three corners have degree exactly 2 in
    the persistent-edge subgraph (only the three triangle-edges,
    no calibrator attached)
  - '3+1' triangles: at least one corner has degree >= 3 (extra
    calibrator-edge attached)

For each 3+1 triangle, compute the "geometry mismatch" between its
signature and the pure-3 baseline averaged on the same regime/seed.
Bin 3+1 triangles by mismatch magnitude into:
  - 'Calibrated' subset: mismatch < median (geometry of 3+1 matches
    pure-3, hypothesis: matter-class outcome)
  - 'Uncalibrated' subset: mismatch >= median (geometry differs,
    hypothesis: antimatter-class outcome)

Geometry signature per triangle: (perimeter, log-area-proxy, mean-Xi-product).
Mismatch = euclidean distance in normalized signature space.

Test predictions:
  P_G1: PPN excess in 'calibrated' subset (matter classification holds)
  P_G2: PNN excess in 'uncalibrated' subset (antimatter classification)
  P_G3: Pure-3 subset itself shows zero asymmetry (pre-calibration neutral)

Output: outputs/audit_test_G_calibration_hypothesis.json
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


def triangle_class_label(triangle, adj):
    """Classify triangle by attached-edge structure:
       'pure3': all three corners have degree exactly 2 (only triangle-edges)
       '3+1':   one corner has degree 3 (one calibrator), others 2
       '3+2':   one corner has degree 4, OR two corners have degree 3
       '3+more': anything else
    """
    i, j, k = triangle
    deg = [len(adj[c]) for c in (i, j, k)]
    extras = [d - 2 for d in deg]  # extras[c] = #attached non-triangle edges
    total_extras = sum(extras)
    if total_extras == 0:
        return "pure3"
    if total_extras == 1:
        return "3+1"
    if total_extras == 2:
        return "3+2"
    return "3+more"


def emergent_gravity_signature(triangle, xi_last, psi_abs2):
    """Triangle 'gravity signature' = (perimeter d_total, area_proxy,
    Pi3 = Xi-tripleproduct, mean |psi|^2 at corners)."""
    i, j, k = triangle
    d_ij = -np.log(max(xi_last[i, j], 1e-9))
    d_jk = -np.log(max(xi_last[j, k], 1e-9))
    d_ki = -np.log(max(xi_last[k, i], 1e-9))
    perim = d_ij + d_jk + d_ki
    # Heron-like log-area proxy via abs M3 deficit
    s = perim / 2
    factor = max((s - d_ij) * (s - d_jk) * (s - d_ki), 1e-30)
    log_area = 0.5 * np.log(s * factor)
    pi3 = xi_last[i, j] * xi_last[j, k] * xi_last[k, i]
    mean_psi2 = (psi_abs2[i] + psi_abs2[j] + psi_abs2[k]) / 3.0
    return np.array([perim, log_area, pi3, mean_psi2])


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


def asym_from_counts(counts):
    n_PPN = counts.get("PPN", 0)
    n_PNN = counts.get("PNN", 0)
    if n_PPN + n_PNN == 0:
        return float("nan"), 0
    return (n_PPN - n_PNN) / (n_PPN + n_PNN), n_PPN + n_PNN


def main():
    print("=" * 80)
    print("Test G: +1 as primary defect, calibration condition hypothesis")
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
            psi_abs2 = np.real(psi_last) ** 2 + np.imag(psi_last) ** 2
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
            # Classify each triangle, compute signature, phase class
            tri_data = []  # (triangle, sig, pclass, extras_sum, label)
            for t in triangles:
                lbl = triangle_class_label(t, adj)
                sig = emergent_gravity_signature(t, xi_last, psi_abs2)
                pclass = triangle_phase_class(t, phi)
                deg = [len(adj[c]) for c in t]
                extras_sum = sum(d - 2 for d in deg)
                tri_data.append((t, sig, pclass, extras_sum, lbl))
            if not tri_data:
                continue
            # Use median signature as baseline (proxy for 'calibrated geometry')
            all_sigs = np.array([d[1] for d in tri_data])
            baseline = np.median(all_sigs, axis=0)
            base_std = all_sigs.std(axis=0)
            base_std[base_std < 1e-9] = 1.0
            # Compute mismatch for each triangle
            mismatches = []
            for (t, sig, pc, ex, lbl) in tri_data:
                mm = float(np.linalg.norm((sig - baseline) / base_std))
                mismatches.append(mm)
            mm_arr = np.array(mismatches)
            median_mm = float(np.median(mm_arr))
            calib_classes = defaultdict(int)
            uncalib_classes = defaultdict(int)
            for (t, sig, pc, ex, lbl), mm in zip(tri_data, mismatches):
                if mm < median_mm:
                    calib_classes[pc] += 1
                else:
                    uncalib_classes[pc] += 1
            asym_calib, n_calib = asym_from_counts(calib_classes)
            asym_uncalib, n_uncalib = asym_from_counts(uncalib_classes)
            # Bin by extras_sum: low (≤ median_extras) vs high
            extras_arr = np.array([d[3] for d in tri_data])
            median_extras = float(np.median(extras_arr))
            low_extras_classes = defaultdict(int)
            high_extras_classes = defaultdict(int)
            for (t, sig, pc, ex, lbl) in tri_data:
                if ex <= median_extras:
                    low_extras_classes[pc] += 1
                else:
                    high_extras_classes[pc] += 1
            asym_low_ex, n_low_ex = asym_from_counts(low_extras_classes)
            asym_high_ex, n_high_ex = asym_from_counts(high_extras_classes)
            # All triangles
            all_classes = defaultdict(int)
            for (t, sig, pc, ex, lbl) in tri_data:
                all_classes[pc] += 1
            asym_all, n_all = asym_from_counts(all_classes)
            # Pure-3 (extras == 0; rare)
            pure3_classes = defaultdict(int)
            for (t, sig, pc, ex, lbl) in tri_data:
                if ex == 0:
                    pure3_classes[pc] += 1
            asym_pure3, n_pure3 = asym_from_counts(pure3_classes)
            n_3plus1 = sum(1 for d in tri_data if d[3] == 1)
            n_3plus2 = sum(1 for d in tri_data if d[3] == 2)
            asym_3plus2 = asym_calib  # placeholder (unused)
            n_pure_actual = sum(1 for d in tri_data if d[3] == 0)
            per_seed.append({
                "seed": s,
                "n_pure3": n_pure_actual,
                "n_3plus1": n_3plus1,
                "n_3plus2": sum(1 for d in tri_data if d[3] == 2),
                "n_total_triangles": len(tri_data),
                "asym_pure3": asym_pure3,
                "asym_calibrated_3plus1": asym_calib,
                "asym_uncalibrated_3plus1": asym_uncalib,
                "asym_low_extras": asym_low_ex,
                "asym_high_extras": asym_high_ex,
                "asym_all": asym_all,
                "n_calib_class": n_calib,
                "n_uncalib_class": n_uncalib,
                "median_extras": median_extras,
                "median_mismatch": median_mm,
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
        a_pure3 = mn("asym_pure3")
        u_pure3 = std_unc("asym_pure3")
        a_calib = mn("asym_calibrated_3plus1")
        u_calib = std_unc("asym_calibrated_3plus1")
        a_uncalib = mn("asym_uncalibrated_3plus1")
        u_uncalib = std_unc("asym_uncalibrated_3plus1")
        a_low_ex = mn("asym_low_extras")
        a_high_ex = mn("asym_high_extras")
        a_all = mn("asym_all")
        n_pure_m = mn("n_pure3")
        n_3p1 = mn("n_3plus1")
        n_3p2 = mn("n_3plus2")
        print(f"--- {regime} N={n_lat} (n_seeds={len(per_seed)}) ---")
        print(f"  Class counts (mean per seed):")
        print(f"    pure-3       = {n_pure_m:.0f}")
        print(f"    3+1          = {n_3p1:.0f}")
        print(f"    3+2          = {n_3p2:.0f}")
        print(f"  P_G3 asym(pure3)         = {a_pure3:+.4f} +/- {u_pure3:.4f}")
        print(f"  P_G1 asym(calibrated)    = {a_calib:+.4f} +/- {u_calib:.4f}")
        print(f"  P_G2 asym(uncalibrated)  = {a_uncalib:+.4f} +/- {u_uncalib:.4f}")
        print(f"      Diff calib - uncalib = "
              f"{a_calib - a_uncalib:+.4f}")
        print(f"      asym(low_extras)     = {a_low_ex:+.4f}")
        print(f"      asym(high_extras)    = {a_high_ex:+.4f}")
        print(f"      Diff low - high      = {a_low_ex - a_high_ex:+.4f}")
        print(f"      asym(all)            = {a_all:+.4f}")
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds": len(per_seed),
            "n_pure3_mean": n_pure_m,
            "n_3plus1_mean": n_3p1,
            "n_3plus2_mean": n_3p2,
            "asym_pure3_mean": a_pure3,
            "asym_pure3_unc": u_pure3,
            "asym_calibrated_mean": a_calib,
            "asym_calibrated_unc": u_calib,
            "asym_uncalibrated_mean": a_uncalib,
            "asym_uncalibrated_unc": u_uncalib,
            "asym_low_extras_mean": a_low_ex,
            "asym_high_extras_mean": a_high_ex,
            "asym_all_mean": a_all,
            "calib_minus_uncalib_mean": a_calib - a_uncalib,
            "low_minus_high_extras_mean": a_low_ex - a_high_ex,
            "per_seed": per_seed,
        })

    print()
    print("=" * 80)
    print("Cross-regime synthesis (Test G predictions)")
    print("=" * 80)
    if rows:
        # P_G3: pure-3 should have ~zero asym
        a3 = np.array([r["asym_pure3_mean"] for r in rows
                         if not np.isnan(r["asym_pure3_mean"])])
        if a3.size:
            print(f"  P_G3 pure-3 asym cross-regime: "
                  f"{a3.mean():+.5f} +/- {a3.std()/np.sqrt(len(a3)):.5f}  "
                  f"({abs(a3.mean())/(max(a3.std()/np.sqrt(len(a3)),1e-9)):.2f}σ vs 0)")
        # P_G1: calibrated should be PPN-positive (matter)
        ac = np.array([r["asym_calibrated_mean"] for r in rows
                         if not np.isnan(r["asym_calibrated_mean"])])
        if ac.size:
            print(f"  P_G1 calibrated 3+1 asym cross-regime: "
                  f"{ac.mean():+.5f} +/- {ac.std()/np.sqrt(len(ac)):.5f}  "
                  f"({abs(ac.mean())/(max(ac.std()/np.sqrt(len(ac)),1e-9)):.2f}σ vs 0)")
        # P_G2: uncalibrated should be PNN-positive (antimatter)
        au = np.array([r["asym_uncalibrated_mean"] for r in rows
                         if not np.isnan(r["asym_uncalibrated_mean"])])
        if au.size:
            print(f"  P_G2 uncalibrated 3+1 asym cross-regime: "
                  f"{au.mean():+.5f} +/- {au.std()/np.sqrt(len(au)):.5f}  "
                  f"({abs(au.mean())/(max(au.std()/np.sqrt(len(au)),1e-9)):.2f}σ vs 0)")
        # Diff: calib - uncalib should be > 0 (matter > antimatter signal)
        diff = np.array([r["calib_minus_uncalib_mean"] for r in rows])
        diff_unc = np.array([np.sqrt(r["asym_calibrated_unc"] ** 2
                                       + r["asym_uncalibrated_unc"] ** 2)
                                for r in rows])
        if diff.size:
            print(f"  Diff (calibrated - uncalibrated) cross-regime: "
                  f"{diff.mean():+.5f} +/- {diff.std()/np.sqrt(len(diff)):.5f}")

    bundle = {
        "method": "test_G_calibration_hypothesis",
        "rows": rows,
    }
    out = REPO / "outputs" / "audit_test_G_calibration_hypothesis.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
