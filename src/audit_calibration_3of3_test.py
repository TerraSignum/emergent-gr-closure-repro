"""Calibration-3/3 hypothesis test.

User hypothesis 2026-05-02 (after autoloop): the +1 single defect
breaks the geometry. Matter forms when 3/3 = 3 perfectly equal
components calibrate (specific case). Antimatter is the GENERIC
3/2+1 = 5/2 mismatch case.

Operationalization on the lattice:
  For each persistent triangle (i,j,k), compute the equilaterality
  of the three principal phase-difference magnitudes:
    {|d_ij|, |d_jk|, |d_ki|}
  - equilateral ratio = 1 - std/mean (1=perfect equilateral, 0=degenerate)
  - 'calibrated 3/3' = equilateral ratio > 0.85
  - 'uncalibrated 3/2+1' = equilateral ratio < 0.5
  - 'transitional' = otherwise

Hypothesis predicts:
  - calibrated 3/3 -> PPN-rich (matter)
  - uncalibrated 3/2+1 -> PNN-rich (antimatter)

Cross-regime tests on 8 P5N regimes, full seeds.

Output: outputs/audit_calibration_3of3_test.json
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
    return triangles


def per_seed_calibration(xi_traj, psi_last, n_lat,
                            high_thr=0.85, low_thr=0.5):
    d_xi = np.abs(np.diff(xi_traj, axis=0))
    offdiag = ~np.eye(n_lat, dtype=bool)
    d_off = d_xi[:, offdiag]
    v_med = (float(np.median(d_off[d_off > 0]))
              if (d_off > 0).any() else 1e-6)
    c_info = 2 * v_med
    persistent_mask_off = (d_off > c_info).mean(axis=0) > 0.5
    ij_idx = np.argwhere(offdiag)
    pers_edges = ij_idx[persistent_mask_off]
    triangles = find_persistent_triangles(pers_edges)
    if not triangles:
        return None
    phi = np.angle(psi_last)
    n_calibrated = 0
    n_uncalibrated = 0
    n_transitional = 0
    cls_calib = defaultdict(int)
    cls_uncalib = defaultdict(int)
    cls_trans = defaultdict(int)
    equilateral_scores = []
    for i, j, k in triangles:
        d_ij = np.angle(np.exp(1j * (phi[j] - phi[i])))
        d_jk = np.angle(np.exp(1j * (phi[k] - phi[j])))
        d_ki = np.angle(np.exp(1j * (phi[i] - phi[k])))
        if (abs(d_ij) < 1e-9 or abs(d_jk) < 1e-9
            or abs(d_ki) < 1e-9):
            continue
        n_pos = ((1 if d_ij > 0 else 0) + (1 if d_jk > 0 else 0)
                  + (1 if d_ki > 0 else 0))
        if n_pos == 3:
            tri_class = "PPP"
        elif n_pos == 2:
            tri_class = "PPN"
        elif n_pos == 1:
            tri_class = "PNN"
        else:
            tri_class = "NNN"
        # Equilateral score: 1 - std/mean of magnitudes
        mags = np.array([abs(d_ij), abs(d_jk), abs(d_ki)])
        m = mags.mean()
        s = mags.std()
        eq = 1 - (s / max(m, 1e-9))
        equilateral_scores.append(eq)
        if eq > high_thr:
            n_calibrated += 1
            cls_calib[tri_class] += 1
        elif eq < low_thr:
            n_uncalibrated += 1
            cls_uncalib[tri_class] += 1
        else:
            n_transitional += 1
            cls_trans[tri_class] += 1
    def asym(d):
        n_PPN = d.get("PPN", 0)
        n_PNN = d.get("PNN", 0)
        if n_PPN + n_PNN == 0:
            return float("nan"), 0
        return (n_PPN - n_PNN) / (n_PPN + n_PNN), n_PPN + n_PNN
    asym_calib, n_calib_class = asym(cls_calib)
    asym_uncalib, n_uncalib_class = asym(cls_uncalib)
    asym_trans, n_trans_class = asym(cls_trans)
    n_total = n_calibrated + n_uncalibrated + n_transitional
    return {
        "n_total": n_total,
        "n_calibrated": n_calibrated,
        "n_uncalibrated": n_uncalibrated,
        "n_transitional": n_transitional,
        "frac_calibrated": n_calibrated / max(n_total, 1),
        "frac_uncalibrated": n_uncalibrated / max(n_total, 1),
        "frac_transitional": n_transitional / max(n_total, 1),
        "asym_calibrated": asym_calib,
        "asym_uncalibrated": asym_uncalib,
        "asym_transitional": asym_trans,
        "n_calib_PPN_PNN": n_calib_class,
        "n_uncalib_PPN_PNN": n_uncalib_class,
        "equilateral_score_mean": (float(np.mean(equilateral_scores))
                                       if equilateral_scores else float("nan")),
        "equilateral_score_std": (float(np.std(equilateral_scores))
                                      if equilateral_scores else float("nan")),
    }


def main():
    print("=" * 80)
    print("Calibration 3/3 vs 3/2+1 hypothesis test")
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
            stats = per_seed_calibration(xi_traj, psi_last, n_lat)
            if stats is None:
                continue
            per_seed.append(stats)
        if not per_seed:
            continue
        def mn(key):
            vals = [d[key] for d in per_seed
                     if not (isinstance(d.get(key), float) and np.isnan(d[key]))]
            return float(np.mean(vals)) if vals else float("nan")
        def std_unc(key):
            vals = [d[key] for d in per_seed
                     if not (isinstance(d.get(key), float) and np.isnan(d[key]))]
            if len(vals) < 2:
                return float("nan")
            return float(np.std(vals)) / np.sqrt(len(vals))
        f_calib = mn("frac_calibrated")
        f_uncalib = mn("frac_uncalibrated")
        f_trans = mn("frac_transitional")
        a_calib = mn("asym_calibrated")
        a_uncalib = mn("asym_uncalibrated")
        a_trans = mn("asym_transitional")
        u_calib = std_unc("asym_calibrated")
        u_uncalib = std_unc("asym_uncalibrated")
        eq_mean = mn("equilateral_score_mean")
        eq_std = mn("equilateral_score_std")
        n_calib_avg = mn("n_calibrated")
        n_uncalib_avg = mn("n_uncalibrated")
        n_trans_avg = mn("n_transitional")
        print(f"--- {regime} N={n_lat} (n_seeds={len(per_seed)}) ---")
        print(f"  Equilateral score: <eq>={eq_mean:.3f} +/- {eq_std:.3f}")
        print(f"  Fractions: calibrated={f_calib:.3f} ({n_calib_avg:.0f}/seed), "
              f"transitional={f_trans:.3f} ({n_trans_avg:.0f}), "
              f"uncalib={f_uncalib:.3f} ({n_uncalib_avg:.0f})")
        print(f"  asym_calibrated   = {a_calib:+.4f} +/- {u_calib:.4f}  "
              f"(prediction: PPN-rich for matter)")
        print(f"  asym_uncalibrated = {a_uncalib:+.4f} +/- {u_uncalib:.4f}  "
              f"(prediction: PNN-rich for antimatter)")
        print(f"  asym_transitional = {a_trans:+.4f}")
        print(f"  Diff (calib - uncalib) = {a_calib - a_uncalib:+.4f}")
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds": len(per_seed),
            "frac_calibrated_mean": f_calib,
            "frac_uncalibrated_mean": f_uncalib,
            "frac_transitional_mean": f_trans,
            "asym_calibrated_mean": a_calib,
            "asym_uncalibrated_mean": a_uncalib,
            "asym_transitional_mean": a_trans,
            "asym_calibrated_unc": u_calib,
            "asym_uncalibrated_unc": u_uncalib,
            "diff_calib_uncalib": a_calib - a_uncalib,
            "equilateral_score_mean": eq_mean,
            "per_seed": per_seed,
        })
    print()
    print("=" * 80)
    print("Cross-regime synthesis")
    print("=" * 80)
    if rows:
        a_calib_arr = np.array([r["asym_calibrated_mean"] for r in rows
                                  if not np.isnan(r["asym_calibrated_mean"])])
        a_uncalib_arr = np.array([r["asym_uncalibrated_mean"] for r in rows
                                    if not np.isnan(r["asym_uncalibrated_mean"])])
        diff_arr = np.array([r["diff_calib_uncalib"] for r in rows
                                if not np.isnan(r["diff_calib_uncalib"])])
        f_calib_arr = np.array([r["frac_calibrated_mean"] for r in rows])
        if a_calib_arr.size:
            print(f"  asym_calibrated cross-regime: "
                  f"{a_calib_arr.mean():+.5f} +/- {a_calib_arr.std()/np.sqrt(len(a_calib_arr)):.5f}  "
                  f"({abs(a_calib_arr.mean()/(a_calib_arr.std()/np.sqrt(len(a_calib_arr)))):.2f}σ)")
        if a_uncalib_arr.size:
            print(f"  asym_uncalibrated cross-regime: "
                  f"{a_uncalib_arr.mean():+.5f} +/- {a_uncalib_arr.std()/np.sqrt(len(a_uncalib_arr)):.5f}  "
                  f"({abs(a_uncalib_arr.mean()/(a_uncalib_arr.std()/np.sqrt(len(a_uncalib_arr)))):.2f}σ)")
        if diff_arr.size:
            print(f"  Diff (calib - uncalib) cross-regime: "
                  f"{diff_arr.mean():+.5f} +/- {diff_arr.std()/np.sqrt(len(diff_arr)):.5f}  "
                  f"({abs(diff_arr.mean()/(diff_arr.std()/np.sqrt(len(diff_arr)))):.2f}σ)")
        print(f"  frac_calibrated cross-regime: "
              f"{f_calib_arr.mean():.4f} +/- {f_calib_arr.std():.4f}")
        # Symanzik fit on f_calibrated to see if there's an asymptote
        if f_calib_arr.size >= 3:
            N_arr = np.array([r["N"] for r in rows], dtype=float)
            x = 1.0 / N_arr
            A = np.column_stack([np.ones_like(x), x])
            try:
                coef, *_ = np.linalg.lstsq(A, f_calib_arr, rcond=None)
                print(f"  Symanzik f_calibrated = a + b/N: "
                      f"a_inf = {coef[0]:.5f}, b = {coef[1]:+.4f}")
            except Exception:
                pass

    bundle = {
        "method": "calibration_3of3_test",
        "rows": rows,
    }
    out = REPO / "outputs" / "audit_calibration_3of3_test.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
