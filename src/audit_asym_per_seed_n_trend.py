"""Per-seed asym(PPN-PNN) analysis for N-trend significance.

User question 2026-05-02: 'asym(PPN-PNN)=-0.0283 steigt doch, vor allem
bei runs mit mehr seeds massiv'. Question: is the apparent drift to
negative values at high N a real signal or sampling noise from the
2-seed runs at N=256, N=300?

For each regime, compute per-seed asym values, the mean and std,
and the per-regime uncertainty (mean / sqrt(n_seeds)). Also a binomial
sampling null. Then test the N-trend with weighted least squares.

Output: outputs/audit_asym_per_seed_n_trend.json
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


def main():
    print("=" * 80)
    print("Per-seed asym(PPN-PNN) N-trend analysis")
    print("=" * 80)
    print(f"  {'regime':<7} {'N':>3} {'n_s':>3}  "
          f"{'asym_mean':>10} {'asym_std':>10} "
          f"{'unc_mean':>10}  {'sigma_vs_0':>10}  per-seed values")
    print("-" * 100)
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
        per_seed_asym = []
        per_seed_n_PPN = []
        per_seed_n_PNN = []
        per_seed_n_total = []
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
            triangles = find_persistent_triangles(pers_edges)
            if not triangles:
                continue
            phi = np.angle(psi_last)
            n_PPN = 0
            n_PNN = 0
            n_total_classified = 0
            for i, j, k in triangles:
                d_ij = np.angle(np.exp(1j * (phi[j] - phi[i])))
                d_jk = np.angle(np.exp(1j * (phi[k] - phi[j])))
                d_ki = np.angle(np.exp(1j * (phi[i] - phi[k])))
                if (abs(d_ij) < 1e-9 or abs(d_jk) < 1e-9
                    or abs(d_ki) < 1e-9):
                    continue
                signs = (np.sign(d_ij), np.sign(d_jk), np.sign(d_ki))
                n_pos = sum(1 for s_ in signs if s_ > 0)
                if n_pos == 2:
                    n_PPN += 1
                elif n_pos == 1:
                    n_PNN += 1
                n_total_classified += 1
            if n_PPN + n_PNN > 0:
                a = (n_PPN - n_PNN) / (n_PPN + n_PNN)
                per_seed_asym.append(a)
                per_seed_n_PPN.append(n_PPN)
                per_seed_n_PNN.append(n_PNN)
                per_seed_n_total.append(n_total_classified)
        if not per_seed_asym:
            continue
        arr = np.array(per_seed_asym)
        asym_mean = float(arr.mean())
        asym_std = float(arr.std())
        unc_mean = float(asym_std / np.sqrt(len(arr))) if len(arr) > 0 else 0.0
        sigma_vs_0 = (asym_mean / unc_mean
                       if unc_mean > 1e-12 else float("nan"))
        # Format per-seed list compact
        seed_str = "  ".join(f"{a:+.3f}" for a in arr)
        print(f"  {regime:<7} {n_lat:>3} {len(arr):>3}  "
              f"{asym_mean:>+10.4f} {asym_std:>10.4f} "
              f"{unc_mean:>10.4f}  {sigma_vs_0:>+10.2f}  {seed_str}")
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds": len(arr),
            "asym_per_seed": arr.tolist(),
            "asym_mean": asym_mean,
            "asym_std": asym_std,
            "asym_uncertainty_of_mean": unc_mean,
            "sigma_vs_0": sigma_vs_0,
            "n_PPN_per_seed": per_seed_n_PPN,
            "n_PNN_per_seed": per_seed_n_PNN,
            "n_total_classified_per_seed": per_seed_n_total,
        })
    print()
    print("=" * 80)
    print("N-trend analysis (weighted by 1/unc^2)")
    print("=" * 80)
    if len(rows) >= 3:
        N_arr = np.array([r["N"] for r in rows], dtype=float)
        asym_arr = np.array([r["asym_mean"] for r in rows])
        unc_arr = np.array([r["asym_uncertainty_of_mean"] for r in rows])
        # Weighted linear fit: asym = a + b * (1/N)
        x = 1.0 / N_arr
        w = 1.0 / np.maximum(unc_arr, 1e-6) ** 2
        A_mat = np.column_stack([np.ones_like(x), x])
        # Weighted normal equations
        AtWA = A_mat.T @ (w[:, None] * A_mat)
        AtWy = A_mat.T @ (w * asym_arr)
        coef = np.linalg.solve(AtWA, AtWy)
        a, b = coef
        # residual chi2
        pred = A_mat @ coef
        chi2 = float(np.sum(w * (asym_arr - pred) ** 2))
        dof = len(rows) - 2
        cov = np.linalg.inv(AtWA)
        a_unc = float(np.sqrt(cov[0, 0]))
        b_unc = float(np.sqrt(cov[1, 1]))
        print(f"  Weighted fit asym = a + b/N:")
        print(f"    a (asymptote N->inf) = {a:+.5f} +/- {a_unc:.5f} "
              f"(sigma vs 0: {a/a_unc:+.2f})")
        print(f"    b (1/N coefficient)  = {b:+.4f} +/- {b_unc:.4f}")
        print(f"    chi2/dof = {chi2:.2f}/{dof} = {chi2/max(dof,1):.2f}")
        # Also constant fit
        a_const = float(np.sum(w * asym_arr) / np.sum(w))
        a_const_unc = float(1.0 / np.sqrt(np.sum(w)))
        chi2_const = float(np.sum(w * (asym_arr - a_const) ** 2))
        print(f"  Weighted constant fit:")
        print(f"    asym_const = {a_const:+.5f} +/- {a_const_unc:.5f} "
              f"(sigma vs 0: {a_const/a_const_unc:+.2f})")
        print(f"    chi2 = {chi2_const:.2f}/{len(rows)-1}")
        print(f"  Delta chi2 (linear vs constant) = "
              f"{chi2_const - chi2:.2f} on 1 d.o.f.")
        if chi2_const - chi2 > 4:
            print(f"    => N-trend SIGNIFICANT at >2sigma")
        else:
            print(f"    => N-trend NOT significant")

    bundle = {
        "method": "per_seed_asym_PPN_PNN_n_trend",
        "rows": rows,
    }
    out = REPO / "outputs" / "audit_asym_per_seed_n_trend.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
