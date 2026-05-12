"""Robustness test: vary c_info threshold (1.5x, 2.0x, 3.0x median |Delta Xi|)
and re-fit the per-seed asym(PPN-PNN) N-trend. If asymptote a_infty
stays stable around -0.0157 (= -pi*gamma^2/2), the result is robust;
otherwise it is a threshold artefact.

Output: outputs/audit_asym_robustness_threshold.json
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

THRESHOLD_FACTORS = [1.5, 2.0, 2.5, 3.0]


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


def per_seed_asym(xi_traj, psi_last, n_lat, threshold_factor):
    d_xi = np.abs(np.diff(xi_traj, axis=0))
    offdiag = ~np.eye(n_lat, dtype=bool)
    d_off = d_xi[:, offdiag]
    v_med = (float(np.median(d_off[d_off > 0]))
              if (d_off > 0).any() else 1e-6)
    c_info = threshold_factor * v_med
    persistent_mask_off = (d_off > c_info).mean(axis=0) > 0.5
    ij_idx = np.argwhere(offdiag)
    pers_edges = ij_idx[persistent_mask_off]
    triangles = find_persistent_triangles(pers_edges)
    if not triangles:
        return None
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
        signs = (np.sign(d_ij), np.sign(d_jk), np.sign(d_ki))
        n_pos = sum(1 for s in signs if s > 0)
        if n_pos == 2:
            n_PPN += 1
        elif n_pos == 1:
            n_PNN += 1
    if n_PPN + n_PNN == 0:
        return None
    return (n_PPN - n_PNN) / (n_PPN + n_PNN)


def main():
    print("=" * 80)
    print("Robustness test: c_info threshold factor variation")
    print("=" * 80)
    print(f"  Reference target: a_infty(2x) = -0.01569 +/- 0.005 = -pi*gamma^2/2")
    print()
    overall_results = {}
    for tf in THRESHOLD_FACTORS:
        print(f"\n--- c_info = {tf} x median |Delta Xi| ---")
        print(f"  {'regime':<7} {'N':>3} {'n_s':>3}  {'asym_mean':>10} "
              f"{'asym_std':>10} {'unc':>10}")
        per_regime = []
        for regime, n_lat, rel in LADDER:
            fp = PARENT / rel
            if not fp.exists():
                continue
            z = np.load(fp, allow_pickle=True)
            snaps = z["edge_xi_snapshots"]
            psi_r = z["psi_real_snapshots"]
            psi_i = z["psi_imag_snapshots"]
            n_seeds = int(snaps.shape[0])
            seed_asym = []
            for s in range(n_seeds):
                xi_traj = np.asarray(snaps[s], dtype=float).copy()
                psi_last = (psi_r[s, -1].astype(float)
                            + 1j * psi_i[s, -1].astype(float))
                a = per_seed_asym(xi_traj, psi_last, n_lat, tf)
                if a is not None:
                    seed_asym.append(a)
            if not seed_asym:
                continue
            arr = np.array(seed_asym)
            mu = float(arr.mean())
            sd = float(arr.std())
            unc = sd / np.sqrt(len(arr))
            per_regime.append({
                "regime": regime, "N": n_lat,
                "n_seeds": len(arr),
                "asym_mean": mu, "asym_std": sd, "unc_mean": unc,
            })
            print(f"  {regime:<7} {n_lat:>3} {len(arr):>3}  "
                  f"{mu:>+10.4f} {sd:>10.4f} {unc:>10.4f}")
        # Weighted fit
        if len(per_regime) >= 3:
            N_arr = np.array([r["N"] for r in per_regime], dtype=float)
            asym_arr = np.array([r["asym_mean"] for r in per_regime])
            unc_arr = np.array([r["unc_mean"] for r in per_regime])
            x = 1.0 / N_arr
            w = 1.0 / np.maximum(unc_arr, 1e-6) ** 2
            A_mat = np.column_stack([np.ones_like(x), x])
            AtWA = A_mat.T @ (w[:, None] * A_mat)
            AtWy = A_mat.T @ (w * asym_arr)
            try:
                coef = np.linalg.solve(AtWA, AtWy)
                a_inf, b = coef
                cov = np.linalg.inv(AtWA)
                a_unc = float(np.sqrt(cov[0, 0]))
                b_unc = float(np.sqrt(cov[1, 1]))
                pred = A_mat @ coef
                chi2 = float(np.sum(w * (asym_arr - pred) ** 2))
                dof = len(per_regime) - 2
                print(f"  Asymptote a_infty = {a_inf:+.5f} +/- {a_unc:.5f} "
                      f"(target -pi*gamma^2/2 = -0.01571, "
                      f"diff = {a_inf - (-np.pi * 0.01 / 2):+.5f})")
                print(f"  b coef = {b:+.4f} +/- {b_unc:.4f}, "
                      f"chi2/dof = {chi2:.2f}/{dof}")
                overall_results[tf] = {
                    "asymptote": a_inf,
                    "asymptote_unc": a_unc,
                    "b_coef": b, "b_unc": b_unc,
                    "chi2": chi2, "dof": dof,
                    "per_regime": per_regime,
                }
            except np.linalg.LinAlgError:
                print(f"  Fit failed (singular matrix)")
                overall_results[tf] = {"per_regime": per_regime}
    print()
    print("=" * 80)
    print("Robustness summary")
    print("=" * 80)
    print(f"  {'c_info_factor':<14} {'a_infty':>10} {'unc':>8} "
          f"{'diff_to_-pi*gamma^2/2':>22}")
    target = -np.pi * 0.01 / 2  # = -pi/200 = -0.015708
    for tf, res in overall_results.items():
        if "asymptote" in res:
            a = res["asymptote"]
            u = res["asymptote_unc"]
            print(f"  {tf:<14.2f} {a:>+10.5f} {u:>8.5f} "
                  f"{a - target:>+22.5f}")
    print()
    print("If asymptote stays around -0.01571 across all 4 thresholds,")
    print("the result is ROBUST against threshold definition.")
    bundle = {
        "method": "asym_robustness_threshold_test",
        "threshold_factors": THRESHOLD_FACTORS,
        "target_value_pi_gamma_squared_over_2": -np.pi * 0.01 / 2,
        "results_by_factor": overall_results,
    }
    out = REPO / "outputs" / "audit_asym_robustness_threshold.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
