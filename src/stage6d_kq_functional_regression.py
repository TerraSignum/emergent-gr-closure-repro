"""Stage 6d: optimization for K/Q metric-functional form via linear
regression on persistent-edge subgraph features.

Approach:
  1. Load k_snapshots, q_snapshots, edge_xi_snapshots, psi_*_snapshots
     across the eight P5N regimes (with full seeds).
  2. For each edge (i,j), build feature vector:
        [1, Xi_ij, |psi_i*psi_j|, |psi_i|^2 + |psi_j|^2, d_ij,
         Re(psi_i*psi_j), Im(psi_i*psi_j), <Xi> per row i,
         <Xi> per row j, ...]
  3. Linear regression for K_ij and Q_ij with these features.
  4. Check fitted coefficients for closeness to System-R
     rationals (1/4, 1/2, gamma, beta_pi, alpha_xi, gamma^2, etc.).
  5. Cross-validate by leaving out one regime at a time.
  6. If linear basis insufficient (R^2 < 0.95), test polynomial
     and bilinear features (e.g., Xi_ij * |psi_i*psi_j|).

Goal: find a clean closed-form K[Xi, psi] and Q[Xi, psi]
expression matching empirical data within 1% rms.

Output: outputs/stage6d_kq_functional_regression.json
"""
from __future__ import annotations
import json
import sys
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

# System-R rational candidates for fit-coefficient matching
SYSTEM_R_CANDIDATES = {
    "0":          0.0,
    "1":          1.0,
    "1/2":        0.5,
    "1/3":        1/3,
    "1/4":        0.25,
    "-1/4":       -0.25,
    "-1/2":       -0.5,
    "alpha_xi":   0.9,
    "beta_pi":    15/16,
    "gamma":      0.10,
    "-gamma":     -0.10,
    "D":          67/80,
    "eps2":       0.05,
    "1-alpha_xi": 0.10,
    "alpha_xi/2": 0.45,
    "beta_pi/2":  15/32,
    "1-beta_pi":  1/16,
    "gamma/2":    0.05,
    "1/16":       1/16,
    "1/8":        1/8,
    "-gamma/2":   -0.05,
}


def closest_system_r(value, tol=0.02):
    best_name, best_diff = "no_match", float("inf")
    for name, target in SYSTEM_R_CANDIDATES.items():
        diff = abs(value - target)
        if diff < best_diff:
            best_diff, best_name = diff, name
    if best_diff < tol:
        return best_name, best_diff
    return "no_match", best_diff


def build_features_for_seed(xi_last, psi_last, n_lat):
    """Build feature matrix for one seed: shape (N*N, n_features).
    Excludes diagonal."""
    psi_re = np.real(psi_last)
    psi_im = np.imag(psi_last)
    psi_abs = np.abs(psi_last)
    psi_abs2 = psi_abs ** 2
    # All (i,j) including diagonal — we'll mask later
    re_outer = np.outer(psi_re, psi_re)
    im_outer = np.outer(psi_im, psi_im)
    abs_outer = np.outer(psi_abs, psi_abs)
    abs2_sum = np.outer(psi_abs2, np.ones(n_lat)) + \
                np.outer(np.ones(n_lat), psi_abs2)
    # psi_i * psi_j* = re_i*re_j + im_i*im_j + i*(im_i*re_j - re_i*im_j)
    re_psi_psi_conj = re_outer + im_outer  # |psi_i|·|psi_j|·cos(arg diff)
    im_psi_psi_conj = (np.outer(psi_im, psi_re)
                        - np.outer(psi_re, psi_im))
    d_ij = -np.log(np.maximum(xi_last, 1e-9))
    # Constant feature
    const = np.ones_like(xi_last)
    feature_names = [
        "const",
        "Xi_ij",
        "|psi_i*psi_j|",
        "|psi_i|^2+|psi_j|^2",
        "d_ij",
        "Re(psi_i psi_j*)",
        "Im(psi_i psi_j*)",
        "Xi_ij * |psi_i psi_j*|",
        "Xi_ij^2",
    ]
    feats = np.stack([
        const,
        xi_last,
        abs_outer,
        abs2_sum,
        d_ij,
        re_psi_psi_conj,
        im_psi_psi_conj,
        xi_last * abs_outer,
        xi_last ** 2,
    ], axis=-1)  # shape (N, N, F)
    return feats, feature_names


def regress_KQ(xi_last, psi_last, k_meas, q_meas, n_lat):
    """Linear regression K_ij = sum_f w_f * f_ij, similarly for Q."""
    feats, names = build_features_for_seed(xi_last, psi_last, n_lat)
    # Mask diagonal
    mask = ~np.eye(n_lat, dtype=bool)
    n_feats = feats.shape[-1]
    X = feats[mask].reshape(-1, n_feats)
    y_K = k_meas[mask]
    y_Q = q_meas[mask]
    # Linear regression: w_K = (X^T X)^{-1} X^T y_K
    XTX = X.T @ X
    coef_K = np.linalg.solve(XTX, X.T @ y_K)
    coef_Q = np.linalg.solve(XTX, X.T @ y_Q)
    # Residuals
    K_pred = X @ coef_K
    Q_pred = X @ coef_Q
    R2_K = 1 - np.sum((y_K - K_pred) ** 2) / np.sum((y_K - y_K.mean()) ** 2)
    R2_Q = 1 - np.sum((y_Q - Q_pred) ** 2) / np.sum((y_Q - y_Q.mean()) ** 2)
    rms_K = float(np.sqrt(np.mean((y_K - K_pred) ** 2)))
    rms_Q = float(np.sqrt(np.mean((y_Q - Q_pred) ** 2)))
    return {
        "feature_names": names,
        "coef_K": [float(c) for c in coef_K],
        "coef_Q": [float(c) for c in coef_Q],
        "R2_K": float(R2_K),
        "R2_Q": float(R2_Q),
        "rms_K": rms_K,
        "rms_Q": rms_Q,
    }


def main():
    print("=" * 80)
    print("Stage 6d: K/Q functional optimization via linear regression")
    print("=" * 80)
    print()
    rows = []
    all_coef_K = []
    all_coef_Q = []
    feature_names_global = None
    for regime, n_lat, rel in LADDER:
        fp = PARENT / rel
        if not fp.exists():
            continue
        z = np.load(fp, allow_pickle=True)
        snaps = z["edge_xi_snapshots"]
        psi_r = z["psi_real_snapshots"]
        psi_i = z["psi_imag_snapshots"]
        k_snaps = z["k_snapshots"]
        q_snaps = z["q_snapshots"]
        n_seeds = int(snaps.shape[0])
        per_seed = []
        for s in range(min(n_seeds, 12)):
            xi_last = snaps[s, -1]
            psi_last = (psi_r[s, -1].astype(float)
                        + 1j * psi_i[s, -1].astype(float))
            k_meas = k_snaps[s, -1]
            q_meas = q_snaps[s, -1]
            res = regress_KQ(xi_last, psi_last, k_meas, q_meas, n_lat)
            per_seed.append(res)
            if feature_names_global is None:
                feature_names_global = res["feature_names"]
        if not per_seed:
            continue
        coef_K_arr = np.array([d["coef_K"] for d in per_seed])
        coef_Q_arr = np.array([d["coef_Q"] for d in per_seed])
        coef_K_mean = coef_K_arr.mean(axis=0)
        coef_K_std = coef_K_arr.std(axis=0)
        coef_Q_mean = coef_Q_arr.mean(axis=0)
        coef_Q_std = coef_Q_arr.std(axis=0)
        R2_K_mean = float(np.mean([d["R2_K"] for d in per_seed]))
        R2_Q_mean = float(np.mean([d["R2_Q"] for d in per_seed]))
        rms_K_mean = float(np.mean([d["rms_K"] for d in per_seed]))
        rms_Q_mean = float(np.mean([d["rms_Q"] for d in per_seed]))
        print(f"--- {regime} N={n_lat} (n_seeds={len(per_seed)}) ---")
        print(f"  R^2 K = {R2_K_mean:.4f}, R^2 Q = {R2_Q_mean:.4f}")
        print(f"  rms K = {rms_K_mean:.4e}, rms Q = {rms_Q_mean:.4e}")
        all_coef_K.append(coef_K_mean)
        all_coef_Q.append(coef_Q_mean)
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds": len(per_seed),
            "feature_names": feature_names_global,
            "coef_K_mean": coef_K_mean.tolist(),
            "coef_K_std": coef_K_std.tolist(),
            "coef_Q_mean": coef_Q_mean.tolist(),
            "coef_Q_std": coef_Q_std.tolist(),
            "R2_K_mean": R2_K_mean,
            "R2_Q_mean": R2_Q_mean,
            "rms_K_mean": rms_K_mean,
            "rms_Q_mean": rms_Q_mean,
        })

    # Cross-regime aggregated coefficient analysis
    print()
    print("=" * 80)
    print("Cross-regime coefficient analysis")
    print("=" * 80)
    if all_coef_K:
        all_coef_K_arr = np.array(all_coef_K)
        all_coef_Q_arr = np.array(all_coef_Q)
        cr_K_mean = all_coef_K_arr.mean(axis=0)
        cr_K_std = all_coef_K_arr.std(axis=0)
        cr_Q_mean = all_coef_Q_arr.mean(axis=0)
        cr_Q_std = all_coef_Q_arr.std(axis=0)
        print(f"\nK coefficients (cross-regime mean +/- std):")
        print(f"  {'feature':<28s} {'mean':>12s} {'std':>12s} {'best System-R match':>30s}")
        for i, fn in enumerate(feature_names_global):
            best, diff = closest_system_r(cr_K_mean[i])
            print(f"  {fn:<28s} {cr_K_mean[i]:>+12.5f} {cr_K_std[i]:>12.5f}  "
                  f"{best:>20s} (diff {diff:.3f})")
        print(f"\nQ coefficients (cross-regime mean +/- std):")
        print(f"  {'feature':<28s} {'mean':>12s} {'std':>12s} {'best System-R match':>30s}")
        for i, fn in enumerate(feature_names_global):
            best, diff = closest_system_r(cr_Q_mean[i])
            print(f"  {fn:<28s} {cr_Q_mean[i]:>+12.5f} {cr_Q_std[i]:>12.5f}  "
                  f"{best:>20s} (diff {diff:.3f})")

    bundle = {
        "method": "stage6d_kq_functional_regression",
        "rows": rows,
        "system_R_candidates": SYSTEM_R_CANDIDATES,
        "cross_regime_K_coef_mean": (
            cr_K_mean.tolist() if all_coef_K else None),
        "cross_regime_Q_coef_mean": (
            cr_Q_mean.tolist() if all_coef_K else None),
    }
    out = REPO / "outputs" / "stage6d_kq_functional_regression.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
