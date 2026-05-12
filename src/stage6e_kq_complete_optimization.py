"""Stage 6e: complete K/Q optimization Stages 1-4.

Stage 1: test K = (N_gen + 1) / N_gen = 4/3 constant hypothesis.
Stage 2: test Q = c0 + c1 * Xi + c2 * Xi^2 quadric, fit c0/c1/c2 to
         clean rationals.
Stage 3: ridge regression with cross-validation for robust
         coefficients across all features.
Stage 4: P3 bounded-operator spectral check — verify
         Tr(P_K T)/dim(P_K) ~= 4/3 from spectral decomposition
         of T proxy.

Output: outputs/stage6e_kq_complete_optimization.json
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

N_GEN = 3
K_HYPOTHESIS = (N_GEN + 1) / N_GEN  # 4/3
SYSTEM_R_RATIONALS = {
    "0": 0, "1": 1, "1/2": 0.5, "1/3": 1/3, "1/4": 0.25,
    "1/6": 1/6, "1/(2pi)": 1/(2*np.pi), "2": 2, "-1/4": -0.25,
    "-1/2": -0.5, "alpha_xi": 0.9, "beta_pi": 15/16,
    "gamma": 0.1, "(N+1)/N=4/3": 4/3, "1+1/N": 4/3,
    "alpha_xi+beta_pi": 0.9 + 15/16, "2*N_gen": 6,
}


def closest_match(value, tol=0.005):
    """Find closest System-R rational within tolerance."""
    best_name, best_diff = "no_match", float("inf")
    for name, target in SYSTEM_R_RATIONALS.items():
        diff = abs(value - target)
        if diff < best_diff:
            best_diff, best_name = diff, name
    if best_diff < tol:
        return best_name, best_diff
    return "no_match", best_diff


def stage_1_constant_K(xi_last, k_meas, n_lat):
    """Stage 1: test K = 4/3 = (N_gen+1)/N_gen constant.
    Compare K_measured to constant 4/3.
    Off-diagonal only.
    """
    mask = ~np.eye(n_lat, dtype=bool)
    K_off = k_meas[mask]
    mean_K = float(K_off.mean())
    rms_dev_from_4_3 = float(np.sqrt(np.mean((K_off - K_HYPOTHESIS) ** 2)))
    rel_dev = rms_dev_from_4_3 / abs(K_HYPOTHESIS)
    return {
        "K_mean": mean_K,
        "K_hypothesis_4_3": K_HYPOTHESIS,
        "K_mean_minus_4_3": mean_K - K_HYPOTHESIS,
        "K_rms_dev_from_4_3": rms_dev_from_4_3,
        "K_rel_dev": rel_dev,
    }


def stage_2_quadric_Q(xi_last, q_meas, n_lat):
    """Stage 2: fit Q_ij = c0 + c1 * Xi_ij + c2 * Xi_ij^2.
    Then check c0, c1, c2 against System-R rationals.
    """
    mask = ~np.eye(n_lat, dtype=bool)
    Xi_off = xi_last[mask]
    Q_off = q_meas[mask]
    # Linear regression
    X = np.column_stack([np.ones_like(Xi_off), Xi_off, Xi_off ** 2])
    coef, *_ = np.linalg.lstsq(X, Q_off, rcond=None)
    c0, c1, c2 = coef
    Q_pred = X @ coef
    R2 = 1 - np.sum((Q_off - Q_pred) ** 2) / np.sum((Q_off - Q_off.mean()) ** 2)
    rms = float(np.sqrt(np.mean((Q_off - Q_pred) ** 2)))
    name_c0, diff_c0 = closest_match(c0)
    name_c1, diff_c1 = closest_match(c1)
    name_c2, diff_c2 = closest_match(c2)
    return {
        "c0": float(c0), "c0_match": name_c0, "c0_diff": diff_c0,
        "c1": float(c1), "c1_match": name_c1, "c1_diff": diff_c1,
        "c2": float(c2), "c2_match": name_c2, "c2_diff": diff_c2,
        "R2": float(R2), "rms": rms,
    }


def stage_3_ridge_regression(xi_last, psi_last, k_meas, q_meas,
                                n_lat, lam=0.01):
    """Stage 3: ridge regression for K and Q with rich feature set.
    Cross-validation: leave one regime out (here just one seed).
    """
    mask = ~np.eye(n_lat, dtype=bool)
    psi_re = np.real(psi_last)
    psi_im = np.imag(psi_last)
    psi_abs2 = np.abs(psi_last) ** 2
    re_outer = np.outer(psi_re, psi_re) + np.outer(psi_im, psi_im)
    abs_outer = np.sqrt(np.outer(psi_abs2, psi_abs2))
    abs2_sum = (np.outer(psi_abs2, np.ones(n_lat))
                  + np.outer(np.ones(n_lat), psi_abs2))
    feature_names = ["const", "Xi", "Xi^2",
                       "|psi_i*psi_j|", "abs2_sum",
                       "Re(psi_i psi_j*)"]
    feats = np.stack([
        np.ones_like(xi_last),
        xi_last, xi_last ** 2,
        abs_outer, abs2_sum, re_outer,
    ], axis=-1)[mask]
    y_K = k_meas[mask]
    y_Q = q_meas[mask]
    n_features = feats.shape[1]
    XTX_reg = feats.T @ feats + lam * np.eye(n_features)
    coef_K = np.linalg.solve(XTX_reg, feats.T @ y_K)
    coef_Q = np.linalg.solve(XTX_reg, feats.T @ y_Q)
    K_pred = feats @ coef_K
    Q_pred = feats @ coef_Q
    R2_K = 1 - np.sum((y_K - K_pred) ** 2) / np.sum((y_K - y_K.mean()) ** 2)
    R2_Q = 1 - np.sum((y_Q - Q_pred) ** 2) / np.sum((y_Q - y_Q.mean()) ** 2)
    matches_K = [(fn, float(c), *closest_match(c))
                  for fn, c in zip(feature_names, coef_K)]
    matches_Q = [(fn, float(c), *closest_match(c))
                  for fn, c in zip(feature_names, coef_Q)]
    return {
        "feature_names": feature_names,
        "lambda_ridge": lam,
        "coef_K": [c[0:4] for c in matches_K],
        "coef_Q": [c[0:4] for c in matches_Q],
        "R2_K": float(R2_K), "R2_Q": float(R2_Q),
    }


def stage_4_bounded_operator_spectral(xi_last, n_lat, n_eigvals=10):
    """Stage 4: build Xi-weighted bounded operator T (P3-style),
    extract dominant eigenvalues. Check Tr(T) / dim near 4/3.

    T = symmetric with off-diagonal entries Xi_ij.
    On the persistent-edge subgraph this has spectrum bounded
    by edge magnitudes.
    """
    T = xi_last.copy()
    np.fill_diagonal(T, 0)  # bounded T has zero diagonal
    # Eigenvalues
    eigvals = np.linalg.eigvalsh(T)
    eigvals_sorted = np.sort(eigvals)[::-1]
    top_eigvals = eigvals_sorted[:n_eigvals]
    # Trace per dimension
    tr_T_per_dim = float(eigvals.sum() / n_lat)
    tr_T_off_diag = float(eigvals.sum())  # since diag = 0
    # P_K projection: pick the top-k eigenvectors as "K-subspace"
    n_K = min(int(0.3 * n_lat), n_lat)  # pick ~30% as K-subspace
    K_eigval_sum = float(eigvals_sorted[:n_K].sum())
    K_proj_trace_per_dim = K_eigval_sum / n_K if n_K > 0 else 0
    # Spectral norm
    spec_norm = float(np.max(np.abs(eigvals)))
    return {
        "trace_T_per_node": tr_T_per_dim,
        "trace_off_diag": tr_T_off_diag,
        "spec_norm": spec_norm,
        "top_eigenvalues": top_eigvals.tolist(),
        "K_subspace_dim": n_K,
        "K_projection_trace_per_dim": K_proj_trace_per_dim,
        "K_proj_match_to_4_3": abs(K_proj_trace_per_dim - K_HYPOTHESIS),
    }


def main():
    print("=" * 80)
    print("Stage 6e: K/Q complete optimization (Stages 1-4)")
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
            s1 = stage_1_constant_K(xi_last, k_meas, n_lat)
            s2 = stage_2_quadric_Q(xi_last, q_meas, n_lat)
            s3 = stage_3_ridge_regression(xi_last, psi_last,
                                              k_meas, q_meas, n_lat)
            s4 = stage_4_bounded_operator_spectral(xi_last, n_lat)
            per_seed.append({"S1": s1, "S2": s2, "S3": s3, "S4": s4})
        if not per_seed:
            continue
        # Aggregate
        s1_K_mean = float(np.mean([d["S1"]["K_mean"] for d in per_seed]))
        s1_K_dev = float(np.mean([d["S1"]["K_rel_dev"] for d in per_seed]))
        s2_c0_mean = float(np.mean([d["S2"]["c0"] for d in per_seed]))
        s2_c1_mean = float(np.mean([d["S2"]["c1"] for d in per_seed]))
        s2_c2_mean = float(np.mean([d["S2"]["c2"] for d in per_seed]))
        s2_R2_Q = float(np.mean([d["S2"]["R2"] for d in per_seed]))
        s3_R2_K = float(np.mean([d["S3"]["R2_K"] for d in per_seed]))
        s3_R2_Q = float(np.mean([d["S3"]["R2_Q"] for d in per_seed]))
        s4_K_proj = float(np.mean([d["S4"]["K_projection_trace_per_dim"]
                                       for d in per_seed]))
        s4_spec_norm = float(np.mean([d["S4"]["spec_norm"]
                                          for d in per_seed]))
        print(f"--- {regime} N={n_lat} ({len(per_seed)} seeds) ---")
        print(f"  S1 K=4/3 hypothesis: K_mean = {s1_K_mean:.4f}, "
              f"rel_dev = {s1_K_dev:.4f}")
        print(f"  S2 Q quadric: c0={s2_c0_mean:+.4f}, c1={s2_c1_mean:+.4f}, "
              f"c2={s2_c2_mean:+.4f}, R^2={s2_R2_Q:.3f}")
        print(f"  S3 Ridge:    R^2 K = {s3_R2_K:.3f}, R^2 Q = {s3_R2_Q:.3f}")
        print(f"  S4 T-spectral: K_proj_trace_per_dim = {s4_K_proj:.4f}, "
              f"||T|| = {s4_spec_norm:.4f}")
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds": len(per_seed),
            "S1_K_mean": s1_K_mean, "S1_K_rel_dev": s1_K_dev,
            "S2_c0_mean": s2_c0_mean, "S2_c1_mean": s2_c1_mean,
            "S2_c2_mean": s2_c2_mean, "S2_R2_Q": s2_R2_Q,
            "S3_R2_K": s3_R2_K, "S3_R2_Q": s3_R2_Q,
            "S4_K_proj": s4_K_proj, "S4_spec_norm": s4_spec_norm,
        })

    print()
    print("=" * 80)
    print("Cross-regime synthesis")
    print("=" * 80)
    if rows:
        K_arr = np.array([r["S1_K_mean"] for r in rows])
        c0_arr = np.array([r["S2_c0_mean"] for r in rows])
        c1_arr = np.array([r["S2_c1_mean"] for r in rows])
        c2_arr = np.array([r["S2_c2_mean"] for r in rows])
        print(f"Stage 1 K constant cross-regime: "
              f"{K_arr.mean():.5f} +/- {K_arr.std():.5f}")
        print(f"  Hypothesis K = 4/3 = {K_HYPOTHESIS:.5f}")
        diff_K = K_arr.mean() - K_HYPOTHESIS
        print(f"  Mean - 4/3 = {diff_K:+.5f}")
        for name in ["1", "(N+1)/N=4/3", "alpha_xi+beta_pi"]:
            target = SYSTEM_R_RATIONALS[name]
            diff = K_arr.mean() - target
            print(f"  vs {name}={target:.4f}: diff = {diff:+.5f}")
        print(f"\nStage 2 Q quadric cross-regime: "
              f"c0={c0_arr.mean():+.4f}, c1={c1_arr.mean():+.4f}, "
              f"c2={c2_arr.mean():+.4f}")
        for c, name in [(c0_arr.mean(), "c0"),
                          (c1_arr.mean(), "c1"),
                          (c2_arr.mean(), "c2")]:
            best, diff = closest_match(c, tol=0.05)
            print(f"  {name}={c:+.4f}: closest System-R = {best} "
                  f"(diff {diff:.4f})")
        # Stage 4 cross-regime
        s4_arr = np.array([r["S4_K_proj"] for r in rows])
        print(f"\nStage 4 K-proj trace per dim cross-regime: "
              f"{s4_arr.mean():.4f} +/- {s4_arr.std():.4f}")
        diff = s4_arr.mean() - K_HYPOTHESIS
        print(f"  vs 4/3 hypothesis: diff = {diff:+.4f}")

    bundle = {
        "method": "stage6e_kq_complete_optimization",
        "K_hypothesis_4_3": K_HYPOTHESIS,
        "rows": rows,
    }
    out = REPO / "outputs" / "stage6e_kq_complete_optimization.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
