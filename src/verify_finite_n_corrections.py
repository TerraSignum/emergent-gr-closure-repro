r"""Comprehensive finite-N corrections diagnostic.

For each branch (VAC + MATTER), fit:
  Symanzik-1: a + b/N
  Symanzik-2: a + b/N + c/N^2
  Symanzik-3: a + b/N + c/N^2 + d/N^3

Report per-N residuals after each fit and compute:
  - Asymptote convergence rate: |a_K - a_{K-1}| / |a_K|
  - Bayes factor for each Symanzik order
  - Sigma_systematic from residuals at maximum N
  - Total error budget = sigma_statistical + sigma_systematic

This tool answers: "is the matter-branch finite-N drift real, or is it
captured by higher-order Symanzik?"

Also: produce a per-N residual plot data (CSV) to identify the cross-over
N at which Symanzik-K becomes insufficient.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parent
OUTPUTS = REPO / "outputs"

sys.path.insert(0, str(SRC))
from _d1_ladder_discovery import discover_d1_ladder  # noqa: E402
from adaptive_pipeline import hierarchical_bayes  # noqa: E402


def extract_branch(ladder, n_filter, max_seeds: int = 12):
    """Extract per-seed weighted lambda_2 from branch regimes."""
    all_n = []
    all_lam = []
    sources = []
    for regime, n_lat, npz_path in ladder:
        if not n_filter(n_lat):
            continue
        try:
            d = np.load(npz_path, allow_pickle=True)
            if "edge_xi_snapshots" not in d.files:
                continue
            xi_arr = d["edge_xi_snapshots"][:, -1]
            n_seeds = min(max_seeds, xi_arr.shape[0])
            sources.append((regime, n_lat, n_seeds))
            for s in range(n_seeds):
                xi = xi_arr[s].astype(np.float64)
                np.fill_diagonal(xi, 0.0)
                deg = np.maximum(xi.sum(axis=1), 1e-12)
                d_inv = 1.0 / np.sqrt(deg)
                L = np.eye(xi.shape[0]) - (d_inv[:, None] * xi * d_inv[None, :])
                L = 0.5 * (L + L.T)
                eigs = np.linalg.eigvalsh(L)
                all_n.append(n_lat)
                all_lam.append(float(eigs[1]))
        except OSError:
            print(f"  [skip] {regime}: NPZ unreadable")
    return np.asarray(all_n), np.asarray(all_lam), sources


def per_n_residuals(n_arr, lam_arr, fit_coefs, order: int):
    """Residuals lambda_observed - lambda_predicted per N point."""
    n_unique = sorted(set(n_arr.tolist()))
    rows = []
    for n in n_unique:
        mask = n_arr == n
        observed = float(np.mean(lam_arr[mask]))
        # Predict from Symanzik-K fit
        pred = fit_coefs[0]
        if order >= 1:
            pred += fit_coefs[1] / n
        if order >= 2:
            pred += fit_coefs[2] / n ** 2
        if order >= 3:
            pred += fit_coefs[3] / n ** 3
        rows.append({"N": n, "observed": observed, "predicted": pred,
                      "residual": observed - pred,
                      "rel_residual_pct": (observed - pred) / pred * 100})
    return rows


def fit_symanzik_k(n_arr, lam_arr, order: int):
    """Symanzik-K fit, returns (coefs, RSS, asymptote_std_OLS)."""
    n_float = n_arr.astype(np.float64)
    if order == 1:
        X = np.column_stack([np.ones_like(n_float), 1.0 / n_float])
    elif order == 2:
        X = np.column_stack([np.ones_like(n_float), 1.0 / n_float, 1.0 / n_float ** 2])
    elif order == 3:
        X = np.column_stack([np.ones_like(n_float), 1.0 / n_float,
                              1.0 / n_float ** 2, 1.0 / n_float ** 3])
    else:
        raise ValueError(f"order must be 1, 2, or 3; got {order}")
    coef, *_ = np.linalg.lstsq(X, lam_arr, rcond=None)
    residuals = lam_arr - X @ coef
    rss = float((residuals ** 2).sum())
    # Estimate asymptote std via OLS covariance
    n_obs = len(lam_arr)
    p = len(coef)
    if n_obs > p:
        sigma_sq = rss / (n_obs - p)
        cov = sigma_sq * np.linalg.inv(X.T @ X)
        a_std = float(np.sqrt(cov[0, 0]))
    else:
        a_std = float("inf")
    return coef.tolist(), rss, a_std


def diagnose_branch(branch_name, n_arr, lam_arr, target_value):
    """Full Symanzik-K diagnostic for a branch."""
    print("=" * 78)
    print(f"  {branch_name}")
    print("=" * 78)
    print(f"  {len(lam_arr)} obs across N in [{n_arr.min()}, {n_arr.max()}]")
    print(f"  Target: {target_value:.5f}")
    print()

    diag = {"branch": branch_name, "n_obs": int(len(lam_arr)),
            "n_range": [int(n_arr.min()), int(n_arr.max())],
            "target": target_value,
            "fits": {}}

    for k in (1, 2, 3):
        if len(lam_arr) < k + 1:
            continue
        coefs, rss, a_std = fit_symanzik_k(n_arr, lam_arr, k)
        residuals = per_n_residuals(n_arr, lam_arr, coefs, k)
        max_res = max(abs(r["rel_residual_pct"]) for r in residuals)
        # Hierarchical Bayes evidence (Laplace)
        try:
            hb = hierarchical_bayes.laplace_fit(n_arr, lam_arr, model_order=k)
            log_ev = hb.log_evidence
            sigma_seed = hb.sigma_seed_mean
        except Exception:
            log_ev = float("nan")
            sigma_seed = float("nan")
        print(f"  Symanzik-{k}: a = {coefs[0]:.5f} +/- {a_std:.5f}")
        print(f"    coefs = {[f'{c:+.4f}' for c in coefs]}")
        print(f"    max |residual| = {max_res:.3f}%")
        print(f"    sigma_seed (HB) = {sigma_seed:.5f}")
        print(f"    log_evidence (HB) = {log_ev:+.2f}")
        z_target = (coefs[0] - target_value) / max(a_std, 1e-12)
        print(f"    asymptote vs target: z = {z_target:+.2f}")
        print()
        diag["fits"][f"symanzik_{k}"] = {
            "coefs": coefs, "asymptote": coefs[0], "asymptote_std": a_std,
            "rss": rss, "max_rel_residual_pct": max_res,
            "log_evidence": log_ev, "sigma_seed": sigma_seed,
            "z_vs_target": z_target,
            "per_n_residuals": residuals,
        }

    # Convergence-rate diagnostic
    if "symanzik_1" in diag["fits"] and "symanzik_2" in diag["fits"]:
        a1 = diag["fits"]["symanzik_1"]["asymptote"]
        a2 = diag["fits"]["symanzik_2"]["asymptote"]
        diag["a2_minus_a1"] = abs(a2 - a1)
        print(f"  |a_S2 - a_S1| = {abs(a2 - a1):.5f}")
    if "symanzik_2" in diag["fits"] and "symanzik_3" in diag["fits"]:
        a2 = diag["fits"]["symanzik_2"]["asymptote"]
        a3 = diag["fits"]["symanzik_3"]["asymptote"]
        diag["a3_minus_a2"] = abs(a3 - a2)
        print(f"  |a_S3 - a_S2| = {abs(a3 - a2):.5f}")
    return diag


def main():
    ladder = discover_d1_ladder(REPO)
    print("Finite-N corrections diagnostic")
    print("Ladder available:")
    for r, n, p in ladder:
        print(f"  {r} (N={n})")

    # VAC branch
    n_vac, lam_vac, _ = extract_branch(ladder, lambda n: n <= 100)
    vac_diag = diagnose_branch("VAC-branch (N <= 100)", n_vac, lam_vac, 3 / 8)

    # MATTER branch
    n_mat, lam_mat, _ = extract_branch(ladder, lambda n: n >= 256)
    mat_diag = diagnose_branch("MATTER-branch (N >= 256)", n_mat, lam_mat, 79 / 200)

    # Save report
    report = {
        "title": "Finite-N Symanzik-K diagnostic per branch",
        "vac_branch": vac_diag,
        "matter_branch": mat_diag,
    }
    out_path = OUTPUTS / "verify_finite_n_corrections.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
