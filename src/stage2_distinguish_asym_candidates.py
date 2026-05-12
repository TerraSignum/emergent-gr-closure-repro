"""Stage 2: distinguish -pi/200 vs -1/64 asym candidates.

Both are System-R-derivable (per Stage 1):
  -pi/200 = -pi*gamma^2/2  (Berry-phase-like)
  -1/64 = (beta_pi - 1)/4  (beta_pi completeness anomaly)

Numerical separation: |(-pi/200) - (-1/64)| = 8.3e-5
Empirical asymptote uncertainty: 5e-3
Ratio: separation / uncertainty = 0.017 -> need 60x more precision

This script:
  1. Tests if a LINEAR COMBINATION fits better than either individually
  2. Tests if a higher-N projection (using Symanzik) can resolve them
  3. Predicts what asymptote uncertainty we need: ~10^-5 to distinguish

Output: outputs/stage2_distinguish_asym_candidates.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main():
    print("=" * 80)
    print("Stage 2: distinguishing -pi/200 vs -1/64 asym candidates")
    print("=" * 80)
    print()
    # Both candidates
    cand_A = -np.pi / 200  # Berry-phase
    cand_B = -1 / 64       # beta_pi anomaly
    sep = abs(cand_A - cand_B)
    print(f"Candidate A: -pi*gamma^2/2 = -pi/200      = {cand_A:+.10f}")
    print(f"Candidate B: (beta_pi-1)/4 = -1/64       = {cand_B:+.10f}")
    print(f"Separation |A - B|                       = {sep:.2e}")
    print()
    # Load existing per-seed asym data
    bundle = json.loads((REPO / "outputs"
                          / "audit_asym_per_seed_n_trend.json").read_text())
    rows = bundle["rows"]
    N = np.array([r["N"] for r in rows], dtype=float)
    y = np.array([r["asym_mean"] for r in rows])
    u = np.array([r["asym_uncertainty_of_mean"] for r in rows])
    n_seeds = np.array([r["n_seeds"] for r in rows])
    print(f"Per-regime data ({len(rows)} regimes):")
    for ni, yi, ui, ns in zip(N, y, u, n_seeds):
        print(f"  N={ni:>4.0f}  asym = {yi:+.5f} +/- {ui:.5f}  ({ns} seeds)")
    print()
    # Symanzik fit a + b/N
    def symanzik_fit(N_arr, y_arr, u_arr):
        x = 1.0 / N_arr
        w = 1.0 / np.maximum(u_arr, 1e-6) ** 2
        A_mat = np.column_stack([np.ones_like(x), x])
        AtWA = A_mat.T @ (w[:, None] * A_mat)
        AtWy = A_mat.T @ (w * y_arr)
        coef = np.linalg.solve(AtWA, AtWy)
        cov = np.linalg.inv(AtWA)
        return coef[0], np.sqrt(cov[0, 0]), coef[1], np.sqrt(cov[1, 1])

    a_inf, a_unc, b, b_unc = symanzik_fit(N, y, u)
    print(f"Current Symanzik fit:")
    print(f"  a_inf = {a_inf:+.5f} +/- {a_unc:.5f}")
    print(f"  b     = {b:+.4f} +/- {b_unc:.4f}")
    sigma_A = abs(a_inf - cand_A) / a_unc
    sigma_B = abs(a_inf - cand_B) / a_unc
    print(f"  Distance to A (-pi/200): {sigma_A:.2f} sigma")
    print(f"  Distance to B (-1/64):   {sigma_B:.2f} sigma")
    print(f"  Resolving power (A vs B): {sep / a_unc:.2f} sigma "
          f"(need >2 to claim distinct)")
    print()
    # Test: weighted combination model a_inf = w*A + (1-w)*B
    # Solve for w:
    if abs(cand_A - cand_B) > 1e-9:
        w_fit = (a_inf - cand_B) / (cand_A - cand_B)
        print(f"Weighted-combination model "
              f"a_inf = w*(-pi/200) + (1-w)*(-1/64):")
        print(f"  w = {w_fit:.3f}  (1 = pure A, 0 = pure B, 0.5 = average)")
    print()
    # Forecast: how many regimes / seeds needed for distinction?
    # If we add P5N512 and P5N1024 with say 12 seeds each, what's the
    # new asymptote uncertainty?
    print("Forecast: adding P5N512 and P5N1024 (12 seeds each)")
    # Per-seed std at high N: extrapolate from current
    # Use measured per-seed std at high-N regimes
    per_seed_stds = []
    for r in rows:
        if r["asym_per_seed"]:
            per_seed_stds.append(np.std(r["asym_per_seed"]))
    typical_std = float(np.mean(per_seed_stds)) if per_seed_stds else 0.03
    print(f"  Typical per-seed std (current): {typical_std:.4f}")
    # Forecast unc at N=512 with 12 seeds: typical_std / sqrt(12)
    unc_512 = typical_std / np.sqrt(12)
    unc_1024 = typical_std / np.sqrt(12)
    print(f"  Forecast unc at N=512 (12 seeds): {unc_512:.5f}")
    print(f"  Forecast unc at N=1024 (12 seeds): {unc_1024:.5f}")
    # Simulated extended fit
    N_ext = np.concatenate([N, [512, 1024]])
    # Assume future asym values land near linear extrapolation
    a_inf_pred = a_inf
    b_pred = b
    y_512_pred = a_inf_pred + b_pred / 512
    y_1024_pred = a_inf_pred + b_pred / 1024
    y_ext = np.concatenate([y, [y_512_pred, y_1024_pred]])
    u_ext = np.concatenate([u, [unc_512, unc_1024]])
    a_inf_ext, a_unc_ext, _, _ = symanzik_fit(N_ext, y_ext, u_ext)
    print(f"  Simulated extended Symanzik:")
    print(f"    a_inf = {a_inf_ext:+.5f} +/- {a_unc_ext:.5f}")
    print(f"    Resolving power (A vs B): {sep / a_unc_ext:.2f} sigma")
    if sep / a_unc_ext >= 2:
        print(f"    -> WOULD distinguish A vs B at >=2sigma")
    else:
        print(f"    -> still NOT enough to distinguish")
    print()
    # What if we have many more seeds?
    print("Sensitivity to seed count at higher N:")
    for n_added_seeds in [12, 24, 48, 96]:
        unc_high = typical_std / np.sqrt(n_added_seeds)
        u_ext = np.concatenate([u, [unc_high, unc_high]])
        a_inf_ext, a_unc_ext, _, _ = symanzik_fit(N_ext, y_ext, u_ext)
        print(f"  {n_added_seeds:>3d} seeds at N=512+1024: "
              f"a_unc = {a_unc_ext:.5f}, "
              f"resolving = {sep / a_unc_ext:.2f}σ")
    bundle = {
        "method": "stage2_distinguish_asym_candidates",
        "candidate_A_pi_gamma2_over_2": cand_A,
        "candidate_B_beta_pi_minus_1_over_4": cand_B,
        "separation_abs": float(sep),
        "current_a_inf_mean": float(a_inf),
        "current_a_inf_unc": float(a_unc),
        "sigma_to_A": float(sigma_A),
        "sigma_to_B": float(sigma_B),
        "current_resolving_power_sigma":
            float(sep / a_unc) if a_unc > 0 else 0,
        "weighted_combination_w_fit":
            float((a_inf - cand_B) / (cand_A - cand_B))
            if abs(cand_A - cand_B) > 1e-9 else None,
    }
    out = REPO / "outputs" / "stage2_distinguish_asym_candidates.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
