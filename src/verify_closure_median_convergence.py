"""Quantify the median-closure convergence rate of the
per-eigendirection residual R_TF (traceless spatial residual)
under N -> infty.

Honest closure statement for the manuscript:

  G_(ii)(a) + Lambda_i - 8 pi G lambda_i(a) -> 0    (in the median, N -> infty)

The MEAN over heavy-tail nodes does NOT converge in the same way;
~10% of nodes dominate the mean-square budget. Per eigendirection
audit (outputs/per_eigendirection_residual_audit.json) shows that
89% of the mean-sq residual is in the TRACELESS spatial part,
which is NOT absorbable by an isotropic Lambda. The median is
the closure observable; the heavy-tail traceless mean is a
separate structural feature (geometric-condensation cluster, see
outputs/geometric_condensation_b2a1a3_audit.json).

This script reads the per_eigendirection_residual_audit.json and
fits |R_TF|_median(N) ~ A * N^(-alpha). A power-law exponent
alpha >= 1.0 with R^2 >= 0.7 demonstrates median closure.

Output: outputs/closure_median_convergence.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def power_law_fit(N: np.ndarray, y: np.ndarray) -> dict:
    """Fit y = A * N^(-alpha)  by log-log linear regression."""
    log_N = np.log(N)
    log_y = np.log(y)
    slope, intercept = np.polyfit(log_N, log_y, 1)
    alpha = -slope
    A = float(np.exp(intercept))
    y_pred = A * N ** (-alpha)
    ss_res = float(np.sum((log_y - np.polyval([slope, intercept], log_N)) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "alpha": float(alpha), "A": A, "r_squared": float(r2),
        "y_pred": [float(v) for v in y_pred],
    }


def main() -> int:
    audit_path = REPO / "outputs" / "per_eigendirection_residual_audit.json"
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    regimes = audit["per_regime"]
    N = np.array([r["N"] for r in regimes], dtype=float)

    series = {
        "R_TF_norm_median_abs":    np.array([r["R_TF_norm_median_abs"]    for r in regimes]),
        "R_trace_median_abs":      np.array([r["R_trace_median_abs"]      for r in regimes]),
        "R_time_median_abs":       np.array([r["R_time_median_abs"]       for r in regimes]),
        "R_off_median_abs":        np.array([r["R_off_median_abs"]        for r in regimes]),
        "R_TF_norm_mean_abs":      np.array([r["R_TF_norm_mean_abs"]      for r in regimes]),
        "R_trace_mean_abs":        np.array([r["R_trace_mean_abs"]        for r in regimes]),
        "R_time_mean_abs":         np.array([r["R_time_mean_abs"]         for r in regimes]),
    }

    print("=" * 90)
    print("Median closure convergence audit")
    print("Equation: G_(ii)(a) + Lambda_i - 8 pi G lambda_i(a) -> 0  (median, N -> infty)")
    print("=" * 90)
    print()
    print("Data points:")
    print(f"{'N':>5} | " + " | ".join([f"{k[:18]:>18}" for k in series]))
    print("-" * 140)
    for i in range(len(N)):
        row = f"{int(N[i]):>5} | " + " | ".join([f"{series[k][i]:>18.6f}" for k in series])
        print(row)

    fits = {}
    for k, v in series.items():
        fits[k] = power_law_fit(N, v)

    print()
    print("Power-law fits  y(N) = A * N^(-alpha):")
    print(f"{'observable':<28} {'alpha':>10} {'A':>14} {'R^2':>8}")
    print("-" * 70)
    for k, f in fits.items():
        print(f"{k:<28} {f['alpha']:>10.3f} {f['A']:>14.6f} {f['r_squared']:>8.3f}")

    print()
    print("Closure verdict (alpha >= 1.0 AND R^2 >= 0.7  ->  median closes):")
    for k, f in fits.items():
        verdict = "CLOSES" if (f["alpha"] >= 1.0 and f["r_squared"] >= 0.7) \
            else ("MARGINAL" if f["r_squared"] >= 0.5 else "DOES_NOT_CLOSE")
        print(f"  {k:<28}  alpha={f['alpha']:>6.2f}  R^2={f['r_squared']:>5.2f}  -> {verdict}")

    out_path = REPO / "outputs" / "closure_median_convergence.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "power_law_fit_per_eigendirection_residual",
            "schema_version": "1.0.0",
            "N_values": [int(n) for n in N],
            "fits": fits,
            "interpretation": (
                "The median closure observable |R_TF|_med(N) and "
                "|R_trace|_med(N) and |R_time|_med(N) are tested for "
                "power-law decay. alpha >= 1.0 with R^2 >= 0.7 signals "
                "convergent median closure. Mean values are reported "
                "but NOT used as primary closure observable, since the "
                "heavy-tail (top ~10% of nodes by T_00 magnitude) "
                "dominates the mean and is identified as a separate "
                "structural feature (geometric-condensation), not a "
                "closure failure."
            ),
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
