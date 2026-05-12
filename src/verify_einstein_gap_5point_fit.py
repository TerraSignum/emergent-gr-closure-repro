r"""
Verify the five-point asymptotic scaling fit on the chirality-balance
deviation observable as an independent witness of the Einstein-
identity-gap exponent alpha_target = 2/3.

The bundled file data/einstein_gap_5point_fit.json carries five
lattice sizes N in {28, 30, 36, 42, 50} with chirality-balance
deviations 1 - <chirality_balance>_N. This script reproduces:

  1. the log-log linear fit of deviation vs N, recovering
     alpha_fit ~ 0.636 (within 4.7% of the theory target 2/3);
  2. the prefactor C_fit;
  3. the R^2 of the log-log fit;
  4. the asymptotic extrapolation to N = 10^6;
  5. the ratio alpha_fit / alpha_target.

This provides an independent five-point witness of the analytical
Delta_E(N) <= C_0 * N^{-2/3} bound, complementing the two-point
Richardson construction at the canonical and extended anchor
regimes (data/einstein_gap_results.json).

Usage:
    python ./src/verify_einstein_gap_5point_fit.py
"""

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_5point_fit():
    with open(DATA / "einstein_gap_5point_fit.json", "r",
              encoding="utf-8") as f:
        return json.load(f)


def loglog_linear_fit(xs, ys):
    """Returns (slope, intercept, r_squared) for a linear fit on
    log(xs) vs log(ys). Slope = -alpha; intercept = log(C)."""
    if any(x <= 0 for x in xs) or any(y <= 0 for y in ys):
        raise ValueError("log-log fit requires strictly positive xs and ys")
    n = len(xs)
    log_xs = [math.log(x) for x in xs]
    log_ys = [math.log(y) for y in ys]
    sx = sum(log_xs)
    sy = sum(log_ys)
    sxx = sum(lx * lx for lx in log_xs)
    sxy = sum(lx * ly for (lx, ly) in zip(log_xs, log_ys))
    syy = sum(ly * ly for ly in log_ys)
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    # R^2:
    y_mean = sy / n
    ss_tot = sum((ly - y_mean) ** 2 for ly in log_ys)
    ss_res = sum(
        (ly - (slope * lx + intercept)) ** 2
        for (lx, ly) in zip(log_xs, log_ys)
    )
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


def main():
    d = load_5point_fit()
    print("=" * 72)
    print("Five-point chirality-deviation log-log fit (Einstein-gap exponent)")
    print("=" * 72)
    print()

    Ns = d["scan"]["N_values"]
    devs = d["scan"]["deviation_means"]
    print(f"  Theory bound: {d['theory']['bound']}")
    print(f"  Observable:   {d['theory']['observable']}")
    print()
    print(f"  {'N':>6} {'deviation':>14}")
    print("  " + "-" * 25)
    for n, dev in zip(Ns, devs):
        print(f"  {n:>6} {dev:>14.6f}")
    print()

    slope, intercept, r2 = loglog_linear_fit(Ns, devs)
    alpha_fit = -slope
    C_fit = math.exp(intercept)
    alpha_target = 2.0 / 3.0
    ratio = alpha_fit / alpha_target

    print("--- Log-log fit ---")
    print(f"  alpha_fit (= -slope) = {alpha_fit:.6f}")
    print(f"  C_fit (= exp(int))   = {C_fit:.6f}")
    print(f"  R^2                  = {r2:.6f}")
    print(f"  alpha_target = 2/3   = {alpha_target:.6f}")
    print(f"  alpha_fit / alpha_target = {ratio:.6f} "
          f"(deviation {abs(1 - ratio)*100:.2f}%)")
    print()

    # Verify against bundled values.
    bundled_alpha = d["fit_result"]["alpha_fit"]
    bundled_C = d["fit_result"]["C_fit"]
    bundled_r2 = d["fit_result"]["r2_loglog"]
    print("--- Cross-check vs bundled fit_result ---")
    print(f"  alpha_fit (recompute) = {alpha_fit:.6f}, "
          f"bundled = {bundled_alpha:.6f}, "
          f"diff = {abs(alpha_fit - bundled_alpha):.6e}")
    print(f"  C_fit (recompute)     = {C_fit:.6f}, "
          f"bundled = {bundled_C:.6f}, "
          f"diff = {abs(C_fit - bundled_C):.6e}")
    print(f"  R^2 (recompute)       = {r2:.6f}, "
          f"bundled = {bundled_r2:.6f}, "
          f"diff = {abs(r2 - bundled_r2):.6e}")
    print()

    # Asymptotic extrapolation.
    N_inf = 1.0e6
    dev_at_N_inf = C_fit * (N_inf ** (-alpha_fit))
    print(f"  Asymptotic extrapolation: deviation at N = 10^6 "
          f"= {dev_at_N_inf:.4e}")
    print()

    out = {
        "criterion": "Five-point chirality-deviation log-log Einstein-gap-exponent fit",
        "alpha_fit": alpha_fit,
        "C_fit": C_fit,
        "r2_loglog": r2,
        "alpha_target": alpha_target,
        "ratio_alpha_fit_to_theory": ratio,
        "tier_alpha": d["fit_result"]["tier_alpha"],
        "deviation_at_N_1e6": dev_at_N_inf,
        "asymptotic_verdict": d["asymptotic_extrapolation"]["asymptotic_verdict"],
        "consistent_with_bundled":
            abs(alpha_fit - bundled_alpha) < 1e-6
            and abs(C_fit - bundled_C) < 1e-6
            and abs(r2 - bundled_r2) < 1e-6,
    }
    out_path = OUTPUTS / "einstein_gap_5point_recompute.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
