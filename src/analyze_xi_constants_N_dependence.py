"""Analysis-only: test the deeper hypothesis that the FIVE
causal-wave-rational coefficients (alpha_xi, gamma, beta_pi,
epsilon_sync^2, D_Omega) themselves carry finite-N drift.

If the constants are static rationals (the framework's claim:
9/10, 1/10, 15/16, 1/20, 67/80), the lattice-derived combinations
should be N-independent within seed noise. If the constants are
finite-N dynamical, the derived combinations should drift.

Indirect tests:
  alpha_xi(N) = sqrt(Lambda_t(N))                under Lambda_t = alpha_xi^2
  alpha_xi(N) + gamma(N) = ?                     check C1
  Lambda_s(N) = -gamma^2/2                       backward extract gamma(N)
  alpha_xi^2 + gamma^2 = ?                       check Pythagoras
  beta_pi(N) - gamma(N) = D_Omega(N)             check C2

Per-N reference values:
  Lambda_t(N) from within-P5 bootstrap (Lambda_t = T_00^rec/T_00 ratio)
  Lambda_t_optimal(N) = T_00 - G_00 from per-regime audit (cross-regime)

DO NOT modify the manuscript; report only.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Static rational targets (the framework's claim)
ALPHA_XI_RAT = 9.0 / 10.0     # 0.9
GAMMA_RAT    = 1.0 / 10.0     # 0.1
BETA_PI_RAT  = 15.0 / 16.0    # 0.9375
EPS2_RAT     = 1.0 / 20.0     # 0.05
D_OMEGA_RAT  = 67.0 / 80.0    # 0.8375

# Lattice-measured (single-point readout from causal-wave bundle)
ALPHA_XI_MEAS = 0.900819
GAMMA_MEAS    = 0.100206
BETA_PI_MEAS  = 0.937913
EPS2_MEAS     = 0.050000
D_OMEGA_MEAS  = 0.839964


def per_n_alpha_xi(lambda_t_N):
    """Under L_t = alpha_xi^2: alpha_xi(N) = sqrt(L_t(N))."""
    return float(math.sqrt(max(lambda_t_N, 0.0)))


def main() -> int:
    print("=" * 78)
    print("Xi-Causal-Wave Constants: N-dependence test (User extension)")
    print("=" * 78)
    print()
    print("Static rationals (framework claim):")
    print(f"  alpha_xi   = 9/10   = {ALPHA_XI_RAT}")
    print(f"  gamma      = 1/10   = {GAMMA_RAT}")
    print(f"  beta_pi    = 15/16  = {BETA_PI_RAT}")
    print(f"  eps_sync^2 = 1/20   = {EPS2_RAT}")
    print(f"  D(Omega)   = 67/80  = {D_OMEGA_RAT}")
    print()
    print("Single-point lattice-measured (causal-wave bundle):")
    print(f"  alpha_xi   = {ALPHA_XI_MEAS:.6f}")
    print(f"  gamma      = {GAMMA_MEAS:.6f}")
    print()

    # --- 1. Within-P5: alpha_xi(N) backward-extracted from Lambda_t ---
    bs = json.loads(
        (REPO / "outputs" / "within_P5_bootstrap_audit.json").read_text())
    seeds = bs["per_regime_seeds"]
    Ns_lt, lt_vals = [], []
    for tag in sorted(seeds.keys(), key=lambda s: int(s.split("=")[1])):
        Ns_lt.append(int(tag.split("=")[1]))
        lt_vals.append(float(np.mean(seeds[tag])))

    print("=" * 78)
    print("alpha_xi(N) backward-extraction within-P5 (under L_t = alpha_xi^2)")
    print("=" * 78)
    print(f"{'N':>5} {'Lambda_t':>10} {'alpha_xi_eff':>14} "
          f"{'drift_vs_meas':>16} {'drift_vs_rat':>14}")
    alpha_xi_eff = []
    for N, lt in zip(Ns_lt, lt_vals):
        a = per_n_alpha_xi(lt)
        alpha_xi_eff.append(a)
        drift_meas = (a - ALPHA_XI_MEAS) / ALPHA_XI_MEAS
        drift_rat = (a - ALPHA_XI_RAT) / ALPHA_XI_RAT
        print(f"{N:>5} {lt:>10.5f} {a:>14.6f} "
              f"{drift_meas:>+15.3%} {drift_rat:>+13.3%}")

    range_alpha = max(alpha_xi_eff) - min(alpha_xi_eff)
    print(f"\n  Drift range over within-P5: {range_alpha:.4f}  "
          f"({range_alpha/ALPHA_XI_MEAS*100:.2f}% relative)")
    if range_alpha > 0.005:
        print(f"  -> alpha_xi_eff drifts SIGNIFICANTLY ({range_alpha:.4f}) "
              f"across within-P5 range; consistent with finite-N dynamics")
    else:
        print(f"  -> alpha_xi_eff is N-stable within seed noise")

    # --- 2. Cross-regime: alpha_xi(N) from optimal Lambda_t per regime ---
    pr = json.loads(
        (REPO / "outputs" / "per_regime_lambda_t_universal_audit.json"
         ).read_text())
    print()
    print("=" * 78)
    print("alpha_xi(N) cross-regime (from Lambda_t_optimal = median(T_00-G_00))")
    print("=" * 78)
    print(f"{'regime':>10} {'N':>4} {'L_t_opt':>10} {'alpha_xi_eff':>14} "
          f"{'kappa_t':>10}")
    rows_cr = []
    for r in pr["per_regime"]:
        if r["regime"].endswith("N128"):
            continue
        a = per_n_alpha_xi(r["Lambda_t_optimal"])
        rows_cr.append((r["N"], r["Lambda_t_optimal"], a,
                        r["Lambda_t_over_T_00_ratio"]))
        print(f"{r['regime']:>10} {r['N']:>4} {r['Lambda_t_optimal']:>10.4f} "
              f"{a:>14.6f} {r['Lambda_t_over_T_00_ratio']:>10.4f}")
    a_cr = [x[2] for x in rows_cr]
    drift_cr = max(a_cr) - min(a_cr)
    print(f"\n  Cross-regime alpha_xi_eff range: [{min(a_cr):.4f}, "
          f"{max(a_cr):.4f}], spread = {drift_cr:.4f}")
    print(f"  Mean: {float(np.mean(a_cr)):.6f}, std: {float(np.std(a_cr)):.6f}")

    # --- 3. Compare drifts ---
    print()
    print("=" * 78)
    print("Aggregated finite-N drift assessment")
    print("=" * 78)
    print()
    print(f"alpha_xi_eff(N) within-P5 (high-stat regime):")
    print(f"  values: {[round(a, 4) for a in alpha_xi_eff]}")
    print(f"  mean = {float(np.mean(alpha_xi_eff)):.6f}, "
          f"std = {float(np.std(alpha_xi_eff)):.6f}")
    print(f"  drift vs rat (9/10) at N=50:  "
          f"{(alpha_xi_eff[0] - ALPHA_XI_RAT)/ALPHA_XI_RAT*100:+.2f}%")
    print(f"  drift vs rat (9/10) at N=300: "
          f"{(alpha_xi_eff[-1] - ALPHA_XI_RAT)/ALPHA_XI_RAT*100:+.2f}%")
    print(f"  asymptotic trend (largest 2): "
          f"{(alpha_xi_eff[-1] - alpha_xi_eff[-2])/alpha_xi_eff[-2]*100:+.3f}%")
    print()
    print("Cross-regime alpha_xi_eff(N):")
    for n_, lt_, a_, k_ in rows_cr:
        print(f"  N={n_:>4}: alpha_xi_eff = {a_:.4f}")

    # --- 4. Check the C1 constraint at each within-P5 N ---
    print()
    print("=" * 78)
    print("Constraint C1 check at each N (under alpha_xi(N) + gamma = 1)")
    print("=" * 78)
    print(f"  Static-gamma assumption: gamma = 0.100206 (single-point readout)")
    print(f"  Then alpha_xi(N) + gamma should equal 1.000")
    print(f"\n{'N':>5} {'alpha_xi_eff':>14} {'+gamma':>10} {'C1_residual':>14}")
    for N, a in zip(Ns_lt, alpha_xi_eff):
        c1 = a + GAMMA_MEAS - 1.0
        print(f"{N:>5} {a:>14.5f} {a + GAMMA_MEAS:>10.5f} "
              f"{c1:>+14.5f}")

    # --- 5. Summary: does the data support the user's hypothesis? ---
    print()
    print("=" * 78)
    print("VERDICT on the User-Erweiterung (Xi-constants dynamic with N)")
    print("=" * 78)
    if range_alpha > 0.020:
        verdict = "STRONG"
        msg = ("alpha_xi_eff drifts >2% over within-P5 — substantial "
               "finite-N dynamics inconsistent with strict-rational claim")
    elif range_alpha > 0.010:
        verdict = "MODERATE"
        msg = ("alpha_xi_eff drifts 1-2% over within-P5 — non-trivial "
               "finite-N drift, possibly consistent with running-rational "
               "or seed-noise but warrants direct testing")
    else:
        verdict = "WEAK"
        msg = ("alpha_xi_eff drifts <1% — could be seed noise; "
               "rational-static claim defensible")
    print(f"  Verdict: {verdict} support for finite-N drift in alpha_xi_eff")
    print(f"  {msg}")
    print()
    print("Caveat: alpha_xi_eff = sqrt(Lambda_t) is itself a forward-")
    print("derived inversion of the structural identification. The true")
    print("test would be to re-run the causal-wave transport-operator")
    print("readout at multiple lattice sizes; that requires running")
    print("`causal_wave_transport_equation_probe.py` per-N, which the")
    print("present analysis does not cover.")

    out = REPO / "outputs" / "xi_constants_N_dependence_analysis.json"
    out.write_text(json.dumps({
        "method": "alpha_xi_eff_N_dependence_analysis",
        "static_rational": ALPHA_XI_RAT,
        "single_point_measured": ALPHA_XI_MEAS,
        "within_P5_alpha_xi_eff_per_N": dict(zip(map(str, Ns_lt),
                                                  alpha_xi_eff)),
        "within_P5_drift_range": float(range_alpha),
        "within_P5_drift_relative_pct":
            float(range_alpha / ALPHA_XI_MEAS * 100),
        "cross_regime_alpha_xi_eff": [
            {"regime": pr["per_regime"][i]["regime"],
             "N": rows_cr[i][0], "alpha_xi_eff": rows_cr[i][2]}
            for i in range(len(rows_cr))
        ],
        "verdict": verdict,
        "interpretation": msg,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
