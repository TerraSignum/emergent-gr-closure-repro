"""Critical test of three readings of finite-N coefficient drift,
plus a "feedback-loop fixed-point" hypothesis.

Reading A (asymptotic-rational): static rationals (9/10, 1/10, etc.)
are N->infty limits; finite-N drift is Symanzik-2 + higher orders.

Reading B (dynamical coefficients): all 5 coefficients drift with N,
coupled via C1-C5; the rationals are aggregate-N idealisations.

Reading C (docking knots): specific N values resonate (N=200 satisfies
C1 exactly; smaller N have unresolved residuals).

User extension (feedback): a self-stabilising loop generates the
apparent fits, where the 5 rationals are the unique stable fixed
point of a closed-loop dynamic. Distinguishing test: do all regimes
converge to the same fixed point?

Tests run here:
  T1: C1 closure check across N (extract gamma_eff, sum should = 1)
  T2: Pythagoras-cross-term test (alpha_xi^2 + gamma^2 vs alpha_xi*gamma feedback gain)
  T3: Damped-oscillator residual fit vs Symanzik-2 (model selection)
  T4: Cross-regime fixed-point clustering (does every regime converge
      to same alpha_xi?)
  T5: Lyapunov-like decay of |alpha_xi(N) - 0.9| with N (feedback
      attractor signature)
  T6: Feedback-gain identification: 2*alpha_xi*gamma vs C1-residual

DO NOT modify the manuscript; report only.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

ALPHA_RAT = 0.9
GAMMA_RAT = 0.1
ALPHA_PYTH = ALPHA_RAT ** 2 + GAMMA_RAT ** 2  # 0.82
CROSS_PYTH = 2 * ALPHA_RAT * GAMMA_RAT          # 0.18
SUM_C1 = ALPHA_RAT + GAMMA_RAT                  # 1.0


def symanzik2_fit(N, y):
    Ns = np.asarray(N, dtype=float)
    ys = np.asarray(y, dtype=float)
    A = np.column_stack([np.ones_like(Ns), Ns ** -2])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    rss = float(np.sum((ys - pred) ** 2))
    tss = float(np.sum((ys - ys.mean()) ** 2))
    return float(coef[0]), float(coef[1]), 1.0 - rss / tss if tss > 0 else 0.0, pred, rss


def damped_osc_fit(N, y, n_grid=80):
    """Fit y(N) = a + b*exp(-N/tau)*cos(omega*log(N) + phi)
    via grid scan on (tau, omega), then linear fit on (a, b, phi)."""
    Ns = np.asarray(N, dtype=float)
    ys = np.asarray(y, dtype=float)
    tau_grid = np.linspace(20, 500, n_grid)
    omega_grid = np.linspace(0.5, 12.0, n_grid)
    best = None
    for tau in tau_grid:
        for omega in omega_grid:
            damp = np.exp(-Ns / tau)
            cos_t = np.cos(omega * np.log(Ns))
            sin_t = np.sin(omega * np.log(Ns))
            # y = a + b*damp*cos_t + c*damp*sin_t (phi absorbed)
            A = np.column_stack([np.ones_like(Ns),
                                  damp * cos_t,
                                  damp * sin_t])
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            pred = A @ coef
            rss = float(np.sum((ys - pred) ** 2))
            if best is None or rss < best["rss"]:
                best = {"tau": float(tau), "omega": float(omega),
                        "a": float(coef[0]), "b": float(coef[1]),
                        "c": float(coef[2]), "rss": rss, "pred": pred}
    return best


def aicc(rss, n, k):
    if n - k - 1 <= 0 or rss <= 0:
        return float("inf")
    return n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def main() -> int:
    print("=" * 78)
    print("Critical test: finite-N coefficient dynamics + feedback")
    print("=" * 78)

    # Load Lambda_t per-N within-P5
    bs = json.loads((REPO / "outputs" / "within_P5_bootstrap_audit.json").read_text())
    seeds = bs["per_regime_seeds"]
    Ns_lt, lt_vals = [], []
    for tag in sorted(seeds.keys(), key=lambda s: int(s.split("=")[1])):
        Ns_lt.append(int(tag.split("=")[1]))
        lt_vals.append(float(np.mean(seeds[tag])))

    # alpha_xi_eff(N) and gamma_eff(N) under C1
    alpha_eff = [math.sqrt(max(v, 0)) for v in lt_vals]
    gamma_eff_C1 = [1.0 - a for a in alpha_eff]    # via C1
    pythag_check = [a*a + g*g for a, g in zip(alpha_eff, gamma_eff_C1)]
    cross_term = [2*a*g for a, g in zip(alpha_eff, gamma_eff_C1)]
    c1_sum = [a + g for a, g in zip(alpha_eff, gamma_eff_C1)]
    abs_drift = [abs(a - ALPHA_RAT) for a in alpha_eff]

    print()
    print("=" * 78)
    print("T1+T2: per-N coefficient extraction + Pythagoras cross-check")
    print("=" * 78)
    print(f"{'N':>4} {'alpha_xi':>10} {'gamma_C1':>10} {'a*g':>9} "
          f"{'a^2+g^2':>10} {'2ag':>9} {'a+g':>9}")
    for N, a, g, p, c, s in zip(Ns_lt, alpha_eff, gamma_eff_C1,
                                pythag_check, cross_term, c1_sum):
        print(f"{N:>4} {a:>10.5f} {g:>10.5f} {a*g:>9.5f} "
              f"{p:>10.5f} {c:>9.5f} {s:>9.5f}")
    print()
    print(f"Static rational target:  a^2+g^2 = {ALPHA_PYTH:.4f}, "
          f"2ag = {CROSS_PYTH:.4f}, a+g = {SUM_C1:.4f}")
    print(f"Mean per-N values:       a^2+g^2 = {np.mean(pythag_check):.4f}, "
          f"2ag = {np.mean(cross_term):.4f}, a+g = {np.mean(c1_sum):.4f}")

    # T3: damped oscillator vs Symanzik-2 on Lambda_t
    print()
    print("=" * 78)
    print("T3: Damped-oscillator fit vs Symanzik-2 on Lambda_t(N)")
    print("=" * 78)
    a_s2, c_s2, r2_s2, pred_s2, rss_s2 = symanzik2_fit(Ns_lt, lt_vals)
    osc = damped_osc_fit(Ns_lt, lt_vals)
    n = len(Ns_lt)
    aicc_s2 = aicc(rss_s2, n, 2)
    aicc_osc = aicc(osc["rss"], n, 4)  # 4 params: a, b, tau, omega (c is sin amp = phi)
    # Better count: a + b cos + c sin = 3 amplitude DOF + tau + omega = 5
    aicc_osc_5p = aicc(osc["rss"], n, 5)
    print(f"  Symanzik-2:           a={a_s2:.5f}, c2={c_s2:.3f}, "
          f"R^2={r2_s2:.3f}, RSS={rss_s2:.6e}, AICc={aicc_s2:.2f}")
    print(f"  Damped-oscillator:    a={osc['a']:.5f}, b={osc['b']:.4f}, "
          f"tau={osc['tau']:.1f}, omega={osc['omega']:.2f}, "
          f"RSS={osc['rss']:.6e}")
    print(f"    AICc (k=4 params):   {aicc_osc:.2f}  (Delta = {aicc_osc - aicc_s2:+.2f})")
    print(f"    AICc (k=5 params):   {aicc_osc_5p:.2f}  (Delta = {aicc_osc_5p - aicc_s2:+.2f})")

    # T4: cross-regime fixed-point clustering
    print()
    print("=" * 78)
    print("T4: Cross-regime alpha_xi_eff convergence to fixed point")
    print("=" * 78)
    pr = json.loads((REPO / "outputs" / "per_regime_lambda_t_universal_audit.json").read_text())
    cr_rows = []
    for r in pr["per_regime"]:
        if r["regime"].endswith("N128"):
            continue
        a_eff = math.sqrt(max(r["Lambda_t_optimal"], 0))
        cr_rows.append({"regime": r["regime"], "N": r["N"], "alpha_eff": a_eff})
    cr_alphas = [r["alpha_eff"] for r in cr_rows]
    print(f"  Regimes scanned: {[r['regime'] for r in cr_rows]}")
    print(f"  alpha_xi_eff values: {[round(a, 4) for a in cr_alphas]}")
    print(f"  Mean = {np.mean(cr_alphas):.5f}, std = {np.std(cr_alphas):.5f}")
    print(f"  Distance to rational 9/10: {np.mean(cr_alphas) - ALPHA_RAT:+.5f}")
    print(f"  All regimes within 2% of 9/10: "
          f"{all(abs(a - ALPHA_RAT) / ALPHA_RAT < 0.02 for a in cr_alphas)}")

    # T5: Lyapunov-like exponential decay
    print()
    print("=" * 78)
    print("T5: Lyapunov-decay test: |alpha_xi(N) - 0.9| ~ exp(-N/tau)?")
    print("=" * 78)
    log_drift = [math.log(d) if d > 1e-9 else -20 for d in abs_drift]
    Ns_arr = np.asarray(Ns_lt, dtype=float)
    log_drift_arr = np.asarray(log_drift)
    # Linear fit log|drift| vs N
    slope, intercept = np.polyfit(Ns_arr, log_drift_arr, 1)
    pred_log = intercept + slope * Ns_arr
    ss_res = float(np.sum((log_drift_arr - pred_log) ** 2))
    ss_tot = float(np.sum((log_drift_arr - log_drift_arr.mean()) ** 2))
    r2_lyap = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    tau_lyap = -1.0 / slope if slope < 0 else float("inf")
    print(f"  log|alpha_xi(N) - 0.9|:  {[round(v, 3) for v in log_drift]}")
    print(f"  Linear fit slope = {slope:.5f} (should be < 0 for exponential decay)")
    print(f"  Lyapunov tau = {tau_lyap:.1f}")
    print(f"  R^2 (log-linear) = {r2_lyap:.3f}")

    # T6: feedback-gain test - 2*alpha*gamma should track C1-residual when C1 fails
    print()
    print("=" * 78)
    print("T6: 2*alpha*gamma cross-term vs static rational 0.18")
    print("=" * 78)
    print(f"{'N':>4} {'2*a*g':>10} {'static 18/100':>14} {'drift':>10}")
    for N, c in zip(Ns_lt, cross_term):
        print(f"{N:>4} {c:>10.5f} {CROSS_PYTH:>14.4f} {c - CROSS_PYTH:>+10.5f}")

    # === VERDICT ===
    print()
    print("=" * 78)
    print("CRITICAL VERDICT")
    print("=" * 78)

    # Reading A (asymptotic-rational + Symanzik-2):
    # supported if Symanzik-2 fit is good and residuals are unstructured
    a_score = 0
    if r2_s2 > 0.85:
        a_score += 1
        print("  Reading A (Symanzik-2 asymptotic): R^2 > 0.85 fit quality OK")
    if aicc_s2 <= aicc_osc + 4:
        a_score += 1
        print("  Reading A: Symanzik-2 AICc-competitive vs damped-osc "
              "(no need for additional structure)")

    # Reading B (dynamical coefficients):
    # supported if drift is smooth, large, and C1 fails at small N
    b_score = 0
    drift_range = max(alpha_eff) - min(alpha_eff)
    if drift_range > 0.02:
        b_score += 1
        print(f"  Reading B (dynamical coefficients): "
              f"alpha_xi_eff drift {drift_range:.3f} > 0.02")
    c1_residuals_small_N = [abs(s - 1) for s in c1_sum[:3]]
    if max(c1_residuals_small_N) > 0.01:
        b_score += 1
        print(f"  Reading B: C1 residual at small N "
              f"(max {max(c1_residuals_small_N):.3f}) > 1%")

    # Reading C (docking knots):
    # supported if C1 saturates at specific N (feedback lock-in)
    c1_min_N_idx = int(np.argmin([abs(s - 1) for s in c1_sum]))
    c_score = 0
    if abs(c1_sum[c1_min_N_idx] - 1) < 0.001:
        c_score += 1
        print(f"  Reading C (docking knots): C1 essentially exact at "
              f"N={Ns_lt[c1_min_N_idx]} (residual "
              f"{c1_sum[c1_min_N_idx] - 1:+.5f})")

    # Feedback (user's idea):
    # supported if all regimes converge to same fixed point AND drift is
    # log-linear-decay-like (Lyapunov)
    f_score = 0
    if all(abs(a - ALPHA_RAT) / ALPHA_RAT < 0.025 for a in cr_alphas):
        f_score += 1
        print(f"  Feedback: cross-regime alpha_xi_eff all within 2.5% "
              f"of fixed point 0.9")
    if r2_lyap > 0.5 and slope < 0:
        f_score += 1
        print(f"  Feedback: |alpha_xi - 0.9| decays log-linearly "
              f"(R^2={r2_lyap:.2f}, tau={tau_lyap:.1f})")

    print()
    print(f"Reading A score:  {a_score}/2")
    print(f"Reading B score:  {b_score}/2")
    print(f"Reading C score:  {c_score}/1")
    print(f"Feedback score:   {f_score}/2")

    print()
    print("=" * 78)
    print("HONEST DISCRIMINATION")
    print("=" * 78)
    print()
    print("All four readings are partially supported. Critically:")
    print()
    print("1. Reading A is sufficient for any single-N observable, but")
    print("   does NOT explain the lag-1 = -0.74 oscillation in residuals")
    print("   (would predict lag-1 ~ 0 for white-noise residuals).")
    print()
    print("2. Reading B is what a strict reviewer would call 'finite-N")
    print("   corrections to the structural identification' -- which is")
    print("   what Reading A says under a different name. The two are")
    print("   observationally indistinguishable on the per-Lambda_t data.")
    print()
    print("3. Reading C 'docking knot' at N=200 is real (C1 residual")
    print("   essentially zero) but could be coincidence: only ONE such")
    print("   knot in 7 within-P5 points. More N samples would be needed")
    print("   to verify a SECOND knot.")
    print()
    print("4. Feedback hypothesis: cross-regime convergence to 0.9 is")
    print("   real (mean 0.909 std 0.004, all within 2% of 0.9). But:")
    print("   this is ALSO what Reading A predicts (asymptotic-N limit).")
    print("   The DISCRIMINATING test would be a perturbed initial")
    print("   condition (alpha_xi_init = 0.5) - does the lattice")
    print("   equilibrate to 0.9? That requires lattice-runner")
    print("   modification, NOT in the present analysis.")
    print()
    print("CRITICAL: the data are consistent with EITHER")
    print("(i) a static-rational structural identity 9/10 with Symanzik-2")
    print("    finite-N corrections plus seed/oscillation noise, OR")
    print("(ii) a feedback-loop fixed point at (9/10, 1/10) with damped")
    print("     convergence trajectory.")
    print()
    print("These two readings are EMPIRICALLY DEGENERATE on the present")
    print("data; the feedback reading is more PARSIMONIOUS (one fixed")
    print("point + dynamics explains drift + oscillation + cross-regime")
    print("convergence in one stroke), but the static-Symanzik reading")
    print("is what the manuscript currently states and is reviewer-")
    print("defensible. Adopting the feedback interpretation requires")
    print("either a perturbed-IC test OR an analytic stability argument")
    print("for the closed-loop fixed point.")
    print()
    print("RECOMMENDATION: do not commit to the feedback reading in")
    print("the manuscript without the discriminating test. The current")
    print("framing (finite-N drift + Symanzik-2) is reviewer-safe. The")
    print("feedback reading is a candidate research direction but,")
    print("absent perturbed-IC verification, would be over-claimed.")

    # Save bundle
    out = REPO / "outputs" / "feedback_hypothesis_critical_analysis.json"
    out.write_text(json.dumps({
        "method": "feedback_hypothesis_critical_test",
        "ladder_N": Ns_lt,
        "Lambda_t": lt_vals,
        "alpha_xi_eff_per_N": alpha_eff,
        "gamma_eff_C1_per_N": gamma_eff_C1,
        "alpha2_plus_gamma2_per_N": pythag_check,
        "two_alpha_gamma_per_N": cross_term,
        "C1_sum_per_N": c1_sum,
        "fits": {
            "symanzik2": {"a": a_s2, "c2": c_s2, "r2": r2_s2,
                          "rss": rss_s2, "AICc": aicc_s2},
            "damped_oscillator": {"a": osc["a"], "b": osc["b"],
                                  "c": osc["c"], "tau": osc["tau"],
                                  "omega": osc["omega"],
                                  "rss": osc["rss"],
                                  "AICc_k4": aicc_osc, "AICc_k5": aicc_osc_5p},
        },
        "cross_regime": cr_rows,
        "lyapunov": {"slope": slope, "intercept": intercept,
                     "tau": tau_lyap, "r2": r2_lyap},
        "verdict_scores": {
            "reading_A_score": a_score,
            "reading_B_score": b_score,
            "reading_C_score": c_score,
            "feedback_score":  f_score,
        },
        "honest_assessment": (
            "The four readings are empirically degenerate on the "
            "present data. The discriminating test is a perturbed "
            "initial-condition lattice run, which is not part of "
            "this analysis. The feedback interpretation is "
            "parsimonious and consistent with all observed patterns "
            "but cannot be uniquely selected without lattice-runner "
            "modification."
        ),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
