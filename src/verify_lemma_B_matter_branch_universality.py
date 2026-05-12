"""Lemma B — matter-branch universality closure.

Tests the hypothesis that on the matter side of the
chirality flip (N >= 256, theta_chir > pi/4, post-N* ~ 110-120),
the weighted-Laplacian asymptote takes the System-R-corrected
value

    lambda_w_inf^mat = (d-1)/(2d) + 2 gamma^2 = 79/200 = 0.395

equivalently:

    shift from vacuum branch = 2 gamma^2 = 1/50 = d gamma^2/2 (d=4)

The vacuum-branch closure
    lambda_w_inf^vac = (d-1)/(2d) = 3/8 = 0.375
is empirically certified at 0.48% on the 5-point pre-flip ladder
N <= 100 (verify_lemma_B_uniform_poincare.py + branch_resolved_fit).

This audit:
  1. Reads per-regime lambda_2 from
     outputs/verify_lemma_B_uniform_poincare.json.
  2. Splits the 10-regime ladder N in [50, 512] into vacuum
     (N <= 100), peri-flip (N in [128, 200]), and matter
     (N >= 256).
  3. Runs Symanzik-1 fits on the matter branch and bootstraps
     the asymptote.
  4. Compares 4 candidate rationals (3/8, 2/5, 79/200, 41/100)
     via force-asymptote diagnostic (implied 1/N coefficient
     should be constant across regimes for the correct
     asymptote).
  5. Reports the verdict.

Verdict (as of 2026-05-13 with 3 stable matter points):
  matter asymptote 0.39571 (LSQ) or 0.39596 (Richardson, N=300/512)
  79/200 = 0.395:    0.18% residual, implied a spread 0.20
  2/5  = 0.400:      1.08% residual, implied a spread 1.10
  3/8  = 0.375:      5.23% residual, implied a spread 5.30 (rejected)

The 79/200 hypothesis is the cleanest fit on the current data.

Output: outputs/verify_lemma_B_matter_branch_universality.json
"""
from __future__ import annotations
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "outputs" / "verify_lemma_B_uniform_poincare.json"
OUT = REPO / "outputs" / "verify_lemma_B_matter_branch_universality.json"

# Branch boundaries (memory: chirality_flip_pi_over_4)
VACUUM_N_MAX = 100
PERI_FLIP_N_LOW = 128
PERI_FLIP_N_HIGH = 200
MATTER_N_MIN = 256

# Candidate rationals to test for the matter-branch asymptote
GAMMA = Fraction(1, 10)
D = 4
CANDIDATES = {
    "3/8 = (d-1)/(2d) [vacuum]":         Fraction(3, 8),
    "2/5 = 2/(d+1)":                     Fraction(2, 5),
    "79/200 = 3/8 + 2*gamma^2":          Fraction(79, 200),
    "41/100 = (d+gamma)/(2d)":           Fraction(41, 100),
}


def symanzik1(N, y):
    """Least-squares fit y = lambda_inf + a/N. Returns (lambda_inf, a, sse)."""
    A = np.column_stack([np.ones_like(N), 1.0/N])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    sse = float(np.sum((y - pred)**2))
    return float(coef[0]), float(coef[1]), sse


def main() -> int:
    if not IN.exists():
        print(f"ERROR: input not found: {IN}")
        return 1
    data = json.loads(IN.read_text(encoding="utf-8"))
    per_regime = [r for r in data["per_regime"] if r.get("status") == "OK"]

    pts = sorted(
        [(int(r["N"]), float(r["lambda_2_mean"])) for r in per_regime],
        key=lambda x: x[0],
    )

    vac = [(n, l) for n, l in pts if n <= VACUUM_N_MAX]
    peri = [(n, l) for n, l in pts
            if PERI_FLIP_N_LOW <= n <= PERI_FLIP_N_HIGH]
    mat = [(n, l) for n, l in pts if n >= MATTER_N_MIN]

    # Matter-branch fit (3 stable points: typically N=256, 300, 512)
    N_mat = np.array([n for n, _ in mat], dtype=float)
    y_mat = np.array([l for _, l in mat], dtype=float)
    if len(N_mat) < 2:
        print(f"ERROR: insufficient stable matter points: {len(N_mat)}")
        return 1
    lam_inf_mat, a_mat, sse_mat = symanzik1(N_mat, y_mat)

    # Richardson 2-point using the two largest N
    sorted_pts = sorted(mat, key=lambda x: -x[0])
    N1, lam1 = sorted_pts[1]  # second-largest
    N2, lam2 = sorted_pts[0]  # largest
    lam_inf_R = (lam2*N2 - lam1*N1) / (N2 - N1)

    # Bootstrap on matter points
    rng = np.random.default_rng(42)
    n_boot = 2000
    asymptotes = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(N_mat), size=len(N_mat))
        try:
            a, _, _ = symanzik1(N_mat[idx], y_mat[idx])
            if 0 < a < 1:
                asymptotes.append(a)
        except Exception:
            continue
    asymp = np.array(asymptotes)
    if len(asymp) > 100:
        boot_median = float(np.median(asymp))
        boot_lo = float(np.percentile(asymp, 2.5))
        boot_hi = float(np.percentile(asymp, 97.5))
    else:
        boot_median = boot_lo = boot_hi = None

    # Force-asymptote diagnostic: assume lambda_inf = X, compute implied a per regime
    cand_results = []
    for label, frac in CANDIDATES.items():
        val = float(frac)
        a_implied = [(float(n), (float(l) - val) * float(n))
                     for n, l in mat]
        a_vals = [x[1] for x in a_implied]
        spread = float(max(a_vals) - min(a_vals))
        a_mean = float(np.mean(a_vals))
        rel_residual_pct = abs(val - lam_inf_mat) / lam_inf_mat * 100
        in_ci = bool(boot_lo <= val <= boot_hi) if boot_lo else None
        cand_results.append({
            "label": label,
            "value": val,
            "rational": f"{frac.numerator}/{frac.denominator}",
            "implied_a_per_regime": [
                {"N": n, "a": a} for n, a in a_implied
            ],
            "implied_a_spread": spread,
            "implied_a_mean": a_mean,
            "relative_residual_pct": rel_residual_pct,
            "in_bootstrap_CI95": in_ci,
        })

    # Pick the best candidate (smallest spread)
    cand_results.sort(key=lambda x: x["implied_a_spread"])
    best = cand_results[0]
    runner_up = cand_results[1] if len(cand_results) > 1 else None

    out = {
        "headline": (
            f"Lemma B matter-branch universality: best candidate is "
            f"{best['rational']} = {best['value']:.4f} with implied-a "
            f"spread {best['implied_a_spread']:.3f} vs runner-up "
            f"{runner_up['rational']} spread {runner_up['implied_a_spread']:.3f}. "
            f"Empirical matter-branch asymptote {lam_inf_mat:.5f}."),
        "branch_boundaries": {
            "vacuum_N_max": VACUUM_N_MAX,
            "peri_flip_range": [PERI_FLIP_N_LOW, PERI_FLIP_N_HIGH],
            "matter_N_min": MATTER_N_MIN,
        },
        "vacuum_points": [{"N": n, "lambda_2": l} for n, l in vac],
        "peri_flip_points": [{"N": n, "lambda_2": l} for n, l in peri],
        "matter_points": [{"N": n, "lambda_2": l} for n, l in mat],
        "matter_branch_fit": {
            "n_points": len(mat),
            "lambda_inf": lam_inf_mat,
            "a_coefficient": a_mat,
            "sse": sse_mat,
            "richardson_2pt_using_two_largest_N": {
                "N1": N1, "N2": N2, "lambda_inf": lam_inf_R,
            },
            "bootstrap": {
                "n_boot": n_boot,
                "asymp_median": boot_median,
                "asymp_CI95": [boot_lo, boot_hi] if boot_lo else None,
            } if boot_median else None,
        },
        "candidate_rationals": cand_results,
        "verdict": {
            "best_candidate": best["rational"],
            "best_candidate_decimal": best["value"],
            "best_residual_pct": best["relative_residual_pct"],
            "shift_from_vacuum_3_8": float(Fraction(best["rational"]) - Fraction(3, 8)),
            "shift_in_gamma2_units": float(
                (Fraction(best["rational"]) - Fraction(3, 8)) / (GAMMA * GAMMA)),
            "interpretation": (
                "Vacuum: lambda_w_inf^vac = (d-1)/(2d) = 3/8. "
                f"Matter: lambda_w_inf^mat = {best['rational']}. "
                f"Shift = {float(Fraction(best['rational']) - Fraction(3, 8)):.4f} "
                f"= "
                f"{float((Fraction(best['rational']) - Fraction(3, 8)) / (GAMMA * GAMMA)):.2f}"
                " * gamma^2 = d/2 * gamma^2 (for d=4)."),
            "status": (
                "MATTER_BRANCH_UNIVERSALITY_CONJECTURED"
                if best["implied_a_spread"] < 0.5
                else "INDETERMINATE_REQUIRES_MORE_N"),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(out)
    return 0


def print_summary(out: dict) -> None:
    print("=" * 90)
    print("Lemma B matter-branch universality audit")
    print("=" * 90)
    print()
    print("Matter-branch (stable post-flip) points:")
    for p in out["matter_points"]:
        print(f"  N = {p['N']:>4d}   lambda_2 = {p['lambda_2']:.5f}")
    print()
    fit = out["matter_branch_fit"]
    print(f"Symanzik-1 fit ({fit['n_points']} pts):")
    print(f"  lambda_inf = {fit['lambda_inf']:.5f},  a = {fit['a_coefficient']:.3f}")
    print(f"Richardson 2pt (N={fit['richardson_2pt_using_two_largest_N']['N1']},"
          f" {fit['richardson_2pt_using_two_largest_N']['N2']}):"
          f" lambda_inf = {fit['richardson_2pt_using_two_largest_N']['lambda_inf']:.5f}")
    if fit.get("bootstrap"):
        b = fit["bootstrap"]
        print(f"Bootstrap median: {b['asymp_median']:.5f}")
        print(f"Bootstrap CI95: [{b['asymp_CI95'][0]:.5f}, {b['asymp_CI95'][1]:.5f}]")
    print()
    print("Candidate rationals (ranked by implied-a spread):")
    print(f"  {'rational':<35} {'value':>8} {'residual':>10} {'spread_a':>10} {'in_CI95':>10}")
    print("  " + "-" * 80)
    for c in out["candidate_rationals"]:
        print(f"  {c['label']:<35} {c['value']:>8.5f} {c['relative_residual_pct']:>9.2f}% "
              f"{c['implied_a_spread']:>10.3f} {str(c.get('in_bootstrap_CI95')):>10}")
    print()
    v = out["verdict"]
    print(f"Verdict: {v['status']}")
    print(f"  best candidate: {v['best_candidate']} = {v['best_candidate_decimal']:.5f}")
    print(f"  shift from vacuum 3/8 = {v['shift_from_vacuum_3_8']:+.4f}"
          f" = {v['shift_in_gamma2_units']:+.2f} * gamma^2")
    print()
    print(f"Interpretation: {v['interpretation']}")
    print()
    print(f"Output: {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    raise SystemExit(main())
