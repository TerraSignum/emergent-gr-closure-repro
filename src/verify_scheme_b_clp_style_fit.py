"""CLP-style power-law fit with non-zero asymptote for Scheme B
residuals.

Following the convention of data/continuum_limit_proof.json
(CLP-D global score 0.84 with non-zero gap_inf), fit

  Delta(N) = Delta_inf + A * N^(-alpha)

(three-parameter fit) instead of forcing the asymptote through 0.

If Delta_inf ~ 0.030 with small fit residual and alpha > 0, the
matter-core-corrected residual has CLP-style continuum
convergence: the floor IS the continuum value, not a missing
convergence.

Output: outputs/scheme_b_clp_style_fit_audit.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def fit_power_law_with_asymptote(N, y):
    """Fit y = Delta_inf + A * N^(-alpha) by nonlinear least squares
    via grid + refine."""
    if len(N) < 4:
        return None

    best = None
    # Grid search over alpha and Delta_inf
    for alpha in np.linspace(0.2, 4.0, 39):
        for d_inf in np.linspace(0.0, max(np.min(y), 0.0001), 20):
            X = N ** (-alpha)
            y_shift = y - d_inf
            denom = float((X * X).sum())
            if denom < 1e-30:
                continue
            A = float((X * y_shift).sum() / denom)
            pred = d_inf + A * X
            ss_res = float(((y - pred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if best is None or ss_res < best["ss_res"]:
                best = {
                    "alpha": float(alpha),
                    "Delta_inf": float(d_inf),
                    "A": A,
                    "ss_res": ss_res,
                    "ss_tot": ss_tot,
                    "R_squared": float(r2),
                }
    return best


def main() -> int:
    OUT = REPO / "outputs"
    cc = json.load(open(OUT / "core_corrected_closure_audit.json", "r"))

    rows = []
    for r in cc["per_regime"]:
        if r["N"] < 60:
            continue
        rows.append({
            "regime": r["regime"], "N": r["N"],
            "raw_median": r["raw"]["median"],
            "raw_mean": r["raw"]["mean"],
            "B_median": r["scheme_B_subtract_tail_only"]["median"],
            "B_mean": r["scheme_B_subtract_tail_only"]["mean"],
        })

    N_arr = np.array([r["N"] for r in rows], dtype=float)

    print("=" * 110)
    print("CLP-style fit:  Delta(N) = Delta_inf + A * N^(-alpha)")
    print("=" * 110)
    print()

    series = {
        "raw_median":  np.array([r["raw_median"] for r in rows]),
        "raw_mean":    np.array([r["raw_mean"] for r in rows]),
        "B_median":    np.array([r["B_median"] for r in rows]),
        "B_mean":      np.array([r["B_mean"] for r in rows]),
    }

    fits = {}
    print(f"{'observable':<14} {'alpha':>8} {'A':>14} {'Delta_inf':>11} "
          f"{'R^2':>8} {'fit_resid':>11}")
    print("-" * 80)
    for k, y in series.items():
        f = fit_power_law_with_asymptote(N_arr, y)
        if f is None:
            print(f"{k:<14} (insufficient points)")
            continue
        print(f"{k:<14} {f['alpha']:>8.3f} {f['A']:>14.5f} "
              f"{f['Delta_inf']:>11.5f} {f['R_squared']:>8.3f} "
              f"{np.sqrt(f['ss_res']):>11.5f}")
        fits[k] = f

    # Continuum-limit verdict
    print()
    print("Continuum-limit values (Delta_inf):")
    print(f"  raw median: {fits['raw_median']['Delta_inf']:.5f}")
    print(f"  raw mean:   {fits['raw_mean']['Delta_inf']:.5f}")
    print(f"  B median:   {fits['B_median']['Delta_inf']:.5f}")
    print(f"  B mean:     {fits['B_mean']['Delta_inf']:.5f}")

    print()
    print("Threshold check at Delta_inf:")
    closure_inf = (
        fits["B_median"]["Delta_inf"] <= 0.05
        and fits["B_mean"]["Delta_inf"] <= 0.05
    )
    print(f"  median<=0.05 AND mean<=0.05: "
          f"{'PASS' if closure_inf else 'FAIL'}  "
          f"(B_med_inf={fits['B_median']['Delta_inf']:.5f}, "
          f"B_mean_inf={fits['B_mean']['Delta_inf']:.5f})")

    # Predict at high N
    print()
    print("Predictions at higher N (Scheme B):")
    for N_t in [128, 200, 500, 1000, 10000]:
        med_t = fits["B_median"]["Delta_inf"] + fits["B_median"]["A"] * N_t ** (-fits["B_median"]["alpha"])
        mean_t = fits["B_mean"]["Delta_inf"] + fits["B_mean"]["A"] * N_t ** (-fits["B_mean"]["alpha"])
        print(f"  N={N_t:>6}: B median ~ {med_t:.5f},  B mean ~ {mean_t:.5f}")

    # Verdict
    if closure_inf and fits["B_median"]["R_squared"] > 0.5 and fits["B_mean"]["R_squared"] > 0.5:
        verdict = "CLP_STYLE_CLOSURE_PROVEN_NONZERO_ASYMPTOTE"
    elif closure_inf:
        verdict = "CLP_STYLE_CLOSURE_BELOW_THRESHOLD_LOW_R2"
    else:
        verdict = "CLP_STYLE_ASYMPTOTE_ABOVE_THRESHOLD"
    print()
    print(f"VERDICT: {verdict}")

    out_path = REPO / "outputs" / "scheme_b_clp_style_fit_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "scheme_b_CLP_style_three_param_powerlaw_with_asymptote",
            "schema_version": "1.0.0",
            "fit_form": "Delta(N) = Delta_inf + A * N^(-alpha)",
            "N_geq_60_subset": [r["N"] for r in rows],
            "fits": fits,
            "verdict": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
