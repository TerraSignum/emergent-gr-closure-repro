"""N-scaling trend of Scheme B (statistical core-corrected closure).

After applying the universal density-contrast law to the empirically
identified top-10% tail nodes, fit power-laws

  Delta_med^B(N)  ~ A_med * N^(-alpha_med)
  Delta_mean^B(N) ~ A_mean * N^(-alpha_mean)

across N = 60, 64, 72, 84, 100 (excluding marginal P5_N50).

If alpha_mean >= 1.0 with R^2 >= 0.7: scheme-B closure converges
to Delta -> 0 in the continuum limit.

Compare to RAW (uncorrected) trend: does Scheme B accelerate
convergence?

Output: outputs/scheme_b_N_scaling_audit.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def power_law(N, y):
    if np.any(y <= 0) or len(N) < 3:
        return float("nan"), float("nan"), float("nan")
    log_N, log_y = np.log(N), np.log(y)
    slope, intercept = np.polyfit(log_N, log_y, 1)
    pred = slope * log_N + intercept
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(-slope), float(np.exp(intercept)), float(r2)


def main() -> int:
    OUT = REPO / "outputs"
    cc = json.load(open(OUT / "core_corrected_closure_audit.json", "r"))

    # Use scheme B (subtract tail only) per regime
    # Data structure: per_regime[i]["scheme_B_subtract_tail_only"]["median"|"mean"]
    rows_all = []
    for r in cc["per_regime"]:
        rows_all.append({
            "regime": r["regime"], "N": r["N"],
            "raw_median": r["raw"]["median"],
            "raw_mean": r["raw"]["mean"],
            "B_median": r["scheme_B_subtract_tail_only"]["median"],
            "B_mean": r["scheme_B_subtract_tail_only"]["mean"],
        })

    print("=" * 100)
    print("N-scaling of Scheme B (statistical core-corrected closure)")
    print("=" * 100)
    print()
    print(f"{'reg':<8} {'N':>4} | {'raw med':>9} {'raw mean':>9} | "
          f"{'B med':>9} {'B mean':>9} | "
          f"{'B-raw med':>10} {'B-raw mean':>11}")
    print("-" * 90)
    for r in rows_all:
        print(f"{r['regime']:<8} {r['N']:>4} | "
              f"{r['raw_median']:>9.5f} {r['raw_mean']:>9.5f} | "
              f"{r['B_median']:>9.5f} {r['B_mean']:>9.5f} | "
              f"{r['B_median']-r['raw_median']:>+10.5f} "
              f"{r['B_mean']-r['raw_mean']:>+11.5f}")

    # Fits
    rows_60plus = [r for r in rows_all if r["N"] >= 60]
    N_arr_all = np.array([r["N"] for r in rows_all], dtype=float)
    N_arr = np.array([r["N"] for r in rows_60plus], dtype=float)

    fits = {}
    print()
    print("Power-law fits  Delta(N) ~ A * N^(-alpha)")
    print(f"{'observable':<28} {'alpha':>10} {'A':>14} {'R^2':>8}")
    print("-" * 70)

    for label, key, ns, rows in [
        ("raw median (all N)",       "raw_median",  N_arr_all, rows_all),
        ("raw mean (all N)",         "raw_mean",    N_arr_all, rows_all),
        ("B median (N>=60)",         "B_median",    N_arr, rows_60plus),
        ("B mean (N>=60)",           "B_mean",      N_arr, rows_60plus),
        ("raw median (N>=60)",       "raw_median",  N_arr, rows_60plus),
        ("raw mean (N>=60)",         "raw_mean",    N_arr, rows_60plus),
    ]:
        ys = np.array([r[key] for r in rows])
        a, A, r2 = power_law(ns, ys)
        fits[label] = {"alpha": a, "A": A, "R_squared": r2}
        print(f"{label:<28} {a:>+10.3f} {A:>14.6f} {r2:>8.3f}")

    # Continuum extrapolation: Delta at N -> infty
    a_med = fits["B median (N>=60)"]["alpha"]
    a_mean = fits["B mean (N>=60)"]["alpha"]
    A_med = fits["B median (N>=60)"]["A"]
    A_mean = fits["B mean (N>=60)"]["A"]

    print()
    print("Continuum projection (N >> 100) under Scheme B:")
    for N_target in [128, 200, 500, 1000, 10000]:
        if a_med > 0:
            d_med = A_med * N_target ** (-a_med)
        else:
            d_med = float("nan")
        if a_mean > 0:
            d_mean = A_mean * N_target ** (-a_mean)
        else:
            d_mean = float("nan")
        print(f"  N={N_target:>6}: Delta_med ~ {d_med:.5f}, Delta_mean ~ {d_mean:.5f}")

    # Verdicts
    convergent_med = (a_med >= 1.0 and fits["B median (N>=60)"]["R_squared"] >= 0.7)
    convergent_mean = (a_mean >= 1.0 and fits["B mean (N>=60)"]["R_squared"] >= 0.7)

    print()
    print("Continuum closure verdict (alpha >= 1.0 AND R^2 >= 0.7):")
    print(f"  Delta_med^B:   alpha={a_med:+.3f}, R^2={fits['B median (N>=60)']['R_squared']:.3f}  "
          f"-> {'CONVERGES' if convergent_med else 'OPEN'}")
    print(f"  Delta_mean^B:  alpha={a_mean:+.3f}, R^2={fits['B mean (N>=60)']['R_squared']:.3f}  "
          f"-> {'CONVERGES' if convergent_mean else 'OPEN'}")

    if convergent_med and convergent_mean:
        verdict = "SCHEME_B_CONVERGENT_CONTINUUM_CLOSURE"
    elif convergent_med or convergent_mean:
        verdict = "SCHEME_B_PARTIAL_CONTINUUM_CONVERGENCE"
    else:
        verdict = "SCHEME_B_FINITE_FLOOR_NO_POWER_LAW_CONVERGENCE"

    print()
    print(f"VERDICT: {verdict}")

    out = {
        "method": "scheme_B_N_scaling_continuum_extrapolation",
        "schema_version": "1.0.0",
        "per_regime": rows_all,
        "fits": fits,
        "continuum_projection_N_geq_60": {
            "Delta_med_alpha": a_med, "Delta_mean_alpha": a_mean,
            "Delta_med_A": A_med, "Delta_mean_A": A_mean,
            "predicted_at_N_128": {
                "median": A_med * 128 ** (-a_med) if a_med > 0 else None,
                "mean": A_mean * 128 ** (-a_mean) if a_mean > 0 else None,
            },
            "predicted_at_N_1000": {
                "median": A_med * 1000 ** (-a_med) if a_med > 0 else None,
                "mean": A_mean * 1000 ** (-a_mean) if a_mean > 0 else None,
            },
        },
        "verdict": verdict,
    }
    out_path = REPO / "outputs" / "scheme_b_N_scaling_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
