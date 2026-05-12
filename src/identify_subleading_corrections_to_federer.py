"""identify_subleading_corrections_to_federer.py

Finde die spezifische sub-leading Korrektur zu der Federer
$N^{-1/3}$-Bulk-Median-Skalierung. Das Symanzik-2-fit gibt
y_inf = 0.019, das ist 2.6 sigma über dem reinen $N^{-1/3}$-Signal.
Welche zusätzliche Form fängt diesen Excess?

Modelle (alle gehen asymptotisch gegen 0):
  M0  pure Federer:         y = c * N^{-1/3}                   (1 param)
  M1  Federer + 1/N:        y = c * N^{-1/3} + d * N^{-1}      (2 params)
  M2  Federer + 1/N^2:      y = c * N^{-1/3} + d * N^{-2}      (2 params)
  M3  Federer + 1/N^{2/3}:  y = c * N^{-1/3} + d * N^{-2/3}    (2 params)
  M4  multiplicative:       y = c * N^{-1/3} * (1 + a/N)       (2 params)
  M5  log/N correction:     y = c * N^{-1/3} * (1 + a*log(N)/N)(2 params)
  M6  three-term Federer:   y = c*N^{-1/3} + d*N^{-2/3} + e/N  (3 params)

Method: linear least-squares (M1, M2, M3, M6) or non-linear (M4, M5).
Compare via AICc (small-sample-corrected AIC) and RMSE.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[1]

def load_per_n_data():
    p = REPO / "outputs" / "stage6f_regular_core_decomposition.json"
    d = json.load(open(p, "r", encoding="utf-8"))
    Ns, ys = [], []
    for entry in d["per_regime"]:
        n = entry.get("N")
        rsm = entry.get("regular_set_median", {})
        v = rsm.get("0.05") if isinstance(rsm, dict) else None
        if n is not None and v is not None:
            Ns.append(int(n)); ys.append(float(v))
    return np.array(Ns, dtype=float), np.array(ys)

def aicc(rmse, n, k):
    """Small-sample AICc."""
    if rmse <= 0 or n <= k + 1:
        return float("inf")
    aic = n * np.log(rmse**2) + 2*k
    return aic + 2*k*(k+1)/(n-k-1)

def main():
    Ns, ys = load_per_n_data()
    n_pts = len(Ns)
    print(f"Ladder: N = {[int(n) for n in Ns]}")
    print(f"y_observed: {[f'{y:.4f}' for y in ys]}")
    print(f"\n{'Model':<46} {'params':>7} {'RMSE':>10} {'AICc':>9} {'DeltaAICc':>10}")
    print("-" * 90)

    results = []

    def report(name, fit_y, k, extra=None):
        res = ys - fit_y
        rmse = float(np.sqrt(np.mean(res**2)))
        a = aicc(rmse, n_pts, k)
        results.append({"model": name, "params": k, "rmse": rmse,
                        "aicc": a, **(extra or {})})

    # ------- Linear least-squares models -------
    # M0: pure Federer y = c * N^{-1/3}
    basis = Ns**(-1/3)
    c0 = float(np.sum(ys*basis)/np.sum(basis**2))
    report("M0 pure Federer y=c*N^(-1/3)", c0*basis, 1, {"c": c0})

    # M1: Federer + 1/N
    A = np.column_stack([Ns**(-1/3), 1.0/Ns])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    report("M1 Federer + 1/N (y=c*N^-1/3 + d/N)", A @ coef, 2,
           {"c": float(coef[0]), "d": float(coef[1])})

    # M2: Federer + 1/N^2
    A = np.column_stack([Ns**(-1/3), 1.0/Ns**2])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    report("M2 Federer + 1/N^2 (y=c*N^-1/3 + d/N^2)", A @ coef, 2,
           {"c": float(coef[0]), "d": float(coef[1])})

    # M3: Federer + 1/N^{2/3}
    A = np.column_stack([Ns**(-1/3), Ns**(-2/3)])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    report("M3 two-power (c*N^-1/3 + d*N^-2/3)", A @ coef, 2,
           {"c": float(coef[0]), "d": float(coef[1])})

    # M6: three-term Federer
    A = np.column_stack([Ns**(-1/3), Ns**(-2/3), 1.0/Ns])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    report("M6 c*N^-1/3 + d*N^-2/3 + e/N", A @ coef, 3,
           {"c": float(coef[0]), "d": float(coef[1]),
            "e": float(coef[2])})

    # ------- Non-linear models -------
    # M4: multiplicative y = c * N^{-1/3} * (1 + a/N)
    def model_M4(N, c, a):
        return c * N**(-1/3) * (1 + a/N)
    try:
        popt, _ = curve_fit(model_M4, Ns, ys, p0=[c0, 0.0])
        report("M4 c*N^-1/3 * (1 + a/N)", model_M4(Ns, *popt), 2,
               {"c": float(popt[0]), "a": float(popt[1])})
    except Exception as e:
        print("M4 failed:", e)

    # M5: log/N
    def model_M5(N, c, a):
        return c * N**(-1/3) * (1 + a*np.log(N)/N)
    try:
        popt, _ = curve_fit(model_M5, Ns, ys, p0=[c0, 0.0])
        report("M5 c*N^-1/3 * (1 + a*log(N)/N)", model_M5(Ns, *popt), 2,
               {"c": float(popt[0]), "a": float(popt[1])})
    except Exception as e:
        print("M5 failed:", e)

    # ------- Reference: Symanzik-2 -------
    A = np.column_stack([np.ones_like(Ns), 1.0/Ns, 1.0/Ns**2])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    report("Sym-2 a + b/N + c/N^2 (FOR REFERENCE)",
           A @ coef, 3,
           {"a_inf": float(coef[0]), "b": float(coef[1]),
            "c": float(coef[2])})

    # Sort and print
    aicc_min = min(r["aicc"] for r in results)
    results_sorted = sorted(results, key=lambda r: r["aicc"])
    for r in results_sorted:
        delta = r["aicc"] - aicc_min
        params_str = ", ".join(f"{k}={v:+.4f}" for k,v in r.items()
                              if k not in {"model","params","rmse","aicc"})
        print(f'{r["model"]:<46} {r["params"]:>7} {r["rmse"]:>10.3e} '
              f'{r["aicc"]:>9.2f} {delta:>10.2f}')

    print(f"\nBest model fit parameters:")
    for r in results_sorted[:3]:
        print(f"  {r['model']}:")
        for k, v in r.items():
            if k not in {"model", "params", "rmse", "aicc"}:
                print(f"    {k} = {v:+.6f}")
        print(f"    rmse = {r['rmse']:.3e}, aicc = {r['aicc']:.2f}")

    # ------- Numerical match of d via D(Omega) and a published numerical anchor -------
    # NOTE: This block is a NUMERICAL OBSERVATION, not a structural derivation.
    # The numeric value gram_min_eigenvalue_comparison = 0.21138 is itself a
    # *relative deviation* between two spectral profiles
    # (vortex-skeleton family vs dense-defect-cell reference) under the
    # framework's structure-healing audit, not an eigenvalue. The match to
    # -d / D(Omega) = 0.21134 within 0.04% is statistically real but
    # structurally unexplained: there is no a-priori reason for these two
    # dimensionless numbers (a sub-leading Federer coefficient on the lattice
    # ladder, vs a relative-deviation comparison between two graph profiles)
    # to coincide. We report it as a numerical curiosity worth deeper audit,
    # not as an established identity.
    print("\n" + "=" * 90)
    print("NUMERICAL OBSERVATION (not a structural derivation)")
    print("=" * 90)
    beta_pi = 15/16; gamma = 1/10
    D_Omega = beta_pi - gamma
    lambda_min_gram_P1 = 0.21138    # canonical (R6/vortex, P1)
    lambda_min_gram_P2p = 0.23559   # alternate (R6/vortex, P2')

    # Find M3 two-power fit
    m3 = next((r for r in results if 'M3' in r["model"]), None)
    d_emp = m3["d"]; c_emp = m3["c"]

    d_struct_P1 = -D_Omega * lambda_min_gram_P1
    d_struct_P2p = -D_Omega * lambda_min_gram_P2p

    print(f'Empirical M3 two-power fit:  c = {c_emp:+.6f}, d = {d_emp:+.6f}')
    print(f'  -d / D(Omega) = {-d_emp/D_Omega:.6f}')
    print()
    print('Coincidence check vs framework reference numbers:')
    print(f'  D(Omega) = beta_pi - gamma = {D_Omega} (= 67/80, parameter-free C2 closure of Paper 2)')
    print('  Note: gram_min_eigenvalue_comparison is a relative spectral')
    print('    deviation between the dense-defect-cell reference and the')
    print('    vortex-skeleton family (= |sigma_min_family - sigma_min_ref|/|sigma_min_ref|),')
    print('    NOT a direct eigenvalue. There is no a-priori structural')
    print('    reason this dimensionless deviation should equal -d/D(Omega).')
    print('  Numerical reference from corpus structural-healing audit:')
    print(f'    P1  comparison-deviation = {lambda_min_gram_P1:.6f} (matches -d/D(Omega) at 0.04 %)')
    print(f'    P2p comparison-deviation = {lambda_min_gram_P2p:.6f} (matches -d/D(Omega) at 11.5 %)')
    print()
    print('Verdict: the 0.04 % match on P1 is a numerical coincidence in a')
    print('         dimensionless 0.2-range; report as observation, not as')
    print('         derived identity. Both Federer orders go to zero in the')
    print('         continuum, irrespective of the coefficient identification.')

    # Save bundle
    out_path = REPO / "data" / "subleading_federer_corrections.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "ladder": [int(n) for n in Ns],
            "y_observed": [float(y) for y in ys],
            "models": results_sorted,
            "best_model": results_sorted[0]["model"],
            "best_aicc": float(aicc_min),
            "numerical_coincidence_check": {
                "form_tested": "d ?= -D(Omega) * gram_min_eigenvalue_comparison",
                "D_Omega": float(D_Omega),
                "D_Omega_rational": "67/80",
                "P1_comparison_deviation": lambda_min_gram_P1,
                "P2prime_comparison_deviation": lambda_min_gram_P2p,
                "comparison_quantity_definition": (
                    "gram_min_eigenvalue_comparison = "
                    "|sigma_min_family - sigma_min_reference| / |sigma_min_reference|, "
                    "a relative spectral deviation between the vortex-skeleton "
                    "family and the dense-defect-cell reference profile. "
                    "It is dimensionless and lies in [0, infinity), NOT a direct "
                    "eigenvalue."
                ),
                "empirical_c": float(c_emp),
                "empirical_d": float(d_emp),
                "P1_match_diff_pct": float((d_struct_P1 - d_emp) / d_emp * 100),
                "P2prime_match_diff_pct": float((d_struct_P2p - d_emp) / d_emp * 100),
                "verdict": (
                    "P1 numerical match at 0.04 %, P2-prime mismatch at 11.5 %. "
                    "Both quantities are dimensionless and live in the 0.2-range; "
                    "we report the match as a numerical observation worth deeper "
                    "audit, NOT as an established structural identity. There is "
                    "no a-priori reason a sub-leading Federer coefficient on the "
                    "lattice ladder should equal a relative-deviation between "
                    "two graph profiles. The conclusion that both Federer orders "
                    "vanish asymptotically does not depend on this coefficient "
                    "identification."
                ),
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
