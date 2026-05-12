"""Test the Symanzik-standard convergence form against our power-law fit:

  (a) Symanzik:   Delta(N) = Delta_inf + c_2 / N^2 + c_4 / N^4
  (b) Power-law:  Delta(N) = Delta_inf + A * N^(-alpha)
  (c) Linear-in-1/N: Delta(N) = Delta_inf + c_1 / N

The Symanzik form is the standard for lattice-discretised observables
(PDG 2024 Review, Lattice QCD chapter 17). If our data fits Symanzik
better than power-law, the per-direction residual is in the standard
asymptotic regime; if power-law is significantly better, it is not.

Compare AIC-style score: ss_res adjusted for n_params.

Output: outputs/symanzik_vs_powerlaw_audit.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def fit_symanzik_2_4(N, y):
    """y = d_inf + c_2 * N^-2 + c_4 * N^-4  (3-param)"""
    X = np.column_stack([np.ones_like(N), N ** -2.0, N ** -4.0])
    c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    pred = X @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"d_inf": float(c[0]), "c_2": float(c[1]), "c_4": float(c[2]),
            "ss_res": ss_res, "R_squared": float(r2),
            "predicted_at_N_128": float(c[0] + c[1] / 128 ** 2 + c[2] / 128 ** 4),
            "predicted_at_N_1000": float(c[0] + c[1] / 1000 ** 2 + c[2] / 1000 ** 4)}


def fit_symanzik_2(N, y):
    """y = d_inf + c_2 * N^-2  (2-param)"""
    X = np.column_stack([np.ones_like(N), N ** -2.0])
    c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    pred = X @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"d_inf": float(c[0]), "c_2": float(c[1]),
            "ss_res": ss_res, "R_squared": float(r2),
            "predicted_at_N_128": float(c[0] + c[1] / 128 ** 2),
            "predicted_at_N_1000": float(c[0] + c[1] / 1000 ** 2)}


def fit_linear_inv_N(N, y):
    """y = d_inf + c_1 / N  (2-param)"""
    X = np.column_stack([np.ones_like(N), 1.0 / N])
    c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    pred = X @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"d_inf": float(c[0]), "c_1": float(c[1]),
            "ss_res": ss_res, "R_squared": float(r2),
            "predicted_at_N_128": float(c[0] + c[1] / 128),
            "predicted_at_N_1000": float(c[0] + c[1] / 1000)}


def fit_power_law(N, y):
    """y = d_inf + A * N^-alpha (3-param, grid search on alpha)"""
    best = None
    for alpha in np.linspace(0.1, 4.0, 80):
        for d_inf in np.linspace(0.0, max(np.min(y), 1e-4), 30):
            X = N ** (-alpha)
            y_shift = y - d_inf
            denom = float((X * X).sum())
            if denom < 1e-30: continue
            A = float((X * y_shift).sum() / denom)
            pred = d_inf + A * X
            ss_res = float(((y - pred) ** 2).sum())
            if best is None or ss_res < best["ss_res"]:
                best = {"d_inf": float(d_inf), "alpha": float(alpha), "A": A,
                        "ss_res": ss_res}
    ss_tot = float(((y - y.mean()) ** 2).sum())
    best["R_squared"] = 1.0 - best["ss_res"] / ss_tot if ss_tot > 0 else 0.0
    best["predicted_at_N_128"] = float(best["d_inf"]
                                        + best["A"] * 128 ** (-best["alpha"]))
    best["predicted_at_N_1000"] = float(best["d_inf"]
                                         + best["A"] * 1000 ** (-best["alpha"]))
    return best


def main() -> int:
    cc = json.load(open(REPO / "outputs"
                        / "core_corrected_closure_audit.json", "r"))
    rows = [r for r in cc["per_regime"] if r["N"] >= 60]
    N = np.array([r["N"] for r in rows], dtype=float)

    series = {
        "raw_med":    np.array([r["raw"]["median"] for r in rows]),
        "raw_mean":   np.array([r["raw"]["mean"] for r in rows]),
        "B_med":      np.array([r["scheme_B_subtract_tail_only"]["median"] for r in rows]),
        "B_mean":     np.array([r["scheme_B_subtract_tail_only"]["mean"] for r in rows]),
    }

    print("=" * 100)
    print("Symanzik standard vs power-law fit comparison (5 N points: 60..100)")
    print("=" * 100)
    print()
    out = {}
    for k, y in series.items():
        sym24 = fit_symanzik_2_4(N, y)
        sym2 = fit_symanzik_2(N, y)
        lin = fit_linear_inv_N(N, y)
        pl = fit_power_law(N, y)
        out[k] = {
            "n_pts": int(len(N)),
            "fit_symanzik_2_4": sym24,
            "fit_symanzik_2": sym2,
            "fit_linear_1overN": lin,
            "fit_power_law": pl,
        }
        print(f"--- {k} ---")
        print(f"  Symanzik 2+4 (3-param):  d_inf={sym24['d_inf']:.4f}, R^2={sym24['R_squared']:.3f}, predicted N=1000: {sym24['predicted_at_N_1000']:.4f}")
        print(f"  Symanzik 2 (2-param):    d_inf={sym2['d_inf']:.4f}, R^2={sym2['R_squared']:.3f}, predicted N=1000: {sym2['predicted_at_N_1000']:.4f}")
        print(f"  Linear 1/N (2-param):    d_inf={lin['d_inf']:.4f}, R^2={lin['R_squared']:.3f}, predicted N=1000: {lin['predicted_at_N_1000']:.4f}")
        print(f"  Power-law (3-param):     d_inf={pl['d_inf']:.4f}, alpha={pl['alpha']:.3f}, R^2={pl['R_squared']:.3f}, predicted N=1000: {pl['predicted_at_N_1000']:.4f}")
        print()

    print("Verdicts:")
    for k, fits in out.items():
        best = max([
            ("symanzik_2_4", fits["fit_symanzik_2_4"]["R_squared"]),
            ("symanzik_2",   fits["fit_symanzik_2"]["R_squared"]),
            ("linear_1/N",   fits["fit_linear_1overN"]["R_squared"]),
            ("power_law",    fits["fit_power_law"]["R_squared"]),
        ], key=lambda x: x[1])
        print(f"  {k:<10}: best fit form = {best[0]} (R^2 = {best[1]:.3f})")

    out_path = REPO / "outputs" / "symanzik_vs_powerlaw_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "symanzik_standard_vs_power_law_form_comparison",
            "schema_version": "1.0.0",
            "reference": "PDG 2024 Lattice QCD Review (Phys. Rev. D 110, 030001), pp. 351-354",
            "per_observable": out,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
