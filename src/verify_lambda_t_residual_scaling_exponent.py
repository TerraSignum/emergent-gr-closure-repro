"""Direct fit of the residual scaling exponent for the
Lambda_t / T_00 ratio against the constant asymptote
(1 - gamma^2 * 4/pi).

Tests the hypothesis that the finite-N approach to the asymptote
goes as N^{-p} with p = 1/2 (the candidate (L) form) versus
p = 1 (linear 1/N) versus p = 2 (Symanzik-2). Both algebraic
gamma=1/10 and lattice-measured gamma=0.100206 are tried.

Fit form (constrained asymptote):
   ratio(N) - kappa_inf = - C * N^{-p}     with kappa_inf fixed
   log[kappa_inf - ratio(N)] = log(C) - p log N

Free fit (3 params):
   ratio(N) = a - b * N^{-p}     with a, b, p free

Writes:
  data/lambda_t_residual_scaling_exponent.json
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent.parent
OUTPUTS = PAPER / "outputs"
DATA = PAPER / "data"


def load_empirical():
    bundle = json.loads(
        (OUTPUTS / "per_regime_lambda_t_universal_audit.json").read_text())
    rows = []
    for r in bundle["per_regime"]:
        if r["regime"].endswith("N128"):
            continue
        rows.append((r["regime"], int(r["N"]),
                     float(r["Lambda_t_over_T_00_ratio"])))
    rows.sort(key=lambda x: x[1])
    return rows


def loglog_fit(N, residual):
    Ns = np.asarray(N, dtype=float)
    rs = np.asarray(residual, dtype=float)
    mask = rs > 0
    if mask.sum() < 3:
        return None
    log_n = np.log(Ns[mask])
    log_r = np.log(rs[mask])
    slope, intercept = np.polyfit(log_n, log_r, 1)
    pred = intercept + slope * log_n
    ss_res = float(np.sum((log_r - pred) ** 2))
    ss_tot = float(np.sum((log_r - log_r.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"p_exponent": float(-slope), "log_C": float(intercept),
            "C": float(np.exp(intercept)), "r2_loglog": r2,
            "n_points": int(mask.sum())}


def fixed_p_fit(N, residual, p):
    Ns = np.asarray(N, dtype=float)
    rs = np.asarray(residual, dtype=float)
    A = (Ns ** -p).reshape(-1, 1)
    coef, *_ = np.linalg.lstsq(A, rs, rcond=None)
    pred = (A @ coef).flatten()
    ss_res = float(np.sum((rs - pred) ** 2))
    ss_tot = float(np.sum((rs - rs.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"p_fixed": p, "C": float(coef[0]),
            "rss": ss_res, "r2": r2}


def free_three_param_fit(N, ratios, p_grid=None):
    """Brute-force grid scan of p in [0.2, 3.0]; for each p fit
    a + b*N^{-p} by linear regression and pick best by RSS."""
    Ns = np.asarray(N, dtype=float)
    rs = np.asarray(ratios, dtype=float)
    if p_grid is None:
        p_grid = np.linspace(0.20, 3.00, 281)
    best = None
    for p in p_grid:
        A = np.column_stack([np.ones_like(Ns), Ns ** -p])
        coef, *_ = np.linalg.lstsq(A, rs, rcond=None)
        pred = A @ coef
        ss = float(np.sum((rs - pred) ** 2))
        if best is None or ss < best["rss"]:
            tss = float(np.sum((rs - rs.mean()) ** 2))
            best = {
                "p_best": float(p), "a": float(coef[0]),
                "b": float(coef[1]), "rss": ss,
                "r2": 1.0 - ss / tss if tss > 0 else 0.0,
            }
    return best


def aicc(rss, n, k):
    if n - k - 1 <= 0:
        return float("inf")
    return n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def main() -> int:
    rows = load_empirical()
    Ns = [r[1] for r in rows]
    ratios = [r[2] for r in rows]
    print("=" * 95)
    print("Lambda_t / T_00 residual scaling exponent")
    print("=" * 95)
    print(f"{'reg':>10} {'N':>4} {'ratio':>8}")
    for lbl, N, r in rows:
        print(f"{lbl:>10} {N:>4} {r:>8.4f}")
    print()

    sets = {
        "algebraic (gamma=1/10)": {"gamma": 0.1, "alpha_xi": 0.9},
        "measured (gamma=0.100206)": {"gamma": 0.100206,
                                      "alpha_xi": 0.900819},
    }
    out = {"empirical": [{"reg": r[0], "N": r[1], "ratio": r[2]}
                         for r in rows], "fits": {}}

    for label, c in sets.items():
        g = c["gamma"]
        kappa_inf = 1 - g**2 * 4/math.pi
        residual = [kappa_inf - r for r in ratios]
        print("=" * 95)
        print(f"{label}: kappa_inf = 1 - gamma^2 * 4/pi = {kappa_inf:.6f}")
        print("=" * 95)
        print()
        print("Residual r(N) = kappa_inf - ratio(N):")
        for N, r in zip(Ns, residual):
            print(f"  N={N:>4}  residual = {r:>+10.5f}")

        loglog = loglog_fit(Ns, residual)
        if loglog is None:
            print("  [residual sign-changes; skipping fit]")
        else:
            print()
            print("Log-log fit on positive residuals:")
            print(f"  exponent p   = {loglog['p_exponent']:.4f}")
            print(f"  prefactor C  = {loglog['C']:.4f}")
            print(f"  R^2 (loglog) = {loglog['r2_loglog']:.4f}")
            print(f"  n_points     = {loglog['n_points']}")
            print()
            print("Fixed-exponent comparison (residual vs N^{-p}):")
            print(f"  {'p':>5} {'C':>9} {'RSS':>11} {'R^2':>7} {'AICc':>9}")
            n = len(Ns)
            for p in [0.5, 1.0, 1.5, 2.0]:
                f = fixed_p_fit(Ns, residual, p)
                a = aicc(f["rss"], n, 1)
                print(f"  {p:>5.1f} {f['C']:>9.4f} {f['rss']:>11.5f}"
                      f" {f['r2']:>7.4f} {a:>9.3f}")

        ff = free_three_param_fit(Ns, ratios)
        a_aicc = aicc(ff["rss"], len(Ns), 3)
        print()
        print(f"Free three-parameter fit a + b*N^{{-p}} (a, b, p free):")
        print(f"  p_best = {ff['p_best']:.3f}")
        print(f"  a      = {ff['a']:.6f}  (vs kappa_inf = {kappa_inf:.6f})")
        print(f"  b      = {ff['b']:.6f}")
        print(f"  R^2    = {ff['r2']:.4f}")
        print(f"  AICc   = {a_aicc:.3f}")
        out["fits"][label] = {
            "kappa_inf_constant": kappa_inf,
            "residual_loglog": loglog,
            "fixed_p_grid": [
                {"p": p, **fixed_p_fit(Ns, residual, p)}
                for p in [0.5, 1.0, 1.5, 2.0]
            ],
            "free_three_param": ff,
        }
        print()

    out_path = DATA / "lambda_t_residual_scaling_exponent.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
