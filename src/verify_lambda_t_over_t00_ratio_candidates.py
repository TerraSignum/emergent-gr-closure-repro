"""Algebraic candidates for the regime-universal ratio
Lambda_t / T_00 ~ 0.97 +- 0.013 across N >= 28.

Empirical per-regime ratios (from per_regime_lambda_t_universal_audit
on standard-configuration runs N <= 100):

  N=18 (P0):     0.944
  N=28 (P1):     0.945
  N=30 (P2'):    0.942
  N=36 (P3):     0.959
  N=42 (P4):     0.969
  N=50 (P5):     0.970
  N=60 (P6):     0.981
  N=64 (P5N64):  0.979
  N=72 (P7):     0.984
  N=84 (P8):     0.989
  N=100 (P5N100):0.987

Cross-regime mean (N>=28): 0.970, std 0.017.

Tested candidates:

  (A) constant 1 - gamma^2 / alpha_xi^2 = 1 - 1/81 = 0.98765
  (B) alpha_xi^2 * (1 + 1/(2N))                  -- finite-N correction
  (C) 1 - 2 gamma^2 (1 - 1/N)                    -- 2-pt-anisotropy form
  (D) 1 - gamma^2 (4/pi)                         -- topological combination
  (E) 1 - gamma / N^{1/2}                        -- sqrt(N) decay
  (F) 1 - alpha_xi gamma^2 / (alpha_xi - gamma)  -- C2-bridge form
  (G) (1 - gamma) + gamma^2                      -- algebraic deformation
  (H) 1 - gamma^2 (1 + 1/N)                      -- linear finite-N

For each candidate the script reports per-regime values, the
cross-regime mean and its deviation from the empirical mean, and
the regime-by-regime residual.

Writes:
  data/lambda_t_over_t00_ratio_candidates.json
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent.parent
OUTPUTS = PAPER / "outputs"
DATA = PAPER / "data"


def load_empirical() -> tuple[list[int], list[float], list[str]]:
    bundle = json.loads(
        (OUTPUTS / "per_regime_lambda_t_universal_audit.json").read_text())
    rows = []
    for r in bundle["per_regime"]:
        if r["regime"].endswith("N128"):
            continue
        rows.append((r["regime"], r["N"],
                     r["Lambda_t_over_T_00_ratio"]))
    rows.sort(key=lambda x: x[1])
    Ns = [r[1] for r in rows]
    rs = [r[2] for r in rows]
    labels = [r[0] for r in rows]
    return Ns, rs, labels


def candidates(Ns, alpha_xi, gamma, beta_pi, D_omega):
    a, g = alpha_xi, gamma
    return {
        "(A) 1 - gamma^2 / alpha_xi^2": [1 - g**2 / a**2 for _ in Ns],
        "(B) alpha_xi^2 (1 + 1/(2N))": [a**2 * (1 + 1.0/(2*N)) for N in Ns],
        "(C) 1 - 2 gamma^2 (1 - 1/N)": [1 - 2 * g**2 * (1 - 1.0/N) for N in Ns],
        "(D) 1 - gamma^2 (4/pi)": [1 - g**2 * 4/math.pi for _ in Ns],
        "(E) 1 - gamma / sqrt(N)": [1 - g / math.sqrt(N) for N in Ns],
        "(F) 1 - alpha_xi gamma^2 / (a-g)": [1 - a * g**2 / (a - g) for _ in Ns],
        "(G) (1 - gamma) + gamma^2": [(1 - g) + g**2 for _ in Ns],
        "(H) 1 - gamma^2 (1 + 1/N)": [1 - g**2 * (1 + 1.0/N) for N in Ns],
        "(I) 1 - gamma (4/pi) / sqrt(N)": [1 - g * (4/math.pi) / math.sqrt(N) for N in Ns],
        "(J) 1 - g^2 - g^3 (1 - 1/sqrt(N))": [1 - g**2 - g**3 * (1 - 1/math.sqrt(N)) for N in Ns],
        # Hybrid: asymptote 1 - g^2 (4/pi), finite-N correction
        "(K) [1 - g^2 (4/pi)] - g/sqrt(N)": [(1 - g**2 * 4/math.pi) - g/math.sqrt(N) for N in Ns],
        "(L) [1 - g^2 (4/pi)] - g (4/pi)/sqrt(N)": [(1 - g**2 * 4/math.pi) - g * 4/math.pi / math.sqrt(N) for N in Ns],
        "(M) [1 - g^2 (4/pi)] - g^2 / N": [(1 - g**2 * 4/math.pi) - g**2 / N for N in Ns],
    }


def free_fit_const_plus_inv_sqrtN(Ns, ratios):
    """Fit: ratio = a + b/sqrt(N); returns (a, b, r2)."""
    x = np.array([1.0/math.sqrt(N) for N in Ns])
    y = np.asarray(ratios, dtype=float)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(coef[0]), float(coef[1]), r2


def metrics(empirical, predicted):
    e = np.asarray(empirical, dtype=float)
    p = np.asarray(predicted, dtype=float)
    diff = p - e
    rms = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    bias = float(np.mean(diff))
    rel_bias_pct = float(np.mean(np.abs(diff) / e) * 100)
    return {"rms": rms, "max_abs_resid": max_abs,
            "mean_bias": bias, "mean_rel_pct": rel_bias_pct}


def main() -> int:
    Ns, ratios, labels = load_empirical()
    print("=" * 95)
    print("Algebraic candidates for Lambda_t / T_00 ratio across regimes")
    print("=" * 95)
    print(f"{'reg':>10} {'N':>4} {'L/T_00':>10}")
    for lbl, N, r in zip(labels, Ns, ratios):
        print(f"{lbl:>10} {N:>4} {r:>10.4f}")
    print()
    e = np.asarray(ratios)
    print(f"Cross-regime mean (all):         {float(np.mean(e)):.4f} +- {float(np.std(e)):.4f}")
    e_28plus = np.asarray([r for r, N in zip(ratios, Ns) if N >= 28])
    print(f"Cross-regime mean (N>=28):       {float(np.mean(e_28plus)):.4f} +- {float(np.std(e_28plus)):.4f}")

    for label, (g_alg, alpha_alg, beta_alg, D_alg) in [
        ("ALGEBRAIC", (0.1, 0.9, 0.9375, 0.8375)),
        ("MEASURED ", (0.100206, 0.900819, 0.937913, 0.839964)),
    ]:
        cands = candidates(Ns, alpha_alg, g_alg, beta_alg, D_alg)
        print()
        print("=" * 95)
        print(f"With {label} constants (gamma={g_alg}, alpha_xi={alpha_alg}):")
        print("=" * 95)
        print(f"{'candidate':<40} {'mean':>8} {'rms':>8} {'max_abs':>10} {'rel_pct':>10}")

        per_cand = {}
        for name, vals in cands.items():
            m = metrics(ratios, vals)
            print(f"{name:<40} {float(np.mean(vals)):>8.4f} {m['rms']:>8.5f} "
                  f"{m['max_abs_resid']:>10.5f} {m['mean_rel_pct']:>10.3f}")
            per_cand[name] = {"mean": float(np.mean(vals)),
                              "values": vals, **m}

        if label.startswith("ALGEBRAIC"):
            algebraic_results = per_cand
        else:
            measured_results = per_cand

    a_fit, b_fit, r2_fit = free_fit_const_plus_inv_sqrtN(Ns, ratios)
    print()
    print("=" * 95)
    print("Free-fit reference (no algebraic constraint):")
    print("  ratio(N) = a + b / sqrt(N)")
    print(f"  a = {a_fit:.6f}    -> compare to 1 - g^2 (4/pi) = "
          f"{1 - 0.01 * 4/math.pi:.6f} (alg) / "
          f"{1 - 0.100206**2 * 4/math.pi:.6f} (meas)")
    print(f"  b = {b_fit:.6f}    -> compare to -g (4/pi) = "
          f"{-0.1 * 4/math.pi:.6f} (alg) / "
          f"{-0.100206 * 4/math.pi:.6f} (meas)")
    print(f"  R^2 = {r2_fit:.4f}")

    bundle = {
        "title": ("Algebraic candidate scan for Lambda_t / T_00 "
                  "regime-mean and per-regime residuals."),
        "empirical": {
            "regimes": labels, "N": Ns, "ratios": ratios,
            "mean_all": float(np.mean(e)),
            "std_all": float(np.std(e)),
            "mean_N28plus": float(np.mean(e_28plus)),
            "std_N28plus": float(np.std(e_28plus)),
        },
        "constants": {
            "algebraic": {"alpha_xi": 0.9, "gamma": 0.1,
                          "beta_pi": 0.9375, "D_omega": 0.8375},
            "measured":  {"alpha_xi": 0.900819, "gamma": 0.100206,
                          "beta_pi": 0.937913, "D_omega": 0.839964},
        },
        "candidates_algebraic": algebraic_results,
        "candidates_measured":  measured_results,
    }
    out = DATA / "lambda_t_over_t00_ratio_candidates.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
