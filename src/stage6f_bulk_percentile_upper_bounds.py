"""Bootstrap-based upper bounds on the bulk-percentile continuum
asymptote y_inf, using two independent model classes.

Reads the Stage 6f per-regime per-percentile data from
outputs/stage6f_full_tensor_norm_audit.json and computes:

  (i) Unconstrained Symanzik-2 fit y(N) = y_inf + b/N with
      bootstrap 95% upper-edge CI on y_inf (the data-conservative
      upper bound).
 (ii) Power-law fit y(N) = c * N^(-alpha) with bootstrap
      95% lower-edge CI on alpha (alpha > 0 forces y_inf = 0
      exactly; the lower-CI bound on alpha quantifies how strict
      the convergence-to-zero is supported).

Output: outputs/stage6f_bulk_percentile_upper_bounds.json

The constrained refit y_inf >= 0 (already in
stage6f_nonneg_residual_extrapolation.py) is the maximum-likelihood
estimate under the physical constraint and saturates at the
boundary; it is NOT a data-derived upper bound. The bounds
reported here are properly data-derived.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "outputs" / "stage6f_full_tensor_norm_audit.json"
OUT = REPO / "outputs" / "stage6f_bulk_percentile_upper_bounds.json"

PERCENTILES = ("median", "mean", "p90", "p95", "p99", "sup")
N_BOOT = 2000
RNG_SEED = 0xC0FFEE


def _fit_symanzik2(n_arr, y_arr):
    x = 1.0 / np.asarray(n_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    a_mat = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(a_mat, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _fit_powerlaw(n_arr, y_arr):
    n_arr = np.asarray(n_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    mask = y_arr > 0
    if mask.sum() < 3:
        return float("nan"), float("nan")
    log_n = np.log(n_arr[mask])
    log_y = np.log(y_arr[mask])
    a_mat = np.column_stack([np.ones_like(log_n), log_n])
    coef, *_ = np.linalg.lstsq(a_mat, log_y, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])
    return -slope, math.exp(intercept)


def _bootstrap(rng, n_arr, y_arr, fit_fn, n_boot=N_BOOT):
    n = len(n_arr)
    samples = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            samples.append(fit_fn([n_arr[i] for i in idx],
                                   [y_arr[i] for i in idx]))
        except (ValueError, np.linalg.LinAlgError):
            continue
    return samples


def _ci_bounds(samples, lo=0.025, hi=0.975):
    s = sorted(samples)
    if not s:
        return None
    lo_i = int(lo * len(s))
    hi_i = int(hi * len(s))
    return s[lo_i], s[hi_i]


def main():
    raw = json.loads(IN.read_text(encoding="utf-8"))
    rows = raw["per_regime"]
    rows.sort(key=lambda r: r["N"])
    n_arr = [r["N"] for r in rows]
    rng = random.Random(RNG_SEED)

    out = {
        "method": "Bootstrap upper-bound on Stage 6f bulk-percentile asymptote",
        "n_bootstrap": N_BOOT,
        "model_class_1": "Symanzik-2: y = y_inf + b/N (unconstrained)",
        "model_class_2": "Power-law: y = c * N^(-alpha)",
        "ladder_size": len(rows),
        "per_percentile": {},
    }

    for pct in PERCENTILES:
        y_arr = [r["delta_full"][pct] for r in rows]

        unc_yinf, unc_b = _fit_symanzik2(n_arr, y_arr)
        pl_alpha, pl_c = _fit_powerlaw(n_arr, y_arr)

        sym_samples = _bootstrap(
            rng, n_arr, y_arr,
            lambda n, y: _fit_symanzik2(n, y)[0])
        pl_samples = _bootstrap(
            rng, n_arr, y_arr,
            lambda n, y: _fit_powerlaw(n, y)[0])

        sym_ci = _ci_bounds(sym_samples)
        pl_ci = _ci_bounds(pl_samples)

        out["per_percentile"][pct] = {
            "symanzik2_unconstrained_y_inf": unc_yinf,
            "symanzik2_y_inf_95CI": list(sym_ci) if sym_ci else None,
            "symanzik2_y_inf_upper95": sym_ci[1] if sym_ci else None,
            "powerlaw_alpha_central": pl_alpha,
            "powerlaw_alpha_95CI": list(pl_ci) if pl_ci else None,
            "powerlaw_alpha_lower95": pl_ci[0] if pl_ci else None,
            "powerlaw_implies_y_inf_zero_if_alpha_strictly_positive": (
                pl_ci[0] > 0 if pl_ci else None),
        }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print()
    print(f"{'pct':>8s} | {'Sym2 y_inf central':>20s} | "
          f"{'Sym2 y_inf upper-95':>20s} | "
          f"{'PL alpha lower-95':>18s}")
    print("-" * 80)
    for pct in PERCENTILES:
        r = out["per_percentile"][pct]
        cu = r["symanzik2_unconstrained_y_inf"]
        ub = r["symanzik2_y_inf_upper95"]
        al = r["powerlaw_alpha_lower95"]
        cu_s = f"{cu:+.4f}" if cu == cu else "nan"
        ub_s = f"{ub:+.4f}" if ub is not None else "nan"
        al_s = f"{al:+.3f}" if al is not None and al == al else "nan"
        print(f"{pct:>8s} | {cu_s:>20s} | {ub_s:>20s} | {al_s:>18s}")


if __name__ == "__main__":
    main()
