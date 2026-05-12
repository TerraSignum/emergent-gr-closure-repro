"""Reviewer follow-up A: sensitivity matrix for the Stage 6f bulk-percentile
full-tensor closure asymptotes.

Reads the existing per-regime Stage 6f data
(outputs/stage6f_full_tensor_norm_audit.json) and recomputes the
Symanzik-2 continuum extrapolation y(N) = y_inf + b/N under five
perturbations:

  (1) cleaned ladder (default; P5N128 excluded by the K/Q-persistence
      pre-registered artefact criterion documented in the parent
      K_Q_BUG_FIX_README);
  (2) include-P5N128 fit (the artefact-included variant);
  (3) robust M-estimator using Huber weights (robust to a single
      outlier regime);
  (4) leave-one-regime-out (LORO) cross-validation: report the
      asymptote spread across all 10 leave-one folds;
  (5) constrained nonneg refit (y_inf >= 0, the physical-residual
      reading documented in the nonneg-extrapolation reproducer).

The output is a single sensitivity matrix that lets a reviewer judge
whether the headline claim is artefact-stable.

Output: outputs/stage6f_sensitivity_matrix.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "outputs" / "stage6f_full_tensor_norm_audit.json"
P5N128_FILE = REPO / "outputs" / "stage6f_full_tensor_norm_audit_with_p5n128.json"
OUT = REPO / "outputs" / "stage6f_sensitivity_matrix.json"


def fit_unweighted(N, y):
    x = 1.0 / np.asarray(N, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def fit_constrained_yinf_zero(N, y):
    x = 1.0 / np.asarray(N, dtype=float)
    y = np.asarray(y, dtype=float)
    num = float(np.sum(x * y))
    den = float(np.sum(x * x))
    return 0.0, num / den


def huber_fit(N, y, c=1.345, n_iter=20):
    """Huber M-estimator (IRLS) for y = a + b/N."""
    x = 1.0 / np.asarray(N, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    for _ in range(n_iter):
        r = y - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        z = r / s
        w = np.where(np.abs(z) <= c, 1.0, c / np.abs(z))
        AW = A * w[:, None]
        yw = y * w
        coef, *_ = np.linalg.lstsq(AW, yw, rcond=None)
    return float(coef[0]), float(coef[1])


def loro(N, y, fit_fn):
    """Leave-one-regime-out spread of y_inf."""
    n = len(N)
    if n <= 3:
        return None
    asym = []
    for i in range(n):
        N_loro = np.delete(N, i)
        y_loro = np.delete(y, i)
        a, _ = fit_fn(N_loro, y_loro)
        asym.append(a)
    return {
        "y_inf_min": float(min(asym)),
        "y_inf_max": float(max(asym)),
        "y_inf_mean": float(np.mean(asym)),
        "y_inf_std": float(np.std(asym, ddof=1) if len(asym) > 1 else 0.0),
    }


def main():
    raw = json.loads(IN.read_text(encoding="utf-8"))
    rows = raw["per_regime"]
    rows.sort(key=lambda r: r["N"])
    n_clean = np.array([r["N"] for r in rows], dtype=float)

    # Try to load the P5N128-included version if it exists
    rows_with_p5n128 = None
    if P5N128_FILE.exists():
        raw_p5n128 = json.loads(P5N128_FILE.read_text(encoding="utf-8"))
        rows_with_p5n128 = sorted(raw_p5n128["per_regime"], key=lambda r: r["N"])

    out = {
        "method": "Stage 6f sensitivity matrix on the bulk-percentile asymptote",
        "input_file": str(IN.relative_to(REPO)),
        "p5n128_inclusion_caveat": (
            "P5N128 is excluded from the cleaned ladder by a "
            "pre-registered artefact criterion: only the old K/Q-"
            "persistence-buggy 4-seed run exists for that regime "
            "(see K_Q_BUG_FIX_README). The exclusion is therefore "
            "by criterion, not by post-hoc residual selection. The "
            "include-P5N128 row of the sensitivity matrix below is "
            "reported only as a robustness-check, with the explicit "
            "warning that the included data point is K/Q-buggy."
        ),
        "sensitivity_per_percentile": {},
    }

    pct_keys = ["median", "mean", "p90", "p95", "p99", "sup"]
    for pct in pct_keys:
        y_clean = np.array([r["delta_full"][pct] for r in rows], dtype=float)
        a_unc, b_unc = fit_unweighted(n_clean, y_clean)
        a_cons, b_cons = (
            fit_constrained_yinf_zero(n_clean, y_clean)
            if a_unc < 0 else (a_unc, b_unc))
        a_huber, b_huber = huber_fit(n_clean, y_clean)
        loro_clean = loro(n_clean, y_clean, fit_unweighted)
        sens = {
            "cleaned_unconstrained": {"y_inf": a_unc, "b": b_unc},
            "cleaned_nonneg_constrained": {
                "y_inf": a_cons, "b": b_cons,
                "constraint_active": a_unc < 0,
            },
            "cleaned_huber_robust": {"y_inf": a_huber, "b": b_huber},
            "cleaned_LORO_spread": loro_clean,
        }
        if rows_with_p5n128 is not None:
            n_with = np.array([r["N"] for r in rows_with_p5n128], dtype=float)
            y_with = np.array([r["delta_full"][pct] for r in rows_with_p5n128],
                               dtype=float)
            a_w, b_w = fit_unweighted(n_with, y_with)
            sens["include_p5n128_unconstrained"] = {"y_inf": a_w, "b": b_w}
            a_w_h, b_w_h = huber_fit(n_with, y_with)
            sens["include_p5n128_huber_robust"] = {"y_inf": a_w_h, "b": b_w_h}
        else:
            sens["include_p5n128_unconstrained"] = (
                "P5N128 K/Q-bug data not retained in cleaned audit "
                "by pre-registered exclusion criterion; "
                "sensitivity-row not computed because including "
                "K/Q-buggy data would propagate the upstream bug "
                "downstream. The exclusion criterion is documented "
                "in K_Q_BUG_FIX_README and is artefact-based, not "
                "residual-based.")
        out["sensitivity_per_percentile"][pct] = sens

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print()
    header = (f"{'pct':>8s} | {'unc y_inf':>10s} | {'cons y_inf':>10s} | "
              f"{'huber y_inf':>11s} | LORO spread (min..max, std)")
    print(header)
    print("-" * 95)
    for pct in pct_keys:
        s = out["sensitivity_per_percentile"][pct]
        unc = s["cleaned_unconstrained"]["y_inf"]
        cons = s["cleaned_nonneg_constrained"]["y_inf"]
        hub = s["cleaned_huber_robust"]["y_inf"]
        lr = s["cleaned_LORO_spread"]
        if lr:
            spread = (f"({lr['y_inf_min']:+.3f}..{lr['y_inf_max']:+.3f}, "
                      f"std={lr['y_inf_std']:.3f})")
        else:
            spread = "(N/A)"
        print(f"{pct:>8s} | {unc:+10.4f} | {cons:+10.4f} | "
              f"{hub:+11.4f} | {spread}")


if __name__ == "__main__":
    main()
