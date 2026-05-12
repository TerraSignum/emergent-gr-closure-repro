"""Lemma B: branch-resolved Symanzik-1 fits for the spectral-gap
asymptote.

Reads the per-regime lambda_2 means produced by
`verify_lemma_B_uniform_poincare.py`
(outputs/verify_lemma_B_uniform_poincare.json) and separates the
ladder by chirality phase:

  vacuum    : N <= 100  (theta_chir < pi/4)
  peri-flip : 110 <= N <= 200  (theta_chir ~ pi/4, mixing zone)
  matter    : N >= 256  (theta_chir > pi/4, post-flip)

The flip itself occurs between N=100 (theta=37.1 deg) and N=128
(theta=46.1 deg), so N=128 and N=200 sit in the mixing zone and
are excluded from both pure-branch fits. The chirality-inversion
endpoint at N_inv ~ 591-600 (where theta -> arctan(N_gen) = 71.6 deg
and alpha_xi saturates) is a separate transition deeper inside the
matter branch.

Symanzik-1 form: lambda_2(N) = lambda_inf + a/N.

Output: outputs/verify_lemma_B_branch_resolved_fit.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "outputs" / "verify_lemma_B_uniform_poincare.json"
OUT = REPO / "outputs" / "verify_lemma_B_branch_resolved_fit.json"

# Chirality-phase boundaries (memory: project_chirality_flip_pi_over_4 2026-05-05).
VACUUM_N_MAX = 100        # last point with theta < pi/4
PERI_FLIP_N_LOW = 110     # first peri-flip N
PERI_FLIP_N_HIGH = 200    # last peri-flip N
MATTER_N_MIN = 256        # first stable post-flip N

# Conjectured asymptote
TARGET_RATIONAL = 3.0 / 8.0  # = (d-1)/(2d), d=4


def symanzik_1_fit(points: list[tuple[int, float]]) -> dict:
    """Least-squares lambda_2 = lambda_inf + a/N. Returns asymptote
    + slope + residuals."""
    if len(points) < 2:
        return {"status": "INSUFFICIENT", "n_points": len(points)}
    N = np.array([n for n, _ in points], dtype=float)
    L = np.array([l for _, l in points], dtype=float)
    X = np.column_stack([np.ones_like(N), 1.0 / N])
    coef, residuals, rank, _ = np.linalg.lstsq(X, L, rcond=None)
    lam_inf, a = float(coef[0]), float(coef[1])
    pred = lam_inf + a / N
    sse = float(((L - pred) ** 2).sum())
    rmse = float(np.sqrt(sse / len(points)))
    rel_to_target = (lam_inf - TARGET_RATIONAL) / TARGET_RATIONAL
    return {
        "status": "OK",
        "n_points": len(points),
        "lambda_inf": lam_inf,
        "a": a,
        "SSE": sse,
        "RMSE": rmse,
        "delta_vs_3_8_relative_pct": float(rel_to_target * 100),
        "points": [{"N": int(n), "lambda_2_mean": float(l)} for n, l in points],
    }


def main() -> int:
    if not IN.exists():
        print(f"ERROR: input not found: {IN}")
        return 1
    data = json.loads(IN.read_text(encoding="utf-8"))
    per_regime = [r for r in data["per_regime"] if r.get("status") == "OK"]
    if not per_regime:
        print("ERROR: no OK regimes in input")
        return 1

    pts: list[tuple[str, int, float]] = [
        (r["regime"], int(r["N"]), float(r["lambda_2_mean"]))
        for r in per_regime
    ]
    pts.sort(key=lambda x: x[1])

    vac = [(n, l) for _, n, l in pts if n <= VACUUM_N_MAX]
    peri = [(n, l) for _, n, l in pts
            if PERI_FLIP_N_LOW <= n <= PERI_FLIP_N_HIGH]
    mat = [(n, l) for _, n, l in pts if n >= MATTER_N_MIN]
    pooled = [(n, l) for _, n, l in pts]

    out = {
        "headline": (
            "Lemma B branch-resolved Symanzik-1 fit: vacuum (N<=100), "
            "matter (N>=256), pooled. Conjectured asymptote 3/8 = "
            f"{TARGET_RATIONAL}. Phase boundaries from "
            "project_chirality_flip_pi_over_4 2026-05-05."),
        "boundaries": {
            "vacuum_N_max": VACUUM_N_MAX,
            "peri_flip_N_low": PERI_FLIP_N_LOW,
            "peri_flip_N_high": PERI_FLIP_N_HIGH,
            "matter_N_min": MATTER_N_MIN,
            "N_inversion_approx": 591,
        },
        "vacuum": symanzik_1_fit(vac),
        "peri_flip": {
            "status": "BANDED",
            "n_points": len(peri),
            "points": [{"N": int(n), "lambda_2_mean": float(l)} for n, l in peri],
            "note": ("Peri-flip points are not fitted; they straddle "
                     "theta_chir = pi/4 mixing zone."),
        },
        "matter": symanzik_1_fit(mat),
        "pooled": symanzik_1_fit(pooled),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print_summary(out)
    return 0


def print_summary(out: dict) -> None:
    print("=" * 80)
    print("Lemma B branch-resolved Symanzik-1 fit")
    print("=" * 80)
    print(f"  target conjecture:    3/8 = {TARGET_RATIONAL:.6f}")
    print(f"  phase boundary:       vacuum N<={VACUUM_N_MAX} | matter N>={MATTER_N_MIN}")
    print()
    for label in ("vacuum", "matter", "pooled"):
        r = out[label]
        if r.get("status") == "OK":
            print(f"  {label:7s} ({r['n_points']} pts): "
                  f"lambda_inf = {r['lambda_inf']:.4f}, a = {r['a']:6.2f}, "
                  f"delta vs 3/8 = {r['delta_vs_3_8_relative_pct']:+6.2f}%, "
                  f"RMSE = {r['RMSE']:.4f}")
        else:
            print(f"  {label:7s}: status={r.get('status')}")
    p = out["peri_flip"]
    if p["points"]:
        np_str = ", ".join(f"N={pt['N']}:{pt['lambda_2_mean']:.4f}" for pt in p["points"])
        print(f"  peri-flip excluded ({p['n_points']} pts): {np_str}")
    print()
    print(f"  Output: {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    raise SystemExit(main())
