"""Decompose what makes the Delta_inf floor.

Per N, split the per-direction relative Frobenius residual into its
four covariant components and Symanzik-fit each separately:

  R_time(a)   : G_00 + Lambda_t - 8 pi G T_00
  R_trace(a)  : (G_(11) + G_(22) + G_(33))/3 + Lambda_s - (lambda_1+lambda_2+lambda_3)/3
  R_TF(a)     : traceless spatial diagonal
  R_off(a)    : off-diagonal Frobenius

Each gets:
  ||component||_med(N) = d_inf + c_2/N^2 + c_4/N^4

The component with the LARGEST d_inf is the SOURCE of the floor.
We then know exactly which physical quantity to target.

Output: outputs/delta_floor_decomposition_audit.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def fit_symanzik_2_4(N, y):
    X = np.column_stack([np.ones_like(N), N ** -2.0, N ** -4.0])
    c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    pred = X @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"d_inf": float(c[0]), "c_2": float(c[1]), "c_4": float(c[2]),
            "R_squared": r2,
            "predicted_at_N_1000": float(c[0] + c[1] / 1000 ** 2 + c[2] / 1000 ** 4)}


def main() -> int:
    audit = json.load(open(REPO / "outputs"
                           / "per_eigendirection_residual_audit.json", "r"))
    rows = [r for r in audit["per_regime"] if r["N"] >= 60]
    N = np.array([r["N"] for r in rows], dtype=float)

    components = {
        "R_time_med":      np.array([r["R_time_median_abs"] for r in rows]),
        "R_trace_med":     np.array([r["R_trace_median_abs"] for r in rows]),
        "R_TF_med":        np.array([r["R_TF_norm_median_abs"] for r in rows]),
        "R_off_med":       np.array([r["R_off_median_abs"] for r in rows]),
        "R_time_mean":     np.array([r["R_time_mean_abs"] for r in rows]),
        "R_trace_mean":    np.array([r["R_trace_mean_abs"] for r in rows]),
        "R_TF_mean":       np.array([r["R_TF_norm_mean_abs"] for r in rows]),
        "R_off_mean":      np.array([r["R_off_mean_abs"] for r in rows]),
    }

    print("=" * 110)
    print("Delta floor decomposition: which component carries the residual floor?")
    print("=" * 110)
    print(f"{'component':<18} | " + " | ".join([f"N={int(n)}" for n in N]) +
          " || d_inf  c_2     c_4     R^2   N=1000")
    print("-" * 130)

    fits = {}
    for k, y in components.items():
        f = fit_symanzik_2_4(N, y)
        fits[k] = f
        vals = " | ".join([f"{v:.4f}" for v in y])
        print(f"{k:<18} | {vals} || "
              f"{f['d_inf']:.4f} {f['c_2']:>+8.1f} {f['c_4']:>+8.1f} "
              f"{f['R_squared']:.3f} {f['predicted_at_N_1000']:.4f}")

    # Identify the dominant floor contribution
    print()
    med_floors = {k: v["d_inf"] for k, v in fits.items() if "med" in k}
    mean_floors = {k: v["d_inf"] for k, v in fits.items() if "mean" in k}
    print(f"MEDIAN floor by component: ")
    for k, v in sorted(med_floors.items(), key=lambda x: -abs(x[1])):
        print(f"   {k:<14}  d_inf = {v:+.5f}")
    print(f"MEAN floor by component:")
    for k, v in sorted(mean_floors.items(), key=lambda x: -abs(x[1])):
        print(f"   {k:<14}  d_inf = {v:+.5f}")

    # The Frobenius combines all four; predict assembled floor
    # |R|_F^2 = R_time^2 + R_trace^2 * 3 + R_TF^2 + R_off^2
    # For median we use median**2 as proxy magnitude
    asm_med = (fits["R_time_med"]["d_inf"] ** 2
               + 3.0 * fits["R_trace_med"]["d_inf"] ** 2
               + fits["R_TF_med"]["d_inf"] ** 2
               + fits["R_off_med"]["d_inf"] ** 2) ** 0.5
    asm_mean = (fits["R_time_mean"]["d_inf"] ** 2
                + 3.0 * fits["R_trace_mean"]["d_inf"] ** 2
                + fits["R_TF_mean"]["d_inf"] ** 2
                + fits["R_off_mean"]["d_inf"] ** 2) ** 0.5
    print()
    print(f"Assembled Frobenius floor (sqrt of squared sums):")
    print(f"  median: {asm_med:.5f}")
    print(f"  mean:   {asm_mean:.5f}")

    # Bottleneck identification
    print()
    print("BOTTLENECK component (largest |d_inf|):")
    print(f"  median: {max(med_floors, key=lambda k: abs(med_floors[k]))}")
    print(f"  mean:   {max(mean_floors, key=lambda k: abs(mean_floors[k]))}")

    out_path = REPO / "outputs" / "delta_floor_decomposition_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "per_component_symanzik_fit_floor_decomposition",
            "schema_version": "1.0.0",
            "fits": fits,
            "assembled_floor": {"median": asm_med, "mean": asm_mean},
            "median_bottleneck": max(med_floors, key=lambda k: abs(med_floors[k])),
            "mean_bottleneck": max(mean_floors, key=lambda k: abs(mean_floors[k])),
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
