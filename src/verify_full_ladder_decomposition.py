"""Full canonical lattice ladder decomposition: extend to all
available D1 NPZ regimes (N=18..100, 11 points) and re-fit
the per-component Symanzik form.

With ~10-11 N-points instead of 5, the 3-parameter Symanzik fit
becomes statistically meaningful. The per-component breakdown
identifies whether R_time, R_trace, R_TF, or R_off carries the
residual floor structurally or whether it was a 5-point noise
artefact.

Output: outputs/full_ladder_decomposition_audit.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, name):
        raise ImportError("cupy disabled")

sys.meta_path.insert(0, _BlockCupy())

from _d1_npz_discovery import find_d1_npz
from verify_galerkin_runner_A_hessian_ricci import (
    edge_to_matrix, per_seed_galerkin)
from verify_per_eigendirection_residual import (
    per_node_eigendirection_residuals, gather_regime)


# Try ALL canonical-ladder regimes
ALL_REGIMES = [
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]


def fit_symanzik_2_4(N, y):
    if len(N) < 4:
        return None
    X = np.column_stack([np.ones_like(N), N ** -2.0, N ** -4.0])
    c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    pred = X @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"d_inf": float(c[0]), "c_2": float(c[1]), "c_4": float(c[2]),
            "R_squared": r2,
            "predicted_at_N_1000": float(c[0] + c[1] / 1000 ** 2 + c[2] / 1000 ** 4)}


def fit_symanzik_2(N, y):
    if len(N) < 3:
        return None
    X = np.column_stack([np.ones_like(N), N ** -2.0])
    c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    pred = X @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"d_inf": float(c[0]), "c_2": float(c[1]),
            "R_squared": r2,
            "predicted_at_N_1000": float(c[0] + c[1] / 1000 ** 2)}


def main() -> int:
    print("=" * 110)
    print("Full canonical-ladder per-component decomposition")
    print("=" * 110)
    print()

    rows = []
    for reg, n_lat in ALL_REGIMES:
        try:
            r = gather_regime(reg, n_lat)
        except Exception as e:
            print(f"  {reg:<8} N={n_lat}: ERROR: {e}")
            continue
        if r is None:
            print(f"  {reg:<8} N={n_lat}: no NPZ found")
            continue
        rows.append(r)

    print(f"{'reg':<8} {'N':>3} | "
          f"{'R_time_med':>11} {'R_trace_med':>12} {'R_TF_med':>10} {'R_off_med':>11} | "
          f"{'R_time_mean':>12} {'R_trace_mean':>13} {'R_TF_mean':>11} {'R_off_mean':>12}")
    print("-" * 130)
    for r in rows:
        print(f"{r['regime']:<8} {r['N']:>3} | "
              f"{r['R_time_median_abs']:>11.5f} {r['R_trace_median_abs']:>12.5f} "
              f"{r['R_TF_norm_median_abs']:>10.5f} {r['R_off_median_abs']:>11.5f} | "
              f"{r['R_time_mean_abs']:>12.5f} {r['R_trace_mean_abs']:>13.5f} "
              f"{r['R_TF_norm_mean_abs']:>11.5f} {r['R_off_mean_abs']:>12.5f}")
    print(f"\nTotal points: {len(rows)}")

    # Symanzik fits per component, with various N-cuts
    components = ["R_time_median_abs", "R_trace_median_abs",
                  "R_TF_norm_median_abs", "R_off_median_abs",
                  "R_time_mean_abs", "R_trace_mean_abs",
                  "R_TF_norm_mean_abs", "R_off_mean_abs"]
    cuts = [(0, "all"), (28, "N>=28"), (42, "N>=42"), (60, "N>=60")]

    print()
    print("Symanzik 2+4 (3-param) fits per cut:")
    print(f"{'component':<22} {'cut':<8} {'N pts':>6} {'d_inf':>10} {'R^2':>7} {'pred N=1000':>13}")
    print("-" * 80)
    fits = {}
    for cut, label in cuts:
        cut_rows = [r for r in rows if r["N"] >= cut]
        N_arr = np.array([r["N"] for r in cut_rows], dtype=float)
        for comp in components:
            y = np.array([r[comp] for r in cut_rows])
            f = fit_symanzik_2_4(N_arr, y)
            if f is None: continue
            print(f"{comp:<22} {label:<8} {len(N_arr):>6} {f['d_inf']:>+10.5f} "
                  f"{f['R_squared']:>7.3f} {f['predicted_at_N_1000']:>13.5f}")
            fits[f"{comp}_{label}"] = f

    print()
    print("Symanzik 2 only (2-param) fits, all N:")
    print(f"{'component':<22} {'N pts':>6} {'d_inf':>10} {'R^2':>7} {'pred N=1000':>13}")
    print("-" * 70)
    fits_s2 = {}
    N_all = np.array([r["N"] for r in rows], dtype=float)
    for comp in components:
        y = np.array([r[comp] for r in rows])
        f = fit_symanzik_2(N_all, y)
        if f is None: continue
        print(f"{comp:<22} {len(N_all):>6} {f['d_inf']:>+10.5f} "
              f"{f['R_squared']:>7.3f} {f['predicted_at_N_1000']:>13.5f}")
        fits_s2[comp] = f

    out_path = REPO / "outputs" / "full_ladder_decomposition_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "full_ladder_per_component_symanzik_decomposition",
            "schema_version": "1.0.0",
            "regimes_loaded": [r["regime"] for r in rows],
            "n_total_points": len(rows),
            "fits_2_4_per_cut": fits,
            "fits_2_only_all_N": fits_s2,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
