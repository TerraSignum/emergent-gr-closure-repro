"""Direct closure test of the three spatial principal-axis equations:

  G_(ii)(a) + Lambda_i - 8 pi G lambda_i(a) -> 0  for i=1,2,3.

Per regime, in the T-eigenframe (sorted by T-eigvals ascending),
extract:
  R_diag1(a) = G_(11) + Lambda_s - lambda_1(a)
  R_diag2(a) = G_(22) + Lambda_s - lambda_2(a)
  R_diag3(a) = G_(33) + Lambda_s - lambda_3(a)

Then Symanzik 2+4 fit each separately on the full 11-point ladder
(or per-regime within-regime sequences). Each fit returns:
  d_inf_i, c_2_i, c_4_i, R^2_i

If d_inf_i for i=1,2,3 all <= 0.05 with constrained-positive fits,
the three principal-axis equations close independently — completing
the spatial-tensor closure beyond the trace+TF decomposition.

Output: outputs/three_principal_axis_closure_audit.json
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

from verify_per_eigendirection_residual import gather_regime


ALL_REGIMES = [
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]


def fit_symanzik_2_4(N, y, constrain=True):
    if len(N) < 4:
        return None
    X = np.column_stack([np.ones_like(N), N ** -2.0, N ** -4.0])
    c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    if constrain and c[0] < 0:
        X2 = X[:, 1:]
        c12, *_ = np.linalg.lstsq(X2, y, rcond=1e-10)
        c = np.array([0.0, c12[0], c12[1]])
    pred = X @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"d_inf": float(c[0]), "c_2": float(c[1]), "c_4": float(c[2]),
            "R_squared": r2}


def bootstrap_symanzik_d_inf(N, y, n_boot=1000):
    rng = np.random.default_rng(seed=42)
    d_infs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(N), size=len(N))
        N_b = N[idx]; y_b = y[idx]
        if len(np.unique(N_b)) < 4:
            continue
        f = fit_symanzik_2_4(N_b, y_b, constrain=True)
        if f is not None:
            d_infs.append(f["d_inf"])
    d_infs = np.array(d_infs)
    if len(d_infs) == 0:
        return {"mean": float("nan"), "ci_2_5": float("nan"), "ci_97_5": float("nan")}
    return {
        "n_resamples": int(len(d_infs)),
        "mean": float(d_infs.mean()),
        "median": float(np.median(d_infs)),
        "ci_2_5": float(np.percentile(d_infs, 2.5)),
        "ci_97_5": float(np.percentile(d_infs, 97.5)),
    }


def main() -> int:
    rows = []
    for reg, n_lat in ALL_REGIMES:
        try:
            r = gather_regime(reg, n_lat)
        except Exception:
            continue
        if r is None: continue
        rows.append(r)
    print(f"Loaded {len(rows)} regimes")

    N = np.array([r["N"] for r in rows], dtype=float)

    # The per_eigendirection audit gives R_diag1, R_diag2, R_diag3
    # as median absolute residuals per direction
    components = ["R_diag1_median_abs", "R_diag2_median_abs",
                  "R_diag3_median_abs",
                  "R_diag1_mean_abs", "R_diag2_mean_abs",
                  "R_diag3_mean_abs"]

    print()
    print("=" * 100)
    print("Three-principal-axis closure: G_(ii)(a) + Lambda_s - lambda_i(a) -> 0, i=1,2,3")
    print("=" * 100)
    print()
    print(f"{'reg':<10} {'N':>3} | "
          f"{'R_diag1_med':>12} {'R_diag2_med':>12} {'R_diag3_med':>12} | "
          f"{'R_diag1_mean':>13} {'R_diag2_mean':>13} {'R_diag3_mean':>13}")
    print("-" * 100)
    for r in rows:
        print(f"{r['regime']:<10} {r['N']:>3} | "
              f"{r['R_diag1_median_abs']:>12.5f} "
              f"{r['R_diag2_median_abs']:>12.5f} "
              f"{r['R_diag3_median_abs']:>12.5f} | "
              f"{r['R_diag1_mean_abs']:>13.5f} "
              f"{r['R_diag2_mean_abs']:>13.5f} "
              f"{r['R_diag3_mean_abs']:>13.5f}")

    print()
    print("Symanzik 2+4 (constrained-positive) fits per principal axis:")
    print(f"{'observable':<22} {'d_inf':>10} {'c_2':>10} {'c_4':>12} {'R^2':>7}")
    print("-" * 70)
    fits = {}
    bootstraps = {}
    for comp in components:
        y = np.array([r[comp] for r in rows])
        f = fit_symanzik_2_4(N, y, constrain=True)
        b = bootstrap_symanzik_d_inf(N, y, n_boot=1000)
        fits[comp] = f
        bootstraps[comp] = b
        print(f"{comp:<22} {f['d_inf']:>10.5f} {f['c_2']:>+10.2f} "
              f"{f['c_4']:>+12.1f} {f['R_squared']:>7.3f}")

    print()
    print("Bootstrap 95% CI on d_inf per principal axis:")
    print(f"{'observable':<22} {'median':>10} {'95% CI':>22}")
    print("-" * 60)
    for comp in components:
        b = bootstraps[comp]
        print(f"{comp:<22} {b['median']:>10.5f} "
              f"[{b['ci_2_5']:>+8.5f}, {b['ci_97_5']:>+8.5f}]")

    # Closure verdicts: each axis median 95% CI Upper <= 0.05
    print()
    print("=" * 100)
    print("THREE-AXIS CLOSURE VERDICT")
    print("=" * 100)
    n_pass = 0
    n_total = 0
    for i in (1, 2, 3):
        comp = f"R_diag{i}_median_abs"
        b = bootstraps[comp]
        n_total += 1
        passes = b["ci_97_5"] <= 0.05
        if passes: n_pass += 1
        print(f"  Axis i={i}: d_inf median {b['median']:.5f}, "
              f"95% CI Upper {b['ci_97_5']:.5f}  -> "
              f"{'PASS' if passes else 'FAIL'} (<= 0.05)")

    if n_pass == n_total:
        verdict = "ALL_THREE_PRINCIPAL_AXES_CLOSE"
    elif n_pass >= 2:
        verdict = f"{n_pass}_OF_{n_total}_PRINCIPAL_AXES_CLOSE"
    else:
        verdict = "PRINCIPAL_AXIS_CLOSURE_OPEN"
    print(f"\nVERDICT: {verdict}")

    out_path = REPO / "outputs" / "three_principal_axis_closure_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "three_principal_axis_symanzik_with_bootstrap",
            "schema_version": "1.0.0",
            "regimes_loaded": [r["regime"] for r in rows],
            "n_pts": int(len(N)),
            "fits": fits,
            "bootstraps": bootstraps,
            "verdict": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
