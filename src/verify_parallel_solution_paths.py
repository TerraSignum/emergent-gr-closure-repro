"""Five parallel data-only solution paths against the residual floor.

  (A) Bootstrap confidence intervals on Symanzik d_inf
  (G) Log-Symanzik fit:  d_inf + (c_2 + c_log * log N) / N^2
  (F) Constrained-positive Symanzik (d_inf >= 0 enforced)
  (M) Model validation: does Delta(N) - d_inf scale as N^(-2)?
  (S) Regime-physics correlation: does d_inf correlate with gamma, alpha_xi?

Goal: tighten the d_inf <= 0.05 claim with statistical CIs (A),
identify whether log-corrections explain sub-Symanzik exponents
(G), restore physical positivity for norm components (F), validate
the Symanzik form is appropriate (M), and rule out regime-physics
contamination (S).

Output: outputs/parallel_solution_paths_audit.json
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


def fit_symanzik_2_4(N, y, constrain_d_inf_nonneg=False):
    if len(N) < 4:
        return None
    X = np.column_stack([np.ones_like(N), N ** -2.0, N ** -4.0])
    if constrain_d_inf_nonneg:
        # NNLS-style: try unconstrained first, if d_inf < 0 then constrain to 0
        c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
        if c[0] < 0:
            # Refit with d_inf = 0
            X2 = X[:, 1:]
            c12, *_ = np.linalg.lstsq(X2, y, rcond=1e-10)
            c = np.array([0.0, c12[0], c12[1]])
    else:
        c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    pred = X @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"d_inf": float(c[0]), "c_2": float(c[1]), "c_4": float(c[2]),
            "R_squared": r2, "ss_res": ss_res}


def fit_log_symanzik(N, y):
    if len(N) < 4:
        return None
    X = np.column_stack([np.ones_like(N), N ** -2.0, np.log(N) * N ** -2.0])
    c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    pred = X @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"d_inf": float(c[0]), "c_2": float(c[1]), "c_log": float(c[2]),
            "R_squared": r2, "ss_res": ss_res}


def bootstrap_symanzik_d_inf(N, y, n_boot=1000, constrain=True):
    """Resample N points with replacement, fit Symanzik, return CI on d_inf."""
    rng = np.random.default_rng(seed=42)
    d_infs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(N), size=len(N))
        N_b = N[idx]; y_b = y[idx]
        if len(np.unique(N_b)) < 4:
            continue
        f = fit_symanzik_2_4(N_b, y_b, constrain_d_inf_nonneg=constrain)
        if f is not None:
            d_infs.append(f["d_inf"])
    d_infs = np.array(d_infs)
    return {
        "n_resamples": int(len(d_infs)),
        "mean": float(d_infs.mean()) if len(d_infs) else float("nan"),
        "median": float(np.median(d_infs)) if len(d_infs) else float("nan"),
        "ci_2_5": float(np.percentile(d_infs, 2.5)) if len(d_infs) else float("nan"),
        "ci_97_5": float(np.percentile(d_infs, 97.5)) if len(d_infs) else float("nan"),
        "std": float(d_infs.std()) if len(d_infs) else float("nan"),
    }


def main() -> int:
    audit = json.load(open(REPO / "outputs"
                           / "per_eigendirection_residual_audit.json", "r"))
    # Load full ladder if available
    full = REPO / "outputs" / "full_ladder_decomposition_audit.json"
    if full.exists():
        full_data = json.load(open(full, "r"))
        regimes_loaded = full_data["regimes_loaded"]
    else:
        regimes_loaded = [r["regime"] for r in audit["per_regime"]]

    # Re-gather per-regime values (full ladder if 11 pts, else 6 pts)
    from verify_per_eigendirection_residual import gather_regime
    ALL_REGIMES = [
        ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
        ("P4", 42), ("P5", 50), ("P6", 60), ("P5N64", 64),
        ("P7", 72), ("P8", 84), ("P5N100", 100),
    ]
    rows = []
    for reg, n_lat in ALL_REGIMES:
        try:
            r = gather_regime(reg, n_lat)
        except Exception:
            continue
        if r is None: continue
        rows.append(r)
    print(f"Loaded {len(rows)} regimes for parallel audit")

    N = np.array([r["N"] for r in rows], dtype=float)
    components = ["R_time_median_abs", "R_trace_median_abs",
                  "R_TF_norm_median_abs", "R_off_median_abs",
                  "R_time_mean_abs", "R_trace_mean_abs",
                  "R_TF_norm_mean_abs", "R_off_mean_abs"]

    out = {"method": "parallel_solution_paths_audit",
           "schema_version": "1.0.0",
           "n_pts": int(len(N))}

    # Path G: Log-Symanzik
    print()
    print("=" * 100)
    print("(G) Log-Symanzik fit comparison")
    print("=" * 100)
    print(f"{'comp':<22} {'sym d_inf':>11} {'sym R^2':>9} | {'log-sym d_inf':>15} {'log-sym R^2':>13} {'c_log':>10}")
    print("-" * 100)
    log_results = {}
    for comp in components:
        y = np.array([r[comp] for r in rows])
        sym = fit_symanzik_2_4(N, y)
        log_sym = fit_log_symanzik(N, y)
        log_results[comp] = {"symanzik": sym, "log_symanzik": log_sym}
        print(f"{comp:<22} {sym['d_inf']:>+11.5f} {sym['R_squared']:>9.3f} | "
              f"{log_sym['d_inf']:>+15.5f} {log_sym['R_squared']:>13.3f} "
              f"{log_sym['c_log']:>+10.2f}")
    out["path_G_log_symanzik"] = log_results

    # Path F: Constrained positive
    print()
    print("=" * 100)
    print("(F) Constrained-positive Symanzik (d_inf >= 0)")
    print("=" * 100)
    print(f"{'comp':<22} {'unconstr d_inf':>15} {'constr d_inf':>14} {'unconstr R^2':>13} {'constr R^2':>12}")
    print("-" * 100)
    constrained = {}
    for comp in components:
        y = np.array([r[comp] for r in rows])
        sym = fit_symanzik_2_4(N, y, constrain_d_inf_nonneg=False)
        sym_c = fit_symanzik_2_4(N, y, constrain_d_inf_nonneg=True)
        constrained[comp] = {"unconstrained": sym, "constrained": sym_c}
        print(f"{comp:<22} {sym['d_inf']:>+15.5f} {sym_c['d_inf']:>+14.5f} "
              f"{sym['R_squared']:>13.3f} {sym_c['R_squared']:>12.3f}")
    out["path_F_constrained_positive"] = constrained

    # Path A: Bootstrap CI
    print()
    print("=" * 100)
    print("(A) Bootstrap 95% CI on Symanzik d_inf (constrained, n_boot=1000)")
    print("=" * 100)
    print(f"{'comp':<22} {'d_inf':>10} {'95% CI':>20} {'std':>10}")
    print("-" * 80)
    bootstraps = {}
    for comp in components:
        y = np.array([r[comp] for r in rows])
        b = bootstrap_symanzik_d_inf(N, y, n_boot=1000, constrain=True)
        bootstraps[comp] = b
        print(f"{comp:<22} {b['median']:>10.5f} "
              f"[{b['ci_2_5']:>+8.5f}, {b['ci_97_5']:>+8.5f}] {b['std']:>10.5f}")
    out["path_A_bootstrap"] = bootstraps

    # Path M: residual scaling validation
    print()
    print("=" * 100)
    print("(M) Model validation: (Delta - d_inf) ~ N^(-alpha)?  (alpha~2 for Symanzik)")
    print("=" * 100)
    print(f"{'comp':<22} {'d_inf used':>12} {'fitted alpha':>15} {'fit R^2':>9}")
    print("-" * 70)
    val_results = {}
    for comp in components:
        y = np.array([r[comp] for r in rows])
        sym_c = fit_symanzik_2_4(N, y, constrain_d_inf_nonneg=True)
        d_inf = sym_c["d_inf"]
        residual = y - d_inf
        if np.all(residual > 0):
            log_N, log_r = np.log(N), np.log(residual)
            slope, intercept = np.polyfit(log_N, log_r, 1)
            alpha = -float(slope)
            pred = slope * log_N + intercept
            ss_res = float(((log_r - pred) ** 2).sum())
            ss_tot = float(((log_r - log_r.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            alpha = float("nan")
            r2 = float("nan")
        val_results[comp] = {"d_inf_used": d_inf, "fitted_alpha": alpha,
                             "R_squared_loglog": r2}
        print(f"{comp:<22} {d_inf:>12.5f} {alpha:>+15.3f} {r2:>9.3f}")
    out["path_M_validation"] = val_results

    # Path S: regime-physics correlation
    # Compute Pearson r between regime physics (gamma fixed at 0.1, alpha_xi varies?)
    # and per-regime |R_time_med|, etc.
    print()
    print("=" * 100)
    print("(S) Per-regime Delta values: how does Delta correlate with regime label?")
    print("=" * 100)
    print(f"{'regime':<10} {'N':>3} {'R_time_med':>11} {'R_trace_med':>12} {'R_TF_med':>10} {'R_off_med':>10}")
    print("-" * 70)
    regime_table = []
    for r in rows:
        print(f"{r['regime']:<10} {r['N']:>3} "
              f"{r['R_time_median_abs']:>11.5f} "
              f"{r['R_trace_median_abs']:>12.5f} "
              f"{r['R_TF_norm_median_abs']:>10.5f} "
              f"{r['R_off_median_abs']:>10.5f}")
        regime_table.append({k: r[k] for k in ["regime", "N",
            "R_time_median_abs", "R_trace_median_abs",
            "R_TF_norm_median_abs", "R_off_median_abs"]})
    # Sort by N and check monotonicity (proxy for regime-physics-vs-N)
    for comp in ["R_time_median_abs", "R_trace_median_abs",
                 "R_TF_norm_median_abs", "R_off_median_abs"]:
        ys = [r[comp] for r in rows]
        # Spearman correlation with N (rank)
        n_arr = N
        rN = np.argsort(np.argsort(n_arr))
        ry = np.argsort(np.argsort(ys))
        if np.std(rN) > 0 and np.std(ry) > 0:
            corr = float(np.corrcoef(rN, ry)[0, 1])
        else:
            corr = float("nan")
        print(f"  Spearman({comp:<22}, N) = {corr:+.3f}")
    out["path_S_regime_table"] = regime_table

    # Final verdict
    print()
    print("=" * 100)
    print("CONSOLIDATED VERDICT")
    print("=" * 100)
    # Use constrained Symanzik d_inf with bootstrap CI
    print()
    print("Median 95% CI for d_inf (constrained Symanzik, n_boot=1000):")
    threshold = 0.05
    n_pass = 0
    n_total = 0
    for comp in components:
        b = bootstraps[comp]
        if "med" in comp:
            n_total += 1
            ci_high = b["ci_97_5"]
            passing = ci_high <= threshold
            if passing: n_pass += 1
            print(f"  {comp:<22}: d_inf median {b['median']:.5f}, "
                  f"CI 95% upper {ci_high:.5f}  -> {'PASS' if passing else 'FAIL'} (<= {threshold})")
    print(f"\n  Median floor closure: {n_pass}/{n_total} components pass with 95% CI upper <= 0.05")

    out_path = REPO / "outputs" / "parallel_solution_paths_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
