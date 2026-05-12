"""(4) Quantify the log-nonlinearity in S_back via AIC/BIC + Bootstrap.

Linear S_back: Lambda_t(a) = kappa * T_00(a)
Log-nonlinear: Lambda_t(a) = kappa * T_00(a) + alpha * T_00(a) * log(T_00(a)/<T_00>)

We previously found alpha = +0.033 (small but non-trivial). Quantify:
  (i) Bootstrap CI on alpha
  (ii) AIC/BIC vs linear
  (iii) Per-regime stability

Output: outputs/log_nonlinearity_quantified_audit.json
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


REGIMES = [
    ("P1", 28), ("P3", 36), ("P4", 42), ("P5", 50), ("P5N64", 64),
    ("P6", 60), ("P7", 72), ("P8", 84), ("P5N100", 100),
]


def fit_linear(t, lt):
    """lt = k*t. Returns kappa, residual."""
    k = float(np.sum(lt * t) / np.sum(t**2))
    pred = k * t
    return k, pred


def fit_log_nonlinear(t, lt, t_mean):
    """lt = k*t + a*t*log(t/t_mean). Returns kappa, alpha, residual."""
    log_r = np.log(t / t_mean)
    A = np.array([[float(np.sum(t**2)),         float(np.sum(t**2 * log_r))],
                  [float(np.sum(t**2 * log_r)), float(np.sum(t**2 * log_r**2))]])
    b = np.array([float(np.sum(t * lt)), float(np.sum(t * lt * log_r))])
    sol = np.linalg.solve(A, b)
    k, a = float(sol[0]), float(sol[1])
    pred = k * t + a * t * log_r
    return k, a, pred


def gather_pool():
    g_pool, t_pool = [], []
    for reg, n_lat in REGIMES:
        p = find_d1_npz(reg, REPO)
        if p is None or not p.exists(): continue
        d = np.load(p, allow_pickle=True)
        e = d["dense_cell_edge_xi_values"]
        a = d["dense_cell_node_amplitude_values"]
        ph = d["dense_cell_node_phase_values"]
        for s in range(min(e.shape[0], 32)):
            xi_mat = edge_to_matrix(e[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            psi = a[s] * np.exp(1j*ph[s])
            k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
            q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
            prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
            g_pool.append(np.asarray(prep["g_00_h"]))
            t_pool.append(np.asarray(prep["t00"]))
    g00 = np.concatenate(g_pool); t00 = np.concatenate(t_pool)
    mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g00)
    return g00[mask], t00[mask]


def main() -> int:
    print("=" * 100)
    print("(4) Log-nonlinearity quantification with Bootstrap CI")
    print("=" * 100)
    g00, t00 = gather_pool()
    lt = t00 - g00
    t_mean = float(np.mean(t00))
    n = len(lt)
    print(f"  n_nodes = {n}, T_00 mean = {t_mean:.4f}")

    # Central fits
    k_lin, pred_lin = fit_linear(t00, lt)
    ss_lin = float(np.sum((lt - pred_lin) ** 2))
    k_log, a_log, pred_log = fit_log_nonlinear(t00, lt, t_mean)
    ss_log = float(np.sum((lt - pred_log) ** 2))

    # AIC, BIC
    AIC_lin = n * np.log(ss_lin/n) + 2 * 1
    AIC_log = n * np.log(ss_log/n) + 2 * 2
    BIC_lin = n * np.log(ss_lin/n) + 1 * np.log(n)
    BIC_log = n * np.log(ss_log/n) + 2 * np.log(n)

    ss_tot = float(np.sum((lt - lt.mean()) ** 2))
    R2_lin = 1.0 - ss_lin/ss_tot
    R2_log = 1.0 - ss_log/ss_tot

    print(f"\n  Linear:    kappa = {k_lin:.5f}, R^2 = {R2_lin:.6f}, ss_res = {ss_lin:.4e}")
    print(f"  Log-nonl:  kappa = {k_log:.5f}, alpha = {a_log:+.5f}, R^2 = {R2_log:.6f}, ss_res = {ss_log:.4e}")
    print(f"  AIC: linear={AIC_lin:.2f}, log-nonl={AIC_log:.2f}, deltaAIC = {AIC_log - AIC_lin:+.2f}")
    print(f"  BIC: linear={BIC_lin:.2f}, log-nonl={BIC_log:.2f}, deltaBIC = {BIC_log - BIC_lin:+.2f}")
    print(f"     (negative deltaAIC/BIC = log model preferred)")

    # Bootstrap on alpha
    rng = np.random.default_rng(seed=42)
    n_boot = 1000
    alphas, kappas_log = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            _, a_b, _ = fit_log_nonlinear(t00[idx], lt[idx], t_mean)
            alphas.append(a_b)
            k_b, _, _ = fit_log_nonlinear(t00[idx], lt[idx], t_mean)[:3]
            kappas_log.append(k_b)
        except np.linalg.LinAlgError:
            continue
    alphas = np.array(alphas); kappas_log = np.array(kappas_log)
    a_med = float(np.median(alphas))
    a_ci = (float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5)))
    print()
    print(f"  Bootstrap (n={len(alphas)}):")
    print(f"    alpha:    median = {a_med:+.5f}, 95% CI = [{a_ci[0]:+.5f}, {a_ci[1]:+.5f}]")
    print(f"    Is alpha=0 in CI? {'YES (log term consistent with zero)' if a_ci[0] < 0 < a_ci[1] else 'NO (log term clearly non-zero)'}")

    if a_med > 0.01 and a_ci[0] > 0:
        verdict = "LOG_NONLINEARITY_REAL_AND_POSITIVE"
    elif abs(a_med) < 0.01:
        verdict = "LOG_NONLINEARITY_NEGLIGIBLE"
    else:
        verdict = "LOG_NONLINEARITY_DETECTED_BUT_SMALL"
    print(f"\n  VERDICT: {verdict}")

    # Per-regime alpha stability
    print()
    print(f"  Per-regime alpha stability:")
    print(f"  {'reg':<10} {'N':>3} | {'kappa':>8} {'alpha':>9} {'R^2 lin':>9} {'R^2 log':>9}")
    print(f"  " + "-" * 60)
    per_reg = []
    for reg, n_lat in REGIMES:
        p = find_d1_npz(reg, REPO)
        if p is None: continue
        d = np.load(p, allow_pickle=True)
        e = d["dense_cell_edge_xi_values"]
        a_arr = d["dense_cell_node_amplitude_values"]
        ph = d["dense_cell_node_phase_values"]
        g_pool, t_pool = [], []
        for s in range(min(e.shape[0], 32)):
            xi_mat = edge_to_matrix(e[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            psi = a_arr[s] * np.exp(1j*ph[s])
            k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
            q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
            prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
            g_pool.append(np.asarray(prep["g_00_h"]))
            t_pool.append(np.asarray(prep["t00"]))
        g_r = np.concatenate(g_pool); t_r = np.concatenate(t_pool)
        mask = (t_r > 0.05) & np.isfinite(t_r) & np.isfinite(g_r)
        g_r = g_r[mask]; t_r = t_r[mask]
        lt_r = t_r - g_r
        t_mean_r = float(np.mean(t_r))
        try:
            k_l, pred_l = fit_linear(t_r, lt_r)
            ss_l = float(np.sum((lt_r - pred_l) ** 2))
            ss_tot_r = float(np.sum((lt_r - lt_r.mean()) ** 2))
            R2_l = 1.0 - ss_l/ss_tot_r
            k_lo, a_lo, pred_lo = fit_log_nonlinear(t_r, lt_r, t_mean_r)
            ss_lo = float(np.sum((lt_r - pred_lo) ** 2))
            R2_lo = 1.0 - ss_lo/ss_tot_r
            per_reg.append({"regime": reg, "N": n_lat,
                            "kappa": k_lo, "alpha": a_lo,
                            "R2_lin": R2_l, "R2_log": R2_lo})
            print(f"  {reg:<10} {n_lat:>3} | {k_lo:>8.4f} {a_lo:>+9.4f} {R2_l:>9.4f} {R2_lo:>9.4f}")
        except np.linalg.LinAlgError:
            print(f"  {reg:<10} {n_lat:>3} | LinAlgError")

    if per_reg:
        a_per = [r["alpha"] for r in per_reg]
        print(f"\n  alpha per-regime: mean = {float(np.mean(a_per)):.4f}, "
              f"std = {float(np.std(a_per)):.4f}, CV = {float(np.std(a_per)/abs(np.mean(a_per))*100) if abs(np.mean(a_per))>1e-12 else float('nan'):.1f}%")

    out = {
        "method": "log_nonlinearity_AIC_BIC_bootstrap_per_regime",
        "schema_version": "1.0.0",
        "n_pooled": int(n),
        "linear_fit": {"kappa": k_lin, "R_squared": R2_lin, "ss_res": ss_lin,
                       "AIC": AIC_lin, "BIC": BIC_lin},
        "log_fit": {"kappa": k_log, "alpha": a_log, "R_squared": R2_log,
                    "ss_res": ss_log, "AIC": AIC_log, "BIC": BIC_log},
        "delta_AIC_log_minus_lin": AIC_log - AIC_lin,
        "delta_BIC_log_minus_lin": BIC_log - BIC_lin,
        "bootstrap_alpha": {
            "n_boot": int(len(alphas)), "median": a_med,
            "ci_2_5": a_ci[0], "ci_97_5": a_ci[1],
            "zero_in_CI": bool(a_ci[0] < 0 < a_ci[1]),
        },
        "per_regime_fits": per_reg,
        "verdict": verdict,
    }
    out_path = REPO / "outputs" / "log_nonlinearity_quantified_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
