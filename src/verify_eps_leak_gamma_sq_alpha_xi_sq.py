"""Test the structural form eps_leak = gamma^2 * alpha_xi^2
against lattice continuum extrapolation.

Hypothesis (NEW, derived from per-regime lattice data):
  eps_leak_continuum = gamma^2 * alpha_xi^2
                     = gamma^2 * Lambda_t_structural
                     = (1/100) * (81/100)
                     = 81/10000
                     = 0.00810

This is a clean System-R rational identification, NOT a Symbolic
Regression match. Both gamma and alpha_xi are System-R structural
constants.

Per-regime: compute eps_leak_N = G_00_med / T_00_med, then Symanzik
2-term continuum extrapolation. Bootstrap CI.

If continuum-limit eps_leak ~ gamma^2 * alpha_xi^2 with bootstrap
CI tight, this is a derived structural form.

Output: outputs/eps_leak_structural_form_audit.json
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


P5_REGIMES = [
    ("P5", 50), ("P5N64", 64), ("P5N72", 72),
    ("P5N84", 84), ("P5N100", 100),
]
ALL_REGIMES = [
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P5N72", 72), ("P7", 72), ("P5N84", 84), ("P8", 84),
    ("P5N100", 100),
]
GAMMA = 1.0/10.0
ALPHA_XI = 9.0/10.0
PRED_EPS_LEAK = GAMMA**2 * ALPHA_XI**2  # 0.00810
PRED_KAPPA_T  = 1.0 - PRED_EPS_LEAK     # 0.99190


def gather_eps_leak(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists(): return None
    d = np.load(p, allow_pickle=True)
    e = d["dense_cell_edge_xi_values"]
    a = d["dense_cell_node_amplitude_values"]
    ph = d["dense_cell_node_phase_values"]
    g_pool, t_pool = [], []
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
    g_med = float(np.median(g00)); t_med = float(np.median(t00))
    eps = g_med / t_med
    return {"regime": reg, "N": n_lat, "G_00_med": g_med,
            "T_00_med": t_med, "eps_leak": eps}


def fit_symanzik_2_with_bootstrap(N, y, n_boot=1000):
    rng = np.random.default_rng(seed=42)
    # Direct fit
    X = np.column_stack([np.ones_like(N), N**-2.0])
    c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    y_inf = float(c[0])
    # Bootstrap
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(N), size=len(N))
        if len(np.unique(N[idx])) < 2: continue
        X_b = np.column_stack([np.ones(len(idx)), N[idx]**-2.0])
        try:
            c_b, *_ = np.linalg.lstsq(X_b, y[idx], rcond=1e-10)
            samples.append(c_b[0])
        except np.linalg.LinAlgError:
            continue
    samples = np.array(samples)
    return {
        "y_inf_central": y_inf,
        "y_inf_median_boot": float(np.median(samples)),
        "ci_2_5":  float(np.percentile(samples, 2.5)),
        "ci_97_5": float(np.percentile(samples, 97.5)),
        "std":     float(samples.std()),
        "n_boot":  int(len(samples)),
    }


def main() -> int:
    print("=" * 100)
    print(f"Structural form: eps_leak = gamma^2 * alpha_xi^2 = (1/100)*(81/100) = 81/10000 = {PRED_EPS_LEAK:.5f}")
    print(f"                 kappa_t   = 1 - eps_leak                                   = {PRED_KAPPA_T:.5f}")
    print("=" * 100)
    print()
    print("--- P5-physics within-regime sequence ---")
    p5_rows = []
    for reg, n_lat in P5_REGIMES:
        r = gather_eps_leak(reg, n_lat)
        if r is None: continue
        p5_rows.append(r)
        print(f"  {reg:<10} N={n_lat:>3}: G_00={r['G_00_med']:.5f}, T_00={r['T_00_med']:.5f}, "
              f"eps_leak={r['eps_leak']:.5f}")

    if len(p5_rows) >= 4:
        N_arr = np.array([r["N"] for r in p5_rows], dtype=float)
        eps_arr = np.array([r["eps_leak"] for r in p5_rows])
        boot = fit_symanzik_2_with_bootstrap(N_arr, eps_arr, n_boot=2000)
        print()
        print(f"  Symanzik 2-term continuum on P5-only ({len(p5_rows)} pts):")
        print(f"    eps_leak_inf central = {boot['y_inf_central']:.6f}")
        print(f"    bootstrap median     = {boot['y_inf_median_boot']:.6f}")
        print(f"    95% CI               = [{boot['ci_2_5']:.6f}, {boot['ci_97_5']:.6f}]")
        rel_err_central = (boot['y_inf_central'] - PRED_EPS_LEAK) / PRED_EPS_LEAK * 100
        print(f"    Predicted gamma^2*alpha_xi^2 = {PRED_EPS_LEAK:.6f}")
        print(f"    Rel error central vs predicted: {rel_err_central:+.3f}%")
        # CI containment
        contains = boot['ci_2_5'] <= PRED_EPS_LEAK <= boot['ci_97_5']
        print(f"    Predicted within bootstrap 95% CI: {'YES' if contains else 'NO'}")

    print()
    print("--- Full canonical ladder (cross-regime, N >= 28) ---")
    full_rows = []
    for reg, n_lat in ALL_REGIMES:
        if n_lat < 28: continue
        r = gather_eps_leak(reg, n_lat)
        if r is None: continue
        full_rows.append(r)
        print(f"  {reg:<10} N={n_lat:>3}: eps_leak={r['eps_leak']:.5f}")

    if len(full_rows) >= 4:
        N_arr = np.array([r["N"] for r in full_rows], dtype=float)
        eps_arr = np.array([r["eps_leak"] for r in full_rows])
        boot_full = fit_symanzik_2_with_bootstrap(N_arr, eps_arr, n_boot=2000)
        print()
        print(f"  Symanzik 2-term continuum on full ladder ({len(full_rows)} pts, N>=28):")
        print(f"    eps_leak_inf central = {boot_full['y_inf_central']:.6f}")
        print(f"    bootstrap median     = {boot_full['y_inf_median_boot']:.6f}")
        print(f"    95% CI               = [{boot_full['ci_2_5']:.6f}, {boot_full['ci_97_5']:.6f}]")
        rel_err_central = (boot_full['y_inf_central'] - PRED_EPS_LEAK) / PRED_EPS_LEAK * 100
        print(f"    Predicted gamma^2*alpha_xi^2 = {PRED_EPS_LEAK:.6f}")
        print(f"    Rel error central vs predicted: {rel_err_central:+.3f}%")
        contains_full = boot_full['ci_2_5'] <= PRED_EPS_LEAK <= boot_full['ci_97_5']
        print(f"    Predicted within bootstrap 95% CI: {'YES' if contains_full else 'NO'}")

    # Verdict
    print()
    print("=" * 100)
    if len(p5_rows) >= 4:
        if abs(rel_err_central) < 5 and contains:
            verdict = "STRUCTURAL_FORM_eps_leak_eq_gamma_sq_alpha_xi_sq_VERIFIED"
        elif abs(rel_err_central) < 15:
            verdict = "STRUCTURAL_FORM_PARTIALLY_SUPPORTED"
        else:
            verdict = "STRUCTURAL_FORM_NOT_SUPPORTED"
    else:
        verdict = "INSUFFICIENT_DATA"
    print(f"VERDICT: {verdict}")

    out_path = REPO / "outputs" / "eps_leak_structural_form_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "structural_form_eps_leak_gamma_sq_alpha_xi_sq_test",
            "schema_version": "1.0.0",
            "predicted_eps_leak": PRED_EPS_LEAK,
            "predicted_kappa_t": PRED_KAPPA_T,
            "predicted_form": "gamma^2 * alpha_xi^2 = (1/100) * (81/100) = 81/10000",
            "p5_regimes": p5_rows,
            "full_regimes_N28plus": full_rows,
            "p5_continuum_extrap": boot if len(p5_rows) >= 4 else None,
            "full_continuum_extrap": boot_full if len(full_rows) >= 4 else None,
            "verdict": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
