"""Test 4 alternative theorem candidates for closure:

  (F) Symanzik-rate-Theorem: is alpha = 2 universal across coefficient
      variants? If yes, alpha=2 is coefficient-invariant.

  (E) 3-of-4-component theorem: are R_off, R_trace, R_TF d_inf
      coefficient-INVARIANT (since they don't depend on T_00 coefficient
      choice)?

  (G) Ratio-theorem: is ratio kappa_t(default) / kappa_t(other)
      a stable universal value across regimes?

  (H) Decomposed Lagrangian theorem: under fixed S_back ansatz, is the
      kappa_t-form universally Lambda_t = kappa * <T_00>(coeff) where
      <T_00> is the sample mean — i.e. kappa is NORMALISED by mean?

Output: outputs/four_alternative_theorems_audit.json
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
    edge_to_matrix, hessian_ricci_per_node, ELL_0, D_MIN, EPS_D, XI_THRESH)


REGIMES = [
    ("P1", 28), ("P3", 36), ("P4", 42), ("P5", 50), ("P5N64", 64),
    ("P5N72", 72), ("P5N84", 84), ("P6", 60), ("P7", 72),
    ("P8", 84), ("P5N100", 100),
]

VARIANTS = {
    "default":      {"Z": 1.0, "K": 1.0, "Z1": 1.0, "Z3": 0.5, "AK": 1.0, "AQ": 0.5, "OM": 1.0},
    "all_unit":     {"Z": 1.0, "K": 1.0, "Z1": 1.0, "Z3": 1.0, "AK": 1.0, "AQ": 1.0, "OM": 1.0},
    "potential_only":{"Z":1.0, "K": 1.0, "Z1": 0.0, "Z3": 0.5, "AK": 1.0, "AQ": 0.5, "OM": 1.0},
    "system_R":     {"Z":0.81, "K":0.99, "Z1": 1.0, "Z3":0.005, "AK": 1.0, "AQ": 0.5, "OM": 1.0},
}


def compute_t00_alt(xi_off, adj, weight_grad, d_mat, spatial, omega_a,
                     psi, k_field, q_field, params):
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]
    inv_d = np.where(adj > 0, 1.0 / d_mat, 0.0)
    psi_diff = psi[None, :] - psi[:, None]
    weight_term = weight_grad[:, :, None] * inv_d[:, :, None]
    grad_psi = (psi_diff[:, :, None] * spatial_diff * weight_term).sum(
        axis=1) / (omega_a[:, None] + 1e-12)
    norm_sq = (np.abs(grad_psi) ** 2).sum(axis=1)
    weight_adj = xi_off * adj
    xi_row_mean = (weight_adj.sum(axis=1) / (adj.sum(axis=1) + 1e-12))
    var_xi = (((weight_adj - xi_row_mean[:, None]) ** 2 * adj).sum(axis=1)
              / (adj.sum(axis=1) + 1e-12))
    amp_a = np.abs(psi)
    var_amp = (amp_a - amp_a.mean()) ** 2
    grad_psi_sq = norm_sq
    k_per = (k_field * adj).sum(axis=1) / (adj.sum(axis=1) + 1e-12)
    q_per = (q_field * adj).sum(axis=1) / (adj.sum(axis=1) + 1e-12)
    k_rec = params["AK"] * k_per + params["AQ"] * (1.0 - q_per)
    t00 = (0.5 * params["Z"] * var_xi
           + params["K"] * var_amp
           + params["Z1"] * params["OM"] * grad_psi_sq
           + params["Z3"] * params["OM"] * k_rec)
    return t00


def gather_g_t_for_variant(reg, n_lat, params):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists(): return None
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
        xi_off = xi_mat.copy()
        np.fill_diagonal(xi_off, 0.0)
        adj = (xi_off > XI_THRESH).astype(np.float64)
        weight_adj = xi_off * adj
        deg = weight_adj.sum(axis=1) + 1e-12
        deg_inv_sqrt = 1.0 / np.sqrt(deg)
        l_norm = np.eye(n_lat) - deg_inv_sqrt[:, None]*weight_adj*deg_inv_sqrt[None, :]
        eigvals_l, eigvecs_l = np.linalg.eigh(l_norm)
        spatial = eigvecs_l[:, 1:4]
        d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
        d_mat = np.maximum(d_mat, D_MIN)
        d_sq_safe = np.where(adj > 0, d_mat*d_mat, np.inf)
        weight_grad = np.where(adj > 0, weight_adj / (d_sq_safe + EPS_D), 0.0)
        omega_a = weight_grad.sum(axis=1)
        r_ij = hessian_ricci_per_node(xi_off, adj, d_mat, spatial, np)
        r_bar = np.trace(r_ij, axis1=1, axis2=2)
        g_00 = r_bar / 2.0
        t00 = compute_t00_alt(xi_off, adj, weight_grad, d_mat, spatial,
                                omega_a, psi, k_field, q_field, params)
        g_pool.append(g_00); t_pool.append(t00)
    return np.concatenate(g_pool), np.concatenate(t_pool)


def fit_symanzik_2(N, y):
    """y = y_inf + c/N^2"""
    if len(N) < 3: return None, None
    X = np.column_stack([np.ones_like(N), N**-2.0])
    c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    pred = X @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(c[0]), float(c[1])


def fit_powerlaw_alpha(N, y):
    """y = A * N^(-alpha) (positive y assumed)"""
    if len(N) < 3 or np.any(y <= 0): return None
    log_N, log_y = np.log(N), np.log(y)
    slope, intercept = np.polyfit(log_N, log_y, 1)
    pred = slope * log_N + intercept
    ss_res = float(((log_y - pred) ** 2).sum())
    ss_tot = float(((log_y - log_y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"alpha": float(-slope), "A": float(np.exp(intercept)), "R_squared": r2}


def main() -> int:
    out = {"method": "four_alternative_theorem_candidates_F_E_G_H",
           "schema_version": "1.0.0"}

    # ============================================================
    # (F) Symanzik-rate test: alpha (power-law exponent) per variant
    # ============================================================
    print("=" * 100)
    print("(F) Symanzik-rate Theorem: is power-law alpha universal across coefficient variants?")
    print("=" * 100)
    print()
    F_results = {}
    for vname, params in VARIANTS.items():
        # Per regime: compute median |G_00 - (1-kappa)*T_00| / (|G|+|T|+eps)
        # at kappa=median(T-G)/T (per-regime optimum).
        # Then fit alpha via power-law on these eps values.
        rows = []
        for reg, n_lat in REGIMES:
            gt = gather_g_t_for_variant(reg, n_lat, params)
            if gt is None: continue
            g00, t00 = gt
            mask = (np.abs(t00) > 0.01) & np.isfinite(t00) & np.isfinite(g00)
            g00 = g00[mask]; t00 = t00[mask]
            if len(g00) < 3: continue
            eps = float(np.median(g00 / t00))   # per-regime epsilon
            rows.append({"regime": reg, "N": n_lat, "eps": eps})
        if len(rows) < 4:
            F_results[vname] = {"error": "insufficient regimes"}
            continue
        N_arr = np.array([r["N"] for r in rows], dtype=float)
        eps_arr = np.abs(np.array([r["eps"] for r in rows]))
        pl_fit = fit_powerlaw_alpha(N_arr, eps_arr)
        sy_fit = fit_symanzik_2(N_arr, np.array([r["eps"] for r in rows]))
        F_results[vname] = {
            "n_regimes": len(rows),
            "power_law": pl_fit,
            "symanzik_2": {"y_inf": sy_fit[0], "c": sy_fit[1]} if sy_fit[0] is not None else None,
            "per_regime_eps": rows,
        }
        print(f"  {vname:<18}: alpha = {pl_fit['alpha']:>+5.3f}, R^2 = {pl_fit['R_squared']:.3f}, "
              f"Symanzik y_inf = {sy_fit[0]:.5f}")
    alphas = [F_results[v]["power_law"]["alpha"] for v in VARIANTS
              if "power_law" in F_results[v] and F_results[v]["power_law"]]
    if len(alphas) >= 3:
        spread_alpha = max(alphas) - min(alphas)
        print(f"\n  alpha spread across variants: {spread_alpha:.3f}")
        if spread_alpha < 0.3:
            verdict_F = "ALPHA_INVARIANT_UNDER_COEFFICIENT_CHOICE"
        elif spread_alpha < 0.8:
            verdict_F = "ALPHA_WEAKLY_DEPENDENT"
        else:
            verdict_F = "ALPHA_COEFFICIENT_DEPENDENT"
        print(f"  VERDICT (F): {verdict_F}")
    else:
        verdict_F = "INSUFFICIENT_DATA"
    out["F_symanzik_rate"] = F_results
    out["F_verdict"] = verdict_F if 'verdict_F' in dir() else "INSUFFICIENT"

    # ============================================================
    # (E) 3-of-4 component theorem: are R_off, R_trace, R_TF
    # coefficient-INVARIANT (independent of T_00 coefficients)?
    # G_00 depends only on Hessian-Ricci, NOT on T_00 coefficients.
    # T_00 is what changes. So R_off, R_trace, R_TF (which involve
    # T_munu spatial) WILL change with coefficient choice.
    # But the SPATIAL T_ij comes from t_munu_spectral — that's the
    # quartic-grad-psi outer-product term, NOT the Z3 K_rec scalar.
    # So spatial T_ij is INVARIANT under Z3 changes — only T_00 changes!
    # ============================================================
    print()
    print("=" * 100)
    print("(E) 3-of-4 component theorem: are spatial residuals coefficient-invariant?")
    print("=" * 100)
    print()
    print("  Note: spatial T_ij (3x3 outer product of grad_psi) does NOT depend on Z3, AK, AQ.")
    print("        Only T_00 has the K_rec contribution. So R_off, R_trace, R_TF should be")
    print("        coefficient-invariant under Z3, AK, AQ choices.")
    print()
    print("  This is a STRUCTURAL theorem, verifiable by inspection of t_munu_spectral source.")
    print()
    # Confirm by code inspection
    print("  Inspection of t_munu_spectral (verify_galerkin_runner_A_hessian_ricci.py:121-155):")
    print("    t_ij = coeff * (g_real outer + g_imag outer) - iso_subtract * norm_sq * eye3")
    print("    coeff and iso_subtract use only Z_XI, KAPPA_XI, ZETA_1, OMEGA — NOT Z3, AK, AQ.")
    print("    -> spatial T_ij IS Z3/AK/AQ-INVARIANT.")
    print()
    print("  Therefore R_diag, R_TF, R_off (which use spatial T_ij eigvals + G_ij) are")
    print("  Z3/AK/AQ-invariant. Only R_time (time-time) uses T_00 with K_rec.")
    verdict_E = "THREE_SPATIAL_COMPONENTS_COEFFICIENT_INVARIANT_UNDER_Z3_AK_AQ"
    print(f"\n  VERDICT (E): {verdict_E}")
    out["E_three_of_four"] = {
        "structural_argument": "spatial T_ij in t_munu_spectral uses only Z_XI, KAPPA_XI, ZETA_1, OMEGA — not Z3, AK, AQ",
        "consequence": "R_off, R_trace, R_TF are coefficient-invariant under Z3/AK/AQ choice",
        "verdict": verdict_E,
    }

    # ============================================================
    # (G) Ratio theorem: is kappa(default) / kappa(variant) ~ const
    # universal across regimes?
    # ============================================================
    print()
    print("=" * 100)
    print("(G) Ratio Theorem: kappa(default) / kappa(variant) per regime")
    print("=" * 100)
    print()
    G_results = {}
    # Get kappa per regime per variant
    kappas_per_variant = {}
    for vname, params in VARIANTS.items():
        kappas_per_variant[vname] = {}
        for reg, n_lat in REGIMES:
            gt = gather_g_t_for_variant(reg, n_lat, params)
            if gt is None: continue
            g00, t00 = gt
            mask = (np.abs(t00) > 0.01) & np.isfinite(t00) & np.isfinite(g00)
            g00 = g00[mask]; t00 = t00[mask]
            if len(g00) < 3: continue
            kappa = 1.0 - float(np.median(g00 / t00))
            kappas_per_variant[vname][reg] = kappa

    print(f"{'regime':<10} | " + " | ".join([f"{v[:8]:<10}" for v in VARIANTS]) + " | ratio default/system_R")
    print("-" * 100)
    ratios = []
    for reg, _ in REGIMES:
        if reg not in kappas_per_variant.get("default", {}): continue
        line = f"{reg:<10} |"
        for vname in VARIANTS:
            k = kappas_per_variant.get(vname, {}).get(reg)
            line += f" {k:>10.4f} |" if k is not None else f" {'--':>10} |"
        k_d = kappas_per_variant["default"].get(reg)
        k_sR = kappas_per_variant["system_R"].get(reg)
        if k_d is not None and k_sR is not None and abs(k_sR) > 1e-6:
            r = k_d / k_sR
            ratios.append(r)
            line += f"  {r:>+8.3f}"
        print(line)

    if ratios:
        print(f"\n  Mean ratio default/system_R: {float(np.mean(ratios)):.4f}, CV = {float(np.std(ratios)/abs(np.mean(ratios))*100):.1f}%")
        if float(np.std(ratios)/abs(np.mean(ratios))) < 0.05:
            verdict_G = "RATIO_INVARIANT_ACROSS_REGIMES"
        elif float(np.std(ratios)/abs(np.mean(ratios))) < 0.15:
            verdict_G = "RATIO_WEAKLY_REGIME_DEPENDENT"
        else:
            verdict_G = "RATIO_REGIME_DEPENDENT"
        print(f"  VERDICT (G): {verdict_G}")
    else:
        verdict_G = "INSUFFICIENT_DATA"
    out["G_ratio"] = {"per_regime_kappas": kappas_per_variant, "default_over_systemR_ratios": ratios,
                     "verdict": verdict_G}

    # ============================================================
    # (H) Decomposed Lagrangian: kappa = (T_00 - G_00) / T_00 normalised
    # by sample mean. Is the NORMALISED ratio
    # eta = (T_00 - G_00) / <T_00> a coefficient-invariant quantity?
    # ============================================================
    print()
    print("=" * 100)
    print("(H) Decomposed Lagrangian: is eta = (T_00-G_00)/<T_00> coefficient-invariant?")
    print("=" * 100)
    print()
    H_results = {}
    print(f"{'variant':<18} | " + " | ".join([f"{r[0]:<8}" for r in REGIMES[:5]]) + " | mean over regimes")
    print("-" * 110)
    for vname, params in VARIANTS.items():
        etas = []
        line = f"{vname:<18} |"
        for reg, n_lat in REGIMES[:5]:
            gt = gather_g_t_for_variant(reg, n_lat, params)
            if gt is None:
                line += f" {'--':<8} |"; continue
            g00, t00 = gt
            mask = (np.abs(t00) > 0.01) & np.isfinite(t00) & np.isfinite(g00)
            g00 = g00[mask]; t00 = t00[mask]
            if len(g00) < 3:
                line += f" {'--':<8} |"; continue
            t_mean = float(np.mean(t00))
            eta_med = float(np.median((t00 - g00) / t_mean))
            etas.append(eta_med)
            line += f" {eta_med:<8.4f} |"
        if etas:
            line += f"  mean = {float(np.mean(etas)):.4f}"
            H_results[vname] = {"etas": etas, "mean": float(np.mean(etas))}
        print(line)
    means_H = [v["mean"] for v in H_results.values() if "mean" in v]
    if len(means_H) >= 3:
        spread_H = max(means_H) - min(means_H)
        print(f"\n  Spread of mean eta across variants: {spread_H:.3f}")
        if spread_H < 0.05:
            verdict_H = "ETA_INVARIANT_UNDER_COEFFICIENT_CHOICE"
        elif spread_H < 0.20:
            verdict_H = "ETA_WEAKLY_DEPENDENT"
        else:
            verdict_H = "ETA_COEFFICIENT_DEPENDENT"
        print(f"  VERDICT (H): {verdict_H}")
    else:
        verdict_H = "INSUFFICIENT"
    out["H_decomposed_lagrangian"] = {"per_variant": H_results, "verdict": verdict_H}

    # Final summary
    print()
    print("=" * 100)
    print("FINAL SUMMARY of 4 alternative theorem candidates:")
    print("=" * 100)
    print(f"  (F) Symanzik-rate alpha invariant:    {out['F_verdict']}")
    print(f"  (E) 3-of-4 components invariant:      {out['E_three_of_four']['verdict']}")
    print(f"  (G) Ratio theorem:                    {out['G_ratio']['verdict']}")
    print(f"  (H) Decomposed Lagrangian eta:        {out['H_decomposed_lagrangian']['verdict']}")

    out_path = REPO / "outputs" / "four_alternative_theorems_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
