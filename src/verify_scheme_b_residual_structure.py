"""Test (E): Sign and structure analysis of the residual after
Scheme B (statistical universal-law subtraction).

Per tail node a:
  Delta_raw(a)        = || R_munu(a) ||_F / || T_munu(a) ||_F
  Delta_pred(a)       = max(0, A0 + A1 * log(T_00(a) / <T_00>))
  Residual_after(a)   = Delta_raw(a) - Delta_pred(a)   (signed)
  Abs_residual(a)     = |Residual_after(a)|

Tests:
  1. Sign distribution: fraction(R > 0) close to 0.5 -> sign-symmetric
     noise -> Scheme B is statistically exact. Sign-biased -> missing
     structural term.
  2. Mean residual: <Residual_after> close to 0 -> exact. Significant
     bias -> universal law is offset.
  3. Distribution shape vs Gaussian: KS-test against
     normal(0, sigma_obs).
  4. Sub-bin structure: split tail into low-T_00, mid-T_00, high-T_00
     thirds; check if mean residual differs by bin.

Output: outputs/scheme_b_residual_structure_audit.json
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
    per_node_eigendirection_residuals)


REGIMES = [
    ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]
LAMBDA_T = 0.81
LAMBDA_S = -0.005
TAIL_FRAC = 0.10
A0 = 0.234   # universal law intercept (excluding P5_N50)
A1 = 1.304   # universal law slope


def per_node_delta_and_t00(prep):
    res = per_node_eigendirection_residuals(prep, LAMBDA_T, LAMBDA_S)
    R_t = res["R_time"]; R_d = res["R_diag"]; R_o = res["R_off"]
    t_e = res["T_eigvals"]; t00 = np.asarray(prep["t00"])
    R = np.sqrt(R_t ** 2 + (R_d ** 2).sum(axis=1) + R_o ** 2)
    T = np.sqrt(t00 ** 2 + (t_e ** 2).sum(axis=1))
    return R / np.maximum(T, 1e-12), np.abs(t00)


def gather_tail(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)
    deltas, t00s = [], []
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        delta, t00 = per_node_delta_and_t00(prep)
        deltas.append(delta)
        t00s.append(t00)
    delta = np.concatenate(deltas)
    t00 = np.concatenate(t00s)
    n_total = len(delta)
    n_tail = max(1, int(n_total * TAIL_FRAC))
    order = np.argsort(-delta)
    tail = order[:n_tail]

    t00_tail = t00[tail]
    delta_tail = delta[tail]
    t00_mean_full = float(np.mean(t00))
    log_ratio_tail = np.log(np.maximum(t00_tail / max(t00_mean_full, 1e-12), 1e-10))
    delta_pred_tail = np.maximum(A0 + A1 * log_ratio_tail, 0.0)
    residual_signed = delta_tail - delta_pred_tail
    return {
        "regime": reg, "N": n_lat,
        "t00_tail": t00_tail,
        "delta_tail": delta_tail,
        "delta_pred_tail": delta_pred_tail,
        "residual_signed": residual_signed,
    }


def ks_against_zero_normal(x):
    """Kolmogorov-Smirnov statistic of x against N(0, sigma_x).
    Returns D and approximate p-value (asymptotic)."""
    if len(x) < 5:
        return float("nan"), float("nan")
    sigma = float(np.std(x))
    if sigma < 1e-12:
        return 0.0, 1.0
    sorted_x = np.sort(x)
    n = len(sorted_x)
    # CDF of N(0, sigma)
    from math import erf, sqrt
    Phi = np.array([0.5 * (1 + erf(v / (sigma * np.sqrt(2.0)))) for v in sorted_x])
    F_emp = np.arange(1, n + 1) / n
    D = float(np.max(np.abs(F_emp - Phi)))
    # Asymptotic p-value (Kolmogorov)
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * D
    if lam < 0.18:
        p = 1.0
    else:
        p = 2 * np.exp(-2 * lam ** 2)
    return D, float(p)


def main() -> int:
    print("=" * 110)
    print("Scheme B residual structure audit")
    print(f"  Universal law: Delta_pred = max(0, {A0:+.4f} + {A1:+.4f} * log(T_00/<T_00>))")
    print(f"  After-residual: r(a) = Delta_raw(a) - Delta_pred(a) on tail nodes")
    print("=" * 110)
    print()
    print(f"{'reg':<8} {'N':>3} {'n_tail':>7} | "
          f"{'mean_r':>9} {'med_r':>9} {'sigma_r':>9} | "
          f"{'frac>0':>7} {'KS_D':>7} {'KS_p':>9} | "
          f"{'low_T_mean':>12} {'mid_T_mean':>12} {'high_T_mean':>12}")
    print("-" * 130)

    rows = []
    pooled_residuals = []
    for reg, n_lat in REGIMES:
        d = gather_tail(reg, n_lat)
        if d is None:
            continue
        r = d["residual_signed"]
        mean_r = float(np.mean(r))
        med_r = float(np.median(r))
        sigma_r = float(np.std(r))
        frac_pos = float((r > 0).sum()) / max(len(r), 1)
        D, p = ks_against_zero_normal(r)

        # Sub-bin by T_00
        t00 = d["t00_tail"]
        t_sorted_idx = np.argsort(t00)
        n = len(t00)
        third = n // 3
        low_idx = t_sorted_idx[:third]
        mid_idx = t_sorted_idx[third:2*third]
        high_idx = t_sorted_idx[2*third:]
        mean_low = float(r[low_idx].mean())
        mean_mid = float(r[mid_idx].mean())
        mean_high = float(r[high_idx].mean())

        rows.append({
            "regime": reg, "N": n_lat, "n_tail": int(len(r)),
            "mean_residual_after_B": mean_r,
            "median_residual_after_B": med_r,
            "sigma_residual_after_B": sigma_r,
            "fraction_positive": frac_pos,
            "KS_D_vs_zero_normal": D,
            "KS_p_value": p,
            "subbin_low_T_mean": mean_low,
            "subbin_mid_T_mean": mean_mid,
            "subbin_high_T_mean": mean_high,
        })
        pooled_residuals.append(r)

        print(f"{reg:<8} {n_lat:>3} {len(r):>7} | "
              f"{mean_r:>+9.4f} {med_r:>+9.4f} {sigma_r:>9.4f} | "
              f"{frac_pos:>7.3f} {D:>7.3f} {p:>9.4f} | "
              f"{mean_low:>+12.4f} {mean_mid:>+12.4f} {mean_high:>+12.4f}")

    pooled = np.concatenate(pooled_residuals)
    pooled_mean = float(np.mean(pooled))
    pooled_med = float(np.median(pooled))
    pooled_sigma = float(np.std(pooled))
    pooled_frac_pos = float((pooled > 0).sum()) / max(len(pooled), 1)
    D_pool, p_pool = ks_against_zero_normal(pooled)

    print()
    print(f"POOLED across all regimes (n_tail_total={len(pooled)}):")
    print(f"  mean = {pooled_mean:+.4f}, median = {pooled_med:+.4f}, sigma = {pooled_sigma:.4f}")
    print(f"  fraction(r > 0) = {pooled_frac_pos:.3f}  (sign-symmetric: 0.500)")
    print(f"  KS_D vs N(0, {pooled_sigma:.4f}) = {D_pool:.4f}, p = {p_pool:.4f}")

    # Verdict logic
    sign_balanced = abs(pooled_frac_pos - 0.5) < 0.10
    mean_zero = abs(pooled_mean) < 0.5 * pooled_sigma  # within half a sigma
    no_subbin_drift = True
    for r in rows:
        spread = abs(r["subbin_high_T_mean"] - r["subbin_low_T_mean"])
        if spread > 0.5 * pooled_sigma:
            no_subbin_drift = False
            break

    if sign_balanced and mean_zero and no_subbin_drift:
        verdict = "SCHEME_B_RESIDUAL_IS_SIGN_SYMMETRIC_NOISE"
        rationale = ("After universal-law subtraction the tail residual is "
                     "sign-balanced, zero-mean noise — the universal law is "
                     "statistically exact for the tail magnitude.")
    elif sign_balanced and not mean_zero:
        verdict = "SCHEME_B_BIASED_BUT_SIGN_SYMMETRIC"
        rationale = ("Sign-balanced residual but with nonzero mean — universal "
                     "law has a small constant offset.")
    elif not no_subbin_drift:
        verdict = "SCHEME_B_RESIDUAL_HAS_T_00_SUB_BIN_STRUCTURE"
        rationale = ("Residual after subtraction varies systematically with "
                     "T_00 sub-bin — the universal slope a_1 may itself be "
                     "T_00-dependent (suggesting nonlinear correction term).")
    else:
        verdict = "SCHEME_B_RESIDUAL_HAS_STRUCTURAL_BIAS"
        rationale = ("Residual after subtraction is sign-biased — a missing "
                     "structural term (probably nonlinear in log) remains.")

    print()
    print(f"VERDICT: {verdict}")
    print(f"  Rationale: {rationale}")

    out_path = REPO / "outputs" / "scheme_b_residual_structure_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "scheme_b_residual_structure",
            "schema_version": "1.0.0",
            "universal_law": {"A0": A0, "A1": A1},
            "tail_fraction": TAIL_FRAC,
            "per_regime": rows,
            "pooled": {
                "n_tail_total": int(len(pooled)),
                "mean": pooled_mean,
                "median": pooled_med,
                "sigma": pooled_sigma,
                "fraction_positive": pooled_frac_pos,
                "KS_D": D_pool,
                "KS_p": p_pool,
            },
            "verdict": verdict,
            "rationale": rationale,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
