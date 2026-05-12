"""Search for a regime-invariant universal form of the tail
functional identity:

  Delta_tail(a)               T_00(a)     |grad T_00(a)|     omega(a)
  -------------- = F( ----- , ----------- , -------- , ... )
   <Delta_tail>      <T_00>    <|grad T|>     <omega>

If Δ_tail / ⟨Δ_tail⟩ is a *universal* function of the
dimensionless ratios (across all N regimes), the matter-core
residual closure is regime-independent.

Test plan:
  1. Per regime, compute tail features and their regime-mean.
  2. Normalize each tail feature by its regime-mean.
  3. Pool ALL regimes into one training set.
  4. Linear regression on normalized features.
  5. Compare pooled R^2 to:
     - raw fit (regime-mean = 1)
     - log-features fit
     - ratio-only fit (no abs scale)

If pooled R^2 >= 0.7 in the dimensionless form, declare
TAIL_UNIVERSAL_FORM_HOLDS.

Output: outputs/tail_universal_form_audit.json
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
    edge_to_matrix, per_seed_galerkin, XI_THRESH, ELL_0, D_MIN, EPS_D)
from verify_per_eigendirection_residual import (
    per_node_eigendirection_residuals)


REGIMES = [
    ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]
LAMBDA_T = 0.81
LAMBDA_S = -0.005
TAIL_FRAC = 0.10


def per_node_features(xi_mat, psi, t00, n_lat):
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    weight_adj = xi_off * adj
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.maximum(d_mat, D_MIN)
    d_sq = d_mat * d_mat
    d_sq_safe = np.where(adj > 0, d_sq, np.inf)
    weight_grad = np.where(adj > 0, weight_adj / (d_sq_safe + EPS_D), 0.0)
    omega_a = weight_grad.sum(axis=1)
    diff_sq = (t00[None, :] - t00[:, None]) ** 2
    grad_T_sq = (weight_grad * diff_sq).sum(axis=1)
    grad_T = np.sqrt(np.maximum(grad_T_sq, 0.0))
    psi_sq = np.abs(psi) ** 2
    return {
        "T_00":      np.abs(t00),
        "grad_T_00": grad_T,
        "omega_a":   omega_a,
        "psi_sq":    psi_sq,
    }


def per_node_delta(prep):
    res = per_node_eigendirection_residuals(prep, LAMBDA_T, LAMBDA_S)
    R_t = res["R_time"]; R_d = res["R_diag"]; R_o = res["R_off"]
    t_e = res["T_eigvals"]; t00 = np.asarray(prep["t00"])
    R = np.sqrt(R_t ** 2 + (R_d ** 2).sum(axis=1) + R_o ** 2)
    T = np.sqrt(t00 ** 2 + (t_e ** 2).sum(axis=1))
    return R / np.maximum(T, 1e-12)


def gather_tail_per_regime(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)
    deltas, feat_pool = [], None
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        delta = per_node_delta(prep)
        t00 = np.asarray(prep["t00"])
        feats = per_node_features(xi_mat, psi, t00, n_lat)
        deltas.append(delta)
        if feat_pool is None:
            feat_pool = {k: [v] for k, v in feats.items()}
        else:
            for k, v in feats.items():
                feat_pool[k].append(v)
    delta = np.concatenate(deltas)
    feats = {k: np.concatenate(v) for k, v in feat_pool.items()}

    # Tail mask: top-10% by Delta
    n_total = len(delta)
    n_tail = max(1, int(n_total * TAIL_FRAC))
    order = np.argsort(-delta)
    tail_idx = order[:n_tail]

    # Regime-means computed over the FULL regime (not just tail) so
    # they are stable
    regime_means = {k: float(np.mean(v)) for k, v in feats.items()}
    regime_mean_delta = float(np.mean(delta))
    return {
        "regime": reg, "N": n_lat,
        "delta_tail": delta[tail_idx],
        "feats_tail": {k: v[tail_idx] for k, v in feats.items()},
        "regime_means": regime_means,
        "regime_mean_delta": regime_mean_delta,
        "n_tail": int(n_tail),
    }


def linreg_with_intercept(X, y):
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    c, *_ = np.linalg.lstsq(Xb, y, rcond=1e-10)
    pred = Xb @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return c, r2


def main() -> int:
    feature_keys = ["T_00", "grad_T_00", "omega_a", "psi_sq"]

    print("=" * 110)
    print("Tail UNIVERSAL FORM search")
    print("Test: Delta_tail(a)/<Delta> = F(T_00(a)/<T>, |grad T|(a)/<|grad T|>, omega(a)/<omega>, ...)")
    print("=" * 110)
    print()

    regime_data = []
    for reg, n_lat in REGIMES:
        rd = gather_tail_per_regime(reg, n_lat)
        if rd is None:
            continue
        regime_data.append(rd)
        print(f"  {reg:<8} N={n_lat:>3}: <Delta>={rd['regime_mean_delta']:.4f}, "
              f"<T_00>={rd['regime_means']['T_00']:.4f}, "
              f"<|grad T|>={rd['regime_means']['grad_T_00']:.4f}, "
              f"<omega>={rd['regime_means']['omega_a']:.4f}")
    print()

    # Build pooled training data in 4 alternative parameterisations:
    #
    # FORM 1: raw absolute features (no normalisation)
    # FORM 2: features divided by regime mean (dimensionless ratios)
    # FORM 3: log-features (multiplicative scaling absorbed)
    # FORM 4: features / regime mean, target / regime-mean-target
    forms = {"raw": [], "ratio": [], "log_ratio": [], "ratio_target_norm": []}
    targets = {"raw": [], "ratio": [], "log_ratio": [], "ratio_target_norm": []}

    for rd in regime_data:
        for k in feature_keys:
            assert rd["regime_means"][k] > 0, f"{k} mean is zero"

        x_raw = np.stack([rd["feats_tail"][k] for k in feature_keys], axis=1)
        x_ratio = np.stack([rd["feats_tail"][k] / rd["regime_means"][k]
                            for k in feature_keys], axis=1)
        eps = 1e-10
        x_log = np.stack([np.log(np.maximum(rd["feats_tail"][k] / rd["regime_means"][k], eps))
                          for k in feature_keys], axis=1)
        forms["raw"].append(x_raw)
        forms["ratio"].append(x_ratio)
        forms["log_ratio"].append(x_log)
        forms["ratio_target_norm"].append(x_ratio)

        targets["raw"].append(rd["delta_tail"])
        targets["ratio"].append(rd["delta_tail"])
        targets["log_ratio"].append(rd["delta_tail"])
        # FORM 4: also normalize target by regime-mean Delta
        targets["ratio_target_norm"].append(
            rd["delta_tail"] / rd["regime_mean_delta"])

    fits = {}
    print("Pooled fit comparison:")
    print(f"{'form':<22} {'R^2':>8} {'intercept':>10}  coeffs")
    print("-" * 100)
    for form, X_list in forms.items():
        X_pool = np.vstack(X_list)
        y_pool = np.concatenate(targets[form])
        c, r2 = linreg_with_intercept(X_pool, y_pool)
        coef_str = ", ".join([f"{k}={c[i+1]:+.4f}" for i, k in enumerate(feature_keys)])
        print(f"{form:<22} {r2:>8.4f} {c[0]:>+10.4f}  {coef_str}")
        fits[form] = {
            "R_squared_pooled": r2,
            "intercept": float(c[0]),
            "coeffs": {k: float(c[i+1]) for i, k in enumerate(feature_keys)},
        }

    # Best form
    best = max(fits.items(), key=lambda kv: kv[1]["R_squared_pooled"])
    print()
    print(f"BEST FORM: {best[0]} with pooled R^2 = {best[1]['R_squared_pooled']:.4f}")
    if best[1]["R_squared_pooled"] >= 0.7:
        verdict = f"TAIL_UNIVERSAL_FORM_HOLDS_{best[0].upper()}"
    elif best[1]["R_squared_pooled"] >= 0.5:
        verdict = f"TAIL_UNIVERSAL_FORM_PARTIAL_{best[0].upper()}"
    else:
        verdict = "TAIL_UNIVERSAL_FORM_FAILS"
    print(f"VERDICT: {verdict}")

    # Per-regime cross-check on best form: how well does the universal
    # coefficient set predict each regime?
    print()
    print(f"Per-regime predictive R^2 using BEST form ({best[0]}) coefficients:")
    print(f"{'reg':<8} {'N':>3} | {'R^2_in_regime':>14}")
    per_regime_r2 = []
    best_coeffs = np.array([fits[best[0]]["intercept"]]
                           + [fits[best[0]]["coeffs"][k] for k in feature_keys])
    for rd in regime_data:
        if best[0] == "raw":
            X = np.stack([rd["feats_tail"][k] for k in feature_keys], axis=1)
            y = rd["delta_tail"]
        elif best[0] == "log_ratio":
            X = np.stack([np.log(np.maximum(rd["feats_tail"][k] / rd["regime_means"][k], 1e-10))
                          for k in feature_keys], axis=1)
            y = rd["delta_tail"]
        elif best[0] == "ratio_target_norm":
            X = np.stack([rd["feats_tail"][k] / rd["regime_means"][k]
                          for k in feature_keys], axis=1)
            y = rd["delta_tail"] / rd["regime_mean_delta"]
        else:  # ratio
            X = np.stack([rd["feats_tail"][k] / rd["regime_means"][k]
                          for k in feature_keys], axis=1)
            y = rd["delta_tail"]
        Xb = np.column_stack([np.ones(X.shape[0]), X])
        pred = Xb @ best_coeffs
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2_reg = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        per_regime_r2.append({"regime": rd["regime"], "N": rd["N"],
                              "R_squared_in_regime": r2_reg})
        print(f"{rd['regime']:<8} {rd['N']:>3} | {r2_reg:>14.4f}")

    out_path = REPO / "outputs" / "tail_universal_form_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "tail_universal_form_search",
            "schema_version": "1.0.0",
            "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
            "tail_fraction": TAIL_FRAC,
            "feature_keys": feature_keys,
            "regime_means_per_regime": [
                {"regime": rd["regime"], "N": rd["N"],
                 "regime_means": rd["regime_means"],
                 "regime_mean_delta": rd["regime_mean_delta"]}
                for rd in regime_data
            ],
            "fits_per_form": fits,
            "best_form": best[0],
            "best_R_squared_pooled": best[1]["R_squared_pooled"],
            "per_regime_R_squared_using_best": per_regime_r2,
            "verdict": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
