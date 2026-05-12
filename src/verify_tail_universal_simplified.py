"""Test the simplified universal tail-form, motivated by the
log_ratio fit in tail_universal_form_audit.json:

  Delta_tail(a) approx const + c_T * log(T_00(a)/<T_00>)

The 4-feature log_ratio fit gave c_T = 1.02 (dominant), with the
other three coefficients (grad_T, omega, psi^2) collectively
contributing < 10% of the variance. Test the 1-parameter form:

  Delta_tail(a) = a_0 + a_1 * log(T_00(a) / <T_00>)

If pooled R^2 with this single feature is >= 0.7, the universal
matter-core identity reduces to a SINGLE power-law:

  Delta_tail(a) ~ (T_00(a))^{a_1}    (modulo overall constant)

Also exclude P5_N50 (identified outlier in the universal-form
audit) and re-fit the full and simplified forms.

Output: outputs/tail_universal_simplified_audit.json
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
EXCLUDE = ["P5"]   # P5_N50 marginal/unterkonvergiert
LAMBDA_T = 0.81
LAMBDA_S = -0.005
TAIL_FRAC = 0.10


def per_node_features(xi_mat, t00, n_lat):
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
    grad_T = np.sqrt(np.maximum((weight_grad * diff_sq).sum(axis=1), 0.0))
    return {"T_00": np.abs(t00), "grad_T_00": grad_T, "omega_a": omega_a}


def per_node_delta(prep):
    res = per_node_eigendirection_residuals(prep, LAMBDA_T, LAMBDA_S)
    R_t = res["R_time"]; R_d = res["R_diag"]; R_o = res["R_off"]
    t_e = res["T_eigvals"]; t00 = np.asarray(prep["t00"])
    R = np.sqrt(R_t ** 2 + (R_d ** 2).sum(axis=1) + R_o ** 2)
    T = np.sqrt(t00 ** 2 + (t_e ** 2).sum(axis=1))
    return R / np.maximum(T, 1e-12)


def gather_tail(reg, n_lat):
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
        feats = per_node_features(xi_mat, t00, n_lat)
        deltas.append(delta)
        if feat_pool is None:
            feat_pool = {k: [v] for k, v in feats.items()}
        else:
            for k, v in feats.items():
                feat_pool[k].append(v)
    delta = np.concatenate(deltas)
    feats = {k: np.concatenate(v) for k, v in feat_pool.items()}
    n_total = len(delta)
    n_tail = max(1, int(n_total * TAIL_FRAC))
    order = np.argsort(-delta)
    tail_idx = order[:n_tail]
    rmeans = {k: float(np.mean(v)) for k, v in feats.items()}
    return {
        "regime": reg, "N": n_lat,
        "delta_tail": delta[tail_idx],
        "log_T_ratio": np.log(np.maximum(feats["T_00"][tail_idx] / rmeans["T_00"], 1e-10)),
        "log_gradT_ratio": np.log(np.maximum(feats["grad_T_00"][tail_idx] / max(rmeans["grad_T_00"], 1e-12), 1e-10)),
        "log_omega_ratio": np.log(np.maximum(feats["omega_a"][tail_idx] / max(rmeans["omega_a"], 1e-12), 1e-10)),
    }


def linreg(X, y):
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    c, *_ = np.linalg.lstsq(Xb, y, rcond=1e-10)
    pred = Xb @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return c, r2


def main() -> int:
    print("=" * 110)
    print("Tail UNIVERSAL FORM simplified test:")
    print("  Delta_tail(a) = a_0 + a_1 * log(T_00(a)/<T_00>)  + ...")
    print("=" * 110)
    print()

    rdata_all = []
    rdata_kept = []
    for reg, n_lat in REGIMES:
        rd = gather_tail(reg, n_lat)
        if rd is None:
            continue
        rdata_all.append(rd)
        if reg not in EXCLUDE:
            rdata_kept.append(rd)
    print(f"All regimes: {[r['regime'] for r in rdata_all]}")
    print(f"Kept (excluding {EXCLUDE}): {[r['regime'] for r in rdata_kept]}")
    print()

    def evaluate(data, label):
        # 1-feature: log T ratio only
        x_T = np.concatenate([rd["log_T_ratio"] for rd in data]).reshape(-1, 1)
        # 3-feature: log T, log gradT, log omega
        x3 = np.column_stack([
            np.concatenate([rd["log_T_ratio"] for rd in data]),
            np.concatenate([rd["log_gradT_ratio"] for rd in data]),
            np.concatenate([rd["log_omega_ratio"] for rd in data]),
        ])
        y = np.concatenate([rd["delta_tail"] for rd in data])
        c1, r2_1 = linreg(x_T, y)
        c3, r2_3 = linreg(x3, y)
        return {
            "label": label,
            "n_pooled": int(y.size),
            "fit_1param_T": {"a0": float(c1[0]), "a1_logT": float(c1[1]), "R_squared": r2_1},
            "fit_3param": {"a0": float(c3[0]),
                            "a_logT": float(c3[1]),
                            "a_loggradT": float(c3[2]),
                            "a_logomega": float(c3[3]),
                            "R_squared": r2_3},
        }

    full = evaluate(rdata_all, "all_regimes")
    kept = evaluate(rdata_kept, f"excluding_{EXCLUDE}")

    for r in (full, kept):
        print(f"--- {r['label']} (n={r['n_pooled']}) ---")
        f1 = r["fit_1param_T"]
        print(f"  1-param: Delta = {f1['a0']:+.4f} + {f1['a1_logT']:+.4f} * log(T/<T>),  R^2 = {f1['R_squared']:.4f}")
        f3 = r["fit_3param"]
        print(f"  3-param: Delta = {f3['a0']:+.4f} + {f3['a_logT']:+.4f}*logT + "
              f"{f3['a_loggradT']:+.4f}*log|gradT| + {f3['a_logomega']:+.4f}*logomega, "
              f"R^2 = {f3['R_squared']:.4f}")

    # Per-regime R^2 with kept-1-param coefficients
    a0 = kept["fit_1param_T"]["a0"]
    a1 = kept["fit_1param_T"]["a1_logT"]
    print()
    print(f"Per-regime R^2 using 1-param universal fit (excluding {EXCLUDE}):")
    print(f"   Delta = {a0:+.4f} + {a1:+.4f} * log(T_00/<T_00>)")
    per_reg_r2_1param = []
    for rd in rdata_all:
        pred = a0 + a1 * rd["log_T_ratio"]
        y = rd["delta_tail"]
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        per_reg_r2_1param.append({"regime": rd["regime"], "N": rd["N"], "R_squared": r2})
        print(f"  {rd['regime']:<8} N={rd['N']:>3}: R^2 = {r2:+.4f}")

    if kept["fit_1param_T"]["R_squared"] >= 0.7:
        verdict_1p = "SIMPLIFIED_UNIVERSAL_FORM_HOLDS_T_00_ONLY"
    elif kept["fit_1param_T"]["R_squared"] >= 0.5:
        verdict_1p = "SIMPLIFIED_UNIVERSAL_FORM_PARTIAL"
    else:
        verdict_1p = "SIMPLIFIED_UNIVERSAL_FORM_FAILS_NEEDS_MORE_FEATURES"
    print()
    print(f"VERDICT (1-param T_00 only, excluding {EXCLUDE}): {verdict_1p}")
    print(f"VERDICT (3-param log_ratio, excluding {EXCLUDE}): "
          f"R^2 = {kept['fit_3param']['R_squared']:.4f}")

    out_path = REPO / "outputs" / "tail_universal_simplified_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "tail_universal_simplified_T_00_log_ratio_test",
            "schema_version": "1.0.0",
            "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
            "tail_fraction": TAIL_FRAC,
            "excluded_regimes": EXCLUDE,
            "all_regimes_fit": full,
            "kept_regimes_fit": kept,
            "per_regime_R_squared_1param": per_reg_r2_1param,
            "verdict_1param": verdict_1p,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
