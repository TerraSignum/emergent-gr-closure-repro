"""Formal matter-core-residual identity test.

Establish (or falsify) the functional identity

  Delta(a) = F(T_00(a), |grad T_00(a)|, Xi_a, omega_a, ...)

for the per-direction RELATIVE Frobenius residual on top-10%
matter-core nodes. If a multivariate fit on training regimes
holds with R^2 >= 0.7 and coefficients stable across regimes,
the matter-core residual is closed by an explicit functional
relation (no hidden parameter).

Predictors per node a:
  T_00(a)            -- matter energy density
  |grad T_00(a)|     -- lattice gradient magnitude of T_00
                         |grad T_00(a)|^2 = sum_b w_ab (T_00(b) - T_00(a))^2
  omega_a            -- weighted node-degree (Ricci-related)
  |psi_a|^2          -- amplitude squared
  Xi_max(a)          -- max edge-Xi at node a (local clumping)
  T_00 * |grad T_00| -- nonlinear coupling
  T_00^2             -- nonlinear self-coupling

Procedure:
  1. Pool tail (top-10% Delta) nodes across N=50,60,64,72,84.
  2. Linear regression Delta(a) = sum c_k * f_k(a)
  3. Hold-out test on N=100 tail.
  4. Same fit on the BULK (90% inner) and check if F is universal
     or tail-specific.

Output: outputs/tail_functional_identity_audit.json
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


def per_node_features(xi_mat, psi, k_field, q_field, t00, n_lat):
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

    # |grad T_00(a)|^2 = sum_b w_ab * (T_00(b) - T_00(a))^2
    diff_sq = (t00[None, :] - t00[:, None]) ** 2  # (n,n)
    grad_T_sq = (weight_grad * diff_sq).sum(axis=1)
    grad_T = np.sqrt(np.maximum(grad_T_sq, 0.0))

    psi_sq = np.abs(psi) ** 2
    xi_max = np.where(adj.sum(axis=1) > 0, xi_off.max(axis=1), 0.0)

    return {
        "T_00":           np.abs(t00),
        "grad_T_00":      grad_T,
        "omega_a":        omega_a,
        "psi_sq":         psi_sq,
        "xi_max":         xi_max,
        "T_grad_T":       np.abs(t00) * grad_T,
        "T_00_sq":        t00 ** 2,
    }


def per_node_delta(prep):
    res = per_node_eigendirection_residuals(prep, LAMBDA_T, LAMBDA_S)
    R_time = res["R_time"]; R_diag = res["R_diag"]; R_off = res["R_off"]
    t_eigs = res["T_eigvals"]; t00 = np.asarray(prep["t00"])
    R_norm = np.sqrt(R_time ** 2 + (R_diag ** 2).sum(axis=1) + R_off ** 2)
    T_norm = np.sqrt(t00 ** 2 + (t_eigs ** 2).sum(axis=1))
    return R_norm / np.maximum(T_norm, 1e-12)


def gather(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)
    deltas, feats_pool = [], None
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        delta = per_node_delta(prep)
        t00 = np.asarray(prep["t00"])
        feats = per_node_features(xi_mat, psi, k_field, q_field, t00, n_lat)
        deltas.append(delta)
        if feats_pool is None:
            feats_pool = {k: [v] for k, v in feats.items()}
        else:
            for k, v in feats.items():
                feats_pool[k].append(v)
    return (np.concatenate(deltas),
            {k: np.concatenate(v) for k, v in feats_pool.items()})


def build_design_matrix(feats, keys):
    cols = [feats[k] for k in keys]
    X = np.stack(cols, axis=1)
    # Standardize for numerical stability
    means = X.mean(axis=0)
    stds = X.std(axis=0) + 1e-12
    X_std = (X - means) / stds
    # Add intercept
    X_with_b = np.column_stack([np.ones(X.shape[0]), X_std])
    return X_with_b, means, stds


def linreg_fit(X, y, ridge: float = 1e-6):
    """Solve X c = y via ridge-regularised normal equations,
    fall back to lstsq with rcond. Returns c, R^2, predictions."""
    try:
        c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    except np.linalg.LinAlgError:
        # Tikhonov ridge: (X^T X + lambda I) c = X^T y
        XtX = X.T @ X
        n_p = XtX.shape[0]
        c = np.linalg.solve(XtX + ridge * np.eye(n_p), X.T @ y)
    pred = X @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return c, r2, pred


def main() -> int:
    feature_keys = ["T_00", "grad_T_00", "omega_a", "psi_sq",
                    "xi_max", "T_grad_T", "T_00_sq"]

    print("=" * 110)
    print("Tail functional identity audit:")
    print("  Delta(a) = F(T_00(a), |grad T_00(a)|, Xi_a, omega_a, ...) ?")
    print("=" * 110)
    print()
    print(f"  Predictors: {feature_keys}")
    print(f"  Tail: top-{TAIL_FRAC*100:.0f}% by Delta(a)")
    print()

    pool_train_tail_X = []
    pool_train_tail_y = []
    pool_train_bulk_X = []
    pool_train_bulk_y = []
    holdout_tail_X = None
    holdout_tail_y = None
    holdout_bulk_X = None
    holdout_bulk_y = None
    holdout_N = 100

    train_means = train_stds = None

    # Per-regime fit (independent + pooled)
    per_reg_fits_tail = {}
    per_reg_fits_bulk = {}
    print(f"{'reg':<8} {'N':>3} | {'tail R^2':>9} {'bulk R^2':>9} | {'tail c_T_00':>12} {'tail c_gradT':>13}")
    print("-" * 90)

    for reg, n_lat in REGIMES:
        gt = gather(reg, n_lat)
        if gt is None:
            continue
        delta, feats = gt
        n_total = len(delta)
        n_tail = max(1, int(n_total * TAIL_FRAC))
        order = np.argsort(-delta)
        tail_idx = order[:n_tail]
        bulk_idx = order[n_tail:]

        feats_tail = {k: feats[k][tail_idx] for k in feature_keys}
        feats_bulk = {k: feats[k][bulk_idx] for k in feature_keys}

        X_tail, mt, st = build_design_matrix(feats_tail, feature_keys)
        X_bulk, mb, sb = build_design_matrix(feats_bulk, feature_keys)
        y_tail = delta[tail_idx]
        y_bulk = delta[bulk_idx]
        c_tail, r2_tail, _ = linreg_fit(X_tail, y_tail)
        c_bulk, r2_bulk, _ = linreg_fit(X_bulk, y_bulk)
        per_reg_fits_tail[f"{reg}_N{n_lat}"] = {
            "n": int(n_tail),
            "r_squared": r2_tail,
            "coeffs_intercept_then_features": [float(x) for x in c_tail],
            "feature_keys": feature_keys,
            "feature_means": [float(x) for x in mt],
            "feature_stds": [float(x) for x in st],
        }
        per_reg_fits_bulk[f"{reg}_N{n_lat}"] = {
            "n": int(len(bulk_idx)),
            "r_squared": r2_bulk,
            "coeffs_intercept_then_features": [float(x) for x in c_bulk],
        }
        print(f"{reg:<8} {n_lat:>3} | {r2_tail:>9.3f} {r2_bulk:>9.3f} | "
              f"{c_tail[1]:>+12.5f} {c_tail[2]:>+13.5f}")

        if n_lat == holdout_N:
            holdout_tail_X = X_tail; holdout_tail_y = y_tail
            holdout_bulk_X = X_bulk; holdout_bulk_y = y_bulk
        else:
            pool_train_tail_X.append(X_tail); pool_train_tail_y.append(y_tail)
            pool_train_bulk_X.append(X_bulk); pool_train_bulk_y.append(y_bulk)

    # Pooled training fit
    Xt = np.vstack(pool_train_tail_X); yt = np.concatenate(pool_train_tail_y)
    Xb = np.vstack(pool_train_bulk_X); yb = np.concatenate(pool_train_bulk_y)
    c_tail_pool, r2_tail_train, _ = linreg_fit(Xt, yt)
    c_bulk_pool, r2_bulk_train, _ = linreg_fit(Xb, yb)

    print()
    print(f"Pooled training fit (N != {holdout_N}):")
    print(f"  Tail R^2 train: {r2_tail_train:.4f}")
    print(f"  Bulk R^2 train: {r2_bulk_train:.4f}")

    # Holdout R^2 (apply pooled-train coefficients to holdout)
    if holdout_tail_X is not None:
        pred_tail_ho = holdout_tail_X @ c_tail_pool
        ss_res = float(((holdout_tail_y - pred_tail_ho) ** 2).sum())
        ss_tot = float(((holdout_tail_y - holdout_tail_y.mean()) ** 2).sum())
        r2_tail_ho = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        r2_tail_ho = float("nan")

    if holdout_bulk_X is not None:
        pred_bulk_ho = holdout_bulk_X @ c_bulk_pool
        ss_res = float(((holdout_bulk_y - pred_bulk_ho) ** 2).sum())
        ss_tot = float(((holdout_bulk_y - holdout_bulk_y.mean()) ** 2).sum())
        r2_bulk_ho = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        r2_bulk_ho = float("nan")

    print(f"  Holdout R^2 tail (N={holdout_N}): {r2_tail_ho:.4f}")
    print(f"  Holdout R^2 bulk (N={holdout_N}): {r2_bulk_ho:.4f}")

    # Coefficient interpretation for tail
    print()
    print(f"Tail-fit coefficients (pooled training, standardized features):")
    print(f"  intercept = {c_tail_pool[0]:+.4f}")
    for i, k in enumerate(feature_keys, 1):
        print(f"  c_{k:<14} = {c_tail_pool[i]:+.4f}")

    # Verdict
    if r2_tail_ho >= 0.7 and r2_tail_train >= 0.7:
        verdict = "TAIL_FUNCTIONAL_IDENTITY_HOLDS"
    elif r2_tail_ho >= 0.5:
        verdict = "TAIL_PARTIALLY_PREDICTABLE"
    else:
        verdict = "TAIL_NOT_REDUCIBLE_TO_LINEAR_FEATURES"
    print()
    print(f"VERDICT: {verdict}")

    out_path = REPO / "outputs" / "tail_functional_identity_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "matter_core_residual_functional_identity_test",
            "schema_version": "1.0.0",
            "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
            "tail_fraction": TAIL_FRAC,
            "feature_keys": feature_keys,
            "per_regime_fits": {
                "tail": per_reg_fits_tail,
                "bulk": per_reg_fits_bulk,
            },
            "pooled_train_tail_coeffs": [float(x) for x in c_tail_pool],
            "pooled_train_bulk_coeffs": [float(x) for x in c_bulk_pool],
            "train_R_squared_tail": r2_tail_train,
            "train_R_squared_bulk": r2_bulk_train,
            "holdout_N": holdout_N,
            "holdout_R_squared_tail": r2_tail_ho,
            "holdout_R_squared_bulk": r2_bulk_ho,
            "verdict": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
