"""Three candidates for R_time closure with full T_00-decomposition.

Decompose T_00 into kinetic (gradient), reconstruction, and frustration:
  T_00^Xi = T_00^grad + zeta_3*Omega*(A_K*K + A_Q*(1-Q))
         = T_00^kin   + T_00^rec

(zeta_2*Omega*f frustration term: tested separately if zeta_2 in code.)

Three candidates:
  (A) No-backreaction baseline:  R_time^(0) = G_00 - T_00^Xi
  (B) Reconstruction-renormalized: Lambda_t^rec = zeta_3*Omega*(A_K*K+A_Q*(1-Q))
      R_time^rec = G_00 + Lambda_t^rec - T_00^Xi = G_00 - T_00^kin
  (C) Density-contrast core correction:
      Lambda_t^core = Lambda_t^rec + c_core * log(T_00/<T_00>)
      R_time^core = G_00 + Lambda_t^core - T_00^Xi
      c_core fitted on TRAIN, applied to HOLDOUT.

DoD per candidate:
  - Train median <= 0.05
  - Train mean <= 0.10
  - Holdout median <= 0.05
  - Holdout mean <= 0.10
  - Tail p90 reduced vs (A) baseline
  - Parameter-CV stable across regimes (ideal <5%)

Critical test: does Lambda_t^rec(a)/T_00(a) -> const across regimes?
If yes -> reconstruction IS the backreaction term, not a free fit.

Output: outputs/t00_decomposition_three_candidates_audit.json
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
    edge_to_matrix, hessian_ricci_per_node, ELL_0, D_MIN, EPS_D, XI_THRESH,
    Z_XI, KAPPA_XI, ZETA_1, OMEGA, ZETA_3, A_K, A_Q)


TRAIN_REGIMES = [
    ("P1", 28), ("P3", 36), ("P4", 42), ("P5", 50), ("P5N64", 64),
    ("P6", 60), ("P7", 72), ("P8", 84), ("P5N100", 100),
]
HOLDOUT_REGIMES = [
    ("P5N72", 72), ("P5N84", 84), ("P6N128", 128), ("P8N128", 128),
]


def gather_decomposed(reg, n_lat):
    """Return per-node arrays of G_00, T_00, T_00_kin, T_00_rec, K_rec(a)."""
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists(): return None
    d = np.load(p, allow_pickle=True)
    e = d["dense_cell_edge_xi_values"]
    a_arr = d["dense_cell_node_amplitude_values"]
    ph = d["dense_cell_node_phase_values"]
    g_pool, t_pool, t_kin_pool, t_rec_pool, k_rec_pool = [], [], [], [], []
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

        # Decomposed T_00
        spatial_diff = spatial[None, :, :] - spatial[:, None, :]
        inv_d = np.where(adj > 0, 1.0 / d_mat, 0.0)
        psi_diff = psi[None, :] - psi[:, None]
        weight_term = weight_grad[:, :, None] * inv_d[:, :, None]
        grad_psi = (psi_diff[:, :, None] * spatial_diff * weight_term).sum(
            axis=1) / (omega_a[:, None] + 1e-12)
        norm_sq = (np.abs(grad_psi) ** 2).sum(axis=1)
        xi_row_mean = (weight_adj.sum(axis=1) / (adj.sum(axis=1) + 1e-12))
        var_xi = (((weight_adj - xi_row_mean[:, None]) ** 2 * adj).sum(axis=1)
                  / (adj.sum(axis=1) + 1e-12))
        amp_a = np.abs(psi)
        var_amp = (amp_a - amp_a.mean()) ** 2
        # T_00^kin: variance + amplitude + gradient (no K_rec)
        t00_kin = (0.5 * Z_XI * var_xi
                   + KAPPA_XI * var_amp
                   + ZETA_1 * OMEGA * norm_sq)
        # T_00^rec: K_rec contribution only
        k_per = (k_field * adj).sum(axis=1) / (adj.sum(axis=1) + 1e-12)
        q_per = (q_field * adj).sum(axis=1) / (adj.sum(axis=1) + 1e-12)
        k_rec = A_K * k_per + A_Q * (1.0 - q_per)
        t00_rec = ZETA_3 * OMEGA * k_rec
        t00_total = t00_kin + t00_rec

        g_pool.append(g_00); t_pool.append(t00_total)
        t_kin_pool.append(t00_kin); t_rec_pool.append(t00_rec)
        k_rec_pool.append(k_rec)
    return {
        "regime": reg, "N": n_lat,
        "G_00": np.concatenate(g_pool),
        "T_00": np.concatenate(t_pool),
        "T_00_kin": np.concatenate(t_kin_pool),
        "T_00_rec": np.concatenate(t_rec_pool),
        "K_rec": np.concatenate(k_rec_pool),
    }


def relative_residual(R, G, T, eps=1e-6):
    """Delta(a) = |R(a)| / (|G(a)| + |T(a)| + eps)"""
    return np.abs(R) / (np.abs(G) + np.abs(T) + eps)


def stats(delta):
    return {
        "median": float(np.median(delta)),
        "mean": float(np.mean(delta)),
        "p90": float(np.percentile(delta, 90)),
    }


def main() -> int:
    print("=" * 110)
    print("Three R_time candidates with T_00 decomposition: T_00 = T_00^kin + T_00^rec")
    print("=" * 110)

    train_pool, holdout_pool = [], []
    for reg, n in TRAIN_REGIMES:
        r = gather_decomposed(reg, n)
        if r is not None: train_pool.append(r)
    for reg, n in HOLDOUT_REGIMES:
        r = gather_decomposed(reg, n)
        if r is not None: holdout_pool.append(r)

    print(f"\nTrain regimes: {len(train_pool)}, Holdout regimes: {len(holdout_pool)}")

    # ============================================================
    # CRITICAL TEST: does Lambda_t^rec(a)/T_00(a) -> const across regimes?
    # If reconstruction IS backreaction, the ratio per regime should
    # approach a stable per-node constant (low CV).
    # ============================================================
    print()
    print("=" * 110)
    print("CRITICAL TEST: Lambda_t^rec(a) / T_00(a) per regime — is it ~ kappa_t per node?")
    print("=" * 110)
    print(f"{'reg':<10} {'N':>3} | {'<T00rec/T00>_med':>17} {'std':>8} {'CV %':>7} | {'<T00rec/T00>_mean':>17}")
    print("-" * 90)
    rows_ratio = []
    for r in train_pool + holdout_pool:
        mask = r["T_00"] > 0.01
        ratio = r["T_00_rec"][mask] / r["T_00"][mask]
        med = float(np.median(ratio))
        std = float(np.std(ratio))
        cv = std/abs(med)*100 if abs(med) > 1e-12 else float("nan")
        mean = float(np.mean(ratio))
        rows_ratio.append({"regime": r["regime"], "N": r["N"],
                           "ratio_med": med, "ratio_std": std,
                           "ratio_CV_pct": cv, "ratio_mean": mean,
                           "subset": "train" if r in train_pool else "holdout"})
        print(f"{r['regime']:<10} {r['N']:>3} | {med:>17.4f} {std:>8.4f} {cv:>7.2f} | {mean:>17.4f}")

    cross_reg_meds = [r["ratio_med"] for r in rows_ratio]
    cross_reg_spread = float(max(cross_reg_meds) - min(cross_reg_meds))
    print(f"\n  Cross-regime spread of median(T_00^rec/T_00): {cross_reg_spread:.4f}")
    if cross_reg_spread < 0.05:
        critical_verdict = "RECONSTRUCTION_RATIO_REGIME_INVARIANT"
    elif cross_reg_spread < 0.15:
        critical_verdict = "RECONSTRUCTION_RATIO_WEAKLY_VARIANT"
    else:
        critical_verdict = "RECONSTRUCTION_RATIO_REGIME_VARIANT"
    print(f"  CRITICAL VERDICT: {critical_verdict}")

    # ============================================================
    # CANDIDATES A, B (no fit), C (fit c_core on TRAIN)
    # ============================================================
    def evaluate_candidate(pool, label_set, lambda_func, c_core=None):
        """For each regime in pool, compute Delta = |R_time| / (|G|+|T|+eps)."""
        rows = []
        for r in pool:
            G = r["G_00"]; T = r["T_00"]
            if lambda_func is not None:
                lt = lambda_func(r, c_core)
                R = G + lt - T
            else:
                R = G - T
            mask = (T > 0.01)
            delta = relative_residual(R[mask], G[mask], T[mask])
            s = stats(delta)
            rows.append({"regime": r["regime"], "N": r["N"], **s,
                         "subset": label_set})
        return rows

    def lambda_rec(r, _c_core=None):
        return r["T_00_rec"]

    # Fit c_core on TRAIN with mean(R_time^core) = 0
    # R_core = G + (T_rec + c_core * log(T/<T>)) - T = G - T_kin + c_core * log(T/<T>)
    # min sum |R_core|^2 over c_core: linear regression
    g_train = np.concatenate([r["G_00"] for r in train_pool])
    t_train = np.concatenate([r["T_00"] for r in train_pool])
    tkin_train = np.concatenate([r["T_00_kin"] for r in train_pool])
    trec_train = np.concatenate([r["T_00_rec"] for r in train_pool])
    mask = (t_train > 0.01)
    g_train = g_train[mask]; t_train = t_train[mask]
    tkin_train = tkin_train[mask]; trec_train = trec_train[mask]
    t_mean_train = float(np.mean(t_train))
    log_r = np.log(t_train / t_mean_train)
    # R_core = G - T_kin + c_core * log_r — we want sum R^2 minimised
    # i.e. c_core = - sum (G - T_kin) * log_r / sum log_r^2
    # but actually we want median |R_core|=0 ideally. For least-squares mean=0:
    # sum(G - T_kin + c_core * log_r) = 0 doesn't constrain c_core (log_r mean ~ 0)
    # Use least-squares min sum (G - T_kin + c_core * log_r)^2:
    base = g_train - tkin_train  # = G - (T - T_rec) = G - T + T_rec
    c_core_fit = float(-np.sum(base * log_r) / np.sum(log_r**2))
    print()
    print(f"Fitted c_core on TRAIN: {c_core_fit:+.5f}")

    def lambda_core(r, c_core):
        t_mean = float(np.mean(r["T_00"]))
        log_r = np.log(np.maximum(r["T_00"]/max(t_mean, 1e-12), 1e-10))
        return r["T_00_rec"] + c_core * log_r

    # Candidate A: no backreaction
    print()
    print("CANDIDATE A — No backreaction (R_time^(0) = G_00 - T_00):")
    rows_A_train = evaluate_candidate(train_pool, "train", None)
    rows_A_holdout = evaluate_candidate(holdout_pool, "holdout", None)
    print(f"{'reg':<10} {'N':>3} {'subset':<10} | {'med':>8} {'mean':>8} {'p90':>8}")
    for r in rows_A_train + rows_A_holdout:
        print(f"  {r['regime']:<10} {r['N']:>3} {r['subset']:<10} | {r['median']:>8.4f} {r['mean']:>8.4f} {r['p90']:>8.4f}")

    # Candidate B: reconstruction
    print()
    print("CANDIDATE B — Reconstruction-renormalized (Lambda_t = T_00^rec):")
    rows_B_train = evaluate_candidate(train_pool, "train", lambda_rec)
    rows_B_holdout = evaluate_candidate(holdout_pool, "holdout", lambda_rec)
    print(f"{'reg':<10} {'N':>3} {'subset':<10} | {'med':>8} {'mean':>8} {'p90':>8}")
    for r in rows_B_train + rows_B_holdout:
        print(f"  {r['regime']:<10} {r['N']:>3} {r['subset']:<10} | {r['median']:>8.4f} {r['mean']:>8.4f} {r['p90']:>8.4f}")

    # Candidate C: rec + c_core * log(T/<T>)
    print()
    print(f"CANDIDATE C — Density-contrast core (Lambda_t = T_00^rec + c_core*log(T/<T>)), c_core = {c_core_fit:+.5f}:")
    rows_C_train = evaluate_candidate(train_pool, "train", lambda_core, c_core_fit)
    rows_C_holdout = evaluate_candidate(holdout_pool, "holdout", lambda_core, c_core_fit)
    print(f"{'reg':<10} {'N':>3} {'subset':<10} | {'med':>8} {'mean':>8} {'p90':>8}")
    for r in rows_C_train + rows_C_holdout:
        print(f"  {r['regime']:<10} {r['N']:>3} {r['subset']:<10} | {r['median']:>8.4f} {r['mean']:>8.4f} {r['p90']:>8.4f}")

    # Aggregate DoD
    def aggregate_dod(rows_train, rows_holdout, label):
        train_med_pass = sum(1 for r in rows_train if r["median"] <= 0.05) / max(len(rows_train), 1)
        train_mean_pass = sum(1 for r in rows_train if r["mean"] <= 0.10) / max(len(rows_train), 1)
        holdout_med_pass = sum(1 for r in rows_holdout if r["median"] <= 0.05) / max(len(rows_holdout), 1)
        holdout_mean_pass = sum(1 for r in rows_holdout if r["mean"] <= 0.10) / max(len(rows_holdout), 1)
        train_p90 = float(np.mean([r["p90"] for r in rows_train]))
        return {
            "label": label,
            "train_med_pass_frac": train_med_pass,
            "train_mean_pass_frac": train_mean_pass,
            "holdout_med_pass_frac": holdout_med_pass,
            "holdout_mean_pass_frac": holdout_mean_pass,
            "train_p90_mean": train_p90,
        }

    A = aggregate_dod(rows_A_train, rows_A_holdout, "A_no_backreaction")
    B = aggregate_dod(rows_B_train, rows_B_holdout, "B_reconstruction")
    C = aggregate_dod(rows_C_train, rows_C_holdout, "C_core_corrected")

    print()
    print("=" * 110)
    print("DoD aggregate")
    print("=" * 110)
    print(f"{'cand':<25} {'train_med<=0.05':>17} {'train_mean<=0.10':>18} {'holdout_med':>12} {'holdout_mean':>13} {'train_p90':>11}")
    for X in (A, B, C):
        print(f"  {X['label']:<23} {X['train_med_pass_frac']*100:>14.0f}%   "
              f"{X['train_mean_pass_frac']*100:>14.0f}%    "
              f"{X['holdout_med_pass_frac']*100:>10.0f}%   "
              f"{X['holdout_mean_pass_frac']*100:>11.0f}%  "
              f"{X['train_p90_mean']:>11.4f}")

    # Verdict
    pass_count = lambda X: int(X["train_med_pass_frac"] == 1.0) + int(X["train_mean_pass_frac"] == 1.0) + \
                            int(X["holdout_med_pass_frac"] == 1.0) + int(X["holdout_mean_pass_frac"] == 1.0)
    print()
    print(f"  A: {pass_count(A)}/4 DoD criteria")
    print(f"  B: {pass_count(B)}/4 DoD criteria")
    print(f"  C: {pass_count(C)}/4 DoD criteria")
    print(f"  Tail-p90 reduction A->B: {A['train_p90_mean']:.4f} -> {B['train_p90_mean']:.4f}  ({(B['train_p90_mean']-A['train_p90_mean'])/A['train_p90_mean']*100:+.1f}%)")
    print(f"  Tail-p90 reduction A->C: {A['train_p90_mean']:.4f} -> {C['train_p90_mean']:.4f}  ({(C['train_p90_mean']-A['train_p90_mean'])/A['train_p90_mean']*100:+.1f}%)")

    out = {
        "method": "T_00_decomposition_three_candidates_train_holdout_DoD",
        "schema_version": "1.0.0",
        "critical_test_T00rec_over_T00": {
            "per_regime": rows_ratio,
            "cross_regime_spread": cross_reg_spread,
            "verdict": critical_verdict,
        },
        "c_core_fitted_on_train": c_core_fit,
        "candidate_A_no_backreaction": {
            "train": rows_A_train, "holdout": rows_A_holdout, "aggregate": A,
        },
        "candidate_B_reconstruction": {
            "train": rows_B_train, "holdout": rows_B_holdout, "aggregate": B,
        },
        "candidate_C_core_corrected": {
            "train": rows_C_train, "holdout": rows_C_holdout, "aggregate": C,
        },
    }
    out_path = REPO / "outputs" / "t00_decomposition_three_candidates_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
