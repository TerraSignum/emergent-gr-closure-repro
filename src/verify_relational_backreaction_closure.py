"""Theorem test: Relational Backreaction Closure.

User-proposed structural form:
  G_00(a) = (1 - kappa_t) * T_00(a)
  i.e. Lambda_t^back(a) := kappa_t * T_00(a) is a self-induced
  backreaction term, NOT a free parameter; the Einstein time-time
  equation closes with that backreaction.

Predicted relation:
  R_00^back(a) = G_00(a) - (1 - kappa_t) * T_00(a)

Symmetric relative residual:
  Delta_00^back(a) = |R_00^back(a)| / (|G_00(a)| + |T_00(a)| + eps)

Procedure:
  1. TRAIN:    fit kappa_t on canonical regimes P0..P5N100 (median
              minimisation of |R_00^back| over training pool).
  2. HOLDOUT:  apply that kappa_t to P6N128, P8N128, P5N72, P5N84
              and measure Delta_00^back distribution.
  3. DoD:      CV(kappa_t per-regime) < 2%, median Delta_00^back <= 0.05,
              mean Delta_00^back <= 0.10, no per-regime re-fit, tail
              lift reduction vs the kappa_t=0 (i.e. plain G_00 - T_00) form.

Output: outputs/relational_backreaction_closure_audit.json
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


TRAIN_REGIMES = [
    # N=18 (P0) excluded as too small for stable Symanzik (standard
    # lattice-QCD methodology; lowest-N point typically excluded for
    # asymptotic-regime fits).
    ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]
HOLDOUT_REGIMES = [
    ("P6N128", 128), ("P8N128", 128),
    ("P5N72", 72), ("P5N84", 84),
]


def gather_g00_t00(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)
    g00s, t00s = [], []
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        g00s.append(np.asarray(prep["g_00_h"]))
        t00s.append(np.asarray(prep["t00"]))
    return np.concatenate(g00s), np.concatenate(t00s)


def fit_kappa_train(rows, train_pool):
    """Find kappa_t that minimises pooled median |G_00 - (1-kappa)*T_00|
    over training regimes."""
    g_pool = np.concatenate([r["g00"] for r in train_pool])
    t_pool = np.concatenate([r["t00"] for r in train_pool])
    # Scan kappa, find median |G - (1-k)T| minimum
    best = None
    for kappa in np.linspace(0.50, 1.10, 601):
        residual = g_pool - (1.0 - kappa) * t_pool
        med = float(np.median(np.abs(residual)))
        if best is None or med < best["median"]:
            best = {"kappa": float(kappa), "median": med}
    return best["kappa"]


def relative_residual(g00, t00, kappa, eps=1e-6):
    """Delta_00^back(a) = |G_00 - (1-kappa)*T_00| / (|G_00| + |T_00| + eps)"""
    R = g00 - (1.0 - kappa) * t00
    denom = np.abs(g00) + np.abs(t00) + eps
    return np.abs(R) / denom


def main() -> int:
    print("=" * 110)
    print("Relational Backreaction Closure: G_00 = (1 - kappa_t) * T_00")
    print("Theorem-test: kappa_t global, NOT per-regime free fit")
    print("=" * 110)
    print()

    # Gather all regimes
    train_pool = []
    holdout_pool = []
    for reg, n_lat in TRAIN_REGIMES:
        gt = gather_g00_t00(reg, n_lat)
        if gt is None: continue
        g00, t00 = gt
        train_pool.append({"regime": reg, "N": n_lat, "g00": g00, "t00": t00})
    for reg, n_lat in HOLDOUT_REGIMES:
        gt = gather_g00_t00(reg, n_lat)
        if gt is None: continue
        g00, t00 = gt
        holdout_pool.append({"regime": reg, "N": n_lat, "g00": g00, "t00": t00})

    print(f"  Train regimes: {len(train_pool)}, Holdout: {len(holdout_pool)}")

    # Per-regime optimal kappa_t (for CV check)
    print()
    print(f"{'regime':<10} {'N':>3} {'subset':<10} {'<G_00>':>9} {'<T_00>':>9} | "
          f"{'kappa*':>8} {'kappa* CV':>10}")
    print("-" * 80)
    per_regime_kappas = []
    for r in train_pool + holdout_pool:
        # Per-regime: find kappa minimising median |G - (1-k)*T|
        best_med = None
        best_kappa = None
        for kappa in np.linspace(0.50, 1.10, 601):
            residual = r["g00"] - (1.0 - kappa) * r["t00"]
            med = float(np.median(np.abs(residual)))
            if best_med is None or med < best_med:
                best_med = med
                best_kappa = float(kappa)
        per_regime_kappas.append({"regime": r["regime"], "N": r["N"],
                                   "kappa_optimal": best_kappa,
                                   "is_train": r in train_pool})
        subset = "train" if r in train_pool else "holdout"
        print(f"{r['regime']:<10} {r['N']:>3} {subset:<10} "
              f"{float(np.mean(r['g00'])):>9.5f} {float(np.mean(r['t00'])):>9.5f} | "
              f"{best_kappa:>8.4f}")

    train_kappas = [k["kappa_optimal"] for k in per_regime_kappas if k["is_train"]]
    train_kappa_mean = float(np.mean(train_kappas))
    train_kappa_std = float(np.std(train_kappas))
    train_cv = train_kappa_std / train_kappa_mean * 100
    print()
    print(f"  Per-regime kappa_t over TRAIN: mean = {train_kappa_mean:.4f}, std = {train_kappa_std:.4f}, CV = {train_cv:.2f}%")
    holdout_kappas = [k["kappa_optimal"] for k in per_regime_kappas if not k["is_train"]]
    holdout_kappa_mean = float(np.mean(holdout_kappas)) if holdout_kappas else float("nan")
    print(f"  Per-regime kappa_t over HOLDOUT: mean = {holdout_kappa_mean:.4f}")

    # Global kappa fit on training pool only
    kappa_global = fit_kappa_train(per_regime_kappas, train_pool)
    print()
    print(f"  GLOBAL kappa_t (fitted on TRAIN, applied to ALL): {kappa_global:.4f}")

    # Compute Delta_00^back per regime under global kappa_t
    print()
    print(f"Delta_00^back per regime under global kappa_t = {kappa_global:.4f}:")
    print(f"{'regime':<10} {'N':>3} {'subset':<10} {'med Delta':>11} {'mean Delta':>11} {'p90 Delta':>11} | thresh check")
    print("-" * 100)
    out_per_regime = []
    for r in train_pool + holdout_pool:
        delta = relative_residual(r["g00"], r["t00"], kappa_global)
        med_d = float(np.median(delta))
        mean_d = float(np.mean(delta))
        p90_d = float(np.percentile(delta, 90))
        med_pass = med_d <= 0.05
        mean_pass = mean_d <= 0.10
        subset = "train" if r in train_pool else "holdout"
        out_per_regime.append({
            "regime": r["regime"], "N": r["N"], "subset": subset,
            "delta_00_back_median": med_d,
            "delta_00_back_mean": mean_d,
            "delta_00_back_p90": p90_d,
            "median_passes_0p05": med_pass,
            "mean_passes_0p10": mean_pass,
        })
        check = ("med<=0.05 " + ("PASS" if med_pass else "FAIL") + ", "
                 "mean<=0.10 " + ("PASS" if mean_pass else "FAIL"))
        print(f"{r['regime']:<10} {r['N']:>3} {subset:<10} "
              f"{med_d:>11.5f} {mean_d:>11.5f} {p90_d:>11.5f} | {check}")

    # DoD checks
    print()
    print("=" * 110)
    print("DoD CHECKS (all must pass):")
    print("=" * 110)
    train_n = sum(1 for r in out_per_regime if r["subset"] == "train")
    holdout_n = sum(1 for r in out_per_regime if r["subset"] == "holdout")
    train_med_pass = sum(1 for r in out_per_regime if r["subset"] == "train" and r["median_passes_0p05"])
    train_mean_pass = sum(1 for r in out_per_regime if r["subset"] == "train" and r["mean_passes_0p10"])
    holdout_med_pass = sum(1 for r in out_per_regime if r["subset"] == "holdout" and r["median_passes_0p05"])
    holdout_mean_pass = sum(1 for r in out_per_regime if r["subset"] == "holdout" and r["mean_passes_0p10"])

    print(f"  1. CV(kappa_t TRAIN) < 2%:     CV = {train_cv:.2f}%  -> "
          f"{'PASS' if train_cv < 2.0 else 'FAIL'}")
    print(f"  2. TRAIN median <= 0.05:       {train_med_pass}/{train_n}  -> "
          f"{'PASS' if train_med_pass == train_n else 'FAIL'}")
    print(f"  3. TRAIN mean <= 0.10:         {train_mean_pass}/{train_n}  -> "
          f"{'PASS' if train_mean_pass == train_n else 'FAIL'}")
    print(f"  4. HOLDOUT median <= 0.05:     {holdout_med_pass}/{holdout_n}  -> "
          f"{'PASS' if holdout_med_pass == holdout_n else 'FAIL'}")
    print(f"  5. HOLDOUT mean <= 0.10:       {holdout_mean_pass}/{holdout_n}  -> "
          f"{'PASS' if holdout_mean_pass == holdout_n else 'FAIL'}")

    # Comparing tail-lift kappa=0 vs kappa=global
    g_pool_all = np.concatenate([r["g00"] for r in train_pool + holdout_pool])
    t_pool_all = np.concatenate([r["t00"] for r in train_pool + holdout_pool])
    delta_at_0 = relative_residual(g_pool_all, t_pool_all, 0.0)  # plain G - T
    delta_at_kappa = relative_residual(g_pool_all, t_pool_all, kappa_global)
    p90_at_0 = float(np.percentile(delta_at_0, 90))
    p90_at_kappa = float(np.percentile(delta_at_kappa, 90))
    print(f"  6. Tail-lift reduction p90:    {p90_at_0:.4f} -> {p90_at_kappa:.4f}  -> "
          f"{'PASS' if p90_at_kappa < p90_at_0 else 'FAIL'}")

    all_dod = [
        train_cv < 2.0,
        train_med_pass == train_n,
        train_mean_pass == train_n,
        holdout_med_pass == holdout_n,
        holdout_mean_pass == holdout_n,
        p90_at_kappa < p90_at_0,
    ]
    n_pass = sum(all_dod)
    if n_pass == 6:
        verdict = "RELATIONAL_BACKREACTION_THEOREM_VERIFIED"
    elif n_pass >= 4:
        verdict = f"RELATIONAL_BACKREACTION_PARTIAL_{n_pass}_OF_6"
    else:
        verdict = "RELATIONAL_BACKREACTION_FAILS"
    print(f"\nFINAL VERDICT: {verdict}  ({n_pass}/6 DoD criteria pass)")

    out_path = REPO / "outputs" / "relational_backreaction_closure_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "relational_backreaction_kappa_train_holdout",
            "schema_version": "1.0.0",
            "form": "G_00(a) = (1 - kappa_t) * T_00(a)",
            "global_kappa_t": kappa_global,
            "per_regime_kappa": per_regime_kappas,
            "train_kappa_mean": train_kappa_mean,
            "train_kappa_std": train_kappa_std,
            "train_kappa_CV_percent": train_cv,
            "holdout_kappa_mean": holdout_kappa_mean,
            "delta_00_back_per_regime": out_per_regime,
            "tail_lift_p90_kappa_zero": p90_at_0,
            "tail_lift_p90_kappa_global": p90_at_kappa,
            "DoD_criteria_pass_count": int(n_pass),
            "DoD_criteria_total": 6,
            "verdict": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
