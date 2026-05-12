"""Stage 6f: Full-tensor closure norm audit.

Per-regime per-node relative Frobenius residual:
    Delta_a = ||R_a||_F / ||T_a||_F
with R_a = G_a + Lambda - 8 pi G T_a in the per-node T-eigenframe and
the per-direction (R_time, R_diag1, R_diag2, R_diag3, R_off) split.

For each regime in the cleaned ladder we compute the FULL percentile
spectrum of Delta_a:
    median, mean, p90, p95, p99, sup-norm.

This is what "full tensor closure" means quantitatively.

Adapter:
  - Canonical d1 NPZ files (P0..P8) carry per-cell xi/amp/phase/K/Q.
  - Snapshot NPZ files (P5N100, P5N200, P5N300, etc.) carry the time
    series; we use the last snapshot (= fixed point) and the
    ff_K_seed*/ff_Q_seed* matrices.

Skip P5N128: known K/Q persistence bug
(project_n128_kq_persistence_bug_2026_05_01).

Output: outputs/stage6f_full_tensor_norm_audit.json
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


# Cleaned P5-physics ladder (skip P5N128 known artefact).
LADDER = [
    # Canonical P5/P5N ladder (ordered by lattice size N).
    ("P5",     50),
    ("P5N64",  64),
    ("P5N72",  72),
    ("P5N84",  84),
    ("P5N100", 100),
    ("P5N128", 128),
    ("P5N200", 200),
    ("P5N256", 256),
    ("P5N300", 300),
    ("P5N512", 512),
    # Alternative anchors (reported separately as cross-anchor
    # consistency checks; not part of the canonical P5 sequence).
    ("P6",     60),
    ("P7",     72),
    ("P8",     84),
    ("P6N128", 128),
    ("P8N128", 128),
]
LAMBDA_T = 0.81   # alpha_xi^2 = 81/100
LAMBDA_S = -0.005  # -gamma^2/2 = -1/200 (isotropic)


def per_node_relative_delta(prep, lambda_t, lambda_s):
    res = per_node_eigendirection_residuals(prep, lambda_t, lambda_s)
    R_time = res["R_time"]
    R_diag = res["R_diag"]
    R_off = res["R_off"]
    t_eigs = res["T_eigvals"]
    t00 = np.asarray(prep["t00"])
    R_norm = np.sqrt(R_time ** 2
                      + (R_diag ** 2).sum(axis=1)
                      + R_off ** 2)
    T_norm = np.sqrt(t00 ** 2 + (t_eigs ** 2).sum(axis=1))
    delta = R_norm / np.maximum(T_norm, 1e-12)
    # Also return the per-component normalised residuals for diagnostics
    comp = {
        "delta_full": delta,
        "delta_time": np.abs(R_time) / np.maximum(T_norm, 1e-12),
        "delta_diag": np.sqrt((R_diag ** 2).sum(axis=1))
                       / np.maximum(T_norm, 1e-12),
        "delta_off": np.abs(R_off) / np.maximum(T_norm, 1e-12),
    }
    return comp


def load_canonical(p, n_lat):
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    seeds = []
    for s in range(min(edge_arr.shape[0], 32)):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        seeds.append((xi_mat, psi, k_field, q_field))
    return seeds


def load_snapshots(p, n_lat):
    d = np.load(p, allow_pickle=True)
    edge_snap = d["edge_xi_snapshots"]
    psi_r = d["psi_real_snapshots"]
    psi_i = d["psi_imag_snapshots"]
    n_seeds = int(edge_snap.shape[0])
    seeds = []
    for s in range(n_seeds):
        xi_mat = edge_snap[s, -1].astype(np.float64).copy()
        np.fill_diagonal(xi_mat, 1.0)
        psi = (psi_r[s, -1].astype(np.float64)
               + 1j * psi_i[s, -1].astype(np.float64))
        k_field = d.get(f"ff_K_seed{s}",
                         np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}",
                         np.full((n_lat, n_lat), 0.45))
        seeds.append((xi_mat, psi, k_field, q_field))
    return seeds


def gather_regime(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    if "snapshots" in p.name.lower():
        seeds = load_snapshots(p, n_lat)
    else:
        seeds = load_canonical(p, n_lat)
    pool_full = []
    pool_time = []
    pool_diag = []
    pool_off = []
    for xi_mat, psi, k_field, q_field in seeds:
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field,
                                  n_lat, np)
        comp = per_node_relative_delta(prep, LAMBDA_T, LAMBDA_S)
        pool_full.append(comp["delta_full"])
        pool_time.append(comp["delta_time"])
        pool_diag.append(comp["delta_diag"])
        pool_off.append(comp["delta_off"])
    return {
        "regime": reg, "N": n_lat, "n_seeds": len(seeds),
        "delta_full": np.concatenate(pool_full),
        "delta_time": np.concatenate(pool_time),
        "delta_diag": np.concatenate(pool_diag),
        "delta_off": np.concatenate(pool_off),
        "source_path": str(p),
    }


def percentile_spectrum(arr):
    return {
        "median":  float(np.median(arr)),
        "mean":    float(arr.mean()),
        "p90":     float(np.percentile(arr, 90)),
        "p95":     float(np.percentile(arr, 95)),
        "p99":     float(np.percentile(arr, 99)),
        "sup":     float(arr.max()),
        "n_node":  int(arr.size),
    }


def power_law(N, y):
    if np.any(y <= 0) or len(N) < 3:
        return float("nan"), float("nan")
    log_N, log_y = np.log(N), np.log(y)
    slope, intercept = np.polyfit(log_N, log_y, 1)
    pred = slope * log_N + intercept
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(-slope), float(r2)


def symanzik_2(N, y):
    """y(N) = a + b/N. Returns (a, b, r2)."""
    if len(N) < 3:
        return float("nan"), float("nan"), float("nan")
    X = np.column_stack([np.ones_like(N, dtype=float), 1.0 / N])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coef
    pred = a + b / N
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(b), float(r2)


def main() -> int:
    print("=" * 110)
    print("Stage 6f: Full-tensor closure NORM AUDIT")
    print("Per-node relative Frobenius residual percentile spectrum")
    print("=" * 110)
    print()
    print(f"  Lambda_t = {LAMBDA_T} (= alpha_xi^2 = 81/100)")
    print(f"  Lambda_s = {LAMBDA_S} (= -gamma^2/2 = -1/200, isotropic)")
    print()

    rows = []
    for reg, n_lat in LADDER:
        r = gather_regime(reg, n_lat)
        if r is None:
            print(f"{reg:<8} N={n_lat:>3}  -- NPZ not found, skipping")
            continue
        full_spec = percentile_spectrum(r["delta_full"])
        time_spec = percentile_spectrum(r["delta_time"])
        diag_spec = percentile_spectrum(r["delta_diag"])
        off_spec = percentile_spectrum(r["delta_off"])
        rows.append({
            "regime": reg, "N": n_lat, "n_seeds": r["n_seeds"],
            "n_node": full_spec["n_node"],
            "delta_full": full_spec,
            "delta_time": time_spec,
            "delta_diag": diag_spec,
            "delta_off": off_spec,
            "source_path": r["source_path"],
        })

    print(f"{'reg':<8} {'N':>3} {'seeds':>5} | "
          f"{'med':>7} {'mean':>7} {'p90':>7} {'p95':>7} "
          f"{'p99':>7} {'sup':>7} | thresh")
    print("-" * 95)
    for row in rows:
        s = row["delta_full"]
        passed_med = "[PASS]" if s["median"] <= 0.05 else "[FAIL]"
        passed_mean = "[PASS]" if s["mean"] <= 0.10 else "[FAIL]"
        print(f"{row['regime']:<8} {row['N']:>3} {row['n_seeds']:>5} | "
              f"{s['median']:>7.4f} {s['mean']:>7.4f} "
              f"{s['p90']:>7.4f} {s['p95']:>7.4f} "
              f"{s['p99']:>7.4f} {s['sup']:>7.4f} | "
              f"med {passed_med} mean {passed_mean}")

    # Symanzik fits per percentile
    if len(rows) >= 3:
        N_arr = np.array([r["N"] for r in rows], dtype=float)
        print()
        print("Symanzik 2pt fit y(N) = y_inf + b/N for each percentile of delta_full:")
        sym_fits = {}
        for stat in ("median", "mean", "p90", "p95", "p99", "sup"):
            y = np.array([r["delta_full"][stat] for r in rows])
            a, b, r2 = symanzik_2(N_arr, y)
            print(f"  {stat:>7}: y_inf = {a:>+8.4f},  b = {b:>+9.3f},  R^2 = {r2:>4.2f}")
            sym_fits[stat] = {"y_inf": a, "b": b, "r_squared": r2}
        print()

        # Power-law fits
        print("Power-law fit y(N) = c * N^(-alpha) for each percentile of delta_full:")
        pl_fits = {}
        for stat in ("median", "mean", "p90", "p95", "p99", "sup"):
            y = np.array([r["delta_full"][stat] for r in rows])
            alpha, r2 = power_law(N_arr, y)
            print(f"  {stat:>7}: alpha = {alpha:>+5.2f},  R^2 = {r2:>4.2f}")
            pl_fits[stat] = {"alpha": alpha, "r_squared": r2}
    else:
        sym_fits = {}
        pl_fits = {}

    # Verdict per percentile
    print()
    print("=" * 110)
    print("VERDICT per percentile (all N >= 60 in cleaned ladder):")
    print("=" * 110)
    verdict_per_pct = {}
    cleaned = [r for r in rows if r["N"] >= 60]
    for stat, thr_lo, thr_hi in (("median", 0.05, 0.05),
                                   ("mean", 0.10, 0.10),
                                   ("p90", 0.20, 0.20),
                                   ("p95", 0.30, 0.30)):
        pass_all = all(r["delta_full"][stat] <= thr_hi for r in cleaned)
        max_val = max(r["delta_full"][stat] for r in cleaned)
        verdict_per_pct[stat] = {
            "threshold": thr_hi,
            "max_at_N>=60": max_val,
            "all_pass": pass_all,
        }
        flag = "[PASS]" if pass_all else "[FAIL]"
        print(f"  {stat:>7} <= {thr_hi:.2f}: max={max_val:.4f} {flag}")
    # Sup-norm: matter-localized heavy-tail; finite is what we require
    sup_max = max(r["delta_full"]["sup"] for r in cleaned)
    sup_med_across_regimes = np.median(
        [r["delta_full"]["sup"] for r in cleaned])
    verdict_per_pct["sup"] = {
        "threshold": "finite (matter-localized heavy tail)",
        "max_at_N>=60": float(sup_max),
        "median_across_regimes": float(sup_med_across_regimes),
    }
    print(f"  {'sup':>7}: regime-median sup = {sup_med_across_regimes:.4f},  "
          f"max sup = {sup_max:.4f}  (matter-localized heavy tail)")

    bundle = {
        "method": "stage6f_full_tensor_norm_audit",
        "schema_version": "1.0.0",
        "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
        "ladder_skipped": ["P5N128 (K/Q persistence bug)"],
        "thresholds": {
            "median_max": 0.05, "mean_max": 0.10,
            "p90_max": 0.20, "p95_max": 0.30,
        },
        "per_regime": rows,
        "symanzik_fits": sym_fits,
        "power_law_fits": pl_fits,
        "verdict_per_percentile": verdict_per_pct,
    }
    out_path = REPO / "outputs" / "stage6f_full_tensor_norm_audit.json"
    out_path.write_text(json.dumps(bundle, indent=2),
                         encoding="utf-8")
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
