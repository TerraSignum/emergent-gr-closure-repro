"""Skeptical end-to-end audit of the Pythagorean Lambda_t = alpha_xi^2 + gamma^2
claim against the now-cleaned 11-point ladder (9 original + 2 repaired large).

Tests:
  T1: Symanzik 2+4 asymptote on full ladder vs 0.820 prediction
  T2: Bootstrap CI on the asymptote — is 0.820 inside the 95% CI?
  T3: Per-regime variance — does the spread VS prediction make sense?
  T4: Compare Pythagorean (alpha^2 + gamma^2) vs naive (alpha^2) vs flat (mean)
  T5: Jackknife sensitivity — does removing any single regime change the verdict?

Output: outputs/skeptical_audit_pythagorean.json
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

PARENT = REPO.parent
ALPHA_XI = 9.0/10.0
GAMMA    = 1.0/10.0
PYTH = ALPHA_XI**2 + GAMMA**2  # 82/100
NAIVE = ALPHA_XI**2            # 81/100

# Standard 9 + repaired 2
SOURCE_LIST = [
    ("P1",     28, "canonical"),
    ("P3",     36, "canonical"),
    ("P4",     42, "canonical"),
    ("P5",     50, "canonical"),
    ("P5N64",  64, "canonical"),
    ("P6",     60, "canonical"),
    ("P7",     72, "canonical"),
    ("P8",     84, "canonical"),
    ("P5N100",100, "canonical"),
    ("P5N72",  72, "snapshot_v2"),
    ("P5N84",  84, "snapshot_v2"),
    ("P6N128",128, "snapshot_v2"),
    ("P8N128",128, "snapshot_v2"),
    ("P5N200",200, "snapshot_v2"),
    ("P5N256",256, "snapshot_v2"),
    ("P5N512", 512, "canonical"),
]


def lambda_from_canonical(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    keys = set(d.files)
    if "dense_cell_edge_xi_values" in keys:
        e = d["dense_cell_edge_xi_values"]
        a = d["dense_cell_node_amplitude_values"]
        ph = d["dense_cell_node_phase_values"]
        n_seeds = e.shape[0]
        get_xi = lambda s: edge_to_matrix(e[s], n_lat)
        get_psi = lambda s: a[s] * np.exp(1j*ph[s])
    elif "edge_xi_snapshots" in keys:
        snaps = d["edge_xi_snapshots"]
        psi_re = d["psi_real_snapshots"]
        psi_im = d["psi_imag_snapshots"]
        last = snaps.shape[1] - 1
        n_seeds = snaps.shape[0]
        get_xi = lambda s: np.asarray(snaps[s, last], dtype=float).copy()
        get_psi = lambda s: (np.asarray(psi_re[s, last], dtype=float)
                              + 1j*np.asarray(psi_im[s, last], dtype=float))
    else:
        return None
    g_pool, t_pool = [], []
    for s in range(min(n_seeds, 32)):
        xi_mat = get_xi(s)
        np.fill_diagonal(xi_mat, 1.0)
        psi = get_psi(s)
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        g_pool.append(np.asarray(prep["g_00_h"]))
        t_pool.append(np.asarray(prep["t00"]))
    g00 = np.concatenate(g_pool); t00 = np.concatenate(t_pool)
    mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g00)
    return float(np.mean(t00[mask] - g00[mask]))


def lambda_from_snapshot_v2(regime, n_lat):
    p = PARENT / f"results_d1_{regime.lower()}_v2" / f"{regime}.snapshots.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    n_seeds = int(d["n_seeds"][0])
    edge_last = d["edge_xi_snapshots"][:, -1, :, :]
    psi_re = d["psi_real_snapshots"][:, -1, :]
    psi_im = d["psi_imag_snapshots"][:, -1, :]
    g_pool, t_pool = [], []
    for s in range(n_seeds):
        xi_mat = edge_last[s].copy()
        np.fill_diagonal(xi_mat, 1.0)
        psi = psi_re[s] + 1j*psi_im[s]
        k_field = d[f"ff_K_seed{s}"]
        q_field = d[f"ff_Q_seed{s}"]
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        g_pool.append(np.asarray(prep["g_00_h"]))
        t_pool.append(np.asarray(prep["t00"]))
    g00 = np.concatenate(g_pool); t00 = np.concatenate(t_pool)
    mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g00)
    return float(np.mean(t00[mask] - g00[mask]))


def symanzik_24(N_arr, y_arr):
    A = np.column_stack([np.ones_like(N_arr), 1.0/N_arr**2, 1.0/N_arr**4])
    coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    pred = A @ coef
    ss_res = np.sum((y_arr - pred)**2)
    ss_tot = np.sum((y_arr - y_arr.mean())**2) + 1e-12
    R2 = 1.0 - ss_res/ss_tot
    return coef, R2


def main() -> int:
    print("="*100)
    print("Skeptical end-to-end audit of Lambda_t = alpha_xi^2 + gamma^2 = 0.820")
    print("="*100)

    rows = []
    for reg, n_lat, src in SOURCE_LIST:
        if src == "canonical":
            v = lambda_from_canonical(reg, n_lat)
        else:
            v = lambda_from_snapshot_v2(reg, n_lat)
        if v is not None:
            rows.append({"regime": reg, "N": n_lat, "src": src, "lambda_t_star": v})
    print(f"  {'regime':<10} {'N':>4} {'src':<14} {'Λt*':>8} {'rel-pyth':>9}")
    print("  " + "-"*55)
    for r in sorted(rows, key=lambda x: x["N"]):
        rel = (r["lambda_t_star"] - PYTH)/PYTH*100
        print(f"  {r['regime']:<10} {r['N']:>4} {r['src']:<14} {r['lambda_t_star']:>8.4f} {rel:>+8.2f}%")

    if len(rows) < 5:
        print("Not enough points for Symanzik fit.")
        return 1

    N_arr = np.array([r["N"] for r in rows], dtype=float)
    L_arr = np.array([r["lambda_t_star"] for r in rows])

    # T1: Symanzik 2+4 fit
    coef, R2 = symanzik_24(N_arr, L_arr)
    L_inf = float(coef[0])
    print()
    print(f"=== T1: Symanzik 2+4 fit (n={len(rows)}) ===")
    print(f"  Lambda_t^infty = {L_inf:.4f} (R^2 = {R2:.3f})")
    print(f"  Pythagorean    = {PYTH:.4f}, distance = {abs(L_inf-PYTH):.4f} ({abs(L_inf-PYTH)/PYTH*100:.2f}% rel)")
    print(f"  Naive alpha^2  = {NAIVE:.4f}, distance = {abs(L_inf-NAIVE):.4f} ({abs(L_inf-NAIVE)/NAIVE*100:.2f}% rel)")
    print(f"  Flat mean      = {L_arr.mean():.4f}, distance = {abs(L_arr.mean()-PYTH):.4f}")

    # T2: Bootstrap CI on asymptote
    rng = np.random.default_rng(42)
    n_boot = 1000
    L_inf_boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rows), size=len(rows))
        try:
            coef_b, _ = symanzik_24(N_arr[idx], L_arr[idx])
            L_inf_boot.append(coef_b[0])
        except np.linalg.LinAlgError:
            continue
    L_inf_boot = np.array(L_inf_boot)
    ci_lo, ci_hi = np.percentile(L_inf_boot, [2.5, 97.5])
    print()
    print(f"=== T2: Bootstrap 95% CI on Lambda_t^infty (n_boot={len(L_inf_boot)}) ===")
    print(f"  median = {np.median(L_inf_boot):.4f}")
    print(f"  95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]  (width {ci_hi-ci_lo:.4f})")
    pyth_in = ci_lo <= PYTH <= ci_hi
    naive_in = ci_lo <= NAIVE <= ci_hi
    print(f"  Pythagorean 0.820 inside CI? {pyth_in}")
    print(f"  Naive       0.810 inside CI? {naive_in}")

    # T3: per-regime spread analysis
    print()
    print(f"=== T3: per-regime spread ===")
    print(f"  mean Lambda_t* = {L_arr.mean():.4f} (std {L_arr.std():.4f}, CV {L_arr.std()/L_arr.mean()*100:.1f}%)")
    print(f"  range = [{L_arr.min():.4f}, {L_arr.max():.4f}]  (full spread {L_arr.max()-L_arr.min():.4f})")
    in_pyth_band = sum(0.80 <= L <= 0.84 for L in L_arr)
    print(f"  Within Pythagorean band [0.80, 0.84]: {in_pyth_band}/{len(L_arr)}")

    # T4: candidate comparison
    pyth_resid = L_arr - PYTH
    naive_resid = L_arr - NAIVE
    flat_resid = L_arr - L_arr.mean()
    print()
    print(f"=== T4: candidate residual comparison (per-regime) ===")
    print(f"  RMS distance to 0.820 (Pyth):   {np.sqrt((pyth_resid**2).mean()):.4f}")
    print(f"  RMS distance to 0.810 (naive):  {np.sqrt((naive_resid**2).mean()):.4f}")
    print(f"  RMS distance to mean (flat):    {np.sqrt((flat_resid**2).mean()):.4f}")
    print(f"  bias to Pyth: mean residual = {pyth_resid.mean():+.4f}")
    print(f"  bias to naive: mean residual = {naive_resid.mean():+.4f}")

    # T5: Jackknife — leave-one-out asymptote
    print()
    print(f"=== T5: Jackknife leave-one-out ===")
    asym_jack = []
    for i in range(len(rows)):
        idx = np.array([j for j in range(len(rows)) if j != i])
        coef_j, _ = symanzik_24(N_arr[idx], L_arr[idx])
        asym_jack.append(coef_j[0])
    asym_jack = np.array(asym_jack)
    print(f"  Asymptote range under leave-one-out: [{asym_jack.min():.4f}, {asym_jack.max():.4f}]")
    print(f"  Worst-case offset to Pyth: {abs(asym_jack - PYTH).max():.4f} ({abs(asym_jack - PYTH).max()/PYTH*100:.2f}% rel)")
    most_influential = int(np.argmax(np.abs(asym_jack - L_inf)))
    print(f"  Most-influential single point: {rows[most_influential]['regime']} (removing flips asymptote to {asym_jack[most_influential]:.4f})")

    # Honest verdict
    print()
    print("="*100)
    print("HONEST VERDICT")
    print("="*100)
    if abs(L_inf - PYTH) < 0.01 and pyth_in and (in_pyth_band > len(L_arr)/3):
        verdict = "PYTH_SUPPORTED"
    elif abs(L_inf - PYTH) < 0.02 and pyth_in:
        verdict = "PYTH_CONSISTENT_WEAK_PER_REGIME"
    else:
        verdict = "PYTH_NOT_CLEARLY_SUPPORTED"
    print(f"  {verdict}")

    out = {
        "method": "skeptical_pythagorean_audit",
        "predictions": {"pyth": PYTH, "naive": NAIVE},
        "per_regime": rows,
        "T1_symanzik24": {
            "Lambda_t_inf": L_inf, "R2": R2,
            "rel_pyth_pct":  abs(L_inf-PYTH)/PYTH*100,
            "rel_naive_pct": abs(L_inf-NAIVE)/NAIVE*100,
        },
        "T2_bootstrap": {
            "n_boot": int(len(L_inf_boot)),
            "median": float(np.median(L_inf_boot)),
            "ci_2_5": float(ci_lo), "ci_97_5": float(ci_hi),
            "pyth_in_CI": bool(pyth_in),
            "naive_in_CI": bool(naive_in),
        },
        "T3_per_regime": {
            "mean": float(L_arr.mean()),
            "std":  float(L_arr.std()),
            "min":  float(L_arr.min()),
            "max":  float(L_arr.max()),
            "in_pyth_band_count": int(in_pyth_band),
        },
        "T4_candidate_comparison": {
            "rms_pyth":  float(np.sqrt((pyth_resid**2).mean())),
            "rms_naive": float(np.sqrt((naive_resid**2).mean())),
            "rms_flat":  float(np.sqrt((flat_resid**2).mean())),
            "bias_pyth": float(pyth_resid.mean()),
            "bias_naive": float(naive_resid.mean()),
        },
        "T5_jackknife": {
            "asymptote_range": [float(asym_jack.min()), float(asym_jack.max())],
            "worst_case_offset_pct": float(abs(asym_jack - PYTH).max()/PYTH*100),
            "most_influential_regime": rows[most_influential]["regime"],
        },
        "verdict": verdict,
    }
    out_path = REPO / "outputs" / "skeptical_audit_pythagorean.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
