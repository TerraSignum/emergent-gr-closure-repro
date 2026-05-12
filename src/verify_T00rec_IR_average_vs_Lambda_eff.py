"""IR-average test: <T_00^rec> across regimes vs Lambda_eff = 0.314 (H3c v9).

If the reconstruction-load reformulation Lambda_t^back = T_00^rec is the
microscopic origin of the bulk-mean Lambda_eff, then per-regime
mean(T_00^rec) should converge across regimes to Lambda_eff.

Output: outputs/T00rec_IR_average_audit.json
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


REGIMES = [
    ("P1", 28), ("P3", 36), ("P4", 42), ("P5", 50), ("P5N64", 64),
    ("P6", 60), ("P7", 72), ("P8", 84), ("P5N100", 100),
    ("P5N72", 72), ("P5N84", 84), ("P6N128", 128), ("P8N128", 128),
    ("P5N128", 128), ("P5N200", 200),
    ("P5N256", 256), ("P5N300", 300), ("P5N512", 512),
]

LAMBDA_EFF_REF = 0.314


def per_regime_T00rec(reg, n_lat):
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
    g_pool, t_pool, trec_pool = [], [], []
    for s in range(min(n_seeds, 32)):
        xi_mat = get_xi(s)
        np.fill_diagonal(xi_mat, 1.0)
        psi = get_psi(s)
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        # T_00^rec = ζ_3*Ω*(A_K*K + A_Q*(1-Q)) — ζ_3=0.5, A_K=1, A_Q=0.5, Ω=1
        adj = prep.get("adj")
        if adj is None:
            xi_off = xi_mat.copy()
            np.fill_diagonal(xi_off, 0.0)
            adj = (xi_off > 0.6).astype(np.float64)
        deg = adj.sum(axis=1) + 1e-12
        K_per = (k_field*adj).sum(axis=1)/deg
        Q_per = (q_field*adj).sum(axis=1)/deg
        trec = 0.5*1.0*(1.0*K_per + 0.5*(1.0 - Q_per))
        g_pool.append(np.asarray(prep["g_00_h"]))
        t_pool.append(np.asarray(prep["t00"]))
        trec_pool.append(trec)
    g00 = np.concatenate(g_pool)
    t00 = np.concatenate(t_pool)
    trec = np.concatenate(trec_pool)
    mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g00) & np.isfinite(trec)
    g00 = g00[mask]; t00 = t00[mask]; trec = trec[mask]
    return {
        "regime": reg, "N": int(n_lat),
        "mean_T00":   float(np.mean(t00)),  "median_T00":   float(np.median(t00)),
        "mean_G00":   float(np.mean(g00)),  "median_G00":   float(np.median(g00)),
        "mean_T00rec":float(np.mean(trec)), "median_T00rec":float(np.median(trec)),
        "mean_T00_minus_G00": float(np.mean(t00-g00)),
    }


def main() -> int:
    print("="*100)
    print("IR-average test: <T_00^rec> per regime vs Lambda_eff = 0.314")
    print("="*100)
    rows = []
    for reg, n in REGIMES:
        r = per_regime_T00rec(reg, n)
        if r is not None:
            rows.append(r)
    print()
    print(f"{'regime':<10} {'N':>3} | {'<T_00>':>8} {'<G_00>':>8} {'<T_00^rec>':>11} {'<T-G>':>8}")
    print("-"*60)
    for r in rows:
        print(f"{r['regime']:<10} {r['N']:>3} | {r['mean_T00']:>8.4f} {r['mean_G00']:>8.4f} "
              f"{r['mean_T00rec']:>11.4f} {r['mean_T00_minus_G00']:>8.4f}")
    arr_T = np.array([r['mean_T00'] for r in rows])
    arr_G = np.array([r['mean_G00'] for r in rows])
    arr_R = np.array([r['mean_T00rec'] for r in rows])
    arr_TG = np.array([r['mean_T00_minus_G00'] for r in rows])
    print()
    print(f"  cross-regime mean(T_00)        = {arr_T.mean():.4f}  (std {arr_T.std():.4f}, CV {arr_T.std()/arr_T.mean()*100:.1f}%)")
    print(f"  cross-regime mean(G_00)        = {arr_G.mean():.4f}  (std {arr_G.std():.4f}, CV {arr_G.std()/arr_G.mean()*100:.1f}%)")
    print(f"  cross-regime mean(T_00^rec)    = {arr_R.mean():.4f}  (std {arr_R.std():.4f}, CV {arr_R.std()/arr_R.mean()*100:.1f}%)")
    print(f"  cross-regime mean(T_00 - G_00) = {arr_TG.mean():.4f}  (std {arr_TG.std():.4f}, CV {arr_TG.std()/abs(arr_TG.mean())*100:.1f}%)")
    print()
    print(f"  reference Lambda_eff (H3c v9)  = {LAMBDA_EFF_REF}")
    rel_R  = (arr_R.mean()  - LAMBDA_EFF_REF)/LAMBDA_EFF_REF*100
    rel_TG = (arr_TG.mean() - LAMBDA_EFF_REF)/LAMBDA_EFF_REF*100
    print(f"  rel-offset <T_00^rec>      vs Lambda_eff = {rel_R:+.1f}%")
    print(f"  rel-offset <T_00 - G_00>   vs Lambda_eff = {rel_TG:+.1f}%")
    consistent = abs(rel_R) < 10.0 and abs(rel_TG) < 10.0
    verdict = ("IR_AVERAGE_CONSISTENT" if consistent
               else "IR_AVERAGE_INCONSISTENT")
    print()
    print(f"  VERDICT: {verdict}  (threshold 10% rel offset)")
    out = {
        "method": "IR_average_T00rec_vs_Lambda_eff",
        "schema_version": "1.0.0",
        "Lambda_eff_reference": LAMBDA_EFF_REF,
        "per_regime": rows,
        "cross_regime": {
            "mean_T00": float(arr_T.mean()),  "std_T00": float(arr_T.std()),
            "mean_G00": float(arr_G.mean()),  "std_G00": float(arr_G.std()),
            "mean_T00rec": float(arr_R.mean()),"std_T00rec": float(arr_R.std()),
            "mean_T_minus_G": float(arr_TG.mean()),"std_T_minus_G": float(arr_TG.std()),
            "rel_offset_T00rec_vs_Lambda_eff_percent": float(rel_R),
            "rel_offset_T_minus_G_vs_Lambda_eff_percent": float(rel_TG),
        },
        "verdict": verdict,
    }
    out_path = REPO / "outputs" / "T00rec_IR_average_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
