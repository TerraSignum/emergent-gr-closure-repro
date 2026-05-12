"""Verify Pythagorean Lambda_t = alpha_xi^2 + gamma^2 = 0.820 on the
re-run large-N regimes that now carry stored ff_K_seed/ff_Q_seed.

We accept the new snapshot-format NPZs at:
  results_d1_p5n72_v2/P5N72.snapshots.npz
  results_d1_p5n84_v2/P5N84.snapshots.npz
  results_d1_p6n128_v2/P6N128.snapshots.npz
  results_d1_p8n128_v2/P8N128.snapshots.npz
  results_d1_p5n200_v2/P5N200.snapshots.npz
  results_d1_p5n256_12seeds/P5N256.snapshots.npz

Take last snapshot per seed, run per_seed_galerkin with the stored
ff_K_seed/ff_Q_seed (NOT defaults), compute per-regime Lambda_t* and
T_00^rec/T_00 ratio. Compare to Pythagorean prediction 0.820.

Output: outputs/repaired_largeN_pythagorean_audit.json
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

from verify_galerkin_runner_A_hessian_ricci import per_seed_galerkin

PARENT = REPO.parent
LATTICE_FILES = [
    ("P5N72",  72,  PARENT / "results_d1_p5n72_24seeds"   / "P5N72.snapshots.npz"),
    ("P5N84",  84,  PARENT / "results_d1_p5n84_24seeds"   / "P5N84.snapshots.npz"),
    ("P6N128", 128, PARENT / "results_d1_p6n128_12seeds"  / "P6N128.snapshots.npz"),
    ("P8N128", 128, PARENT / "results_d1_p8n128_12seeds"  / "P8N128.snapshots.npz"),
    ("P5N200", 200, PARENT / "results_d1_p5n200_8seeds"   / "P5N200.snapshots.npz"),
    ("P5N256", 256, PARENT / "results_d1_p5n256_12seeds"  / "P5N256.snapshots.npz"),
    ("P5N300", 300, PARENT / "results_d1_p5n300_12seeds"  / "P5N300.snapshots.npz"),
    ("P5N512", 512, PARENT / "results_d1_p5n512_12seeds"  / "P5N512.snapshots.npz"),
]
ALPHA_XI = 9.0/10.0
GAMMA    = 1.0/10.0
LAMBDA_T_R_PYTH = ALPHA_XI**2 + GAMMA**2  # 82/100 = 0.820
LAMBDA_T_R_NAIVE = ALPHA_XI**2            # 81/100 = 0.810


def analyze(reg, n_lat, p):
    if not p.exists():
        return None, "FILE_MISSING"
    try:
        d = np.load(p, allow_pickle=True)
    except Exception as e:
        return None, f"LOAD_ERROR:{e}"
    keys = list(d.keys())
    has_ff = any(k.startswith("ff_K_seed") for k in keys)
    if not has_ff:
        return None, "NO_FF_K_SEED"
    if "edge_xi_snapshots" not in keys:
        return None, "NO_SNAPSHOTS"
    n_seeds = int(d["n_seeds"][0]) if hasattr(d["n_seeds"], "__len__") else int(d["n_seeds"])
    edge_last = d["edge_xi_snapshots"][:, -1, :, :]
    psi_re = d["psi_real_snapshots"][:, -1, :]
    psi_im = d["psi_imag_snapshots"][:, -1, :]

    g_pool, t_pool, trec_pool = [], [], []
    K_means = []; Q_means = []
    for s in range(n_seeds):
        xi_mat = edge_last[s].copy()
        np.fill_diagonal(xi_mat, 1.0)
        psi = psi_re[s] + 1j*psi_im[s]
        k_field = d[f"ff_K_seed{s}"]
        q_field = d[f"ff_Q_seed{s}"]
        K_means.append(float(k_field.mean()))
        Q_means.append(float(q_field.mean()))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        # Decompose
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
    if len(t00) == 0:
        return None, "EMPTY_AFTER_MASK"

    ratio = trec/t00
    lambda_t_star = float(np.mean(t00 - g00))
    cand_B = np.abs(g00 + trec - t00) / np.maximum(np.abs(t00), 1e-9)

    return {
        "regime": reg, "N": int(n_lat),
        "n_seeds": int(n_seeds),
        "n_nodes": int(len(t00)),
        "mean_K_field": float(np.mean(K_means)),
        "mean_Q_field": float(np.mean(Q_means)),
        "mean_T00":    float(np.mean(t00)),
        "mean_T00rec": float(np.mean(trec)),
        "mean_G00":    float(np.mean(g00)),
        "lambda_t_star": lambda_t_star,
        "T00rec_over_T00_med": float(np.median(ratio)),
        "T00rec_over_T00_CV":  float(np.std(ratio)/np.mean(ratio)*100),
        "candidate_B_residual": {
            "median": float(np.median(cand_B)),
            "mean":   float(np.mean(cand_B)),
            "p90":    float(np.percentile(cand_B, 90)),
        },
        "rel_offset_pyth_pct":  float((lambda_t_star - LAMBDA_T_R_PYTH)/LAMBDA_T_R_PYTH*100),
        "rel_offset_naive_pct": float((lambda_t_star - LAMBDA_T_R_NAIVE)/LAMBDA_T_R_NAIVE*100),
    }, "OK"


def main() -> int:
    print("=" * 110)
    print("Repaired large-N regimes: Pythagorean Lambda_t = alpha_xi^2 + gamma^2 = 0.820 verification")
    print("=" * 110)
    print(f"  Pythagorean prediction: {LAMBDA_T_R_PYTH:.4f}")
    print(f"  Naive prediction:       {LAMBDA_T_R_NAIVE:.4f}")
    print()
    print(f"{'regime':<8} {'N':>4} {'status':<18} {'mean_K':>7} {'<T00>':>7} {'<T00^rec>':>10} {'<G00>':>7} {'Λt*':>7} {'rel-pyth':>9} {'B-med':>7}")
    print("-" * 105)
    rows = []
    for reg, n, p in LATTICE_FILES:
        r, status = analyze(reg, n, p)
        if r is None:
            print(f"{reg:<8} {n:>4} {status:<18}  —")
            continue
        rows.append(r)
        print(f"{reg:<8} {n:>4} {'OK':<18} {r['mean_K_field']:>7.3f} {r['mean_T00']:>7.4f} {r['mean_T00rec']:>10.4f} {r['mean_G00']:>7.4f} {r['lambda_t_star']:>7.4f} {r['rel_offset_pyth_pct']:>+8.2f}% {r['candidate_B_residual']['median']:>7.4f}")

    if rows:
        L_arr = np.array([r["lambda_t_star"] for r in rows])
        N_arr = np.array([r["N"] for r in rows])
        print()
        print(f"  cross-regime mean(Lambda_t*) = {L_arr.mean():.4f} (std {L_arr.std():.4f}, CV {L_arr.std()/abs(L_arr.mean())*100:.1f}%)")
        print(f"  vs Pythagorean 0.820:    rel = {(L_arr.mean()-LAMBDA_T_R_PYTH)/LAMBDA_T_R_PYTH*100:+.2f}%")
        print(f"  vs naive    0.810:       rel = {(L_arr.mean()-LAMBDA_T_R_NAIVE)/LAMBDA_T_R_NAIVE*100:+.2f}%")
        # Is each regime now in the Pythagorean band [0.80, 0.84]?
        in_band = sum(0.80 <= r["lambda_t_star"] <= 0.84 for r in rows)
        print(f"  Regimes within Pythagorean band [0.80, 0.84]: {in_band}/{len(rows)}")

    out = {
        "method": "repaired_largeN_pythagorean_verification",
        "pythagorean_prediction":  LAMBDA_T_R_PYTH,
        "naive_prediction":        LAMBDA_T_R_NAIVE,
        "per_regime": rows,
    }
    out_path = REPO / "outputs" / "repaired_largeN_pythagorean_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
