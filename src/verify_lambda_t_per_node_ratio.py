"""Critical falsification test: is Lambda_t(a)/T_00(a) ~ 0.97
PER-NODE or only globally?

If the ratio holds per-node:
  Lambda_t(a) := T_00(a) - G_00(a)        (per-node optimal)
  ratio(a)    := Lambda_t(a) / T_00(a)
  std(ratio(a)) << 0.1   ->  STRUCTURAL relation
  std(ratio(a)) >> 0.1   ->  global-fit artefact only

This decides whether the regime-universal observation
<Lambda_t/T_00> ≈ 0.97 is a STRUCTURAL identity Lambda_t(a) ∝ T_00(a)
node-by-node, or merely a coincidence of regime-medians.

Output: outputs/lambda_t_per_node_ratio_audit.json
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
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
    ("P6N128", 128), ("P8N128", 128),
]


def main() -> int:
    print("=" * 110)
    print("Per-node Lambda_t(a)/T_00(a) ratio — falsification test against global-fit artefact")
    print("=" * 110)
    print()
    print(f"{'reg':<10} {'N':>3} {'n_total':>8} | "
          f"{'<ratio>':>9} {'med ratio':>10} {'std ratio':>10} {'CV%':>7} | "
          f"{'spearman':>10} {'pearson':>9}")
    print("-" * 100)

    rows = []
    pool_t00 = []
    pool_lambda_t = []
    for reg, n_lat in REGIMES:
        p = find_d1_npz(reg, REPO)
        if p is None or not p.exists():
            continue
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
        g00 = np.concatenate(g00s)
        t00 = np.concatenate(t00s)
        # Per-node optimal Lambda_t(a) = T_00(a) - G_00(a)
        lambda_t = t00 - g00
        # Filter: only nodes with T_00 > small threshold (avoid div-by-near-zero)
        mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(lambda_t)
        lambda_t = lambda_t[mask]
        t00 = t00[mask]
        ratio = lambda_t / t00

        mean_r = float(np.mean(ratio))
        med_r = float(np.median(ratio))
        std_r = float(np.std(ratio))
        cv = (std_r / abs(mean_r)) * 100 if abs(mean_r) > 1e-12 else float("nan")

        # Per-node correlation: does Lambda_t(a) track T_00(a) linearly?
        if len(t00) > 2 and np.std(t00) > 0 and np.std(lambda_t) > 0:
            pearson_r = float(np.corrcoef(t00, lambda_t)[0, 1])
            rt = np.argsort(np.argsort(t00))
            rl = np.argsort(np.argsort(lambda_t))
            spearman_r = float(np.corrcoef(rt, rl)[0, 1])
        else:
            pearson_r = float("nan")
            spearman_r = float("nan")

        rows.append({
            "regime": reg, "N": n_lat, "n_nodes": int(len(t00)),
            "mean_ratio": mean_r, "median_ratio": med_r,
            "std_ratio": std_r, "CV_percent": cv,
            "spearman_lambda_t_vs_T_00": spearman_r,
            "pearson_lambda_t_vs_T_00": pearson_r,
        })
        pool_t00.append(t00)
        pool_lambda_t.append(lambda_t)

        print(f"{reg:<10} {n_lat:>3} {len(t00):>8} | "
              f"{mean_r:>9.4f} {med_r:>10.4f} {std_r:>10.4f} {cv:>7.2f} | "
              f"{spearman_r:>+10.4f} {pearson_r:>+9.4f}")

    # Pooled across all regimes
    t00_all = np.concatenate(pool_t00)
    lt_all = np.concatenate(pool_lambda_t)
    ratio_all = lt_all / t00_all
    pearson_all = float(np.corrcoef(t00_all, lt_all)[0, 1])
    rt = np.argsort(np.argsort(t00_all))
    rl = np.argsort(np.argsort(lt_all))
    spearman_all = float(np.corrcoef(rt, rl)[0, 1])

    # Linear fit lambda_t = a + b * t00
    slope, intercept = np.polyfit(t00_all, lt_all, 1)
    pred = slope * t00_all + intercept
    ss_res = float(((lt_all - pred) ** 2).sum())
    ss_tot = float(((lt_all - lt_all.mean()) ** 2).sum())
    r2_lin = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print()
    print(f"POOLED across all {len(rows)} regimes ({len(t00_all)} nodes):")
    print(f"  Pearson(Lambda_t, T_00):  {pearson_all:+.4f}")
    print(f"  Spearman(Lambda_t, T_00): {spearman_all:+.4f}")
    print(f"  Linear fit Lambda_t = {intercept:+.4f} + {slope:+.4f}·T_00,  R^2 = {r2_lin:.4f}")
    print(f"  Per-node ratio Lambda_t/T_00:")
    print(f"    mean = {float(np.mean(ratio_all)):.4f}, median = {float(np.median(ratio_all)):.4f}")
    print(f"    std  = {float(np.std(ratio_all)):.4f}, CV = {float(np.std(ratio_all)/abs(np.mean(ratio_all))*100):.2f}%")

    # Verdict
    cv_pooled = float(np.std(ratio_all) / abs(np.mean(ratio_all))) * 100
    if cv_pooled <= 10 and pearson_all >= 0.9:
        verdict = "STRUCTURAL_PER_NODE_PROPORTIONALITY_CONFIRMED"
    elif cv_pooled <= 25:
        verdict = "MODERATE_PER_NODE_PROPORTIONALITY"
    else:
        verdict = "GLOBAL_FIT_ONLY_NO_PER_NODE_STRUCTURE"
    print()
    print(f"VERDICT: {verdict}")
    print(f"  CV pooled = {cv_pooled:.2f}%, Pearson = {pearson_all:+.4f}")

    out_path = REPO / "outputs" / "lambda_t_per_node_ratio_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "per_node_lambda_t_over_T_00_falsification_test",
            "schema_version": "1.0.0",
            "per_regime": rows,
            "pooled": {
                "n_nodes_total": int(len(t00_all)),
                "pearson_lambda_t_vs_T_00": pearson_all,
                "spearman_lambda_t_vs_T_00": spearman_all,
                "linear_fit_intercept": float(intercept),
                "linear_fit_slope": float(slope),
                "linear_fit_R_squared": float(r2_lin),
                "mean_ratio": float(np.mean(ratio_all)),
                "median_ratio": float(np.median(ratio_all)),
                "std_ratio": float(np.std(ratio_all)),
                "CV_percent": cv_pooled,
            },
            "verdict": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
