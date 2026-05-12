"""
Reviewer follow-up C: regular/core decomposition.

For each regime in the cleaned ten-regime ladder we partition the
per-node residuals into a regular set R_N(tau) and a core set
C_N(tau) at threshold tau:

    R_N(tau) = { nodes a : delta_full_a <= tau }
    C_N(tau) = { nodes a : delta_full_a >  tau }

We measure mu_N(C_N(tau)) := |C_N(tau)| / N_node_a (the fraction of
nodes belonging to the core) at four canonical thresholds
tau in {0.05, 0.10, 0.20, 0.50} per regime, and report whether the
core fraction decays with N. Symanzik-2 fit f(N) = f_inf + b/N is
applied to the core-fraction trajectory under each tau.

The complementary regular-set residual statistic is the median of
delta_full restricted to R_N(tau); its convergence to zero in the
continuum corroborates the bulk-closure verdict.

Output: outputs/stage6f_regular_core_decomposition.json
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

from stage6f_full_tensor_norm_audit import LADDER, gather_regime  # noqa: E402

OUT = REPO / "outputs" / "stage6f_regular_core_decomposition.json"
THRESHOLDS = [0.05, 0.10, 0.20, 0.50]


def symanzik_2(n_arr, y_arr):
    if len(n_arr) < 3:
        return float("nan"), float("nan"), float("nan")
    x = 1.0 / np.asarray(n_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    coef = np.polyfit(x, y, 1)
    b, a = coef[0], coef[1]
    pred = a + b * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), r2


def main():
    rows = []
    for regime, n in LADDER:
        data = gather_regime(regime, n)
        if data is None:
            continue
        delta_full = data["delta_full"]
        n_node = len(delta_full)
        rec = {
            "regime": regime,
            "N": n,
            "n_seeds": data["n_seeds"],
            "n_node_total": n_node,
            "core_fraction": {},
            "regular_set_median": {},
            "regular_set_max": {},
            "core_residual_sum_per_node": {},
            "core_residual_sumsq_per_node": {},
        }
        for tau in THRESHOLDS:
            mask_core = delta_full > tau
            n_core = int(mask_core.sum())
            rec["core_fraction"][f"{tau:.2f}"] = n_core / n_node
            if n_core < n_node:
                regular = delta_full[~mask_core]
                rec["regular_set_median"][f"{tau:.2f}"] = float(np.median(regular))
                rec["regular_set_max"][f"{tau:.2f}"] = float(regular.max())
            else:
                rec["regular_set_median"][f"{tau:.2f}"] = float("nan")
                rec["regular_set_max"][f"{tau:.2f}"] = float("nan")
            if n_core > 0:
                core_vals = delta_full[mask_core]
                rec["core_residual_sum_per_node"][f"{tau:.2f}"] = (
                    float(core_vals.sum()) / n_node)
                rec["core_residual_sumsq_per_node"][f"{tau:.2f}"] = (
                    float((core_vals ** 2).sum()) / n_node)
            else:
                rec["core_residual_sum_per_node"][f"{tau:.2f}"] = 0.0
                rec["core_residual_sumsq_per_node"][f"{tau:.2f}"] = 0.0
        rows.append(rec)

    rows.sort(key=lambda r: r["N"])

    summary = {
        "method": "regular/core decomposition with Symanzik-2 extrapolation",
        "thresholds": THRESHOLDS,
        "ladder_used": [{"regime": r["regime"], "N": r["N"], "n_seeds": r["n_seeds"]}
                         for r in rows],
        "per_regime": rows,
        "core_fraction_symanzik_fits": {},
        "regular_set_median_symanzik_fits": {},
        "regular_set_max_symanzik_fits": {},
        "core_residual_sum_per_node_symanzik_fits": {},
        "core_residual_sumsq_per_node_symanzik_fits": {},
    }

    n_arr = np.array([r["N"] for r in rows], dtype=float)
    for tau in THRESHOLDS:
        key = f"{tau:.2f}"
        cf = np.array([r["core_fraction"][key] for r in rows], dtype=float)
        a, b, r2 = symanzik_2(n_arr, cf)
        summary["core_fraction_symanzik_fits"][key] = {
            "f_inf": a, "b": b, "r_squared": r2}
        rm = np.array([r["regular_set_median"][key] for r in rows], dtype=float)
        if not np.isnan(rm).any():
            a, b, r2 = symanzik_2(n_arr, rm)
            summary["regular_set_median_symanzik_fits"][key] = {
                "y_inf": a, "b": b, "r_squared": r2}
        rmax = np.array([r["regular_set_max"][key] for r in rows], dtype=float)
        if not np.isnan(rmax).any():
            a, b, r2 = symanzik_2(n_arr, rmax)
            summary["regular_set_max_symanzik_fits"][key] = {
                "y_inf": a, "b": b, "r_squared": r2}
        cs = np.array([r["core_residual_sum_per_node"][key] for r in rows],
                       dtype=float)
        a, b, r2 = symanzik_2(n_arr, cs)
        summary["core_residual_sum_per_node_symanzik_fits"][key] = {
            "y_inf": a, "b": b, "r_squared": r2}
        cs2 = np.array([r["core_residual_sumsq_per_node"][key] for r in rows],
                        dtype=float)
        a, b, r2 = symanzik_2(n_arr, cs2)
        summary["core_residual_sumsq_per_node_symanzik_fits"][key] = {
            "y_inf": a, "b": b, "r_squared": r2}

    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print()
    print("Per-regime core fractions mu_N(C_N(tau)):")
    print(f"{'regime':>10s} {'N':>4s}", end="")
    for tau in THRESHOLDS:
        print(f"  tau={tau:4.2f}", end="")
    print()
    for r in rows:
        print(f"{r['regime']:>10s} {r['N']:>4d}", end="")
        for tau in THRESHOLDS:
            print(f"   {r['core_fraction'][f'{tau:.2f}']:.4f}", end="")
        print()
    print()
    print("Symanzik-2 continuum fits f(N) = f_inf + b/N for core fraction:")
    for tau in THRESHOLDS:
        f = summary["core_fraction_symanzik_fits"][f"{tau:.2f}"]
        print(f"  tau={tau:4.2f}: f_inf = {f['f_inf']:+8.4f},  "
              f"b = {f['b']:+8.3f},  R^2 = {f['r_squared']:.2f}")
    print()
    print("Core residual SUM per node (Sigma_{C_N} Delta_a / |V_N|):")
    for tau in THRESHOLDS:
        f = summary["core_residual_sum_per_node_symanzik_fits"][f"{tau:.2f}"]
        print(f"  tau={tau:4.2f}: y_inf = {f['y_inf']:+8.4f},  "
              f"b = {f['b']:+8.3f},  R^2 = {f['r_squared']:.2f}")
    print()
    print("Core residual SUM-OF-SQUARES per node (Sigma_{C_N} Delta_a^2 / |V_N|):")
    for tau in THRESHOLDS:
        f = summary["core_residual_sumsq_per_node_symanzik_fits"][f"{tau:.2f}"]
        print(f"  tau={tau:4.2f}: y_inf = {f['y_inf']:+8.5f},  "
              f"b = {f['b']:+8.3f},  R^2 = {f['r_squared']:.2f}")
    print()
    print("Regular-set MAX residual (max_{R_N} Delta_a):")
    for tau in THRESHOLDS:
        if f"{tau:.2f}" in summary["regular_set_max_symanzik_fits"]:
            f = summary["regular_set_max_symanzik_fits"][f"{tau:.2f}"]
            print(f"  tau={tau:4.2f}: y_inf = {f['y_inf']:+8.4f},  "
                  f"b = {f['b']:+8.3f},  R^2 = {f['r_squared']:.2f}")


if __name__ == "__main__":
    main()
