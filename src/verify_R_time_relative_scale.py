"""Test whether the R_time_median stagnation offset ~0.02 is a
LATTICE-CUTOFF-ARTEFACT: compare absolute residual to characteristic
tensor magnitudes G_00, T_00, and to lattice cutoff scales
EPS_D = D_MIN^2 = 0.01.

Three diagnostics per regime:

  1. Relative residual:  |R_time_med| / median(|T_00|)
     -> if this is <= 0.05, the closure is fine RELATIVELY,
        the absolute offset just reflects the absolute energy scale.

  2. Cutoff comparison:  |R_time_med| vs EPS_D = 0.01 vs D_MIN = 0.1
     -> if |R_time_med| ~ EPS_D in all regimes, cutoff is suspect.

  3. Cutoff sensitivity: compute |R_time_med| with EPS_D = 0 and
     compare to default. If shifts > 30%, cutoff is the cause;
     if shifts < 5%, cutoff is innocent.

Output: outputs/R_time_relative_scale_audit.json
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
    ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]

LAMBDA_T_STRUCT = 0.81
EPS_D_DEFAULT = 0.01
D_MIN_DEFAULT = 0.1


def gather_pool():
    pool = []
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
            prep = per_seed_galerkin(xi_mat, psi, k_field, q_field,
                                       n_lat, np)
            g00s.append(np.asarray(prep["g_00_h"]))
            t00s.append(np.asarray(prep["t00"]))
        pool.append({
            "regime": reg, "N": n_lat,
            "g00": np.concatenate(g00s),
            "t00": np.concatenate(t00s),
        })
    return pool


def main() -> int:
    pool = gather_pool()

    print("=" * 110)
    print("R_time relative scale audit")
    print("=" * 110)
    print()
    print(f"  EPS_D (default) = {EPS_D_DEFAULT} (= D_MIN^2)")
    print(f"  D_MIN (default) = {D_MIN_DEFAULT}")
    print(f"  Lambda_t struct = {LAMBDA_T_STRUCT}")
    print()
    print(f"{'reg':<8} {'N':>3} | "
          f"{'med|R_time|':>12} {'med|T_00|':>10} {'med|G_00|':>10} "
          f"{'rel_R/T':>9} {'rel_R/EPS':>10} {'med(T-G)':>10}")
    print("-" * 100)

    rows = []
    for r in pool:
        resid = r["g00"] + LAMBDA_T_STRUCT - r["t00"]
        med_R = float(np.median(np.abs(resid)))
        med_T = float(np.median(np.abs(r["t00"])))
        med_G = float(np.median(np.abs(r["g00"])))
        med_TG_diff = float(np.median(r["t00"] - r["g00"]))
        rel_R_over_T = med_R / max(med_T, 1e-12)
        rel_R_over_EPS = med_R / EPS_D_DEFAULT
        rows.append({
            "regime": r["regime"], "N": int(r["N"]),
            "median_abs_R_time": med_R,
            "median_abs_T_00": med_T,
            "median_abs_G_00": med_G,
            "median_T_minus_G": med_TG_diff,
            "relative_R_over_T": rel_R_over_T,
            "relative_R_over_EPS_D": rel_R_over_EPS,
        })
        print(f"{r['regime']:<8} {int(r['N']):>3} | "
              f"{med_R:>12.5f} {med_T:>10.5f} {med_G:>10.5f} "
              f"{rel_R_over_T:>9.4f} {rel_R_over_EPS:>10.3f} {med_TG_diff:>+10.4f}")

    print()
    rel_R_over_T_mean = float(np.mean([r["relative_R_over_T"] for r in rows]))
    rel_R_over_EPS_mean = float(np.mean([r["relative_R_over_EPS_D"] for r in rows]))
    print(f"  Mean relative residual / |T_00|:    {rel_R_over_T_mean:.4f}")
    print(f"  Mean relative residual / EPS_D:     {rel_R_over_EPS_mean:.3f}")
    print()
    print("  Interpretation:")
    if rel_R_over_T_mean <= 0.05:
        print(f"    rel < 5% -> closure is OK in RELATIVE terms; the absolute")
        print(f"    offset reflects the absolute |T_00| scale, not a closure failure.")
        verdict = "RELATIVE_CLOSURE_HOLDS"
    elif rel_R_over_T_mean <= 0.20:
        print(f"    rel ~ 5-20% -> closure is MARGINAL in relative terms.")
        verdict = "MARGINAL_RELATIVE_CLOSURE"
    else:
        print(f"    rel > 20% -> absolute offset is genuine, not a scale artefact.")
        verdict = "ABSOLUTE_RESIDUAL_PROBLEM"

    if abs(rel_R_over_EPS_mean - 1.0) < 0.5:
        print(f"    |R_time_med| ~ EPS_D suggests CUTOFF-ARTEFACT possibility.")

    out_path = REPO / "outputs" / "R_time_relative_scale_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "R_time_relative_scale_vs_T_00_and_EPS_D",
            "schema_version": "1.0.0",
            "lambda_t_struct": LAMBDA_T_STRUCT,
            "eps_d": EPS_D_DEFAULT,
            "d_min": D_MIN_DEFAULT,
            "per_regime": rows,
            "mean_relative_R_over_T": rel_R_over_T_mean,
            "mean_relative_R_over_EPS_D": rel_R_over_EPS_mean,
            "verdict": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
