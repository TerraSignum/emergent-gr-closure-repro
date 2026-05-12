"""Test if R_time becomes regime-invariant when Lambda_t is taken
as the regime-specific median(T_00 - G_00) instead of the
P5-canonical alpha_xi^2 = 0.81.

Hypothesis: Lambda_t = median_a(T_00(a) - G_00(a))_regime is the
correct per-regime asymptote. The System-R value 0.81 = alpha_xi^2
is the special-case asymptote in P5-physics where T_00 ~ 0.84.
For P6/P8 with T_00 ~ 0.42, Lambda_t should be ~0.42.

If this hypothesis holds:
  - per-regime median R_time at Lambda_t^regime should be regime-invariant
  - Lambda_t^regime / T_00_med^regime should be ~constant ratio across regimes

Output: outputs/per_regime_lambda_t_universal_audit.json
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
    ("P7", 72), ("P5N72", 72), ("P8", 84), ("P5N84", 84),
    ("P5N100", 100), ("P5N128", 128),
    ("P6N128", 128), ("P8N128", 128),
    ("P5N200", 200), ("P5N300", 300), ("P5N512", 512),
]
ALPHA_XI = 9.0 / 10.0


def main() -> int:
    rows = []
    for reg, n_lat in REGIMES:
        p = find_d1_npz(reg, REPO)
        if p is None or not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        # Accept both d1.npz (dense_cell_edge_xi_values) and snapshot
        # (edge_xi_snapshots) formats. For snapshots, use last frame.
        if "edge_xi_snapshots" in d.files:
            snaps = d["edge_xi_snapshots"]
            psi_re_snaps = d["psi_real_snapshots"]
            psi_im_snaps = d["psi_imag_snapshots"]
            last = snaps.shape[1] - 1
            edge_arr = snaps[:, last]            # (n_seeds, N, N)
            psi_arr = psi_re_snaps[:, last] + 1j * psi_im_snaps[:, last]
            edge_format = "snap"
        else:
            edge_arr = d["dense_cell_edge_xi_values"]
            amp_arr = d["dense_cell_node_amplitude_values"]
            phase_arr = d["dense_cell_node_phase_values"]
            psi_arr = amp_arr * np.exp(1j * phase_arr)
            edge_format = "d1"
        n_seeds = min(edge_arr.shape[0], 32)
        g00s, t00s = [], []
        for s in range(n_seeds):
            if edge_format == "snap":
                xi_mat = np.asarray(edge_arr[s], dtype=float).copy()
            else:
                xi_mat = edge_to_matrix(edge_arr[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            psi = psi_arr[s]
            k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
            q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
            prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
            g00s.append(np.asarray(prep["g_00_h"]))
            t00s.append(np.asarray(prep["t00"]))
        g00 = np.concatenate(g00s)
        t00 = np.concatenate(t00s)
        # Per-regime optimal Lambda_t
        lt_optimal = float(np.median(t00 - g00))
        t00_med = float(np.median(t00))
        # R_time at System-R Lambda_t = 0.81
        R_time_at_0_81 = g00 + 0.81 - t00
        R_time_med_at_0_81 = float(np.median(np.abs(R_time_at_0_81)))
        # R_time at regime-optimal Lambda_t
        R_time_at_opt = g00 + lt_optimal - t00
        R_time_med_at_opt = float(np.median(np.abs(R_time_at_opt)))
        # Ratio Lambda_t / T_00 — is this regime-invariant?
        ratio = lt_optimal / t00_med if t00_med > 0 else float("nan")
        rows.append({
            "regime": reg, "N": n_lat,
            "T_00_med": t00_med,
            "G_00_med": float(np.median(g00)),
            "Lambda_t_optimal": lt_optimal,
            "Lambda_t_over_T_00_ratio": ratio,
            "R_time_med_at_System_R_0p81": R_time_med_at_0_81,
            "R_time_med_at_optimal": R_time_med_at_opt,
        })

    print("=" * 110)
    print("Per-regime optimal Lambda_t (= median(T_00 - G_00)) vs System-R 0.81")
    print("=" * 110)
    print(f"{'reg':<10} {'N':>3} | {'T_00_med':>9} {'G_00_med':>9} {'Λ_t_opt':>10} {'Λ_t/T_00':>10} | "
          f"{'R_t @0.81':>11} {'R_t @opt':>10}")
    print("-" * 90)
    for r in rows:
        print(f"{r['regime']:<10} {r['N']:>3} | "
              f"{r['T_00_med']:>9.4f} {r['G_00_med']:>9.4f} "
              f"{r['Lambda_t_optimal']:>10.4f} {r['Lambda_t_over_T_00_ratio']:>10.4f} | "
              f"{r['R_time_med_at_System_R_0p81']:>11.5f} "
              f"{r['R_time_med_at_optimal']:>10.5f}")

    # Aggregate
    p5_regimes = [r for r in rows if "P5" in r["regime"]]
    other_regimes = [r for r in rows if "P5" not in r["regime"]]
    p5_ratios = [r["Lambda_t_over_T_00_ratio"] for r in p5_regimes]
    other_ratios = [r["Lambda_t_over_T_00_ratio"] for r in other_regimes]
    print()
    print(f"Mean Lambda_t/T_00 ratio across all regimes: {float(np.mean([r['Lambda_t_over_T_00_ratio'] for r in rows])):.4f}")
    print(f"  P5-only regimes: mean={float(np.mean(p5_ratios)):.4f}, std={float(np.std(p5_ratios)):.4f}")
    print(f"  non-P5 regimes:  mean={float(np.mean(other_ratios)):.4f}, std={float(np.std(other_ratios)):.4f}")
    print()
    R_at_0_81 = [r["R_time_med_at_System_R_0p81"] for r in rows]
    R_at_opt = [r["R_time_med_at_optimal"] for r in rows]
    print(f"R_time_med spread:")
    print(f"  at Lambda_t = 0.81 (System-R): max={float(max(R_at_0_81)):.5f}, range={float(max(R_at_0_81) - min(R_at_0_81)):.5f}")
    print(f"  at Lambda_t = optimal:         max={float(max(R_at_opt)):.5f}, range={float(max(R_at_opt) - min(R_at_opt)):.5f}")

    # Verdict
    if float(max(R_at_opt)) < 0.05 and float(max(R_at_opt) - min(R_at_opt)) < 0.02:
        verdict = "REGIME_INVARIANT_AT_PER_REGIME_LAMBDA_T"
    elif float(max(R_at_opt)) < 0.05:
        verdict = "ALL_BELOW_THRESHOLD_AT_PER_REGIME_LAMBDA_T"
    else:
        verdict = "PER_REGIME_LAMBDA_T_DOES_NOT_CLOSE"
    print(f"\nVERDICT: {verdict}")

    out = {
        "method": "per_regime_optimal_lambda_t_test",
        "schema_version": "1.0.0",
        "alpha_xi_squared_System_R": 0.81,
        "per_regime": rows,
        "verdict": verdict,
    }
    out_path = REPO / "outputs" / "per_regime_lambda_t_universal_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
